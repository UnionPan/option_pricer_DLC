"""
PPO core for the JAX POMARL stack.

This is the strong cumulative-reward RL baseline the project's negative
result needs to be defensible: "RL underperforms supervised deep hedging"
is only credible if the RL side includes PPO (clipped surrogate + GAE +
a learned value baseline), not just REINFORCE / naive pathwise-SVG.

The two ingredients our earlier cumulative-reward runs lacked are exactly
what PPO adds:

    * a learned critic V(x̂)  — a *per-state* control-variate baseline,
      strictly more expressive than the batch-mean baseline that the
      Buehler/supervised loss gets for free. This is the direct attack on
      the gradient-estimator-variance gap.
    * GAE                     — credit assignment for the temporally
      concentrated (gamma-spike) reward.

Recurrent-PPO-done-right, made tractable by one observation: in this
exogenous-noise hedging MDP the **observation stream is a deterministic
function of the stored actions** (positions = cumsum of trades), so a
collected trajectory can be *replayed* through the encoder/policy/critic to
recompute log-probs, entropies and values under updated params — without
re-sampling the market or the actions. That lets us train the GRU encoder
end-to-end through the actor+critic losses (so no AIS auxiliary losses are
needed; this matches the "end-to-end gradient is what works" finding).

Natural-gradient note: preconditioning this PPO update with KFAC is exactly
ACKTR (A2C + KFAC) — the reconnection to the project's original
``kfac_optimizer.py`` goal. Left as a follow-on; this file uses Adam.

All functions are pure (closures over the flax ``apply`` callables) so they
JIT cleanly and differentiate against any subset of params.
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from .ais import AISGRUEncoder
from .utils import build_pomdp_obs

# Differential entropy of a unit-variance 1-D Gaussian: 0.5·log(2πe).
_GAUSSIAN_ENTROPY_CONST = 0.5 * math.log(2.0 * math.pi * math.e)


class ValueCritic(nn.Module):
    """State-value head V(x̂): 2-layer MLP over the AIS state → scalar."""

    hidden_size: int = 64

    @nn.compact
    def __call__(self, x_hat):
        h = nn.relu(nn.Dense(self.hidden_size, name="fc1")(x_hat))
        h = nn.relu(nn.Dense(self.hidden_size, name="fc2")(h))
        v = nn.Dense(1, name="head")(h)
        return v[..., 0]


def tanh_gaussian_log_prob_entropy(
    mu: jnp.ndarray,            # (..., N)
    log_std: jnp.ndarray,       # (..., N)
    action: jnp.ndarray,        # (..., N) — post-tanh, scaled by position_limit
    mask: jnp.ndarray,          # (..., N) — float {0,1}; masked dims contribute 0
    position_limit: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Recompute log π(a|s) and per-step entropy for a *fixed* tanh-squashed
    Gaussian action.

    Mirrors :func:`policy.sample_action` exactly but evaluates the density of
    a given ``action`` (recovering the pre-tanh ``z`` via ``arctanh``) instead
    of sampling. Masked dims are zeroed in both the log-prob and the entropy,
    so neither the surrogate nor the entropy bonus moves forbidden slots.
    """
    sigma = jnp.exp(log_std)
    # Recover tanh(z) = action / L; clip into the open interval before arctanh.
    a_scaled = jnp.clip(action / position_limit, -1.0 + 1e-6, 1.0 - 1e-6)
    z = jnp.arctanh(a_scaled)

    log_prob_z = -0.5 * (((z - mu) / sigma) ** 2 + 2.0 * log_std
                         + jnp.log(2.0 * jnp.pi))
    # action = L·tanh(z) ⇒ log|da/dz| = log(L) + log(1 − tanh(z)²)
    jacobian = jnp.log(position_limit) + jnp.log(1.0 - a_scaled * a_scaled + 1e-6)
    per_dim = (log_prob_z - jacobian) * mask
    log_prob = per_dim.sum(axis=-1)

    # Entropy of the *pre-tanh* diagonal Gaussian (standard PPO entropy bonus;
    # the tanh Jacobian term is action-dependent and conventionally dropped).
    ent_per_dim = (log_std + _GAUSSIAN_ENTROPY_CONST) * mask
    entropy = ent_per_dim.sum(axis=-1)
    return log_prob, entropy


def compute_gae(
    rewards: jnp.ndarray,    # (T, B)
    values: jnp.ndarray,     # (T+1, B) — includes bootstrap V_T
    discount: float,
    gae_lambda: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generalized Advantage Estimation.

    Returns ``(advantages, returns)``, both ``(T, B)`` with
    ``returns = advantages + V_t`` (the value-regression targets).
    """
    v_t = values[:-1]      # (T, B)
    v_tp1 = values[1:]     # (T, B)
    deltas = rewards + discount * v_tp1 - v_t

    def step(adv_next, delta_t):
        adv = delta_t + discount * gae_lambda * adv_next
        return adv, adv

    _, advs_rev = jax.lax.scan(step, jnp.zeros_like(deltas[0]), deltas[::-1])
    advantages = advs_rev[::-1]
    returns = advantages + v_t
    return advantages, returns


def replay_forward(
    *,
    encoder_apply: Callable,
    policy_apply: Callable,
    critic_apply: Callable,
    encoder_params,
    policy_params,
    critic_params,
    spots: jnp.ndarray,            # (B, T+1)
    prices: jnp.ndarray,           # (B, T+1, N)
    masks: jnp.ndarray,            # (T+1, N) float
    actions: jnp.ndarray,          # (T, B, N) — stored trades (post-override)
    instrument_mask_static: jnp.ndarray | None,
    position_limit: float,
    horizon: int,
    n_instruments: int,
    encoder_hidden_size: int,
    encoder_n_layers: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Replay a collected trajectory through encoder+policy+critic.

    Because positions evolve deterministically from the stored ``actions``,
    the observation at every step is reconstructable, so this re-runs the
    full recurrent forward pass differentiably (gradients flow into the GRU
    encoder, the policy, and the critic) to produce, for the *same* actions:

        new_log_probs  (T, B)     — log π_θ(a_t | x̂_t) under current params
        entropies      (T, B)     — per-step policy entropy
        values         (T+1, B)   — V(x̂_t), with a bootstrap V_T at the end
    """
    B = spots.shape[0]
    spots_t = jnp.swapaxes(spots[:, :horizon], 0, 1)     # (T, B)
    prices_t = jnp.swapaxes(prices[:, :horizon], 0, 1)   # (T, B, N)
    masks_t = masks[:horizon]                            # (T, N)
    masks_t_b = jnp.broadcast_to(
        masks_t[:, None, :], (horizon, B, n_instruments),
    )
    time_idx = jnp.arange(horizon, dtype=jnp.int32)

    init_positions = jnp.zeros((B, n_instruments), dtype=jnp.float32)
    init_prev = jnp.zeros((B, n_instruments), dtype=jnp.float32)
    init_hidden = AISGRUEncoder.init_hidden(
        B, encoder_hidden_size, encoder_n_layers,
    )

    def body(carry, xs):
        positions, prev_trades, hidden = carry
        spot_t, prices_step, mask_t, action_t, t_idx = xs

        obs = build_pomdp_obs(
            spot_t=spot_t,
            option_prices_t=prices_step[:, 1:],
            positions=positions,
            previous_trades=prev_trades,
            time_index=t_idx,
            horizon=horizon,
        )
        x_hat, new_hidden = encoder_apply(encoder_params, obs, hidden)

        if instrument_mask_static is not None:
            mask_eff = mask_t * instrument_mask_static
        else:
            mask_eff = mask_t

        mu, log_std = policy_apply(policy_params, x_hat, mask_eff)
        log_prob, entropy = tanh_gaussian_log_prob_entropy(
            mu, log_std, action_t, mask_eff, position_limit,
        )
        value = critic_apply(critic_params, x_hat)   # (B,)

        new_positions = positions + action_t
        new_carry = (new_positions, action_t, new_hidden)
        return new_carry, (log_prob, entropy, value)

    xs = (spots_t, prices_t, masks_t_b, actions, time_idx)
    final_carry, (log_probs, entropies, values_t) = jax.lax.scan(
        body, (init_positions, init_prev, init_hidden), xs,
    )

    # Bootstrap value V_T from the post-terminal encoder state.
    final_positions, final_prev, final_hidden = final_carry
    obs_T = build_pomdp_obs(
        spot_t=spots[:, horizon],
        option_prices_t=prices[:, horizon, 1:],
        positions=final_positions,
        previous_trades=final_prev,
        time_index=horizon,
        horizon=horizon,
    )
    x_hat_T, _ = encoder_apply(encoder_params, obs_T, final_hidden)
    value_T = critic_apply(critic_params, x_hat_T)              # (B,)
    values = jnp.concatenate([values_t, value_T[None]], axis=0)  # (T+1, B)
    return log_probs, entropies, values


class PPOLossInfo(NamedTuple):
    loss: jnp.ndarray
    policy_loss: jnp.ndarray
    value_loss: jnp.ndarray
    entropy: jnp.ndarray
    approx_kl: jnp.ndarray
    clip_frac: jnp.ndarray


def ppo_loss(
    *,
    new_log_probs: jnp.ndarray,   # (T, B)
    old_log_probs: jnp.ndarray,   # (T, B)
    advantages: jnp.ndarray,      # (T, B) — pre-normalization
    returns: jnp.ndarray,         # (T, B)
    new_values: jnp.ndarray,      # (T+1, B)
    old_values: jnp.ndarray,      # (T+1, B)
    entropies: jnp.ndarray,       # (T, B)
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
) -> tuple[jnp.ndarray, PPOLossInfo]:
    """Clipped PPO surrogate + clipped value loss − entropy bonus."""
    # Normalize advantages over the minibatch (standard PPO).
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    ratio = jnp.exp(new_log_probs - old_log_probs)
    unclipped = ratio * adv
    clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))

    # Clipped value loss (PPO2-style).
    v_pred = new_values[:-1]
    v_old = old_values[:-1]
    v_clipped = v_old + jnp.clip(v_pred - v_old, -clip_eps, clip_eps)
    value_loss = jnp.mean(
        jnp.maximum((v_pred - returns) ** 2, (v_clipped - returns) ** 2)
    )

    entropy = jnp.mean(entropies)
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    # Diagnostics.
    approx_kl = jnp.mean(old_log_probs - new_log_probs)
    clip_frac = jnp.mean(
        (jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32)
    )
    info = PPOLossInfo(
        loss=loss, policy_loss=policy_loss, value_loss=value_loss,
        entropy=entropy, approx_kl=approx_kl, clip_frac=clip_frac,
    )
    return loss, info
