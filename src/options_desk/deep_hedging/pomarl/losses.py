"""
REINFORCE policy loss + AIS reward / transition auxiliary losses.

Both losses are pure functions of ``(log_probs, rewards, ...)`` and the
detached prediction targets, so they can be wrapped in ``jax.value_and_grad``
against any subset of params without surprises.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .utils import discounted_returns


def reinforce_loss(
    log_probs: jnp.ndarray,        # (T, B)
    rewards: jnp.ndarray,          # (T, B)
    discount: float = 1.0,
    baseline_kind: str = "batch_mean",
    entropy: jnp.ndarray | None = None,   # (T, B) optional
    entropy_coef: float = 0.0,
) -> tuple[jnp.ndarray, dict]:
    """Score-function policy loss with a batch-mean baseline.

    Negates the expected discounted return so ``jax.grad`` against
    policy params gives the standard REINFORCE update::

        ∇_θ J ≈ -∇_θ E[ Σ_t log π_θ(a_t | x̂_t) · (G_t − b_t) ]
    """
    G = discounted_returns(rewards, discount)
    if baseline_kind == "batch_mean":
        baseline = G.mean(axis=1, keepdims=True)
    elif baseline_kind == "zero":
        baseline = jnp.zeros_like(G)
    else:
        raise ValueError(
            f"baseline_kind must be 'batch_mean' or 'zero', got {baseline_kind!r}"
        )
    advantage = jax.lax.stop_gradient(G - baseline)
    policy_term = -(log_probs * advantage).sum(axis=0).mean()
    loss = policy_term
    info = {
        "policy_term": policy_term,
        "mean_return": G[0].mean(),     # E[G_0] = expected total reward
        "mean_log_prob": log_probs.mean(),
        "mean_advantage": advantage.mean(),
        "std_advantage": advantage.std(),
    }
    if entropy is not None and entropy_coef > 0.0:
        entropy_term = -entropy_coef * entropy.mean()
        loss = loss + entropy_term
        info["entropy"] = entropy.mean()
        info["entropy_term"] = entropy_term
    return loss, info


def ais_reward_loss(
    x_hat: jnp.ndarray,           # (T, B, H) – encoder output at each step
    actions: jnp.ndarray,         # (T, B, N)
    rewards: jnp.ndarray,         # (T, B)
    reward_model_apply,           # closure: (params, x_hat, action) -> r_hat
    reward_params,
) -> jnp.ndarray:
    """Mean-squared reward-prediction loss.

    Targets are detached (no gradient through ``rewards``).
    """
    target = jax.lax.stop_gradient(rewards)
    r_hat = reward_model_apply(reward_params, x_hat, actions)
    return jnp.mean((r_hat - target) ** 2)


def ais_transition_loss(
    x_hat: jnp.ndarray,           # (T+1, B, H)
    actions: jnp.ndarray,         # (T, B, N)
    transition_model_apply,       # closure: (params, x_hat, action) -> (mu, log_sigma)
    transition_params,
) -> jnp.ndarray:
    """Diagonal-Gaussian NLL on next-AIS prediction.

    Prediction target ``x̂_{t+1}`` is stop-gradded so the encoder is updated
    only via its appearance in ``x̂_t`` (input side), not as the regression
    label — matching Algorithm 1's auxiliary self-prediction loss.
    """
    x_t = x_hat[:-1]
    x_next = jax.lax.stop_gradient(x_hat[1:])
    mu, log_sigma = transition_model_apply(transition_params, x_t, actions)
    sigma = jnp.exp(log_sigma)
    # Per-dim Gaussian NLL up to constant.
    per_dim = 0.5 * ((x_next - mu) / sigma) ** 2 + log_sigma
    return per_dim.mean()
