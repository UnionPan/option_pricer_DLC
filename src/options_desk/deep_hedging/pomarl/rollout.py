"""
JAX-native POMARL rollout.

Drives the AIS encoder + tanh-Gaussian policy through a batch of pre-sampled
market trajectories via ``jax.lax.scan``. Two variants share the same body:

    * :func:`stochastic_rollout` — samples actions (training);
    * :func:`greedy_rollout`     — uses ``tanh(μ)`` (evaluation).

Reward construction follows the standard rebalancing-period MTM convention::

    V_t   = cash_t + positions_t · prices_{t+1}      (cash already net of costs)
    r_t   = V_t − V_{t-1}                            (cost implicit in cash flow)
    r_T-1 −= payoff_T                                (terminal liability settle)

so ``Σ_t r_t = terminal_PnL − initial_cash − payoff_T``, which is exactly
the Buehler-form ``mean_reward`` reported by :class:`BuehlerTrainer.evaluate`
— giving us a clean identity between POMARL's optimization objective
(``E[Σ r]``) and the existing eval pipeline.

Encoder gradients flow through ``x_hat_seq``; the policy receives
``stop_gradient(x_hat)``, so policy updates do not perturb the encoder
(Algorithm 1, Single-loop AIS-based REINFORCE).
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import jax.scipy.stats as jss

from ..utils.contracts import LiabilitySpec
from .ais import AISGRUEncoder
from .policy import mean_action, sample_action
from .utils import build_pomdp_obs, jax_total_payoff


def bs_call_jax(
    S: jnp.ndarray,
    K: float,
    T_rem: jnp.ndarray,
    sigma: jnp.ndarray,
    r: float = 0.0,
) -> jnp.ndarray:
    """JAX-native BS call price (vectorized, jit-compatible)."""
    eps = 1e-8
    T_rem = jnp.maximum(T_rem, eps)
    sigma = jnp.maximum(sigma, eps)
    sqrt_T = jnp.sqrt(T_rem)
    d1 = (jnp.log(S / K) + (r + 0.5 * sigma * sigma) * T_rem) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return S * jss.norm.cdf(d1) - K * jnp.exp(-r * T_rem) * jss.norm.cdf(d2)


def bs_put_jax(
    S: jnp.ndarray,
    K: float,
    T_rem: jnp.ndarray,
    sigma: jnp.ndarray,
    r: float = 0.0,
) -> jnp.ndarray:
    """JAX-native BS put price via put-call parity."""
    call = bs_call_jax(S, K, T_rem, sigma, r)
    eps = 1e-8
    T_rem = jnp.maximum(T_rem, eps)
    return call - S + K * jnp.exp(-r * T_rem)


def _liability_value_at_steps(
    legs: Sequence[LiabilitySpec],
    spots: jnp.ndarray,       # (B, T+1)
    variances: jnp.ndarray,   # (B, T+1)
    horizon: int,
    dt: float,
) -> jnp.ndarray:
    """Approximate per-step BS value of the liability portfolio.

    Returns (T+1, B) array of liability values, where index 0 = t=0
    and index T = terminal payoff.

    For vanilla call/put legs, uses BS with σ = √v_t and remaining T = (maturity − t)·dt.
    Other leg kinds get only the terminal payoff (set to NaN at t<T to flag).
    This is intended as a SHAPING potential, not an exact price, so the
    BS approximation in Heston is acceptable (errors are second-order in v).
    """
    B, T1 = spots.shape
    L = jnp.zeros((T1, B), dtype=spots.dtype)
    for leg in legs:
        if leg.kind not in ("call", "put"):
            # Path-dependent legs: shaping not supported, contributes only at T
            continue
        K = float(leg.strike)
        mat = int(leg.maturity)
        qty = float(leg.quantity)
        kind = leg.kind
        t_grid = jnp.arange(T1, dtype=spots.dtype)        # (T+1,)
        steps_remaining = jnp.maximum(mat - t_grid, 0.0)  # (T+1,)
        T_rem = steps_remaining * dt                      # (T+1,)
        # Broadcast: shapes need to be (T+1, B)
        S = spots.T                                       # (T+1, B)
        sigma = jnp.sqrt(jnp.maximum(variances.T, 1e-12))  # (T+1, B)
        T_rem_b = T_rem[:, None]                          # (T+1, 1)
        if kind == "call":
            leg_value = bs_call_jax(S, K, T_rem_b, sigma)
        else:
            leg_value = bs_put_jax(S, K, T_rem_b, sigma)
        # At expiry (t >= maturity), force intrinsic payoff exactly
        expired = (t_grid >= mat)[:, None]
        if kind == "call":
            intrinsic = jnp.maximum(S - K, 0.0)
        else:
            intrinsic = jnp.maximum(K - S, 0.0)
        leg_value = jnp.where(expired, intrinsic, leg_value)
        L = L + qty * leg_value
    return L  # (T+1, B)


class RolloutOutputs(NamedTuple):
    """Per-trajectory outputs from a POMARL rollout."""
    x_hat_seq: jnp.ndarray        # (T+1, B, H)
    actions: jnp.ndarray          # (T, B, N)
    log_probs: jnp.ndarray        # (T, B)
    rewards: jnp.ndarray          # (T, B)
    costs: jnp.ndarray            # (T, B)
    terminal_pnl: jnp.ndarray     # (B,)
    total_costs: jnp.ndarray      # (B,)
    payoff: jnp.ndarray           # (B,)


def _step_body(
    *,
    encoder_apply: Callable,
    policy_apply: Callable,
    encoder_params,
    policy_params,
    transaction_cost_rates: jnp.ndarray,    # (N,)
    instrument_mask_static: jnp.ndarray | None,
    position_limit: float,
    horizon: int,
    n_instruments: int,
    sample: bool,
    discrete_bucket_size: float = 0.0,
    encoder_gradient_from_policy: bool = False,
):
    """Build a scan body closed over network + config."""

    def body(carry, xs):
        positions, cash, prev_trades, hidden, prev_V = carry
        spot_t, prices_t, prices_tp1, mask_t, key_t, time_idx = xs

        # ── 1. Build variance-hidden observation ───────────────────────────
        option_prices_t = prices_t[:, 1:]
        obs = build_pomdp_obs(
            spot_t=spot_t,
            option_prices_t=option_prices_t,
            positions=positions,
            previous_trades=prev_trades,
            time_index=time_idx,
            horizon=horizon,
        )

        # ── 2. AIS encoder ─────────────────────────────────────────────────
        x_hat, new_hidden = encoder_apply(encoder_params, obs, hidden)

        # ── 3. Policy on detached AIS ──────────────────────────────────────
        if instrument_mask_static is not None:
            mask_eff = mask_t * instrument_mask_static
        else:
            mask_eff = mask_t

        # Standard AIS keeps encoder updated only via aux losses
        # (stop_gradient). Hybrid mode lets the policy loss flow through
        # the encoder too — only enable when the encoder is meant to be
        # trained end-to-end with the policy objective.
        x_hat_for_policy = (
            x_hat if encoder_gradient_from_policy
            else jax.lax.stop_gradient(x_hat)
        )
        mu, log_std = policy_apply(
            policy_params, x_hat_for_policy, mask_eff,
        )

        # ── 4. Sample / greedy action + mask-liquidate forbidden slots ────
        if sample:
            action, log_prob = sample_action(
                mu, log_std, mask_eff, key_t, position_limit,
            )
        else:
            action = mean_action(mu, mask_eff, position_limit)
            log_prob = jnp.zeros(action.shape[0], dtype=action.dtype)

        # Masked dimensions: liquidate any residual position (mirrors Buehler).
        mask_bool = mask_eff > 0.5
        action = jnp.where(mask_bool, action, -positions)
        if discrete_bucket_size > 0.0:
            # NON-DIFFERENTIABLE: round trades to lot size before applying.
            # REINFORCE-style POMARL still gets gradient via log_prob of the
            # CONTINUOUS pre-rounding action; pathwise gradient through this
            # round() is zero — same failure mode as LSTM with hard rounding.
            action = jnp.round(action / discrete_bucket_size) * discrete_bucket_size

        # ── 5. Trade physics ───────────────────────────────────────────────
        traded_notional = action * prices_t
        notional = traded_notional.sum(axis=-1)
        cost = (transaction_cost_rates * jnp.abs(traded_notional)).sum(axis=-1)
        new_cash = cash - notional - cost
        new_positions = positions + action

        # ── 6. MTM at next prices + reward ─────────────────────────────────
        V = new_cash + (new_positions * prices_tp1).sum(axis=-1)
        reward = V - prev_V

        new_carry = (new_positions, new_cash, action, new_hidden, V)
        outputs = (x_hat, action, log_prob, reward, cost, V)
        return new_carry, outputs

    return body


def _final_obs(
    *,
    encoder_apply: Callable,
    encoder_params,
    final_positions: jnp.ndarray,
    final_prev_trades: jnp.ndarray,
    final_hidden,
    spot_T: jnp.ndarray,
    prices_T: jnp.ndarray,
    horizon: int,
) -> jnp.ndarray:
    """Encoder pass at the post-terminal observation (provides x̂_T target)."""
    obs_T = build_pomdp_obs(
        spot_t=spot_T,
        option_prices_t=prices_T[:, 1:],
        positions=final_positions,
        previous_trades=final_prev_trades,
        time_index=horizon,
        horizon=horizon,
    )
    x_hat_T, _ = encoder_apply(encoder_params, obs_T, final_hidden)
    return x_hat_T


def pomarl_rollout(
    *,
    encoder_apply: Callable,
    policy_apply: Callable,
    encoder_params,
    policy_params,
    spots: jnp.ndarray,            # (B, T+1)
    prices: jnp.ndarray,           # (B, T+1, N)
    masks: jnp.ndarray,            # (T+1, N) float
    transaction_cost_rates: jnp.ndarray,   # (N,)
    instrument_mask_static: jnp.ndarray | None,
    liability_legs: Sequence[LiabilitySpec],
    initial_cash: float,
    position_limit: float,
    horizon: int,
    n_instruments: int,
    encoder_hidden_size: int,
    encoder_n_layers: int,
    sample: bool,
    key: jax.Array,
    reward_kind: str = "mean_pnl",
    risk_aversion: float = 1.0,
    variances: jnp.ndarray | None = None,   # (B, T+1) — needed for hedging_shaped
    dt: float = 1.0 / 252.0,
    discrete_bucket_size: float = 0.0,
    encoder_gradient_from_policy: bool = False,
) -> RolloutOutputs:
    """Run one POMARL rollout over a batch of market paths."""
    B = spots.shape[0]
    keys = jax.random.split(key, horizon)

    # Per-step inputs the scan walks through
    spots_t   = jnp.swapaxes(spots[:, :horizon], 0, 1)    # (T, B)
    prices_t  = jnp.swapaxes(prices[:, :horizon], 0, 1)   # (T, B, N)
    prices_tp = jnp.swapaxes(prices[:, 1:], 0, 1)         # (T, B, N)
    masks_t   = masks[:horizon]                           # (T, N)
    masks_t_b = jnp.broadcast_to(masks_t[:, None, :], (horizon, B, n_instruments))
    time_idx  = jnp.arange(horizon, dtype=jnp.int32)

    initial_positions = jnp.zeros((B, n_instruments), dtype=jnp.float32)
    initial_prev = jnp.zeros((B, n_instruments), dtype=jnp.float32)
    initial_hidden = AISGRUEncoder.init_hidden(
        B, encoder_hidden_size, encoder_n_layers,
    )
    initial_cash_b = jnp.full((B,), float(initial_cash), dtype=jnp.float32)
    initial_V = initial_cash_b   # V_{−1} := initial_cash (no positions)

    body = _step_body(
        encoder_apply=encoder_apply,
        policy_apply=policy_apply,
        encoder_params=encoder_params,
        policy_params=policy_params,
        transaction_cost_rates=transaction_cost_rates,
        instrument_mask_static=instrument_mask_static,
        position_limit=position_limit,
        horizon=horizon,
        n_instruments=n_instruments,
        sample=sample,
        discrete_bucket_size=discrete_bucket_size,
        encoder_gradient_from_policy=encoder_gradient_from_policy,
    )

    carry_init = (initial_positions, initial_cash_b, initial_prev,
                  initial_hidden, initial_V)
    xs = (spots_t, prices_t, prices_tp, masks_t_b, keys, time_idx)
    final_carry, (x_hat_t, actions, log_probs, rewards, costs, V_per_step) = jax.lax.scan(
        body, carry_init, xs,
    )
    final_positions, final_cash, final_prev, final_hidden, V_final = final_carry

    # The scan's raw `rewards` output is the per-step hedge MTM minus cost:
    #   rewards_scan[t] = ΔV_hedge_t − cost_t   (derived in mean_pnl notes)
    # Recover the pure MTM hedge increment for the local-risk reward.
    hedge_pnl_increment = rewards + costs            # (T, B) = ΔV_hedge_t

    # Reward shaping: original 'mean_pnl' vs hedging-aware alternatives.
    payoff = jax_total_payoff(tuple(liability_legs), spots)
    if reward_kind == "mean_pnl":
        # Original: per-step r = ΔV, terminal r -= payoff.
        # E[Σ r] = E[V_T − payoff] = E[hedging_error] — alpha-harvesting.
        rewards = rewards.at[-1].add(-payoff)
    elif reward_kind == "hedging_mse":
        # Föllmer-Schweizer minimum-variance hedging.
        # Per-step r = -cost; terminal r += -γ·(V_T − payoff)².
        # E[Σ r] = -E[cost] − γ·E[(error)²]. Maximizing → minimize
        # E[cost] + γ·MSE(error). Terminal-only spike has high variance.
        terminal_error = V_final - payoff
        rewards = -costs
        rewards = rewards.at[-1].add(-risk_aversion * (terminal_error ** 2))
    elif reward_kind == "hedging_var":
        # Pure variance (no mean penalty), centered on batch mean.
        terminal_error = V_final - payoff
        batch_mean_err = jnp.mean(terminal_error)
        rewards = -costs
        centered = terminal_error - batch_mean_err
        rewards = rewards.at[-1].add(-risk_aversion * (centered ** 2))
    elif reward_kind == "hedging_shaped":
        # Ng-Russell potential-function shaping. Let Φ(s_t) = -γ·(V_t - L_t)²
        # where L_t = BS-priced liability value at time t (computed outside
        # the policy gradient — purely path-dependent).
        # Per-step reward = -cost_t + (Φ(s_t) - Φ(s_{t-1})).
        # Σ reward = -Σ cost + (Φ_T - Φ_0)
        #          = -Σ cost - γ·(V_T - payoff)² + γ·(V_0 - L_0)²
        # Same expected sum as hedging_mse plus a constant — but the gradient
        # is distributed across all T steps, giving the AIS encoder a dense
        # low-variance target instead of one terminal spike.
        if variances is None:
            raise ValueError(
                "reward_kind='hedging_shaped' requires variances argument "
                "(needed to BS-price the liability at each step)"
            )
        L_all = _liability_value_at_steps(
            liability_legs, spots, variances, horizon, dt,
        )  # (T+1, B)
        # V_per_step from scan is shape (T, B) — V at end of each step
        # V_{-1} := initial_cash (no positions). V_0 ... V_{T-1} from scan.
        V_init_row = jnp.full((1, B), float(initial_cash), dtype=V_per_step.dtype)
        V_full = jnp.concatenate([V_init_row, V_per_step], axis=0)  # (T+1, B)
        phi_full = -risk_aversion * (V_full - L_all) ** 2          # (T+1, B)
        shaping = phi_full[1:] - phi_full[:-1]                     # (T, B)
        rewards = -costs + shaping
    elif reward_kind == "local_risk":
        # Föllmer-Schweizer / local risk minimization via the Doob
        # martingale decomposition.  Let e_t = V_t − L_t (tracking error),
        # L_t = E_Q[payoff | F_t] the option price path. Per-step error
        # increment:
        #     Δe_t = ΔV_hedge_t − ΔL_t
        # Under Q (r=0), both ΔV_hedge_t (instrument prices are Q-martingales)
        # and ΔL_t (claim price is a Q-martingale) are martingale differences,
        # so they're uncorrelated across t. Therefore:
        #     E[ Σ_t (Δe_t)² ] = Var( Σ_t Δe_t ) = Var(e_T) = Var(hedging error)
        # i.e. the SUM OF SQUARED INCREMENTS equals the terminal variance —
        # NOT the square of the summed increments (which would be the
        # mean-inclusive MSE that 'hedging_shaped' telescopes to).
        # Reward = -γ·(Δe_t)² - cost_t  →  E[Σ r] = -(γ·Var + E[cost]).
        if variances is None:
            raise ValueError(
                "reward_kind='local_risk' requires variances argument "
                "(needed to price the liability path L_t at each step)"
            )
        L_all = _liability_value_at_steps(
            liability_legs, spots, variances, horizon, dt,
        )  # (T+1, B)
        dL = L_all[1:] - L_all[:-1]                       # (T, B) = ΔL_t
        delta_e = hedge_pnl_increment - dL               # (T, B) = Δe_t
        rewards = -risk_aversion * (delta_e ** 2) - costs
    else:
        raise ValueError(
            f"unknown reward_kind={reward_kind!r}; expected one of "
            f"'mean_pnl', 'hedging_mse', 'hedging_var', 'hedging_shaped', "
            f"'local_risk'"
        )

    # Post-terminal encoder pass — gives x̂_T for the transition NLL target.
    x_hat_T = _final_obs(
        encoder_apply=encoder_apply,
        encoder_params=encoder_params,
        final_positions=final_positions,
        final_prev_trades=final_prev,
        final_hidden=final_hidden,
        spot_T=spots[:, horizon],
        prices_T=prices[:, horizon],
        horizon=horizon,
    )
    x_hat_seq = jnp.concatenate([x_hat_t, x_hat_T[None]], axis=0)

    terminal_pnl = V_final
    total_costs = costs.sum(axis=0)

    return RolloutOutputs(
        x_hat_seq=x_hat_seq,
        actions=actions,
        log_probs=log_probs,
        rewards=rewards,
        costs=costs,
        terminal_pnl=terminal_pnl,
        total_costs=total_costs,
        payoff=payoff,
    )


def stochastic_rollout(*args, **kwargs) -> RolloutOutputs:
    """Sampled-action rollout for training."""
    kwargs["sample"] = True
    return pomarl_rollout(*args, **kwargs)


def greedy_rollout(*args, **kwargs) -> RolloutOutputs:
    """Deterministic (mean-action) rollout for evaluation."""
    kwargs["sample"] = False
    return pomarl_rollout(*args, **kwargs)
