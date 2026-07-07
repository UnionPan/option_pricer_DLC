"""
Batched evaluation of analytical Black-Scholes hedgers.

Mirrors :meth:`BuehlerTrainer.evaluate` so the returned metric dict is
shape-compatible — same keys, same definitions — and can be compared
directly against trained-policy evals on the same eval seed (and hence
byte-identical market paths via the shared sampler).

Supports three hedger variants:

    'delta'             — underlying only, BS delta hedge of liability
    'delta_gamma'       — underlying + one ATM-ish option from the grid;
                          option position matches liability gamma, then
                          underlying position is adjusted for residual delta
    'delta_gamma_vega'  — underlying + two options; jointly solve a 2x2
                          system to neutralize {gamma, vega}, then adjust
                          underlying for residual delta

All variants vectorize across the path batch in NumPy and use the
multi-instrument cash/position/cost accounting from
:func:`_replay_multi_instrument_batch`. Action-mask semantics match the
trainer's ``_enforce_trade_mask`` (masked → liquidate).

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Literal

import numpy as np
from scipy.stats import norm

from .contracts import LiabilitySpec, MarketTrajectory, TrajectoryBatch


MarketBatchSampler = Callable[[int, Any], TrajectoryBatch]
HedgeKind = Literal["delta", "delta_gamma", "delta_gamma_vega"]


def _bs_delta_batch(
    S: np.ndarray,
    K: float,
    tau: np.ndarray,
    sigma: np.ndarray,
    r: float,
    kind: str,
) -> np.ndarray:
    """Black-Scholes delta, vectorized over (B, T) arrays.

    Matches :func:`AnalyticalDeltaHedger._bs_delta_gamma` semantics for the
    delta component (gamma is unused here — the underlying-only baseline
    does not gamma-hedge).
    """
    delta = np.zeros_like(S, dtype=np.float64)
    expired = tau <= 0.0
    if kind == "call":
        delta[expired] = np.where(
            S[expired] > K, 1.0, np.where(S[expired] == K, 0.5, 0.0)
        )
    else:
        delta[expired] = np.where(
            S[expired] < K, -1.0, np.where(S[expired] == K, -0.5, 0.0)
        )

    alive = ~expired
    if alive.any():
        S_a = S[alive]
        sigma_a = np.maximum(sigma[alive], 1e-8)
        tau_a = np.maximum(tau[alive], 1e-12)
        sqrt_tau = np.sqrt(tau_a)
        d1 = (np.log(S_a / K) + (r + 0.5 * sigma_a ** 2) * tau_a) / (
            sigma_a * sqrt_tau + 1e-12
        )
        if kind == "call":
            delta[alive] = norm.cdf(d1)
        else:
            delta[alive] = norm.cdf(d1) - 1.0
    return delta


def _bs_greeks_batch(
    S: np.ndarray,
    K: float | np.ndarray,
    tau: np.ndarray,
    sigma: np.ndarray,
    r: float,
    kind: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Black-Scholes (delta, gamma, vega), vectorized over (B, T) arrays.

    K may be a scalar or broadcast-compatible with S. Returns three arrays
    of the same shape as S.
    """
    K_arr = np.broadcast_to(K, S.shape)
    delta = np.zeros_like(S, dtype=np.float64)
    gamma = np.zeros_like(S, dtype=np.float64)
    vega = np.zeros_like(S, dtype=np.float64)

    expired = tau <= 0.0
    if kind == "call":
        delta[expired] = np.where(
            S[expired] > K_arr[expired], 1.0,
            np.where(S[expired] == K_arr[expired], 0.5, 0.0),
        )
    else:
        delta[expired] = np.where(
            S[expired] < K_arr[expired], -1.0,
            np.where(S[expired] == K_arr[expired], -0.5, 0.0),
        )

    alive = ~expired
    if alive.any():
        S_a = S[alive]
        K_a = K_arr[alive]
        sigma_a = np.maximum(sigma[alive], 1e-8)
        tau_a = np.maximum(tau[alive], 1e-12)
        sqrt_tau = np.sqrt(tau_a)
        d1 = (np.log(S_a / K_a) + (r + 0.5 * sigma_a ** 2) * tau_a) / (
            sigma_a * sqrt_tau + 1e-12
        )
        phi = np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi)
        if kind == "call":
            delta[alive] = norm.cdf(d1)
        else:
            delta[alive] = norm.cdf(d1) - 1.0
        gamma[alive] = phi / (S_a * sigma_a * sqrt_tau + 1e-12)
        vega[alive] = S_a * sqrt_tau * phi
    return delta, gamma, vega


def _select_hedging_options(
    option_grid_spec: list[dict],
    liability_maturity: int,
    n_options: int,
) -> list[dict]:
    """Pick the best ``n_options`` calls for hedging from the grid spec.

    Heuristic: score = abs(tau_steps - liability_maturity) + 100·|moneyness − 1|.
    Returns ``n_options`` calls sorted ascending by score (best first).
    """
    calls = [s for s in option_grid_spec if s["kind"] == "call"]
    if not calls:
        raise ValueError("option_grid_spec contains no call options")
    if len(calls) < n_options:
        raise ValueError(
            f"need {n_options} hedging options, grid only has {len(calls)} calls"
        )
    calls_sorted = sorted(
        calls,
        key=lambda s: (
            abs(s["tau_steps"] - liability_maturity)
            + 100.0 * abs(s["moneyness"] - 1.0)
        ),
    )
    # Prefer DIVERSE tenors when picking 2+ (avoids near-singular {gamma,vega} matrix)
    if n_options >= 2:
        chosen = [calls_sorted[0]]
        for cand in calls_sorted[1:]:
            if all(c["tau_steps"] != cand["tau_steps"] for c in chosen):
                chosen.append(cand)
            if len(chosen) == n_options:
                break
        if len(chosen) < n_options:
            chosen += [
                c for c in calls_sorted
                if c not in chosen
            ][: n_options - len(chosen)]
        return chosen
    return calls_sorted[:n_options]


def _replay_multi_instrument_batch(
    target_positions: np.ndarray,         # (T, B, N)
    prices: np.ndarray,                   # (B, T+1, N)
    masks: np.ndarray,                    # (T+1, N)  (or (B, T+1, N))
    transaction_cost_rates: np.ndarray,   # (N,)
    initial_cash: float,
    horizon: int,
    n_instruments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run multi-instrument cash/position accounting vectorized over the batch.

    Returns (terminal_pnl, total_costs, terminal_positions), each (B,) or
    (B, N) as appropriate. Mask semantics match the trainer's
    ``_enforce_trade_mask`` — when an instrument is not tradable at time t,
    the trade is forced to ``-positions`` (liquidate to zero).
    """
    B = prices.shape[0]
    cash = np.full(B, float(initial_cash), dtype=np.float64)
    positions = np.zeros((B, n_instruments), dtype=np.float64)
    total_costs = np.zeros(B, dtype=np.float64)
    tc = np.asarray(transaction_cost_rates, dtype=np.float64)

    if masks.ndim == 2:
        # broadcast (T+1, N) -> per-step lookup
        get_mask = lambda t: masks[t]
    elif masks.ndim == 3:
        get_mask = lambda t: masks[:, t]
    else:
        raise ValueError(f"masks must be 2-D or 3-D, got shape {masks.shape}")

    for t in range(horizon):
        mask_t = get_mask(t)
        target_t = target_positions[t]                  # (B, N)
        if mask_t.ndim == 1:
            tradable = mask_t.astype(bool)              # (N,)
            target_masked = np.where(tradable[None, :], target_t, 0.0)
        else:
            tradable = mask_t.astype(bool)              # (B, N)
            target_masked = np.where(tradable, target_t, 0.0)
        # Standard target_position semantics: trade = target - current,
        # then for masked instruments force trade = -positions (liquidate).
        trade = target_masked - positions
        if mask_t.ndim == 1:
            trade = np.where(tradable[None, :], trade, -positions)
        else:
            trade = np.where(tradable, trade, -positions)

        prices_t = prices[:, t]                         # (B, N)
        notional = (trade * prices_t).sum(axis=1)
        step_cost = (tc[None, :] * np.abs(trade * prices_t)).sum(axis=1)
        cash = cash - notional - step_cost
        positions = positions + trade
        total_costs = total_costs + step_cost

    prices_T = prices[:, horizon]
    terminal_pnl = cash + (positions * prices_T).sum(axis=1)
    return terminal_pnl, total_costs, positions


def _terminal_call_put_payoff(spots_terminal: np.ndarray, leg: LiabilitySpec) -> np.ndarray:
    if leg.kind == "call":
        return leg.quantity * np.maximum(spots_terminal - leg.strike, 0.0)
    if leg.kind == "put":
        return leg.quantity * np.maximum(leg.strike - spots_terminal, 0.0)
    raise ValueError(
        f"evaluate_analytical_delta_hedger_batch supports only call/put liabilities, "
        f"got kind={leg.kind!r}"
    )


def _left_tail_mean(values: np.ndarray, quantile: float) -> float:
    threshold = float(np.quantile(values, quantile))
    tail = values[values <= threshold]
    if tail.size == 0:
        return threshold
    return float(tail.mean())


def _error_distribution_metrics(
    errors: np.ndarray, prefix: str = ""
) -> Dict[str, float]:
    abs_err = np.abs(errors)
    mse = float(np.mean(errors * errors))
    return {
        f"{prefix}mean_hedging_error": float(errors.mean()),
        f"{prefix}std_hedging_error": float(errors.std(ddof=0)),
        f"{prefix}mae_hedging_error": float(abs_err.mean()),
        f"{prefix}mse_hedging_error": mse,
        f"{prefix}rmse_hedging_error": float(np.sqrt(mse)),
        f"{prefix}p01_hedging_error": float(np.quantile(errors, 0.01)),
        f"{prefix}p05_hedging_error": float(np.quantile(errors, 0.05)),
        f"{prefix}p50_hedging_error": float(np.quantile(errors, 0.50)),
        f"{prefix}p95_hedging_error": float(np.quantile(errors, 0.95)),
        f"{prefix}p99_hedging_error": float(np.quantile(errors, 0.99)),
        f"{prefix}cvar_05_hedging_error": _left_tail_mean(errors, 0.05),
    }


def evaluate_analytical_hedger_batch(
    market_sampler: MarketBatchSampler,
    horizon: int,
    n_instruments: int,
    transaction_cost_rates: np.ndarray,
    liability: LiabilitySpec,
    hedge_kind: HedgeKind = "delta",
    option_grid_spec: list[dict] | None = None,
    dt: float = 1.0 / 252.0,
    n_paths: int = 4096,
    eval_seed: int = 9999,
    risk_free_rate: float = 0.0,
    initial_cash: float = 0.0,
    risk_aversion: float = 1000.0,
    variance_source: str = "oracle",
    ewma_lambda: float = 0.94,
    initial_variance_prior: float = 0.04,
) -> Dict[str, float]:
    """Run a BS-{delta | delta-gamma | delta-gamma-vega} hedger over a batched
    JAX market trajectory.

    Returns a metric dict with the same keys as
    :meth:`BuehlerTrainer.evaluate` so it can be compared directly against a
    trained-policy eval on the *same* ``eval_seed``.

    Args:
        hedge_kind: which analytical hedger to use:
            'delta'             — underlying only
            'delta_gamma'       — underlying + 1 option (gamma-neutral)
            'delta_gamma_vega'  — underlying + 2 options ({gamma, vega}-neutral)
        option_grid_spec: required for delta_gamma / delta_gamma_vega — the
            output of :func:`build_option_grid_spec(grid)`. Slot ordering
            must match the trainer's instrument-vector layout.
        variance_source: how the hedger gets σ²_t at each step:
            'oracle'        — uses the true Heston v_t (MDP observability,
                              default — what classical eval assumed)
            'realized_vol'  — EWMA of squared log-returns (RiskMetrics-style
                              estimator a real trading desk would compute
                              from price history alone — POMDP-realistic)
            'implied_vol'   — back out IV from the closest-to-ATM option
                              in the grid via Brent inversion of BS-call.
                              Requires option_grid_spec. P2-only.
        ewma_lambda: decay for the EWMA realized-variance estimator
            (0.94 = RiskMetrics standard, ~16-day half-life). Ignored
            unless variance_source='realized_vol'.
        initial_variance_prior: seed value for the EWMA recursion at t=0
            (the hedger doesn't see v_0 in the POMDP regime; this is the
            agent's prior). Defaults to long-run Heston θ.

    All other args match :func:`BuehlerTrainer.evaluate` semantics.
    """
    if hedge_kind not in ("delta", "delta_gamma", "delta_gamma_vega"):
        raise ValueError(f"unknown hedge_kind={hedge_kind!r}")
    if hedge_kind in ("delta_gamma", "delta_gamma_vega") and option_grid_spec is None:
        raise ValueError(
            f"hedge_kind={hedge_kind!r} requires option_grid_spec"
        )
    if variance_source not in ("oracle", "realized_vol", "implied_vol"):
        raise ValueError(
            f"variance_source must be 'oracle' | 'realized_vol' | 'implied_vol', "
            f"got {variance_source!r}"
        )
    if variance_source == "implied_vol" and option_grid_spec is None:
        raise ValueError("variance_source='implied_vol' requires option_grid_spec")
    if liability.kind not in ("call", "put"):
        raise ValueError(
            f"only call/put liabilities supported, got kind={liability.kind!r}"
        )
    if liability.maturity <= 0:
        raise ValueError(
            f"liability.maturity must be positive, got {liability.maturity}"
        )

    import jax

    key = jax.random.PRNGKey(eval_seed)
    batch = market_sampler(n_paths, key)

    spots = np.asarray(batch.spots, dtype=np.float64)              # (B, T+1)
    variances = np.asarray(batch.variances, dtype=np.float64)      # (B, T+1)
    prices = np.asarray(batch.instrument_prices, dtype=np.float64) # (B, T+1, N)
    masks = np.asarray(batch.action_masks, dtype=bool)             # (T+1, N) or (B, T+1, N)
    B = n_paths
    if spots.shape != (B, horizon + 1):
        raise ValueError(
            f"spots shape {spots.shape} != ({B}, {horizon + 1})"
        )
    tc = np.asarray(transaction_cost_rates, dtype=np.float64)

    # ── Build target_positions[t, B, n_instruments] for the chosen hedge ──
    target_positions = np.zeros((horizon, B, n_instruments), dtype=np.float64)

    # ── Variance estimator: oracle | EWMA-realized-vol | ATM-implied-vol ──
    if variance_source == "oracle":
        sigma_t = np.sqrt(np.maximum(variances[:, :horizon], 1e-12))
    elif variance_source == "realized_vol":
        # RiskMetrics EWMA: v̂_t = λ·v̂_{t-1} + (1-λ)·r_t² / dt
        # Forward-recursive, always defined from t=0 onward (with prior).
        log_spots = np.log(np.maximum(spots, 1e-12))
        log_returns_sq = np.diff(log_spots, axis=1) ** 2 / dt    # (B, T)
        v_hat = np.empty_like(log_returns_sq)
        v_prev = np.full(B, float(initial_variance_prior), dtype=np.float64)
        for t in range(horizon):
            v_prev = ewma_lambda * v_prev + (1.0 - ewma_lambda) * log_returns_sq[:, t]
            v_hat[:, t] = v_prev
        # CRITICAL CAUSALITY FIX: at time t (start of step t), the hedger has
        # observed r_1, ..., r_{t-1} (returns up to t-1). So the estimator
        # available for forming the decision at step t uses returns through
        # t-1. Shift by 1: σ_t at step t uses v̂ computed from r_1..r_t-1.
        # For t=0, no returns observed yet → use the prior.
        sigma_t = np.empty_like(v_hat)
        sigma_t[:, 0] = np.sqrt(float(initial_variance_prior))
        if horizon > 1:
            sigma_t[:, 1:] = np.sqrt(np.maximum(v_hat[:, :-1], 1e-12))
    elif variance_source == "implied_vol":
        # IV-from-ATM: invert BS at the closest-to-ATM call price in the grid.
        # Vectorized Brent root-find isn't straightforward in numpy; use a
        # Newton iteration over (B, T) which converges quickly for liquid calls.
        if not option_grid_spec:
            raise ValueError("implied_vol needs option_grid_spec")
        atm = min(
            (s for s in option_grid_spec if s["kind"] == "call"),
            key=lambda s: abs(s["moneyness"] - 1.0) * 100 + s["tau_steps"],
        )
        atm_idx = atm["idx"]
        atm_K = atm["moneyness"] * spots[:, :horizon]              # (B, T)
        atm_tau = np.full_like(spots[:, :horizon], atm["tau_steps"] * dt)
        atm_call_prices = prices[:, :horizon, atm_idx]             # (B, T)
        # Newton's method on σ for BS call price (vectorized)
        sigma = np.full_like(spots[:, :horizon], np.sqrt(float(initial_variance_prior)))
        for _ in range(20):
            S_ = spots[:, :horizon]
            sqrtT = np.sqrt(np.maximum(atm_tau, 1e-12))
            d1 = (np.log(S_ / atm_K) + 0.5 * sigma * sigma * atm_tau) / (sigma * sqrtT + 1e-12)
            phi_d1 = np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi)
            vega = S_ * sqrtT * phi_d1                              # (B, T)
            bs_price = S_ * norm.cdf(d1) - atm_K * norm.cdf(d1 - sigma * sqrtT)
            update = (bs_price - atm_call_prices) / np.maximum(vega, 1e-8)
            sigma = np.clip(sigma - update, 1e-4, 5.0)
        sigma_t = sigma
    steps_remaining = np.maximum(
        liability.maturity - np.arange(horizon, dtype=np.float64), 0.0
    )                                                              # (T,)
    tau_t = np.broadcast_to(steps_remaining * dt, (B, horizon))    # (B, T)
    spots_t = spots[:, :horizon]                                   # (B, T)

    liab_delta, liab_gamma, liab_vega = _bs_greeks_batch(
        S=spots_t, K=float(liability.strike), tau=tau_t,
        sigma=sigma_t, r=float(risk_free_rate), kind=liability.kind,
    )
    qty = float(liability.quantity)
    target_underlying = qty * liab_delta                           # (B, T)

    if hedge_kind == "delta":
        target_positions[:, :, 0] = target_underlying.T            # transpose to (T, B)
    elif hedge_kind == "delta_gamma":
        # Pick 1 hedging option
        opt = _select_hedging_options(option_grid_spec, liability.maturity, 1)[0]
        opt_idx = opt["idx"]
        opt_tau_steps = opt["tau_steps"]
        opt_moneyness = opt["moneyness"]
        # opt strike floats with spot (rolling-tenor moneyness convention)
        opt_K = opt_moneyness * spots_t
        opt_tau = np.full_like(spots_t, opt_tau_steps * dt)
        opt_delta, opt_gamma, _ = _bs_greeks_batch(
            S=spots_t, K=opt_K, tau=opt_tau, sigma=sigma_t,
            r=float(risk_free_rate), kind="call",
        )
        liab_gamma_qty = qty * liab_gamma
        # Avoid divide-by-tiny-gamma issues at near-expiry
        safe_opt_gamma = np.where(np.abs(opt_gamma) > 1e-8, opt_gamma, np.inf)
        opt_pos = liab_gamma_qty / safe_opt_gamma                  # (B, T)
        underlying_pos = target_underlying - opt_pos * opt_delta
        target_positions[:, :, 0] = underlying_pos.T
        target_positions[:, :, opt_idx] = opt_pos.T
    elif hedge_kind == "delta_gamma_vega":
        opts = _select_hedging_options(option_grid_spec, liability.maturity, 2)
        opt_indices = [o["idx"] for o in opts]
        # Compute greeks for both options
        opt_data = []
        for o in opts:
            opt_K = o["moneyness"] * spots_t
            opt_tau = np.full_like(spots_t, o["tau_steps"] * dt)
            d_, g_, v_ = _bs_greeks_batch(
                S=spots_t, K=opt_K, tau=opt_tau, sigma=sigma_t,
                r=float(risk_free_rate), kind="call",
            )
            opt_data.append({"delta": d_, "gamma": g_, "vega": v_})
        # Solve 2x2 system per (B, T) cell for option positions:
        #   [γ_1  γ_2] [p_1]   [Γ_liab]
        #   [ν_1  ν_2] [p_2] = [ν_liab]
        # vectorized via direct inverse formula
        g1, g2 = opt_data[0]["gamma"], opt_data[1]["gamma"]
        v1, v2 = opt_data[0]["vega"], opt_data[1]["vega"]
        det = g1 * v2 - g2 * v1
        # Sanitize near-singular cells
        safe_det = np.where(np.abs(det) > 1e-10, det, np.inf)
        liab_gamma_qty = qty * liab_gamma
        liab_vega_qty = qty * liab_vega
        p1 = (v2 * liab_gamma_qty - g2 * liab_vega_qty) / safe_det
        p2 = (-v1 * liab_gamma_qty + g1 * liab_vega_qty) / safe_det
        # Residual delta absorbed into underlying
        d1, d2 = opt_data[0]["delta"], opt_data[1]["delta"]
        underlying_pos = target_underlying - p1 * d1 - p2 * d2
        target_positions[:, :, 0] = underlying_pos.T
        target_positions[:, :, opt_indices[0]] = p1.T
        target_positions[:, :, opt_indices[1]] = p2.T

    # ── Replay with the multi-instrument accounting ─────────────────────
    terminal_pnl, total_costs, _ = _replay_multi_instrument_batch(
        target_positions=target_positions,
        prices=prices,
        masks=masks,
        transaction_cost_rates=tc,
        initial_cash=initial_cash,
        horizon=horizon,
        n_instruments=n_instruments,
    )

    # ── Liability payoff + Buehler-style metrics ────────────────────────
    spot_T = spots[:, horizon]
    liability_payoffs = _terminal_call_put_payoff(spot_T, liability)
    hedging_errors = terminal_pnl - liability_payoffs
    zero_errors = float(initial_cash) - liability_payoffs

    eval_loss = (
        risk_aversion * float(np.var(hedging_errors, ddof=1))
        + float(total_costs.mean())
    )

    metrics: Dict[str, float] = {
        "n_paths": int(B),
        "hedge_kind": hedge_kind,
        "eval_loss": eval_loss,
        "mean_pnl": float(terminal_pnl.mean()),
        "std_pnl": float(terminal_pnl.std(ddof=0)),
        "mean_reward": float((hedging_errors - float(initial_cash)).mean()),
        "mean_cost": float(total_costs.mean()),
        "std_cost": float(total_costs.std(ddof=0)),
        "mean_liability_payoff": float(liability_payoffs.mean()),
        "std_liability_payoff": float(liability_payoffs.std(ddof=0)),
    }
    metrics.update(_error_distribution_metrics(hedging_errors))
    metrics.update(_error_distribution_metrics(zero_errors, "zero_hedge_"))
    metrics["std_improvement_vs_zero"] = (
        metrics["zero_hedge_std_hedging_error"] - metrics["std_hedging_error"]
    )
    metrics["rmse_improvement_vs_zero"] = (
        metrics["zero_hedge_rmse_hedging_error"] - metrics["rmse_hedging_error"]
    )
    return metrics


def evaluate_analytical_delta_hedger_batch(
    market_sampler: MarketBatchSampler,
    horizon: int,
    n_instruments: int,
    transaction_cost_rates: np.ndarray,
    liability: LiabilitySpec,
    dt: float = 1.0 / 252.0,
    n_paths: int = 4096,
    eval_seed: int = 9999,
    risk_free_rate: float = 0.0,
    initial_cash: float = 0.0,
    risk_aversion: float = 1000.0,
) -> Dict[str, float]:
    """Backward-compatible delta-only wrapper around
    :func:`evaluate_analytical_hedger_batch`.

    Returns a metric dict with the same keys as
    :meth:`BuehlerTrainer.evaluate` so it can be compared directly against
    a trained LSTM eval on the *same* ``eval_seed``.

    The hedger trades only the underlying (slot 0). All option slots are
    held flat. This matches ``AnalyticalDeltaHedger(gamma_hedge=False)``.

    Args:
        market_sampler: callable ``(batch_size, key) -> TrajectoryBatch``
            (typically the same sampler the trainer uses).
        horizon: number of env steps (e.g. 252).
        n_instruments: total instruments in the action vector.
        transaction_cost_rates: ``(n_instruments,)`` proportional rates.
        liability: vanilla European call or put spec.
        dt: time per env step (e.g. 1/252 for daily).
        n_paths: eval batch size — keep equal to the trainer's eval call
            for byte-identical paths.
        eval_seed: PRNG seed; identical seed → identical paths.
        risk_free_rate: r used in BS delta.
        initial_cash: starting cash.
        risk_aversion: gamma in the Buehler eval_loss term.
    """
    return evaluate_analytical_hedger_batch(
        market_sampler=market_sampler,
        horizon=horizon,
        n_instruments=n_instruments,
        transaction_cost_rates=transaction_cost_rates,
        liability=liability,
        hedge_kind="delta",
        option_grid_spec=None,
        dt=dt,
        n_paths=n_paths,
        eval_seed=eval_seed,
        risk_free_rate=risk_free_rate,
        initial_cash=initial_cash,
        risk_aversion=risk_aversion,
    )
