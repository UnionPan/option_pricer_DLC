"""
Integrated JAX rollout: fused Heston path simulation + COS option pricing.

The core function ``simulate_heston_market`` runs a single-path Heston
Euler-Maruyama simulation and prices the entire floating option grid at
every time step via the COS method — all inside a single ``lax.scan``.

This is the "counterfactual instrument simulator": given calibrated Heston
parameters, it produces a realistic but synthetic trajectory of spot,
variance, and the full instrument price vector (underlying + options).

A separate ``replay_rollout`` function takes pre-computed trades and the
price trajectory to compute portfolio P&L, so any agent (NumPy or JAX)
can be used without touching the JIT-compiled market kernel.

author: Yunian Pan
email: yp1170@nyu.edu
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from options_desk.processes._jax_backend import configure_jax_runtime

from ..utils.contracts import MarketTrajectory
from .env import (
    DeepHedgingEnvConfig,
    DeepHedgingRollout,
    DeepHedgingState,
    build_action_mask,
    build_transaction_cost_vector,
)
from .pricing import HestonMarketParams, _PaddedGrid, compile_padded_grid, price_option_grid

configure_jax_runtime()


# ============================================================================
# Heston single-step (inlined for lax.scan — no class dispatch)
# ============================================================================

def _heston_euler_step(
    spot: jnp.ndarray,
    variance: jnp.ndarray,
    dt: float,
    mu: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    cholesky: jnp.ndarray,
    key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray, jax.Array]:
    """Single Euler-Maruyama step for the Heston model.

    Args:
        spot: scalar — current spot price
        variance: scalar — current instantaneous variance
        dt: time step size
        mu: risk-neutral drift (r - q)
        kappa, theta, sigma_v: Heston vol dynamics parameters
        cholesky: (2, 2) Cholesky factor of [[1, rho], [rho, 1]]
        key: JAX PRNG key

    Returns:
        (next_spot, next_variance, next_key)
    """
    key, k_bm = jax.random.split(key)
    dW_raw = jax.random.normal(k_bm, shape=(2,)) * jnp.sqrt(dt)
    dW = cholesky @ dW_raw  # correlated increments

    v_pos = jnp.maximum(variance, 0.0)
    sqrt_v = jnp.sqrt(v_pos)

    # Spot: dS = mu*S*dt + sqrt(v)*S*dW_1
    next_spot = spot + mu * spot * dt + sqrt_v * spot * dW[0]

    # Variance: dv = kappa*(theta - v)*dt + sigma_v*sqrt(v)*dW_2
    next_variance = variance + kappa * (theta - variance) * dt + sigma_v * sqrt_v * dW[1]

    # Truncation scheme: clamp variance to zero
    next_variance = jnp.maximum(next_variance, 0.0)

    # Absorbing barrier at zero for spot (safety)
    next_spot = jnp.maximum(next_spot, 1e-8)

    return next_spot, next_variance, key


# ============================================================================
# Core: fused Heston simulation + COS pricing via lax.scan
# ============================================================================

def _simulate_market_core(
    market: HestonMarketParams,
    padded_grid: _PaddedGrid,
    initial_spot: float,
    initial_variance: float,
    key: jax.Array,
    dt: float,
    horizon_steps: int,
    N_cos: int,
    L_cos: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pure JAX core of the market simulation (vmap-safe, not JIT'd).

    Callers should use ``_jit_simulate_single`` or ``_jit_simulate_batch``
    which wrap this function with ``jax.jit`` and appropriate static args.

    ``horizon_steps`` and ``N_cos`` must be static (they determine array shapes).

    Returns:
        (spots, variances, instrument_prices) as JAX arrays with shapes
        (horizon+1,), (horizon+1,), (horizon+1, n_instruments).
    """
    mu = market.r - market.q
    horizon = horizon_steps

    rho = market.rho
    cholesky = jnp.array([
        [1.0, 0.0],
        [rho, jnp.sqrt(1.0 - rho**2)],
    ])

    spot_0 = jnp.asarray(initial_spot)
    var_0 = jnp.asarray(initial_variance)

    prices_0 = price_option_grid(
        spot_0, var_0, padded_grid, market,
        time_index=0, horizon_steps=horizon,
        N_cos=N_cos, L_cos=L_cos,
    )

    def scan_body(carry, time_idx):
        spot, variance, key = carry

        next_spot, next_var, key = _heston_euler_step(
            spot, variance, dt, mu,
            market.kappa, market.theta, market.sigma_v,
            cholesky, key,
        )

        prices = price_option_grid(
            next_spot, next_var, padded_grid, market,
            time_index=time_idx + 1, horizon_steps=horizon,
            N_cos=N_cos, L_cos=L_cos,
        )

        return (next_spot, next_var, key), (next_spot, next_var, prices)

    time_indices = jnp.arange(horizon, dtype=jnp.int32)
    _, (spots_body, vars_body, prices_body) = jax.lax.scan(
        scan_body,
        (spot_0, var_0, key),
        time_indices,
    )

    spots = jnp.concatenate([spot_0[jnp.newaxis], spots_body])
    variances = jnp.concatenate([var_0[jnp.newaxis], vars_body])
    instrument_prices = jnp.concatenate([prices_0[jnp.newaxis], prices_body])

    return spots, variances, instrument_prices


# ============================================================================
# Module-level JIT'd wrappers
# ============================================================================
# horizon_steps (arg 6) and N_cos (arg 7) are static — they determine
# array shapes (jnp.arange) and must be compile-time constants.

_jit_simulate_single = jax.jit(
    _simulate_market_core,
    static_argnums=(6, 7),
)

_jit_simulate_batch = jax.jit(
    jax.vmap(
        _simulate_market_core,
        in_axes=(None, None, None, None, 0, None, None, None, None),
    ),
    static_argnums=(6, 7),
)


def simulate_heston_market(
    config: DeepHedgingEnvConfig,
    market: HestonMarketParams,
    padded_grid: _PaddedGrid,
    initial_spot: float,
    initial_variance: float,
    key: jax.Array,
    N_cos: int = 128,
    L_cos: float = 12.0,
) -> MarketTrajectory:
    """Simulate a single Heston path and price the full option grid at each step.

    This is the core counterfactual instrument simulator. The entire
    Heston SDE + COS pricing pipeline is JIT-compiled into a single
    reusable kernel via ``_jit_simulate_single``.

    Args:
        config: environment configuration (horizon, grid, costs, etc.)
        market: risk-neutral Heston parameters for both simulation and pricing
        padded_grid: pre-compiled option grid (from ``compile_padded_grid``)
        initial_spot: starting spot price
        initial_variance: starting instantaneous variance
        key: JAX PRNG key
        N_cos: COS expansion terms per pricing call
        L_cos: COS truncation parameter

    Returns:
        ``MarketTrajectory`` with spots, variances, instrument prices, and
        action masks for all horizon_steps + 1 time points (including t=0).
    """
    horizon = config.horizon_steps

    spots, variances, instrument_prices = _jit_simulate_single(
        market, padded_grid,
        initial_spot, initial_variance, key,
        config.dt, horizon, N_cos, L_cos,
    )

    # Use precomputed masks if available, otherwise build on the fly
    if padded_grid.action_masks is not None:
        masks = padded_grid.action_masks
    else:
        masks = np.stack([
            build_action_mask(config.option_grid, t, horizon)
            for t in range(horizon + 1)
        ], axis=0)

    return MarketTrajectory(
        spots=np.asarray(spots, dtype=np.float64),
        variances=np.asarray(variances, dtype=np.float64),
        instrument_prices=np.asarray(instrument_prices, dtype=np.float64),
        action_masks=masks,
    )


def simulate_heston_market_batch(
    config: DeepHedgingEnvConfig,
    market: HestonMarketParams,
    padded_grid: _PaddedGrid,
    initial_spot: float,
    initial_variance: float,
    keys: jax.Array,
    N_cos: int = 128,
    L_cos: float = 12.0,
) -> MarketTrajectory:
    """Simulate multiple independent Heston paths via ``vmap``.

    Same as ``simulate_heston_market`` but vectorized over PRNG keys.
    The entire batch is JIT-compiled into a single kernel via
    ``_jit_simulate_batch``.

    Args:
        keys: (n_paths,) array of JAX PRNG keys (from ``jax.random.split``)

    Returns:
        ``MarketTrajectory`` with shapes (n_paths, horizon+1, ...).
        ``action_masks`` has shape (horizon+1, n_instruments) (shared
        across paths since they depend only on time, not state).
    """
    horizon = config.horizon_steps

    spots, variances, instrument_prices = _jit_simulate_batch(
        market, padded_grid,
        initial_spot, initial_variance, keys,
        config.dt, horizon, N_cos, L_cos,
    )

    # Use precomputed masks if available, otherwise build on the fly
    if padded_grid.action_masks is not None:
        masks = padded_grid.action_masks
    else:
        masks = np.stack([
            build_action_mask(config.option_grid, t, horizon)
            for t in range(horizon + 1)
        ], axis=0)

    return MarketTrajectory(
        spots=np.asarray(spots, dtype=np.float64),
        variances=np.asarray(variances, dtype=np.float64),
        instrument_prices=np.asarray(instrument_prices, dtype=np.float64),
        action_masks=masks,
    )


# ============================================================================
# Replay rollout: pre-computed trades → portfolio P&L
# ============================================================================

def replay_rollout(
    config: DeepHedgingEnvConfig,
    trajectory: MarketTrajectory,
    trades: np.ndarray,
    initial_cash: float = 0.0,
) -> DeepHedgingRollout:
    """Roll pre-computed trades through a market trajectory.

    Takes the output of ``simulate_heston_market`` and a matrix of trades
    (from any agent) and computes the portfolio evolution.

    Args:
        config: environment configuration
        trajectory: market trajectory from ``simulate_heston_market``
        trades: (horizon_steps, n_instruments) — trades at each step
        initial_cash: initial cash position

    Returns:
        ``DeepHedgingRollout`` with positions, cash, portfolio values,
        and the final state.
    """
    horizon = config.horizon_steps
    n_instruments = config.option_grid.n_instruments
    trades = np.asarray(trades, dtype=np.float64)

    if trades.shape != (horizon, n_instruments):
        raise ValueError(
            f"trades shape {trades.shape} != expected ({horizon}, {n_instruments})"
        )

    tc_vector = build_transaction_cost_vector(config).astype(np.float64)

    # Initialize state
    cash = initial_cash
    positions = np.zeros(n_instruments, dtype=np.float64)
    previous_action = np.zeros(n_instruments, dtype=np.float64)

    # Trajectory arrays
    all_positions = np.zeros((horizon, n_instruments), dtype=np.float64)
    all_cash = np.zeros(horizon, dtype=np.float64)
    portfolio_values = np.zeros(horizon, dtype=np.float64)

    for t in range(horizon):
        trade = trades[t]
        prices_t = trajectory.instrument_prices[t]  # prices at time t (before step)

        # Transaction accounting
        notional = np.dot(trade, prices_t)
        tc = np.dot(tc_vector, np.abs(trade))
        cash = cash - notional - tc

        # Update positions
        positions = positions + trade
        previous_action = trade

        # Mark-to-market at next period's prices
        prices_next = trajectory.instrument_prices[t + 1]
        pv = cash + np.dot(positions, prices_next)

        all_positions[t] = positions
        all_cash[t] = cash
        portfolio_values[t] = pv

    # Build final DeepHedgingState
    final_state = DeepHedgingState(
        time_index=horizon,
        spot=jnp.asarray([trajectory.spots[horizon]], dtype=jnp.float32),
        variance=jnp.asarray([trajectory.variances[horizon]], dtype=jnp.float32),
        cash=jnp.asarray([cash], dtype=jnp.float32),
        positions=jnp.asarray(positions, dtype=jnp.float32),
        previous_action=jnp.asarray(previous_action, dtype=jnp.float32),
    )

    return DeepHedgingRollout(
        positions=all_positions.astype(np.float32),
        cash=all_cash.astype(np.float32),
        portfolio_values=portfolio_values.astype(np.float32),
        final_state=final_state,
    )
