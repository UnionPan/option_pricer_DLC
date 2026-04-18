"""
Configuration and utilities for the JAX deep hedging environment line.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from options_desk.processes._jax_backend import configure_jax_runtime


configure_jax_runtime()


@dataclass(frozen=True)
class FloatingOptionGrid:
    """
    Floating-grid option specification from the deep hedging notes.

    ``maturities`` are measured in environment steps. Each maturity maps to a
    tuple of relative strikes / moneyness levels. For each grid point the
    action space contains both a call and a put, plus the underlying.
    """

    maturities: tuple[int, ...]
    moneyness_by_maturity: Mapping[int, tuple[float, ...]]

    def __post_init__(self) -> None:
        canonical_maturities = tuple(sorted(int(m) for m in self.maturities))
        canonical_grid = {
            int(maturity): tuple(float(k) for k in self.moneyness_by_maturity[maturity])
            for maturity in canonical_maturities
        }

        if not canonical_maturities:
            raise ValueError("FloatingOptionGrid requires at least one maturity")

        for maturity, moneyness in canonical_grid.items():
            if maturity <= 0:
                raise ValueError("Option maturities must be positive")
            if not moneyness:
                raise ValueError(f"Maturity {maturity} must have at least one strike")

        object.__setattr__(self, "maturities", canonical_maturities)
        object.__setattr__(self, "moneyness_by_maturity", canonical_grid)

    @property
    def n_options(self) -> int:
        return 2 * sum(len(self.moneyness_by_maturity[m]) for m in self.maturities)

    @property
    def n_instruments(self) -> int:
        return 1 + self.n_options


@dataclass(frozen=True)
class DeepHedgingEnvConfig:
    """
    High-level configuration for the new JAX deep hedging rollout stack.
    """

    horizon_steps: int
    option_grid: FloatingOptionGrid
    dt: float = 1.0 / 250.0
    transaction_cost_underlying: float = 1.0e-4
    transaction_cost_option: float = 1.0e-2
    risk_aversion: float = 1_000.0

    def __post_init__(self) -> None:
        if self.horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")


@dataclass(frozen=True)
class DeepHedgingState:
    """
    Minimal immutable environment state for the JAX deep hedging line.
    """

    time_index: int
    spot: jnp.ndarray
    variance: jnp.ndarray
    cash: jnp.ndarray
    positions: jnp.ndarray
    previous_action: jnp.ndarray


@dataclass(frozen=True)
class DeepHedgingRollout:
    """
    Collected trajectory outputs from a pure trade rollout.
    """

    positions: np.ndarray
    cash: np.ndarray
    portfolio_values: np.ndarray
    final_state: DeepHedgingState


GRID_COARSE = FloatingOptionGrid(
    maturities=(5, 20),
    moneyness_by_maturity={
        5:  (0.95, 1.00, 1.05),
        20: (0.95, 1.00, 1.05),
    },
)

GRID_FINE = FloatingOptionGrid(
    maturities=(5, 10, 25, 63),
    moneyness_by_maturity={
        5:  (0.95, 0.975, 1.00, 1.025),
        10: (0.93, 0.97, 1.00, 1.03),
        25: (0.90, 0.95, 1.00, 1.05),
        63: (0.85, 0.90, 0.95, 1.00, 1.05),
    },
)


def _as_jax_vector(values: np.ndarray | jnp.ndarray) -> jnp.ndarray:
    """Convert 1-D instrument arrays to JAX float32 vectors."""
    return jnp.asarray(values, dtype=jnp.float32)


def build_action_mask(
    option_grid: FloatingOptionGrid,
    time_index: int,
    horizon_steps: int,
) -> np.ndarray:
    """
    Build the floating-grid action mask at a given time.

    The underlying is always tradable. Option actions are available only when
    their maturity does not exceed the remaining episode horizon, matching the
    convention in the notes.
    """

    if time_index < 0:
        raise ValueError("time_index must be non-negative")
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive")

    remaining_steps = max(horizon_steps - time_index, 0)
    mask = [True]

    for maturity in option_grid.maturities:
        available = maturity <= remaining_steps
        for _ in option_grid.moneyness_by_maturity[maturity]:
            mask.extend([available, available])

    return np.asarray(mask, dtype=bool)


def build_transaction_cost_vector(config: DeepHedgingEnvConfig) -> np.ndarray:
    """
    Per-instrument proportional transaction costs.

    Instrument 0 is the underlying. All remaining instruments are options on
    the floating grid.
    """

    costs = np.full(
        config.option_grid.n_instruments,
        config.transaction_cost_option,
        dtype=np.float32,
    )
    costs[0] = np.float32(config.transaction_cost_underlying)
    return costs


def reset_state(
    config: DeepHedgingEnvConfig,
    initial_spot: float,
    initial_variance: float,
    initial_cash: float = 0.0,
) -> DeepHedgingState:
    """
    Create the initial immutable state for a new trajectory rollout.
    """

    n_instruments = config.option_grid.n_instruments
    return DeepHedgingState(
        time_index=0,
        spot=jnp.asarray([initial_spot], dtype=jnp.float32),
        variance=jnp.asarray([initial_variance], dtype=jnp.float32),
        cash=jnp.asarray([initial_cash], dtype=jnp.float32),
        positions=jnp.zeros((n_instruments,), dtype=jnp.float32),
        previous_action=jnp.zeros((n_instruments,), dtype=jnp.float32),
    )


def step_state(
    state: DeepHedgingState,
    trade: np.ndarray | jnp.ndarray,
    current_prices: np.ndarray | jnp.ndarray,
    next_spot: float,
    next_variance: float,
    transaction_costs: np.ndarray | jnp.ndarray,
) -> DeepHedgingState:
    """
    Advance the environment state under trade-based control.

    ``trade`` is the action in the deep hedging notes: the amount bought or
    sold at the current time step. Positions accumulate over time.
    """

    trade_vec = _as_jax_vector(trade)
    price_vec = _as_jax_vector(current_prices)
    cost_vec = _as_jax_vector(transaction_costs)

    notional = jnp.dot(trade_vec, price_vec)
    transaction_cost = jnp.dot(cost_vec, jnp.abs(trade_vec))
    next_cash = state.cash - jnp.asarray([notional + transaction_cost], dtype=jnp.float32)

    return DeepHedgingState(
        time_index=state.time_index + 1,
        spot=jnp.asarray([next_spot], dtype=jnp.float32),
        variance=jnp.asarray([next_variance], dtype=jnp.float32),
        cash=next_cash,
        positions=state.positions + trade_vec,
        previous_action=trade_vec,
    )


def compute_portfolio_value(
    state: DeepHedgingState,
    instrument_prices: np.ndarray | jnp.ndarray,
) -> float:
    """
    Mark-to-market portfolio value at the current step.
    """

    price_vec = _as_jax_vector(instrument_prices)
    value = state.cash[0] + jnp.dot(state.positions, price_vec)
    return float(value)


def rollout_trades(
    initial_state: DeepHedgingState,
    trades: np.ndarray,
    step_prices: np.ndarray,
    next_spots: np.ndarray,
    next_variances: np.ndarray,
    transaction_costs: np.ndarray | jnp.ndarray,
) -> DeepHedgingRollout:
    """
    Roll a sequence of trade actions through the pure state transition.
    """

    states = []
    portfolio_values = []
    state = initial_state

    for trade, prices, next_spot, next_variance in zip(
        np.asarray(trades, dtype=np.float32),
        np.asarray(step_prices, dtype=np.float32),
        np.asarray(next_spots, dtype=np.float32),
        np.asarray(next_variances, dtype=np.float32),
    ):
        state = step_state(
            state=state,
            trade=trade,
            current_prices=prices,
            next_spot=float(next_spot),
            next_variance=float(next_variance),
            transaction_costs=transaction_costs,
        )

        # Value the updated portfolio at the next spot and the current option grid.
        mark_prices = np.asarray(prices, dtype=np.float32).copy()
        mark_prices[0] = np.float32(next_spot)
        portfolio_values.append(compute_portfolio_value(state, mark_prices))
        states.append(state)

    positions = np.stack([np.asarray(s.positions, dtype=np.float32) for s in states], axis=0)
    cash = np.asarray([np.asarray(s.cash, dtype=np.float32)[0] for s in states], dtype=np.float32)
    portfolio_values = np.asarray(portfolio_values, dtype=np.float32)

    return DeepHedgingRollout(
        positions=positions,
        cash=cash,
        portfolio_values=portfolio_values,
        final_state=state,
    )


def build_observation(
    state: DeepHedgingState,
    option_features: np.ndarray,
    action_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Construct a recurrent-policy-friendly observation bundle.
    """

    time_ratio = np.float32(state.time_index)
    observation = {
        "spot": np.asarray(state.spot, dtype=np.float32),
        "variance": np.asarray(state.variance, dtype=np.float32),
        "time_features": np.asarray([time_ratio, 1.0], dtype=np.float32),
        "positions": np.asarray(state.positions, dtype=np.float32),
        "previous_action": np.asarray(state.previous_action, dtype=np.float32),
        "option_features": np.asarray(option_features, dtype=np.float32),
        "action_mask": np.asarray(action_mask, dtype=bool),
    }
    return observation
