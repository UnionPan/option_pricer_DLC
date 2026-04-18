"""
Shared data contracts for deep hedging env/agent interoperability.

All contract types use NumPy arrays only -- no JAX or PyTorch dependencies.
This keeps the contract layer import-safe for Gym, JAX, and PyTorch stacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ObservationBatch:
    """
    Common observation boundary for Gym and JAX deep hedging stacks.

    Shape conventions:
        spot:               (B, 1)
        time_index:         (B, 1)
        option_features:    (B, F_option)
        portfolio_features: (B, N)
        previous_action:    (B, N)
        context_features:   (B, F_context)
        action_mask:        (B, N) or None
    """

    spot: NDArray[np.float32]
    time_index: NDArray[np.float32]
    option_features: NDArray[np.float32]
    portfolio_features: NDArray[np.float32]
    previous_action: NDArray[np.float32]
    context_features: NDArray[np.float32]
    action_mask: Optional[NDArray[np.bool_]] = None

    @property
    def n_episodes(self) -> int:
        return int(self.spot.shape[0])


@dataclass(frozen=True)
class ActionBatch:
    """
    Common action boundary.

    Shape: (B, N)
    """

    actions: NDArray[np.float32]

    @property
    def n_episodes(self) -> int:
        return int(self.actions.shape[0])


@dataclass(frozen=True)
class LiabilitySpec:
    """
    Terminal liability definition for hedging objectives.
    """

    kind: str
    strike: float
    maturity: int
    quantity: float = 1.0

    def terminal_payoff(self, terminal_spot: np.ndarray | float) -> NDArray[np.float32]:
        """Compute terminal payoff under a vanilla call/put liability."""
        spot = np.asarray(terminal_spot, dtype=np.float32)
        if self.kind == "call":
            intrinsic = np.maximum(spot - np.float32(self.strike), 0.0)
        elif self.kind == "put":
            intrinsic = np.maximum(np.float32(self.strike) - spot, 0.0)
        else:
            raise ValueError(f"unsupported liability kind: {self.kind}")
        return (np.float32(self.quantity) * intrinsic).astype(np.float32)


@dataclass(frozen=True)
class MarketTrajectory:
    """
    Counterfactual market trajectory emitted by the JAX market kernel.

    Single-path shapes:
        spots:             (T + 1,)
        variances:         (T + 1,)
        instrument_prices: (T + 1, N)
        action_masks:      (T + 1, N)

    Batched shapes:
        spots:             (B, T + 1)
        variances:         (B, T + 1)
        instrument_prices: (B, T + 1, N)
        action_masks:      (T + 1, N) or (B, T + 1, N)
    """

    spots: NDArray[np.float64]
    variances: NDArray[np.float64]
    instrument_prices: NDArray[np.float64]
    action_masks: NDArray[np.bool_]


@dataclass(frozen=True)
class TrajectoryBatch:
    """
    Common trajectory/result boundary.

    Batch-first shape conventions:
        spots:                     (B, T + 1)
        variances:                 (B, T + 1)
        instrument_prices:         (B, T + 1, N)
        action_masks:              (B, T + 1, N)
        rewards:                   (B, T)
        dones:                     (B, T)
        positions:                 (B, T, N)
        portfolio_values:          (B, T)
        actions:                   (B, T, N)
        trades:                    (B, T, N)
        terminal_liability_payoffs:(B,)
        initial_cash:              (B,)
    """

    rewards: Optional[NDArray[np.float32]] = None
    dones: Optional[NDArray[np.bool_]] = None
    positions: Optional[NDArray[np.float32]] = None
    portfolio_values: Optional[NDArray[np.float32]] = None
    observations: Optional[ObservationBatch] = None
    actions: Optional[NDArray[np.float32]] = None
    trades: Optional[NDArray[np.float32]] = None
    spots: Optional[NDArray[np.float32]] = None
    variances: Optional[NDArray[np.float32]] = None
    instrument_prices: Optional[NDArray[np.float32]] = None
    action_masks: Optional[NDArray[np.bool_]] = None
    terminal_liability_payoffs: Optional[NDArray[np.float32]] = None
    initial_cash: Optional[NDArray[np.float32]] = None

    @property
    def batch_size(self) -> int:
        for array in (
            self.rewards,
            self.positions,
            self.actions,
            self.trades,
            self.portfolio_values,
            self.spots,
            self.variances,
            self.instrument_prices,
            self.action_masks,
        ):
            if array is not None:
                return int(array.shape[0])
        return 0

    @property
    def horizon(self) -> int:
        for array in (
            self.rewards,
            self.dones,
            self.positions,
            self.actions,
            self.trades,
            self.portfolio_values,
        ):
            if array is not None:
                return int(array.shape[1])
        for array in (self.spots, self.variances, self.instrument_prices, self.action_masks):
            if array is not None:
                return int(array.shape[1] - 1)
        return 0
