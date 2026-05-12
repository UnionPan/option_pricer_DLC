"""
Shared data contracts for deep hedging env/agent interoperability.

All contract types use NumPy arrays only -- no JAX or PyTorch dependencies.
This keeps the contract layer import-safe for Gym, JAX, and PyTorch stacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

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


_BARRIER_TYPES = ("up_and_out", "up_and_in", "down_and_out", "down_and_in")
_AVERAGE_TYPES = ("arithmetic", "geometric")
_LOOKBACK_TYPES = ("floating", "fixed")
_OPTION_TYPES = ("call", "put")


@dataclass(frozen=True)
class LiabilitySpec:
    """
    Terminal liability definition for hedging objectives.

    Supported ``kind`` values (single-asset European-style; early-exercise,
    multi-asset, and rates products are out of scope for the current Buehler
    env):

    * ``'call'`` / ``'put'`` — vanilla European: payoff = ``(S_T − K)⁺`` or
      ``(K − S_T)⁺``.
    * ``'cliquet'`` — locally-capped, globally-floored cliquet
      (Buehler et al. 2019, eq. ψ in dh.tex)::

          ψ(x) = max[ Σ_{i=1..m} min(x_{τ_i}/x_{τ_{i-1}} − 1, cap), floor ]

      ``τ_1 < … < τ_m = maturity`` are reset-date indices in ``reset_dates``;
      ``τ_0 = 0`` is implicit.
    * ``'barrier'`` — knock-in/knock-out call/put. Requires ``option_type``
      (``'call'``/``'put'``), ``strike``, ``barrier``, and ``barrier_type``
      (one of ``'up_and_out'``/``'up_and_in'``/``'down_and_out'``/
      ``'down_and_in'``). Barrier is monitored at every env step.
    * ``'asian'`` — average-price call/put on the path mean. Requires
      ``option_type``, ``strike``, and ``average_type``
      (``'arithmetic'``/``'geometric'``). Averages over every env step from
      ``t=0`` through ``t=maturity``.
    * ``'lookback'`` — payoff against path max/min. Requires ``option_type``
      and ``lookback_type`` (``'floating'``/``'fixed'``). For ``'fixed'``,
      also requires ``strike``; for ``'floating'``, ``strike`` is unused.

    All non-required fields default to neutral values so older
    constructions (e.g. ``LiabilitySpec(kind='call', strike=…, maturity=…)``)
    continue to work unchanged.
    """

    kind: str
    strike: float = 0.0
    maturity: int = 0
    quantity: float = 1.0
    # --- cliquet-only fields ---
    cap: float = 0.0
    floor: float = 0.0
    reset_dates: tuple[int, ...] = ()
    # --- barrier/asian/lookback option direction (call vs put) ---
    option_type: str = ""
    # --- barrier-only fields ---
    barrier: float = 0.0
    barrier_type: str = ""  # one of _BARRIER_TYPES
    # --- asian-only fields ---
    average_type: str = "arithmetic"  # one of _AVERAGE_TYPES
    # --- lookback-only fields ---
    lookback_type: str = ""  # one of _LOOKBACK_TYPES

    def __post_init__(self) -> None:
        if self.kind in ("call", "put"):
            if self.strike <= 0:
                raise ValueError(f"{self.kind!r} requires strike > 0")
            if self.maturity <= 0:
                raise ValueError(f"{self.kind!r} requires maturity > 0")
        elif self.kind == "cliquet":
            if not self.reset_dates:
                raise ValueError("cliquet requires non-empty reset_dates")
            resets = list(self.reset_dates)
            if any(d <= 0 for d in resets):
                raise ValueError("cliquet reset_dates must be positive")
            if resets != sorted(resets) or len(set(resets)) != len(resets):
                raise ValueError("cliquet reset_dates must be strictly increasing")
            if self.cap <= 0:
                raise ValueError("cliquet requires cap > 0")
            if self.maturity != resets[-1]:
                raise ValueError(
                    f"cliquet maturity ({self.maturity}) must equal last reset "
                    f"date ({resets[-1]})"
                )
        elif self.kind == "barrier":
            if self.option_type not in _OPTION_TYPES:
                raise ValueError(
                    f"barrier requires option_type in {_OPTION_TYPES}, "
                    f"got {self.option_type!r}"
                )
            if self.strike <= 0:
                raise ValueError("barrier requires strike > 0")
            if self.barrier <= 0:
                raise ValueError("barrier requires barrier > 0")
            if self.barrier_type not in _BARRIER_TYPES:
                raise ValueError(
                    f"barrier requires barrier_type in {_BARRIER_TYPES}, "
                    f"got {self.barrier_type!r}"
                )
            if self.maturity <= 0:
                raise ValueError("barrier requires maturity > 0")
        elif self.kind == "asian":
            if self.option_type not in _OPTION_TYPES:
                raise ValueError(
                    f"asian requires option_type in {_OPTION_TYPES}, "
                    f"got {self.option_type!r}"
                )
            if self.strike <= 0:
                raise ValueError("asian requires strike > 0")
            if self.average_type not in _AVERAGE_TYPES:
                raise ValueError(
                    f"asian requires average_type in {_AVERAGE_TYPES}, "
                    f"got {self.average_type!r}"
                )
            if self.maturity <= 0:
                raise ValueError("asian requires maturity > 0")
        elif self.kind == "lookback":
            if self.option_type not in _OPTION_TYPES:
                raise ValueError(
                    f"lookback requires option_type in {_OPTION_TYPES}, "
                    f"got {self.option_type!r}"
                )
            if self.lookback_type not in _LOOKBACK_TYPES:
                raise ValueError(
                    f"lookback requires lookback_type in {_LOOKBACK_TYPES}, "
                    f"got {self.lookback_type!r}"
                )
            if self.lookback_type == "fixed" and self.strike <= 0:
                raise ValueError("fixed-strike lookback requires strike > 0")
            if self.maturity <= 0:
                raise ValueError("lookback requires maturity > 0")
        else:
            raise ValueError(f"unsupported liability kind: {self.kind!r}")

    @property
    def is_path_dependent(self) -> bool:
        """True if payoff depends on the full path, not just terminal spot."""
        return self.kind in ("cliquet", "barrier", "asian", "lookback")

    def terminal_payoff(self, terminal_spot: np.ndarray | float) -> NDArray[np.float32]:
        """Compute terminal payoff for vanilla call/put only.

        For path-dependent kinds (cliquet, barrier, asian, lookback) use
        :meth:`terminal_payoff_from_path` instead.
        """
        if self.is_path_dependent:
            raise ValueError(
                f"{self.kind!r} payoff is path-dependent; call "
                "terminal_payoff_from_path(spots) instead"
            )
        spot = np.asarray(terminal_spot, dtype=np.float32)
        if self.kind == "call":
            intrinsic = np.maximum(spot - np.float32(self.strike), 0.0)
        elif self.kind == "put":
            intrinsic = np.maximum(np.float32(self.strike) - spot, 0.0)
        else:
            raise ValueError(f"unsupported liability kind: {self.kind!r}")
        return (np.float32(self.quantity) * intrinsic).astype(np.float32)

    def terminal_payoff_from_path(
        self, spots: np.ndarray
    ) -> NDArray[np.float32]:
        """Compute terminal payoff from the full spot path.

        Works for all liability kinds. ``spots`` may be shape ``(T+1,)`` for a
        single path or ``(B, T+1)`` for a batch.
        """
        spots_arr = np.asarray(spots, dtype=np.float32)
        if self.kind in ("call", "put"):
            return self.terminal_payoff(spots_arr[..., -1])

        if self.kind == "cliquet":
            reset_idx = [0, *self.reset_dates]
            spots_at = spots_arr[..., reset_idx]                        # (..., m+1)
            period_ret = spots_at[..., 1:] / spots_at[..., :-1] - 1.0   # (..., m)
            capped = np.minimum(period_ret, np.float32(self.cap))
            summed = capped.sum(axis=-1)
            floored = np.maximum(summed, np.float32(self.floor))
            return (np.float32(self.quantity) * floored).astype(np.float32)

        if self.kind == "barrier":
            S_T = spots_arr[..., -1]
            if self.option_type == "call":
                european = np.maximum(S_T - np.float32(self.strike), 0.0)
            else:
                european = np.maximum(np.float32(self.strike) - S_T, 0.0)
            barrier = np.float32(self.barrier)
            if self.barrier_type.startswith("up"):
                triggered = (spots_arr >= barrier).any(axis=-1)
            else:
                triggered = (spots_arr <= barrier).any(axis=-1)
            if self.barrier_type.endswith("out"):
                alive = (~triggered).astype(np.float32)
            else:  # *_in
                alive = triggered.astype(np.float32)
            return (np.float32(self.quantity) * european * alive).astype(np.float32)

        if self.kind == "asian":
            if self.average_type == "arithmetic":
                avg = spots_arr.mean(axis=-1)
            else:  # geometric
                avg = np.exp(np.log(spots_arr).mean(axis=-1))
            if self.option_type == "call":
                intrinsic = np.maximum(avg - np.float32(self.strike), 0.0)
            else:
                intrinsic = np.maximum(np.float32(self.strike) - avg, 0.0)
            return (np.float32(self.quantity) * intrinsic).astype(np.float32)

        if self.kind == "lookback":
            S_T = spots_arr[..., -1]
            S_max = spots_arr.max(axis=-1)
            S_min = spots_arr.min(axis=-1)
            if self.lookback_type == "floating":
                if self.option_type == "call":
                    payoff = S_T - S_min          # always >= 0
                else:
                    payoff = S_max - S_T          # always >= 0
            else:  # fixed
                if self.option_type == "call":
                    payoff = np.maximum(S_max - np.float32(self.strike), 0.0)
                else:
                    payoff = np.maximum(np.float32(self.strike) - S_min, 0.0)
            return (np.float32(self.quantity) * payoff).astype(np.float32)

        raise ValueError(f"unsupported liability kind: {self.kind!r}")


# Multi-leg liability: either a single LiabilitySpec or a sequence of legs.
LiabilityPortfolio = Union[LiabilitySpec, Sequence[LiabilitySpec]]


def _normalize_legs(liability: LiabilityPortfolio) -> tuple[LiabilitySpec, ...]:
    """Normalize a single spec or a sequence of specs to a tuple."""
    if isinstance(liability, LiabilitySpec):
        return (liability,)
    legs = tuple(liability)
    if not legs:
        raise ValueError("multi-leg liability requires at least one LiabilitySpec")
    for i, leg in enumerate(legs):
        if not isinstance(leg, LiabilitySpec):
            raise TypeError(
                f"leg {i} must be a LiabilitySpec, got {type(leg).__name__}"
            )
    return legs


def total_payoff_from_path(
    liability: LiabilityPortfolio, spots: np.ndarray
) -> NDArray[np.float32]:
    """Sum of leg payoffs along the spot path. Accepts single or multi-leg."""
    legs = _normalize_legs(liability)
    total = legs[0].terminal_payoff_from_path(spots)
    for leg in legs[1:]:
        total = total + leg.terminal_payoff_from_path(spots)
    return total.astype(np.float32)


def liability_max_maturity(liability: LiabilityPortfolio) -> int:
    """Max maturity across all legs (env-step index)."""
    return max(leg.maturity for leg in _normalize_legs(liability))


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
