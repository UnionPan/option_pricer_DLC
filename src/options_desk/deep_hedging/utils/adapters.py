"""
Adapters between environment-specific formats and shared deep hedging contracts.

This module avoids runtime JAX imports so Gym+PyTorch workflows can use it
without pulling in JAX backend initialization. Torch imports are function-local.
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

import numpy as np

from .contracts import MarketTrajectory, ObservationBatch, TrajectoryBatch

if TYPE_CHECKING:
    from ..jax import DeepHedgingRollout, FloatingOptionGrid

# Shared empty feature placeholder -- (1, 0) float32.
# Always copy before handing to a frozen dataclass to avoid aliasing.
_EMPTY_FEATURE = np.zeros((1, 0), dtype=np.float32)


def _as_batch_2d(values: Any) -> np.ndarray:
    """Coerce scalars and 1-D arrays to ``(1, features)`` shape."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array.reshape(1, -1)
    return array.astype(np.float32)


def _as_batch_first(array: np.ndarray, expected_ndim: int, name: str) -> np.ndarray:
    """Normalize single-path arrays to batch-first arrays."""
    result = np.asarray(array)
    if result.ndim == expected_ndim - 1:
        result = result[np.newaxis, ...]
    if result.ndim != expected_ndim:
        raise ValueError(
            f"{name} must have {expected_ndim - 1} or {expected_ndim} dims, got shape {result.shape}"
        )
    return result


def adapt_observation_batch(
    observation: dict[str, Any],
    info: Optional[dict[str, Any]] = None,
) -> ObservationBatch:
    """
    Convert Gym-style or JAX-style observations into a shared batch contract.

    Gym envs use ``"spot_price"`` as their spot key; JAX envs use ``"spot"``.
    The adapter dispatches on this key to extract fields appropriately.
    """

    info = info or {}

    if "spot_price" in observation:
        spot = _as_batch_2d(observation["spot_price"])
        time_index = _as_batch_2d(
            observation.get("time_step", np.array([0.0], dtype=np.float32))
        )
        option_features = _as_batch_2d(
            observation.get("option_features", _EMPTY_FEATURE)
        )

        if "portfolio_weights" in observation:
            portfolio_features = _as_batch_2d(observation["portfolio_weights"])
        elif "position" in observation and "portfolio_value" in observation:
            portfolio_features = _as_batch_2d(
                np.concatenate(
                    [
                        np.asarray(observation["position"], dtype=np.float32).ravel(),
                        np.asarray(observation["portfolio_value"], dtype=np.float32).ravel(),
                    ]
                )
            )
        elif "positions" in info:
            portfolio_features = _as_batch_2d(info["positions"])
        else:
            portfolio_features = _EMPTY_FEATURE.copy()

        previous_action = _EMPTY_FEATURE.copy()
        context_features = _EMPTY_FEATURE.copy()
        action_mask = None
    else:
        spot = _as_batch_2d(
            observation.get("spot", np.array([0.0], dtype=np.float32))
        )

        if "time_index" in observation:
            time_index = _as_batch_2d(observation["time_index"])
        elif "time_features" in observation:
            tf = np.asarray(observation["time_features"], dtype=np.float32)
            time_index = _as_batch_2d(tf[0:1])
        else:
            time_index = _as_batch_2d(np.array([0.0], dtype=np.float32))

        option_features = _as_batch_2d(
            observation.get("option_features", _EMPTY_FEATURE)
        )

        if "positions" in observation:
            portfolio_features = _as_batch_2d(observation["positions"])
        else:
            portfolio_features = _EMPTY_FEATURE.copy()

        previous_action = _as_batch_2d(
            observation.get("previous_action", _EMPTY_FEATURE)
        )

        context_parts: list[np.ndarray] = []
        if "variance" in observation:
            context_parts.append(
                np.asarray(observation["variance"], dtype=np.float32).ravel()
            )
        if "time_features" in observation:
            context_parts.append(
                np.asarray(observation["time_features"], dtype=np.float32).ravel()
            )
        context_features = (
            _as_batch_2d(np.concatenate(context_parts, axis=0))
            if context_parts
            else _EMPTY_FEATURE.copy()
        )

        action_mask_value = observation.get("action_mask")
        action_mask = (
            None
            if action_mask_value is None
            else np.asarray(action_mask_value, dtype=bool).reshape(1, -1)
        )

    return ObservationBatch(
        spot=spot,
        time_index=time_index,
        option_features=option_features,
        portfolio_features=portfolio_features,
        previous_action=previous_action,
        context_features=context_features,
        action_mask=action_mask,
    )


def floating_grid_to_gym_grid(
    grid: FloatingOptionGrid,
    dt: float = 1.0 / 250.0,
) -> dict[int, list[float]]:
    """
    Convert a JAX ``FloatingOptionGrid`` (maturities in env steps) to the
    Gym ``option_grid`` format (maturities in calendar days).
    """
    return {
        int(round(maturity * dt * 252)): list(grid.moneyness_by_maturity[maturity])
        for maturity in grid.maturities
    }


def adapt_rollout_batch(
    rollout: DeepHedgingRollout,
    rewards: np.ndarray,
) -> TrajectoryBatch:
    """
    Convert a JAX replay result into a shared batch-first trajectory batch.
    """

    rewards_array = np.asarray(rewards, dtype=np.float32).reshape(1, -1)
    positions = np.asarray(rollout.positions, dtype=np.float32).reshape(
        1, rollout.positions.shape[0], rollout.positions.shape[1]
    )
    portfolio_values = np.asarray(rollout.portfolio_values, dtype=np.float32).reshape(1, -1)

    horizon = rewards_array.shape[1]
    if positions.shape[1] != horizon:
        raise ValueError(
            f"positions length {positions.shape[1]} != rewards length {horizon}"
        )
    if portfolio_values.shape[1] != horizon:
        raise ValueError(
            f"portfolio_values length {portfolio_values.shape[1]} != rewards length {horizon}"
        )

    dones = np.zeros((1, horizon), dtype=bool)
    if horizon > 0:
        dones[0, -1] = True

    return TrajectoryBatch(
        rewards=rewards_array,
        dones=dones,
        positions=positions,
        portfolio_values=portfolio_values,
        observations=None,
        actions=None,
    )


def adapt_market_trajectory_batch(trajectory: MarketTrajectory) -> TrajectoryBatch:
    """
    Convert a JAX market trajectory into the shared batch-first contract.
    """

    spots = _as_batch_first(np.asarray(trajectory.spots, dtype=np.float32), 2, "spots")
    variances = _as_batch_first(
        np.asarray(trajectory.variances, dtype=np.float32),
        2,
        "variances",
    )
    instrument_prices = _as_batch_first(
        np.asarray(trajectory.instrument_prices, dtype=np.float32),
        3,
        "instrument_prices",
    )
    action_masks = _as_batch_first(
        np.asarray(trajectory.action_masks, dtype=bool),
        3,
        "action_masks",
    )

    if spots.shape != variances.shape:
        raise ValueError(
            f"variances shape {variances.shape} != spots shape {spots.shape}"
        )
    if instrument_prices.shape[0] != spots.shape[0] and instrument_prices.shape[0] != 1:
        raise ValueError(
            "instrument_prices batch dimension must match spots or be singleton"
        )
    if instrument_prices.shape[1] != spots.shape[1]:
        raise ValueError(
            f"instrument_prices time dimension {instrument_prices.shape[1]} != spots time dimension {spots.shape[1]}"
        )

    batch_size = max(spots.shape[0], instrument_prices.shape[0], action_masks.shape[0])

    if spots.shape[0] == 1 and batch_size > 1:
        spots = np.repeat(spots, batch_size, axis=0)
        variances = np.repeat(variances, batch_size, axis=0)
    if instrument_prices.shape[0] == 1 and batch_size > 1:
        instrument_prices = np.repeat(instrument_prices, batch_size, axis=0)
    if action_masks.shape[0] == 1 and batch_size > 1:
        action_masks = np.repeat(action_masks, batch_size, axis=0)

    if variances.shape[0] != batch_size:
        raise ValueError("variances batch dimension does not match normalized batch size")
    if instrument_prices.shape[0] != batch_size:
        raise ValueError("instrument_prices batch dimension does not match normalized batch size")
    if action_masks.shape[0] != batch_size:
        raise ValueError("action_masks batch dimension does not match normalized batch size")
    if action_masks.shape[1:] != instrument_prices.shape[1:]:
        raise ValueError(
            f"action_masks shape {action_masks.shape} incompatible with instrument_prices shape {instrument_prices.shape}"
        )

    return TrajectoryBatch(
        spots=spots,
        variances=variances,
        instrument_prices=instrument_prices,
        action_masks=action_masks,
    )


def trajectory_batch_to_torch(
    batch: TrajectoryBatch,
    device: str | None = None,
) -> dict[str, Any]:
    """
    Convert a shared trajectory batch into Torch tensors.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only when torch missing
        raise ImportError("trajectory_batch_to_torch requires PyTorch") from exc

    tensor_batch: dict[str, Any] = {}
    for key in (
        "rewards",
        "dones",
        "positions",
        "portfolio_values",
        "actions",
        "trades",
        "spots",
        "variances",
        "instrument_prices",
        "action_masks",
        "terminal_liability_payoffs",
        "initial_cash",
    ):
        value = getattr(batch, key)
        if value is None:
            continue

        array = np.asarray(value)
        if array.dtype == np.bool_:
            tensor = torch.as_tensor(array, dtype=torch.bool, device=device)
        else:
            tensor = torch.as_tensor(array, dtype=torch.float32, device=device)
        tensor_batch[key] = tensor

    return tensor_batch
