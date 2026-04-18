"""
Boundary rollout helpers between market trajectories and agent APIs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .contracts import LiabilitySpec, MarketTrajectory, ObservationBatch, TrajectoryBatch


def _ensure_single_path_trajectory(trajectory: MarketTrajectory) -> MarketTrajectory:
    """Normalize a market trajectory to single-path arrays."""
    spots = np.asarray(trajectory.spots, dtype=np.float64)
    variances = np.asarray(trajectory.variances, dtype=np.float64)
    prices = np.asarray(trajectory.instrument_prices, dtype=np.float64)
    masks = np.asarray(trajectory.action_masks, dtype=bool)

    if spots.ndim == 2:
        if spots.shape[0] != 1:
            raise ValueError("collect_agent_rollout_from_market currently supports single-path trajectories only")
        spots = spots[0]
        variances = variances[0]
        prices = prices[0]
    if masks.ndim == 3:
        if masks.shape[0] != 1:
            raise ValueError("collect_agent_rollout_from_market currently supports single-path masks only")
        masks = masks[0]

    return MarketTrajectory(
        spots=spots,
        variances=variances,
        instrument_prices=prices,
        action_masks=masks,
    )


def _build_market_observation(
    trajectory: MarketTrajectory,
    time_index: int,
    horizon_steps: int,
    positions: np.ndarray,
    previous_action: np.ndarray,
) -> ObservationBatch:
    """Build the shared observation contract from a market state snapshot."""
    time_ratio = np.float32(time_index / horizon_steps) if horizon_steps > 0 else np.float32(0.0)
    return ObservationBatch(
        spot=np.asarray([[trajectory.spots[time_index]]], dtype=np.float32),
        time_index=np.asarray([[time_ratio]], dtype=np.float32),
        option_features=np.asarray(
            trajectory.instrument_prices[time_index, 1:],
            dtype=np.float32,
        ).reshape(1, -1),
        portfolio_features=np.asarray(positions, dtype=np.float32).reshape(1, -1),
        previous_action=np.asarray(previous_action, dtype=np.float32).reshape(1, -1),
        context_features=np.asarray(
            [[trajectory.variances[time_index], time_ratio, 1.0]],
            dtype=np.float32,
        ),
        action_mask=np.asarray(trajectory.action_masks[time_index], dtype=bool).reshape(1, -1),
    )


def _coerce_trade(
    action: np.ndarray,
    positions: np.ndarray,
    action_mask: np.ndarray,
    action_mode: str,
) -> np.ndarray:
    """Interpret agent output as target positions or direct trades."""
    action_array = np.asarray(action, dtype=np.float32).reshape(-1)
    positions_array = np.asarray(positions, dtype=np.float32).reshape(-1)
    mask = np.asarray(action_mask, dtype=bool).reshape(-1)

    if action_array.shape != positions_array.shape or action_array.shape != mask.shape:
        raise ValueError(
            f"action, positions, and action_mask must share shape, got {action_array.shape}, {positions_array.shape}, {mask.shape}"
        )

    if action_mode == "target_positions":
        target_positions = np.where(mask, action_array, 0.0).astype(np.float32)
        return (target_positions - positions_array).astype(np.float32)
    if action_mode == "trades":
        masked_trades = np.where(mask, action_array, 0.0).astype(np.float32)
        return np.where(mask, masked_trades, -positions_array).astype(np.float32)

    raise ValueError(f"unsupported action_mode: {action_mode}")


def _compute_rewards(
    portfolio_values: np.ndarray,
    initial_cash: float,
    terminal_liability_payoff: float,
) -> np.ndarray:
    """Convert mark-to-market portfolio values into per-step rewards."""
    previous_values = np.concatenate(
        [np.asarray([initial_cash], dtype=np.float32), portfolio_values[:-1].astype(np.float32)]
    )
    rewards = portfolio_values.astype(np.float32) - previous_values
    rewards[-1] = rewards[-1] - np.float32(terminal_liability_payoff)
    return rewards.astype(np.float32)


def collect_agent_rollout_from_market(
    agent: Any,
    config: Any,
    trajectory: MarketTrajectory,
    liability: LiabilitySpec | None = None,
    initial_cash: float = 0.0,
    action_mode: str = "target_positions",
    update_agent: bool = True,
) -> TrajectoryBatch:
    """
    Run an agent step-by-step against a market trajectory and replay resulting trades.
    """
    from ..jax.rollout import replay_rollout

    single_path = _ensure_single_path_trajectory(trajectory)
    horizon = int(config.horizon_steps)
    n_instruments = int(config.option_grid.n_instruments)

    if single_path.instrument_prices.shape != (horizon + 1, n_instruments):
        raise ValueError(
            f"trajectory instrument_prices shape {single_path.instrument_prices.shape} != expected {(horizon + 1, n_instruments)}"
        )
    if single_path.action_masks.shape != (horizon + 1, n_instruments):
        raise ValueError(
            f"trajectory action_masks shape {single_path.action_masks.shape} != expected {(horizon + 1, n_instruments)}"
        )

    raw_actions = np.zeros((horizon, n_instruments), dtype=np.float32)
    trades = np.zeros((horizon, n_instruments), dtype=np.float32)
    positions_after = np.zeros((horizon, n_instruments), dtype=np.float32)
    next_observations: list[ObservationBatch] = []

    positions = np.zeros(n_instruments, dtype=np.float32)
    previous_action = np.zeros(n_instruments, dtype=np.float32)

    initial_observation = _build_market_observation(
        single_path,
        time_index=0,
        horizon_steps=horizon,
        positions=positions,
        previous_action=previous_action,
    )
    agent.reset(initial_observation, {})

    for time_index in range(horizon):
        observation = _build_market_observation(
            single_path,
            time_index=time_index,
            horizon_steps=horizon,
            positions=positions,
            previous_action=previous_action,
        )
        raw_action = np.asarray(agent.act(observation, {}), dtype=np.float32).reshape(-1)
        if raw_action.shape != (n_instruments,):
            raise ValueError(
                f"agent action shape {raw_action.shape} != expected {(n_instruments,)}"
            )

        trade = _coerce_trade(
            action=raw_action,
            positions=positions,
            action_mask=single_path.action_masks[time_index],
            action_mode=action_mode,
        )

        raw_actions[time_index] = raw_action
        trades[time_index] = trade

        positions = positions + trade
        previous_action = trade
        positions_after[time_index] = positions
        next_observations.append(
            _build_market_observation(
                single_path,
                time_index=time_index + 1,
                horizon_steps=horizon,
                positions=positions,
                previous_action=previous_action,
            )
        )

    replay = replay_rollout(
        config=config,
        trajectory=single_path,
        trades=trades,
        initial_cash=initial_cash,
    )
    terminal_liability_payoff = np.float32(0.0)
    if liability is not None:
        terminal_liability_payoff = np.float32(liability.terminal_payoff(single_path.spots[-1]))

    rewards = _compute_rewards(
        portfolio_values=np.asarray(replay.portfolio_values, dtype=np.float32),
        initial_cash=initial_cash,
        terminal_liability_payoff=float(terminal_liability_payoff),
    )
    dones = np.zeros((horizon,), dtype=bool)
    if horizon > 0:
        dones[-1] = True

    if update_agent and hasattr(agent, "update"):
        for time_index in range(horizon):
            agent.update(
                next_observations[time_index],
                float(rewards[time_index]),
                bool(dones[time_index]),
                False,
                {},
            )

    return TrajectoryBatch(
        rewards=rewards.reshape(1, -1),
        dones=dones.reshape(1, -1),
        positions=np.asarray(replay.positions, dtype=np.float32).reshape(1, horizon, n_instruments),
        portfolio_values=np.asarray(replay.portfolio_values, dtype=np.float32).reshape(1, -1),
        actions=raw_actions.reshape(1, horizon, n_instruments),
        trades=trades.reshape(1, horizon, n_instruments),
        spots=np.asarray(single_path.spots, dtype=np.float32).reshape(1, -1),
        variances=np.asarray(single_path.variances, dtype=np.float32).reshape(1, -1),
        instrument_prices=np.asarray(single_path.instrument_prices, dtype=np.float32).reshape(
            1, horizon + 1, n_instruments
        ),
        action_masks=np.asarray(single_path.action_masks, dtype=bool).reshape(
            1, horizon + 1, n_instruments
        ),
        terminal_liability_payoffs=np.asarray([terminal_liability_payoff], dtype=np.float32),
        initial_cash=np.asarray([initial_cash], dtype=np.float32),
    )
