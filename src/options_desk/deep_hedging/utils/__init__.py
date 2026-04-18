"""
Shared deep-hedging utilities.

This subpackage holds the framework-neutral pieces that both agents and
training algorithms depend on:

- ``contracts``: dataclass contracts (ObservationBatch, ActionBatch,
  LiabilitySpec, MarketTrajectory, TrajectoryBatch). Pure NumPy.
- ``adapters``: conversions between Gym/JAX observations and the shared
  contracts, and between contracts and Torch tensors.
- ``rollouts``: boundary helpers that drive an inference-only agent against
  a market trajectory and replay trades through the JAX rollout kernel.
"""

from .contracts import (
    ActionBatch,
    LiabilitySpec,
    MarketTrajectory,
    ObservationBatch,
    TrajectoryBatch,
)
from .adapters import (
    adapt_market_trajectory_batch,
    adapt_observation_batch,
    adapt_rollout_batch,
    floating_grid_to_gym_grid,
    trajectory_batch_to_torch,
)
from .rollouts import collect_agent_rollout_from_market

__all__ = [
    "ActionBatch",
    "LiabilitySpec",
    "MarketTrajectory",
    "ObservationBatch",
    "TrajectoryBatch",
    "adapt_market_trajectory_batch",
    "adapt_observation_batch",
    "adapt_rollout_batch",
    "floating_grid_to_gym_grid",
    "trajectory_batch_to_torch",
    "collect_agent_rollout_from_market",
]
