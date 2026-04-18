"""
Deep Hedging Framework

Provides agents and utilities for option hedging:
- Analytical hedging (Delta, Delta-Gamma)
- Deep hedging with neural networks
- Reinforcement learning hedging agents

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

from .agents import (
    BaseHedgingAgent,
    RandomAgent,
    DoNothingAgent,
    DeltaHedgingAgent,
    DeepHedgingAgent,
)
from .utils import (
    ObservationBatch,
    ActionBatch,
    LiabilitySpec,
    MarketTrajectory,
    TrajectoryBatch,
    adapt_market_trajectory_batch,
    adapt_observation_batch,
    adapt_rollout_batch,
    floating_grid_to_gym_grid,
    trajectory_batch_to_torch,
    collect_agent_rollout_from_market,
)

try:
    from .agents.torch_policy import HedgingMLPPolicy, TorchPolicyAgent
    from .training import (
        BaseTrainer,
        BuehlerTrainer,
        TrainerConfig,
        differentiable_rollout,
        buehler_loss,
    )

    _TORCH_TRAINER_AVAILABLE = True
except ImportError:
    _TORCH_TRAINER_AVAILABLE = False

__all__ = [
    'BaseHedgingAgent',
    'RandomAgent',
    'DoNothingAgent',
    'DeltaHedgingAgent',
    'DeepHedgingAgent',
    'ObservationBatch',
    'ActionBatch',
    'LiabilitySpec',
    'MarketTrajectory',
    'TrajectoryBatch',
    'adapt_market_trajectory_batch',
    'adapt_observation_batch',
    'adapt_rollout_batch',
    'floating_grid_to_gym_grid',
    'trajectory_batch_to_torch',
    'collect_agent_rollout_from_market',
]

if _TORCH_TRAINER_AVAILABLE:
    __all__ += [
        'HedgingMLPPolicy',
        'TorchPolicyAgent',
        'BaseTrainer',
        'BuehlerTrainer',
        'TrainerConfig',
        'differentiable_rollout',
        'buehler_loss',
    ]
