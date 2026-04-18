"""
JAX-native deep hedging scaffold.

This package is the start of the note-aligned deep hedging line:
- pure configuration objects
- floating-grid action masking
- policy/trainer configs for a functional rollout stack

The legacy Gym environments and OO agents remain under
``options_desk.simulations`` and ``options_desk.deep_hedging.agents``.
"""

from .env import (
    DeepHedgingEnvConfig,
    DeepHedgingRollout,
    DeepHedgingState,
    FloatingOptionGrid,
    GRID_COARSE,
    GRID_FINE,
    build_action_mask,
    build_observation,
    build_transaction_cost_vector,
    compute_portfolio_value,
    reset_state,
    rollout_trades,
    step_state,
)
from .policies import (
    PolicyConfig,
    PolicyParams,
    PolicyState,
    apply_action_mask_to_action,
    apply_linear_policy,
    init_linear_policy_params,
    init_policy_state,
)
from .pricing import (
    HestonMarketParams,
    compile_padded_grid,
    price_option_grid,
)
from .rollout import (
    MarketTrajectory,
    replay_rollout,
    simulate_heston_market,
    simulate_heston_market_batch,
)
from .trainers import HedgingLossBreakdown, TrainerConfig, compute_hedging_loss

__all__ = [
    "DeepHedgingEnvConfig",
    "DeepHedgingRollout",
    "DeepHedgingState",
    "FloatingOptionGrid",
    "GRID_COARSE",
    "GRID_FINE",
    "PolicyConfig",
    "PolicyParams",
    "PolicyState",
    "TrainerConfig",
    "apply_action_mask_to_action",
    "apply_linear_policy",
    "build_action_mask",
    "build_observation",
    "build_transaction_cost_vector",
    "compute_hedging_loss",
    "compute_portfolio_value",
    "init_linear_policy_params",
    "init_policy_state",
    "reset_state",
    "rollout_trades",
    "step_state",
    "HedgingLossBreakdown",
    "HestonMarketParams",
    "compile_padded_grid",
    "price_option_grid",
    "MarketTrajectory",
    "simulate_heston_market",
    "simulate_heston_market_batch",
    "replay_rollout",
]
