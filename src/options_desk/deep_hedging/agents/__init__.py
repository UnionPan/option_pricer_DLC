"""
Hedging Agents

Provides agent implementations for option hedging:
- BaseHedgingAgent: Abstract base class for all agents
- RandomAgent: Baseline random agent
- DoNothingAgent: Baseline no-trading agent
- DeltaHedgingAgent: Analytical delta hedging using Black-Scholes
- DeepHedgingAgent: Neural network-based deep hedging (REINFORCE)

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

from .base import BaseHedgingAgent, RandomAgent, DoNothingAgent
from .delta_hedging import DeltaHedgingAgent
from .deep_hedging_agent import DeepHedgingAgent

try:
    from .torch_policy import HedgingMLPPolicy, TorchPolicyAgent, obs_batch_to_tensor

    _TORCH_POLICY_AVAILABLE = True
except ImportError:
    _TORCH_POLICY_AVAILABLE = False

__all__ = [
    'BaseHedgingAgent',
    'RandomAgent',
    'DoNothingAgent',
    'DeltaHedgingAgent',
    'DeepHedgingAgent',
]

if _TORCH_POLICY_AVAILABLE:
    __all__ += ['HedgingMLPPolicy', 'TorchPolicyAgent', 'obs_batch_to_tensor']
