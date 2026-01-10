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
)

__all__ = [
    'BaseHedgingAgent',
    'RandomAgent',
    'DoNothingAgent',
    'DeltaHedgingAgent',
]
