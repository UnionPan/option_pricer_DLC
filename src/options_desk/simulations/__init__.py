"""
Simulation Environments for Reinforcement Learning

Provides gym-compatible environments for:
- Heston stochastic volatility model with multi-dimensional trading
- Option hedging and trading on fixed grid
- Portfolio management under partial observability (POMDP)

Features:
- Professional processes.Heston simulation (Milstein/Euler)
- Multi-dimensional action space (underlying + options)
- Fixed grid representation (moneyness × TTM)
- Synthetic equity option chains with realistic pricing
- T=246 steps, dt=1/252 (~1 trading year)

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

from .heston_env import HestonEnv, HestonParams, Liability, make_heston_env
from .merton_env import MertonEnv, MertonParams
from .rough_bergomi_env import RoughBergomiEnv, RoughBergomiParams
from .heston_cache import CachedHestonEnv, build_heston_cache
from .renderer import HestonEnvRenderer


def make_heston_trading_env(**kwargs):
    """Convenience wrapper: Heston trading env without options."""
    return HestonEnv(task='trading', include_options=False, **kwargs)


def make_heston_hedging_env(**kwargs):
    """Convenience wrapper: Heston hedging env with options."""
    return HestonEnv(task='hedging', include_options=True, **kwargs)


def make_heston_no_options_env(**kwargs):
    """Convenience wrapper: Heston env without options."""
    return HestonEnv(include_options=False, **kwargs)

__all__ = [
    'HestonEnv',
    'HestonParams',
    'Liability',
    'make_heston_env',
    'MertonEnv',
    'MertonParams',
    'RoughBergomiEnv',
    'RoughBergomiParams',
    'CachedHestonEnv',
    'build_heston_cache',
    'make_heston_trading_env',
    'make_heston_hedging_env',
    'make_heston_no_options_env',
    'HestonEnvRenderer',
]
