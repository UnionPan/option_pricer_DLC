"""
Pricer module - Comprehensive derivative pricing methods

author: Yunian Pan
email: yp1170@nyu.edu
"""

# Base classes
from .base import Pricer, PricingResult

# Monte Carlo pricer
from .monte_carlo import MonteCarloPricer

# Analytical pricers
from .analytical import BlackScholesPricer, HestonAnalyticalPricer

# Fourier-based pricers
from .fourier import COSPricer, CarrMadanPricer

# PDE pricers
from .finite_difference import FiniteDifferencePricer, AdaptiveFiniteDifferencePricer
from .finite_element import FiniteElementPricer, HighOrderFiniteElementPricer

# MGF-based pricers
from .heston_mgf_pricer import heston_price_vanilla, heston_price_slice
from .merton_mgf_pricer import merton_price_vanilla, merton_price_slice

__all__ = [
    # Base
    "Pricer",
    "PricingResult",
    # Monte Carlo
    "MonteCarloPricer",
    # Analytical
    "BlackScholesPricer",
    "HestonAnalyticalPricer",
    # Fourier
    "COSPricer",
    "CarrMadanPricer",
    # PDE
    "FiniteDifferencePricer",
    "AdaptiveFiniteDifferencePricer",
    "FiniteElementPricer",
    "HighOrderFiniteElementPricer",
    # MGF
    "heston_price_vanilla",
    "heston_price_slice",
    "merton_price_vanilla",
    "merton_price_slice",
]
