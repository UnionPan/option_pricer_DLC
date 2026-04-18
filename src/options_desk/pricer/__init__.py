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
from .bachelier_pricer import bachelier_price_vanilla, bachelier_price_slice

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
    # Bachelier
    "bachelier_price_vanilla",
    "bachelier_price_slice",
]

# --- JAX-accelerated pricers (optional, require jax) ---
try:
    from ._jax_mgf_pricer import (
        heston_price_grid_jax, heston_price_slice_fast,
        merton_price_grid_jax, merton_price_slice_fast,
        bates_price_grid_jax, bates_price_slice_fast,
        bs_price_grid_jax,
    )
    __all__ += [
        "heston_price_grid_jax", "heston_price_slice_fast",
        "merton_price_grid_jax", "merton_price_slice_fast",
        "bates_price_grid_jax", "bates_price_slice_fast",
        "bs_price_grid_jax",
    ]
except ImportError:
    pass

try:
    from ._jax_fourier_pricer import (
        # COS pricers (single-strike)
        jax_cos_price_gbm, jax_cos_price_heston, jax_cos_price_merton,
        jax_cos_price_bates, jax_cos_price_kou, jax_cos_price_vg,
        jax_cos_price_nig,
        # COS pricers (multi-strike)
        jax_cos_price_gbm_multi, jax_cos_price_heston_multi,
        jax_cos_price_merton_multi, jax_cos_price_bates_multi,
        jax_cos_price_kou_multi, jax_cos_price_vg_multi,
        jax_cos_price_nig_multi,
        # Carr-Madan FFT pricers
        jax_carr_madan_price_gbm, jax_carr_madan_price_heston,
        jax_carr_madan_price_merton,
    )
    __all__ += [
        "jax_cos_price_gbm", "jax_cos_price_heston", "jax_cos_price_merton",
        "jax_cos_price_bates", "jax_cos_price_kou", "jax_cos_price_vg",
        "jax_cos_price_nig",
        "jax_cos_price_gbm_multi", "jax_cos_price_heston_multi",
        "jax_cos_price_merton_multi", "jax_cos_price_bates_multi",
        "jax_cos_price_kou_multi", "jax_cos_price_vg_multi",
        "jax_cos_price_nig_multi",
        "jax_carr_madan_price_gbm", "jax_carr_madan_price_heston",
        "jax_carr_madan_price_merton",
    ]
except ImportError:
    pass

try:
    from ._jax_fd_pricer import jax_fd_price
    __all__ += ["jax_fd_price"]
except ImportError:
    pass

try:
    from ._jax_fe_pricer import jax_fe_price
    __all__ += ["jax_fe_price"]
except ImportError:
    pass
