"""
Neural Posterior Estimation (NPE) infrastructure for Heston calibration.

Provides:
- Prior distributions and parameter transforms
- JAX-based Heston simulator
- Summary feature extraction for NPE training
"""
from .simulate import (
    PRIOR_LOW,
    PRIOR_HIGH,
    FEATURE_NAMES,
    sample_prior,
    simulate_heston_paths,
    summary_features,
    to_unconstrained,
    to_natural,
)

__all__ = [
    "PRIOR_LOW",
    "PRIOR_HIGH",
    "FEATURE_NAMES",
    "sample_prior",
    "simulate_heston_paths",
    "summary_features",
    "to_unconstrained",
    "to_natural",
]
