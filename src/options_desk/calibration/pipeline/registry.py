"""
Model registry: model name -> fit function over one asset's 1-D price array.

Adding a model to the universe pipeline is one ``register_model`` call.
Fit functions return a FLAT dict of scalars (params + diagnostics); the
runner adds bookkeeping columns (ticker, sector, error, calibration_date).

Phase 2 will register JAX ``fit_batch`` implementations under the same
names; the per-asset ``fit`` path here is the scipy reference route.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Callable

import numpy as np

# (prices_1d, dt) -> flat dict of scalar params/diagnostics
FitFn = Callable[[np.ndarray, float], dict]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    fit: FitFn
    min_obs: int = 60


_REGISTRY: dict[str, ModelSpec] = {}


def register_model(spec: ModelSpec) -> None:
    if spec.name in _REGISTRY:
        raise ValueError(f"model '{spec.name}' already registered")
    _REGISTRY[spec.name] = spec


def get_model(name: str) -> ModelSpec:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown model '{name}'; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_models() -> list[str]:
    return sorted(_REGISTRY)


def _scalars(result) -> dict:
    """Flatten a calibration result dataclass to its scalar fields."""
    d = asdict(result) if is_dataclass(result) else dict(result)
    return {k: v for k, v in d.items()
            if isinstance(v, (int, float, bool, str, np.floating, np.integer))}


def _fit_gbm(prices: np.ndarray, dt: float) -> dict:
    from ..physical.gbm_calibrator import GBMCalibrator
    return _scalars(GBMCalibrator().fit(prices, dt=dt))


def _fit_garch(prices: np.ndarray, dt: float) -> dict:
    from ..physical.garch_calibrator import GARCHCalibrator
    return _scalars(GARCHCalibrator().fit(prices, dt=dt))


def _fit_heston_qmle(prices: np.ndarray, dt: float) -> dict:
    from ..physical.heston_qmle import HestonQMLECalibrator
    return _scalars(HestonQMLECalibrator(smooth_window=10).fit(prices, dt=dt))


register_model(ModelSpec(name="gbm", fit=_fit_gbm, min_obs=60))
register_model(ModelSpec(name="garch", fit=_fit_garch, min_obs=250))
register_model(ModelSpec(name="heston_qmle", fit=_fit_heston_qmle, min_obs=60))
