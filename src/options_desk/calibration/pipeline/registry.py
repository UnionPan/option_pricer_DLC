"""
Model registry: model name -> fit function over one asset's 1-D price array.

Adding a model to the universe pipeline is one ``register_model`` call.
Fit functions return a FLAT dict of scalars (params + diagnostics); the
runner adds bookkeeping columns (ticker, sector, error, calibration_date).

Phase 2 will register JAX ``fit_batch`` implementations under the same
names; the per-asset ``fit`` path here is the scipy reference route.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Callable

import numpy as np

# (prices_1d, dt) -> flat dict of scalar params/diagnostics
FitFn = Callable[[np.ndarray, float], dict]

# (list[prices_1d], dt) -> dict of (N,)-arrays
BatchFitFn = Callable[[list[np.ndarray], float], dict]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    fit: FitFn
    min_obs: int = 60
    fit_batch: BatchFitFn | None = None
    needs_ohlc: bool = False


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
            if isinstance(v, (int, float, bool, str, np.floating, np.integer, np.bool_))}


def _fit_gbm(prices: np.ndarray, dt: float) -> dict:
    from ..physical.gbm_calibrator import GBMCalibrator
    return _scalars(GBMCalibrator().fit(prices, dt=dt))


def _fit_garch(prices: np.ndarray, dt: float) -> dict:
    from ..physical.garch_calibrator import GARCHCalibrator
    return _scalars(GARCHCalibrator().fit(prices, dt=dt))


def _fit_heston_qmle(prices: np.ndarray, dt: float) -> dict:
    from ..physical.heston_qmle import HestonQMLECalibrator
    return _scalars(HestonQMLECalibrator(smooth_window=10).fit(prices, dt=dt))


def _fit_ou(prices: np.ndarray, dt: float) -> dict:
    """
    Per-asset OU fit on LOG-PRICE levels (matches the batch adapter).

    The OU level series is log(prices), not raw prices. method='discretization'
    (AR(1) OLS, ddof=2 residual variance) is the estimator that batched/ou.py
    implements in closed form, so batch and fallback paths agree.
    """
    from ..physical.ou_calibrator import OUCalibrator
    levels = np.log(prices)
    return _scalars(OUCalibrator(method='discretization').fit(levels, dt=dt))


def _fit_merton(prices: np.ndarray, dt: float) -> dict:
    from ..physical.merton_calibrator import MertonJumpCalibrator
    result = MertonJumpCalibrator(k_max=5).fit(prices, dt=dt)
    d = _scalars(result)
    # Rename lambda_ to lam for consistency with batched version
    if "lambda_" in d:
        d["lam"] = d.pop("lambda_")
    return d


def _fit_rbergomi(prices: np.ndarray, dt: float) -> dict:
    from ..physical.rough_bergomi_calibrator import RoughBergomiCalibrator
    return _scalars(RoughBergomiCalibrator(window=20, max_lag=10).fit(prices, dt=dt))


def _fit_heston_qmle_gk(prices: np.ndarray, dt: float) -> dict:
    """
    Per-asset Heston QMLE GK fit: not implemented (OHLC models have no scipy fallback).

    This function should never be called — the runner only uses fit_batch for
    needs_ohlc models and raises an error on adapter exception.
    """
    raise NotImplementedError(
        "heston_qmle_gk requires OHLC data and has no per-asset scipy fallback"
    )


# Batch adapters: take list of price arrays, compute returns/levels internally,
# pad, call module fit_batch, return dict of (N,)-arrays


def _batch_gbm(price_arrays: list[np.ndarray], dt: float) -> dict:
    """
    Batch adapter for GBM: prices -> log-returns -> fit_batch.
    """
    from ..physical.batched import gbm, common

    returns_list = [np.diff(np.log(prices)) for prices in price_arrays]
    returns, mask = common.pad_returns(returns_list)
    return gbm.fit_batch(returns, mask, dt)


def _batch_garch(price_arrays: list[np.ndarray], dt: float) -> dict:
    """
    Batch adapter for GARCH: prices -> log-returns -> fit_batch.
    """
    from ..physical.batched import garch, common

    returns_list = [np.diff(np.log(prices)) for prices in price_arrays]
    returns, mask = common.pad_returns(returns_list)
    return garch.fit_batch(returns, mask, dt)


def _batch_heston_qmle(price_arrays: list[np.ndarray], dt: float) -> dict:
    """
    Batch adapter for Heston QMLE: prices -> log-returns -> fit_batch.

    Note: Heston QMLE requires OHLC data; this adapter only has close prices.
    We use close as a proxy for all OHLC values (suboptimal but matches
    the per-asset path when only close is available).
    """
    from ..physical.batched import heston_qmle, common

    returns_list = [np.diff(np.log(prices)) for prices in price_arrays]
    returns, mask = common.pad_returns(returns_list)
    return heston_qmle.fit_batch(returns, mask, dt)


def _batch_ou(price_arrays: list[np.ndarray], dt: float) -> dict:
    """
    Batch adapter for OU: prices -> log-prices as levels -> fit_batch.

    OU is a mean-reverting process on levels, so we use log-prices as the
    level series (not log-returns). This matches the per-asset fallback
    (_fit_ou), which fits the scipy calibrator on np.log(prices).

    Each asset's level series is centered (mean subtracted) before padding:
    the batched AR(1) OLS runs in float32 and its normal-equation determinant
    n*sum(X^2) - sum(X)^2 cancels catastrophically for un-centered levels.
    Centering is exact for AR(1) with intercept — b, kappa, sigma and the
    log-likelihood are translation-invariant; only theta shifts, and the
    center is added back below.
    """
    from ..physical.batched import ou, common

    # Log-prices as the level series (OU process values), centered per asset
    levels_list = [np.log(prices) for prices in price_arrays]
    centers = np.array([lv.mean() for lv in levels_list], dtype=np.float64)
    centered = [lv - c for lv, c in zip(levels_list, centers)]
    levels, mask = common.pad_returns(centered)  # pad_returns works for any 1-D arrays
    out = ou.fit_batch(levels, mask, dt)
    out["theta"] = out["theta"] + centers
    return out


def _batch_merton(price_arrays: list[np.ndarray], dt: float) -> dict:
    """
    Batch adapter for Merton: prices -> log-returns -> fit_batch.
    """
    from ..physical.batched import merton, common

    returns_list = [np.diff(np.log(prices)) for prices in price_arrays]
    returns, mask = common.pad_returns(returns_list)
    return merton.fit_batch(returns, mask, dt)


def _batch_rbergomi(price_arrays: list[np.ndarray], dt: float) -> dict:
    """
    Batch adapter for rough Bergomi: prices -> log-returns -> fit_batch.
    """
    from ..physical.batched import rbergomi, common

    returns_list = [np.diff(np.log(prices)) for prices in price_arrays]
    returns, mask = common.pad_returns(returns_list)
    return rbergomi.fit_batch(returns, mask, dt)


def _batch_heston_qmle_gk(ohlc_list: list[dict[str, np.ndarray]], dt: float) -> dict:
    """
    Batch adapter for Heston QMLE with Garman-Klass OHLC variance proxy.

    Args:
        ohlc_list: List of dicts with keys 'open', 'high', 'low', 'close'
                   (already adjustment-scaled by the runner)
        dt: Time increment in years

    Returns:
        Dictionary with Heston parameters (same keys as fit_batch)
    """
    from ..physical.batched import heston_qmle, common

    # Pad each of the four OHLC series
    open_list = [ohlc["open"] for ohlc in ohlc_list]
    high_list = [ohlc["high"] for ohlc in ohlc_list]
    low_list = [ohlc["low"] for ohlc in ohlc_list]
    close_list = [ohlc["close"] for ohlc in ohlc_list]

    open_padded, mask_open = common.pad_returns(open_list)
    high_padded, mask_high = common.pad_returns(high_list)
    low_padded, mask_low = common.pad_returns(low_list)
    close_padded, mask_close = common.pad_returns(close_list)

    # All masks should be identical (same valid periods across OHLC)
    # Use the close mask as the primary mask
    mask = mask_close

    return heston_qmle.fit_batch_ohlc(
        open_padded, high_padded, low_padded, close_padded, mask, dt, smooth_window=10
    )


def _fit_heston_npe(prices: np.ndarray, dt: float) -> dict:
    """
    Per-asset Heston NPE fit: not implemented (batch-only amortized model).

    This function should never be called — heston_npe is batch-only.
    The runner uses fit_batch when available; this fallback raises a clear error.
    """
    raise NotImplementedError(
        "heston_npe is batch-only; train the model first: scripts/train_npe_heston.py"
    )


def _batch_heston_npe(price_arrays: list[np.ndarray], dt: float) -> dict:
    """
    Batch adapter for Heston NPE: prices -> log-returns -> fit_batch.

    Catches FileNotFoundError from missing checkpoint and re-raises with
    the same clear message (train-first instruction).
    """
    from ..physical.batched.npe import estimator
    from ..physical.batched import common

    try:
        returns_list = [np.diff(np.log(prices)) for prices in price_arrays]
        returns, mask = common.pad_returns(returns_list)
        return estimator.fit_batch(returns, mask, dt)
    except FileNotFoundError as e:
        # Re-raise with same message to surface in runner error handling
        raise


register_model(ModelSpec(name="gbm", fit=_fit_gbm, min_obs=60, fit_batch=_batch_gbm))
register_model(ModelSpec(name="garch", fit=_fit_garch, min_obs=250, fit_batch=_batch_garch))
register_model(ModelSpec(name="heston_qmle", fit=_fit_heston_qmle, min_obs=60, fit_batch=_batch_heston_qmle))
register_model(ModelSpec(name="ou", fit=_fit_ou, min_obs=60, fit_batch=_batch_ou))
register_model(ModelSpec(name="merton", fit=_fit_merton, min_obs=250, fit_batch=_batch_merton))
register_model(ModelSpec(name="rbergomi", fit=_fit_rbergomi, min_obs=250, fit_batch=_batch_rbergomi))
register_model(ModelSpec(
    name="heston_qmle_gk",
    fit=_fit_heston_qmle_gk,
    min_obs=60,
    fit_batch=_batch_heston_qmle_gk,
    needs_ohlc=True,
))
register_model(ModelSpec(
    name="heston_npe",
    fit=_fit_heston_npe,
    min_obs=250,
    fit_batch=_batch_heston_npe,
    needs_ohlc=False,
))
