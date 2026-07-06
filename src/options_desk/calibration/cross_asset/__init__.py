"""Cross-asset calibration models."""

from options_desk.calibration.cross_asset.factor_model import (
    FactorCov,
    FactorModel,
    fit_factor_model,
)
from options_desk.calibration.cross_asset.pooling import pool_parameters
from options_desk.calibration.cross_asset.dcc import (
    DCCResult,
    fit_dcc,
    dcc_corr_path,
)

__all__ = [
    "FactorCov",
    "FactorModel",
    "fit_factor_model",
    "pool_parameters",
    "DCCResult",
    "fit_dcc",
    "dcc_corr_path",
]
