"""
Model Calibration Module

Calibrate stochastic models to market data.

Organized by measure type:
- physical: P-measure calibrators (historical data → real-world dynamics)
- risk_neutral: Q-measure calibrators (option prices → risk-neutral dynamics)

author: Yunian Pan
email: yp1170@nyu.edu
"""

# Data providers
from .data.yfinance_fetcher import YFinanceFetcher
from .data.data_provider import MarketData, OptionChain

# P-measure (physical) calibrators
from . import physical

# Q-measure (risk-neutral) calibrators
from . import risk_neutral

# Legacy imports for backwards compatibility
try:
    from .historical import (
        GBMCalibrator,
        GBMCalibrationResult,
        OUCalibrator,
        OUCalibrationResult,
    )
except ImportError:
    # Use new location
    from .physical import (
        GBMCalibrator,
        GBMCalibrationResult,
        OUCalibrator,
        OUCalibrationResult,
    )

try:
    from .models import (
        HestonCalibrator,
        CalibrationResult,
        RegimeSwitchingCalibrator,
        RegimeSwitchingSimulator,
        RegimeSwitchingCalibrationResult,
        RegimeParameters,
        RegimeSwitchingHestonCalibrator,
        RegimeSwitchingHestonSimulator,
        RegimeSwitchingHestonResult,
        RegimeHestonParameters,
    )
except ImportError:
    # Use new locations
    from .physical import (
        RegimeSwitchingCalibrator,
        RegimeSwitchingSimulator,
        RegimeSwitchingCalibrationResult,
        RegimeParameters,
    )
    from .risk_neutral import (
        HestonCalibrator,
        CalibrationResult,
        RegimeSwitchingHestonCalibrator,
        RegimeSwitchingHestonSimulator,
        RegimeSwitchingHestonResult,
        RegimeHestonParameters,
    )

__all__ = [
    # Data providers
    'YFinanceFetcher',
    'MarketData',
    'OptionChain',

    # Module namespaces
    'physical',  # P-measure calibrators
    'risk_neutral',  # Q-measure calibrators

    # Legacy exports (for backwards compatibility)
    'GBMCalibrator',
    'GBMCalibrationResult',
    'OUCalibrator',
    'OUCalibrationResult',
    'HestonCalibrator',
    'CalibrationResult',
    'RegimeSwitchingCalibrator',
    'RegimeSwitchingSimulator',
    'RegimeSwitchingCalibrationResult',
    'RegimeParameters',
    'RegimeSwitchingHestonCalibrator',
    'RegimeSwitchingHestonSimulator',
    'RegimeSwitchingHestonResult',
    'RegimeHestonParameters',
]
