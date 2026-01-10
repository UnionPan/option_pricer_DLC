"""
Risk-neutral (Q-measure) calibrators.

Calibrate models from option market prices (bids, asks, implied volatilities).
Used for derivatives pricing, hedging, and risk-neutral scenario analysis.

Key characteristic: Drift = r (risk-free rate), not real-world drift μ.
"""

from .heston_calibrator import HestonCalibrator, CalibrationResult
from .sabr_calibrator import SABRCalibrator, SABRCalibrationResult
from .dupire_calibrator import DupireCalibrator, DupireResult
from .regime_switching_heston_calibrator import (
    RegimeSwitchingHestonCalibrator,
    RegimeSwitchingHestonSimulator,
    RegimeSwitchingHestonResult,
    RegimeHestonParameters,
)

__all__ = [
    # Heston stochastic volatility
    'HestonCalibrator',
    'CalibrationResult',

    # SABR stochastic volatility
    'SABRCalibrator',
    'SABRCalibrationResult',

    # Dupire local volatility
    'DupireCalibrator',
    'DupireResult',

    # Regime-switching Heston
    'RegimeSwitchingHestonCalibrator',
    'RegimeSwitchingHestonSimulator',
    'RegimeSwitchingHestonResult',
    'RegimeHestonParameters',
]
