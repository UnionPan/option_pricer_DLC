"""
Schemas for calibration endpoints
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import date


class FetchDataRequest(BaseModel):
    """Request to fetch OHLCV data"""
    tickers: List[str] = Field(..., min_items=1, description="List of ticker symbols")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")


class OHLCVData(BaseModel):
    """OHLCV data for a single ticker"""
    ticker: str
    dates: List[str]
    open: List[float]
    high: List[float]
    low: List[float]
    close: List[float]
    volume: List[float]


class CalibrationRequest(BaseModel):
    """Request to calibrate a model"""
    tickers: List[str] = Field(..., min_items=1, description="List of ticker symbols")
    model: str = Field(
        ...,
        description=(
            "Model name (gbm, ou, heston, rough_bergomi, regime_switching_gbm, "
            "merton_jump, garch)"
        ),
    )
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    method: str = Field(
        default="mle",
        description=(
            "Calibration method:\n"
            "- 'mle': Maximum Likelihood (for gbm, ou)\n"
            "- 'particle_filter': Particle Filter (for heston, rough_bergomi)\n"
            "- 'moment_matching': Moment Matching (for rough_bergomi)\n"
            "- 'em': EM Algorithm (for regime_switching_gbm)\n"
            "- 'mle': Maximum Likelihood (for merton_jump)\n"
            "- 'mle': Quasi-MLE (for garch)"
        )
    )
    include_drift: bool = Field(default=True, description="Whether to include drift parameter")


class CalibrationDiagnostics(BaseModel):
    """Calibration diagnostics"""
    logLikelihood: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    mean_ess: Optional[float] = None
    n_particles: Optional[int] = None
    variogram_r2: Optional[float] = None
    errorMetrics: Optional[Dict[str, float]] = None
    note: Optional[str] = None


class CalibrationResult(BaseModel):
    """Result of model calibration"""
    ticker: str
    model: str
    parameters: Dict[str, float]
    diagnostics: CalibrationDiagnostics
    timestamp: str
    method: Optional[str] = None
    measure: Optional[str] = Field(default="P-measure")


# ========== Q-measure (Risk-Neutral) Calibration Schemas ==========

class OptionQuoteData(BaseModel):
    """Single option quote"""
    strike: float
    expiry: str
    option_type: str
    bid: float
    ask: float
    mid: float
    last: float
    volume: int
    open_interest: int
    implied_volatility: Optional[float] = None
    moneyness: float


class FetchOptionChainRequest(BaseModel):
    """Request to fetch option chain data"""
    ticker: str = Field(..., description="Ticker symbol")
    reference_date: Optional[str] = Field(None, description="Reference date (YYYY-MM-DD), defaults to today")
    risk_free_rate: float = Field(default=0.05, description="Risk-free rate (e.g., 0.05 for 5%)")
    expiry: Optional[str] = Field(None, description="Specific expiry date (YYYY-MM-DD) or None for all")


class OptionChainData(BaseModel):
    """Option chain data for a ticker"""
    ticker: str
    spot_price: float
    reference_date: str
    risk_free_rate: float
    dividend_yield: float
    options: List[OptionQuoteData]
    expiries: List[str]
    n_options: int


class QMeasureCalibrationRequest(BaseModel):
    """Request to calibrate Q-measure model from option chain"""
    ticker: str = Field(..., description="Ticker symbol")
    model: str = Field(..., description="Model name (heston)")
    reference_date: Optional[str] = Field(None, description="Reference date (YYYY-MM-DD)")
    risk_free_rate: float = Field(default=0.05, description="Risk-free rate")
    filter_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional filtering: min_volume, min_open_interest, max_spread_pct, moneyness_range (list)"
    )
    calibration_method: str = Field(default="differential_evolution", description="Optimization method")
    maxiter: int = Field(default=1000, description="Maximum iterations")


class QMeasureCalibrationResult(BaseModel):
    """Result of Q-measure calibration"""
    ticker: str
    model: str
    parameters: Dict[str, float]
    diagnostics: Dict[str, Any]
    timestamp: str
    measure: str = "Q-measure"


class VolSurfaceRequest(BaseModel):
    """Request to generate volatility surface"""
    ticker: str
    model: str
    parameters: Dict[str, float]
    spot_price: float
    risk_free_rate: float
    dividend_yield: float = 0.0
    n_strikes: int = Field(default=30, description="Number of strike points")
    n_maturities: int = Field(default=20, description="Number of maturity points")
    strike_range: List[float] = Field(default=[0.7, 1.3], description="Strike range as [min_moneyness, max_moneyness]")
    maturity_range: List[float] = Field(default=[0.05, 2.0], description="Maturity range in years [min, max]")


class VolSurfaceResponse(BaseModel):
    """Volatility surface data"""
    strikes: List[float]
    maturities: List[float]
    vols: List[List[float]]  # 2D array: maturities x strikes
    surface_type: str  # 'implied_vol' or 'local_vol'
    ticker: str
    model: str  # Distinguish from P-measure results
