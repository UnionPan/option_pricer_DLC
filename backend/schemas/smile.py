"""Volatility smile schemas with model comparison."""
from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class VolSmileRequest(BaseModel):
    """Request for volatility smile with model comparison."""
    symbol: str
    expiration_date: str
    models: List[Literal["black_scholes", "heston", "sabr", "merton"]] = Field(
        default=["black_scholes"],
        description="Pricing models to compare"
    )
    # Heston parameters
    heston_v0: Optional[float] = Field(default=0.04, description="Heston initial variance")
    heston_theta: Optional[float] = Field(default=0.04, description="Heston long-term variance")
    heston_kappa: Optional[float] = Field(default=2.0, description="Heston mean reversion")
    heston_sigma_v: Optional[float] = Field(default=0.3, description="Heston vol of vol")
    heston_rho: Optional[float] = Field(default=-0.7, description="Heston correlation")

    # SABR parameters
    sabr_alpha: Optional[float] = Field(default=0.2, description="SABR volatility level")
    sabr_beta: Optional[float] = Field(default=0.7, description="SABR CEV parameter")
    sabr_rho: Optional[float] = Field(default=-0.3, description="SABR correlation")
    sabr_nu: Optional[float] = Field(default=0.4, description="SABR vol of vol")

    # Merton parameters
    merton_lambda: Optional[float] = Field(default=0.1, description="Merton jump intensity")
    merton_mu_j: Optional[float] = Field(default=-0.05, description="Merton mean jump size")
    merton_sigma_j: Optional[float] = Field(default=0.15, description="Merton jump volatility")


class VolSmileDataPoint(BaseModel):
    """Single point on volatility smile."""
    strike: float
    moneyness: float
    market_iv: float
    calculated_ivs: dict[str, float] = Field(
        default_factory=dict,
        description="Calculated IVs for each model"
    )


class VolSmileComparisonResponse(BaseModel):
    """Volatility smile with model comparison."""
    symbol: str
    expiration_date: str
    time_to_expiry: float
    spot_price: float
    risk_free_rate: float
    data_points: List[VolSmileDataPoint]
    models_used: List[str]
