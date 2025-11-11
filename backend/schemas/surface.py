"""Volatility surface request/response schemas."""
from pydantic import BaseModel, Field
from typing import List, Optional


class VolSurfacePoint(BaseModel):
    """Single point on volatility surface."""

    strike: float
    expiry: float = Field(..., description="Time to expiration in years")
    implied_vol: float
    moneyness: Optional[float] = Field(None, description="Strike/Spot ratio")


class VolSurfaceRequest(BaseModel):
    """Request to build volatility surface."""

    symbol: str = Field(..., description="Underlying ticker symbol")
    spot_price: Optional[float] = Field(None, description="Current spot price. If None, fetches live.")
    min_expiry_days: int = Field(default=7, description="Minimum days to expiration")
    max_expiry_days: int = Field(default=365, description="Maximum days to expiration")


class VolSurfaceResponse(BaseModel):
    """Response with volatility surface data."""

    symbol: str
    spot_price: float
    surface_points: List[VolSurfacePoint]
    num_expirations: int
    num_strikes: int


class VolSmileRequest(BaseModel):
    """Request for volatility smile at specific expiry."""

    symbol: str
    expiration_date: str = Field(..., description="Target expiration date (YYYY-MM-DD)")


class VolSmileResponse(BaseModel):
    """Response with volatility smile."""

    symbol: str
    expiration_date: str
    time_to_expiry: float
    spot_price: float
    strikes: List[float]
    implied_vols: List[float]
