"""Data fetching request/response schemas."""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class OptionChainRequest(BaseModel):
    """Request for option chain data."""

    symbol: str = Field(..., description="Underlying ticker symbol (e.g., 'AAPL')")
    expiration_date: Optional[str] = Field(
        None, description="Specific expiration date (YYYY-MM-DD). If None, gets nearest."
    )


class OptionContractData(BaseModel):
    """Single option contract data."""

    strike: float
    last_price: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    implied_volatility: Optional[float] = None
    option_type: str


class OptionChainResponse(BaseModel):
    """Response with option chain data."""

    symbol: str
    expiration_date: str
    spot_price: float
    contracts: List[OptionContractData]
    fetched_at: datetime = Field(default_factory=datetime.now)


class AvailableExpirationsResponse(BaseModel):
    """Response with available expiration dates."""

    symbol: str
    expirations: List[str] = Field(..., description="List of available expiration dates")
