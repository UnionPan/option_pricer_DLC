"""Pricing and Greeks request/response schemas."""
from pydantic import BaseModel, Field
from typing import Optional
from backend.schemas.common import OptionType


class PricingRequest(BaseModel):
    """Request for option pricing."""

    spot: float = Field(..., gt=0, description="Current spot price of underlying")
    strike: float = Field(..., gt=0, description="Strike price")
    time_to_expiry: float = Field(..., gt=0, description="Time to expiration in years")
    volatility: float = Field(..., gt=0, description="Implied volatility (annualized)")
    risk_free_rate: float = Field(default=0.05, description="Risk-free interest rate")
    dividend_yield: float = Field(default=0.0, description="Continuous dividend yield")
    option_type: OptionType = Field(..., description="Option type (call/put)")


class PricingResponse(BaseModel):
    """Response with option price."""

    price: float = Field(..., description="Option price")
    model: str = Field(default="black_scholes", description="Pricing model used")


class GreeksRequest(BaseModel):
    """Request for Greeks calculation."""

    spot: float = Field(..., gt=0, description="Current spot price of underlying")
    strike: float = Field(..., gt=0, description="Strike price")
    time_to_expiry: float = Field(..., gt=0, description="Time to expiration in years")
    volatility: float = Field(..., gt=0, description="Implied volatility (annualized)")
    risk_free_rate: float = Field(default=0.05, description="Risk-free interest rate")
    dividend_yield: float = Field(default=0.0, description="Continuous dividend yield")
    option_type: OptionType = Field(..., description="Option type (call/put)")


class GreeksResponse(BaseModel):
    """Response with Greeks values."""

    delta: float = Field(..., description="Delta (price sensitivity to spot)")
    gamma: float = Field(..., description="Gamma (delta sensitivity to spot)")
    vega: float = Field(..., description="Vega (price sensitivity to volatility)")
    theta: float = Field(..., description="Theta (time decay)")
    rho: float = Field(..., description="Rho (interest rate sensitivity)")


class ImpliedVolRequest(BaseModel):
    """Request for implied volatility calculation."""

    spot: float = Field(..., gt=0, description="Current spot price of underlying")
    strike: float = Field(..., gt=0, description="Strike price")
    time_to_expiry: float = Field(..., gt=0, description="Time to expiration in years")
    market_price: float = Field(..., gt=0, description="Observed market price")
    risk_free_rate: float = Field(default=0.05, description="Risk-free interest rate")
    dividend_yield: float = Field(default=0.0, description="Continuous dividend yield")
    option_type: OptionType = Field(..., description="Option type (call/put)")


class ImpliedVolResponse(BaseModel):
    """Response with implied volatility."""

    implied_vol: float = Field(..., description="Implied volatility")
    iterations: Optional[int] = Field(None, description="Number of iterations for convergence")
