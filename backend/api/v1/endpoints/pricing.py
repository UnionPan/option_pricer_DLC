"""Pricing and Greeks endpoints."""
from fastapi import APIRouter, HTTPException
from backend.schemas.pricing import (
    PricingRequest,
    PricingResponse,
    GreeksRequest,
    GreeksResponse,
)
from backend.services.pricing_service import PricingService

router = APIRouter()
pricing_service = PricingService()


@router.post("/price", response_model=PricingResponse, tags=["Pricing"])
async def calculate_price(request: PricingRequest):
    """
    Calculate option price using Black-Scholes model.

    Returns the theoretical price for a European-style option.
    """
    try:
        return pricing_service.calculate_price(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/greeks", response_model=GreeksResponse, tags=["Pricing"])
async def calculate_greeks(request: GreeksRequest):
    """
    Calculate all Greeks (delta, gamma, vega, theta, rho) for an option.

    Greeks measure the sensitivity of option price to various factors:
    - Delta: Sensitivity to underlying price
    - Gamma: Rate of change of delta
    - Vega: Sensitivity to volatility
    - Theta: Time decay (per day)
    - Rho: Sensitivity to interest rate
    """
    try:
        return pricing_service.calculate_greeks(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
