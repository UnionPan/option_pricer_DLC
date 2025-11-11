"""Volatility surface endpoints."""
from fastapi import APIRouter, HTTPException
from backend.schemas.surface import (
    VolSurfaceRequest,
    VolSurfaceResponse,
    VolSmileRequest,
    VolSmileResponse,
)
from backend.services.surface_service import SurfaceService

router = APIRouter()
surface_service = SurfaceService()


@router.post("/build", response_model=VolSurfaceResponse, tags=["Volatility Surface"])
async def build_vol_surface(request: VolSurfaceRequest):
    """
    Build a volatility surface from market data.

    Fetches option chains across multiple expirations and constructs
    a complete volatility surface with strikes, maturities, and implied vols.
    """
    try:
        return surface_service.build_vol_surface(request)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/smile", response_model=VolSmileResponse, tags=["Volatility Surface"])
async def get_vol_smile(request: VolSmileRequest):
    """
    Get volatility smile for a specific expiration date.

    Returns strikes and implied volatilities for a single expiration,
    showing the characteristic smile/skew pattern.
    """
    try:
        return surface_service.get_vol_smile(request)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
