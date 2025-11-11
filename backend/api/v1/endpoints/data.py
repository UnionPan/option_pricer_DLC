"""Data fetching endpoints."""
from fastapi import APIRouter, HTTPException, Query
from backend.schemas.data import (
    OptionChainRequest,
    OptionChainResponse,
    AvailableExpirationsResponse,
)
from backend.services.data_service import DataService

router = APIRouter()
data_service = DataService()


@router.post("/option-chain", response_model=OptionChainResponse, tags=["Data"])
async def get_option_chain(request: OptionChainRequest):
    """
    Fetch option chain data for a given symbol and expiration.

    Returns all call and put contracts with strikes, prices, and implied volatilities.
    """
    try:
        return data_service.get_option_chain(request)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/expirations/{symbol}", response_model=AvailableExpirationsResponse, tags=["Data"]
)
async def get_expirations(symbol: str):
    """
    Get list of available expiration dates for an underlying symbol.

    Returns all available option expiration dates.
    """
    try:
        return data_service.get_available_expirations(symbol)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
