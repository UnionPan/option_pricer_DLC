"""Volatility smile endpoints."""
from fastapi import APIRouter, HTTPException
from backend.services.smile_service import SmileService
from backend.schemas.smile import VolSmileRequest, VolSmileComparisonResponse

router = APIRouter()


@router.post("/compare", response_model=VolSmileComparisonResponse)
async def get_volatility_smile_comparison(request: VolSmileRequest):
    """Get volatility smile with market IV vs model-calculated IVs."""
    try:
        return SmileService.get_volatility_smile_comparison(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
