"""Market data endpoints."""
from fastapi import APIRouter, HTTPException, Query
from backend.services.market_service import MarketService
from backend.schemas.market import MarketOverviewResponse, IndexChartsResponse, OHLCChartResponse

router = APIRouter()


@router.get("/overview", response_model=MarketOverviewResponse)
async def get_market_overview():
    """Get market overview with indices, stocks, and commodities."""
    try:
        return MarketService.get_market_overview()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/charts", response_model=IndexChartsResponse)
async def get_index_charts():
    """Get historical chart data for major indices."""
    try:
        return MarketService.get_index_charts()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ohlc/{symbol}", response_model=OHLCChartResponse)
async def get_ohlc_data(
    symbol: str,
    period: str = Query(default="6mo", description="Time period (e.g., 1mo, 3mo, 6mo, 1y)")
):
    """Get OHLC (candlestick) data for a symbol."""
    try:
        return MarketService.get_ohlc_data(symbol, period)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
