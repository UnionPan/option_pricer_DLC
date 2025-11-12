"""Market data schemas."""
from pydantic import BaseModel
from typing import List, Optional


class QuoteData(BaseModel):
    """Individual quote data."""
    symbol: str
    name: str
    last_price: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None
    volume: Optional[int] = None


class HistoricalDataPoint(BaseModel):
    """Single point in historical data."""
    date: str
    close: float


class OHLCDataPoint(BaseModel):
    """OHLC data point."""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class IndexChartData(BaseModel):
    """Chart data for an index."""
    symbol: str
    name: str
    data: List[HistoricalDataPoint]


class OHLCChartResponse(BaseModel):
    """OHLC chart data response."""
    symbol: str
    data: List[OHLCDataPoint]


class MarketOverviewResponse(BaseModel):
    """Market overview with indices, stocks, and commodities."""
    indices: List[QuoteData]
    magnificent7: List[QuoteData]
    commodities: List[QuoteData]


class IndexChartsResponse(BaseModel):
    """Historical chart data for indices."""
    charts: List[IndexChartData]
