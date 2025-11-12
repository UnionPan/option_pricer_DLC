"""Market data service using yfinance."""
import yfinance as yf
from typing import List
from datetime import datetime, timedelta
from backend.schemas.market import (
    QuoteData,
    MarketOverviewResponse,
    IndexChartData,
    IndexChartsResponse,
    HistoricalDataPoint,
    OHLCDataPoint,
    OHLCChartResponse,
)


class MarketService:
    """Service for fetching market overview data."""

    # Symbol mappings
    INDICES = [
        ("^GSPC", "S&P 500"),
        ("ES=F", "ESZ5"),
        ("^IXIC", "NASDAQ"),
    ]

    MAGNIFICENT7 = [
        ("AAPL", "Apple"),
        ("MSFT", "Microsoft"),
        ("GOOGL", "Google"),
        ("AMZN", "Amazon"),
        ("NVDA", "NVIDIA"),
        ("META", "Meta"),
        ("TSLA", "Tesla"),
    ]

    COMMODITIES = [
        ("GC=F", "Gold"),
        ("CL=F", "Crude Oil"),
        ("NG=F", "Natural Gas"),
    ]

    @staticmethod
    def _get_quote_data(symbol: str, name: str) -> QuoteData:
        """Fetch quote data for a single symbol."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="2d")

            if hist.empty or len(hist) < 1:
                # Try to get current price from info
                last_price = info.get("currentPrice") or info.get("regularMarketPrice")
                return QuoteData(
                    symbol=symbol,
                    name=name,
                    last_price=last_price,
                    change=None,
                    change_percent=None,
                    volume=info.get("volume"),
                )

            last_close = hist["Close"].iloc[-1]
            prev_close = hist["Close"].iloc[-2] if len(hist) >= 2 else last_close

            change = last_close - prev_close
            change_percent = (change / prev_close * 100) if prev_close != 0 else 0

            return QuoteData(
                symbol=symbol,
                name=name,
                last_price=float(last_close),
                change=float(change),
                change_percent=float(change_percent),
                volume=int(hist["Volume"].iloc[-1]) if "Volume" in hist.columns else None,
            )
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return QuoteData(
                symbol=symbol,
                name=name,
                last_price=None,
                change=None,
                change_percent=None,
                volume=None,
            )

    @staticmethod
    def get_market_overview() -> MarketOverviewResponse:
        """Fetch market overview data."""
        indices = [
            MarketService._get_quote_data(symbol, name)
            for symbol, name in MarketService.INDICES
        ]

        magnificent7 = [
            MarketService._get_quote_data(symbol, name)
            for symbol, name in MarketService.MAGNIFICENT7
        ]

        commodities = [
            MarketService._get_quote_data(symbol, name)
            for symbol, name in MarketService.COMMODITIES
        ]

        return MarketOverviewResponse(
            indices=indices,
            magnificent7=magnificent7,
            commodities=commodities,
        )

    @staticmethod
    def get_index_charts() -> IndexChartsResponse:
        """Fetch 1-year historical data for major indices."""
        charts = []

        # Only fetch charts for S&P 500 and NASDAQ
        chart_symbols = [
            ("^GSPC", "S&P 500"),
            ("^IXIC", "NASDAQ"),
        ]

        for symbol, name in chart_symbols:
            try:
                ticker = yf.Ticker(symbol)
                # Get 1 year of data
                hist = ticker.history(period="1y")

                if not hist.empty:
                    data_points = [
                        HistoricalDataPoint(
                            date=date.strftime("%Y-%m-%d"),
                            close=float(row["Close"]),
                        )
                        for date, row in hist.iterrows()
                    ]

                    charts.append(
                        IndexChartData(
                            symbol=symbol,
                            name=name,
                            data=data_points,
                        )
                    )
            except Exception as e:
                print(f"Error fetching chart for {symbol}: {e}")
                continue

        return IndexChartsResponse(charts=charts)

    @staticmethod
    def get_ohlc_data(symbol: str, period: str = "6mo") -> OHLCChartResponse:
        """Fetch OHLC (candlestick) data for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)

            if hist.empty:
                raise ValueError(f"No historical data available for {symbol}")

            data_points = [
                OHLCDataPoint(
                    date=date.strftime("%Y-%m-%d"),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=int(row["Volume"]),
                )
                for date, row in hist.iterrows()
            ]

            return OHLCChartResponse(symbol=symbol, data=data_points)
        except Exception as e:
            print(f"Error fetching OHLC data for {symbol}: {e}")
            raise
