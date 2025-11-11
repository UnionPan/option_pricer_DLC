"""Data fetching service using yfinance."""
import yfinance as yf
from typing import List, Optional
from datetime import datetime
from backend.schemas.data import (
    OptionChainRequest,
    OptionChainResponse,
    OptionContractData,
    AvailableExpirationsResponse,
)


class DataService:
    """Service for fetching market data."""

    @staticmethod
    def get_option_chain(request: OptionChainRequest) -> OptionChainResponse:
        """Fetch option chain data from yfinance."""
        ticker = yf.Ticker(request.symbol)

        # Get available expiration dates
        expirations = ticker.options
        if not expirations:
            raise ValueError(f"No options available for {request.symbol}")

        # Use specified expiration or nearest one
        if request.expiration_date:
            if request.expiration_date not in expirations:
                raise ValueError(
                    f"Expiration {request.expiration_date} not available. "
                    f"Available: {', '.join(expirations)}"
                )
            expiration = request.expiration_date
        else:
            expiration = expirations[0]  # Nearest expiration

        # Fetch option chain
        chain = ticker.option_chain(expiration)

        # Get current spot price
        info = ticker.info
        spot_price = info.get("currentPrice") or info.get("regularMarketPrice", 0.0)

        # Parse calls and puts
        contracts = []

        def safe_float(value):
            """Convert to float, handling NaN."""
            import pandas as pd
            if pd.isna(value):
                return None
            try:
                return float(value)
            except (ValueError, TypeError):
                return None

        def safe_int(value):
            """Convert to int, handling NaN."""
            import pandas as pd
            if pd.isna(value):
                return None
            try:
                return int(value)
            except (ValueError, TypeError):
                return None

        for _, row in chain.calls.iterrows():
            contracts.append(
                OptionContractData(
                    strike=float(row["strike"]),
                    last_price=safe_float(row.get("lastPrice")),
                    bid=safe_float(row.get("bid")),
                    ask=safe_float(row.get("ask")),
                    volume=safe_int(row.get("volume")),
                    open_interest=safe_int(row.get("openInterest")),
                    implied_volatility=safe_float(row.get("impliedVolatility")),
                    option_type="call",
                )
            )

        for _, row in chain.puts.iterrows():
            contracts.append(
                OptionContractData(
                    strike=float(row["strike"]),
                    last_price=safe_float(row.get("lastPrice")),
                    bid=safe_float(row.get("bid")),
                    ask=safe_float(row.get("ask")),
                    volume=safe_int(row.get("volume")),
                    open_interest=safe_int(row.get("openInterest")),
                    implied_volatility=safe_float(row.get("impliedVolatility")),
                    option_type="put",
                )
            )

        return OptionChainResponse(
            symbol=request.symbol,
            expiration_date=expiration,
            spot_price=spot_price,
            contracts=contracts,
        )

    @staticmethod
    def get_available_expirations(symbol: str) -> AvailableExpirationsResponse:
        """Get list of available expiration dates for a symbol."""
        ticker = yf.Ticker(symbol)
        expirations = ticker.options

        if not expirations:
            raise ValueError(f"No options available for {symbol}")

        return AvailableExpirationsResponse(symbol=symbol, expirations=list(expirations))
