"""Volatility surface building service."""
import yfinance as yf
import pandas as pd
from typing import List
from datetime import datetime
from backend.schemas.surface import (
    VolSurfaceRequest,
    VolSurfaceResponse,
    VolSurfacePoint,
    VolSmileRequest,
    VolSmileResponse,
)


def safe_float(value):
    """Convert to float, handling NaN."""
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


class SurfaceService:
    """Service for building volatility surfaces."""

    @staticmethod
    def build_vol_surface(request: VolSurfaceRequest) -> VolSurfaceResponse:
        """Build volatility surface from market data."""
        ticker = yf.Ticker(request.symbol)

        # Get spot price
        if request.spot_price:
            spot_price = request.spot_price
        else:
            info = ticker.info
            spot_price = info.get("currentPrice") or info.get("regularMarketPrice", 0.0)

        if spot_price == 0:
            raise ValueError(f"Could not fetch spot price for {request.symbol}")

        # Get all available expirations
        expirations = ticker.options
        if not expirations:
            raise ValueError(f"No options available for {request.symbol}")

        surface_points: List[VolSurfacePoint] = []

        # Filter expirations by date range
        today = datetime.now()
        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            days_to_expiry = (exp_date - today).days

            if days_to_expiry < request.min_expiry_days or days_to_expiry > request.max_expiry_days:
                continue

            time_to_expiry = days_to_expiry / 365.25

            # Get option chain for this expiration
            try:
                chain = ticker.option_chain(exp_str)

                # Process calls
                for _, row in chain.calls.iterrows():
                    iv = safe_float(row.get("impliedVolatility"))
                    if iv is not None and iv > 0:
                        strike = float(row["strike"])
                        surface_points.append(
                            VolSurfacePoint(
                                strike=strike,
                                expiry=time_to_expiry,
                                implied_vol=iv,
                                moneyness=strike / spot_price,
                            )
                        )

                # Process puts
                for _, row in chain.puts.iterrows():
                    iv = safe_float(row.get("impliedVolatility"))
                    if iv is not None and iv > 0:
                        strike = float(row["strike"])
                        surface_points.append(
                            VolSurfacePoint(
                                strike=strike,
                                expiry=time_to_expiry,
                                implied_vol=iv,
                                moneyness=strike / spot_price,
                            )
                        )
            except Exception as e:
                # Skip this expiration if there's an error
                continue

        if not surface_points:
            raise ValueError(f"No valid surface points found for {request.symbol}")

        # Count unique expirations and strikes
        unique_expiries = len(set(p.expiry for p in surface_points))
        unique_strikes = len(set(p.strike for p in surface_points))

        return VolSurfaceResponse(
            symbol=request.symbol,
            spot_price=spot_price,
            surface_points=surface_points,
            num_expirations=unique_expiries,
            num_strikes=unique_strikes,
        )

    @staticmethod
    def get_vol_smile(request: VolSmileRequest) -> VolSmileResponse:
        """Get volatility smile for a specific expiration."""
        ticker = yf.Ticker(request.symbol)

        # Get spot price
        info = ticker.info
        spot_price = info.get("currentPrice") or info.get("regularMarketPrice", 0.0)

        # Verify expiration exists
        if request.expiration_date not in ticker.options:
            raise ValueError(
                f"Expiration {request.expiration_date} not available for {request.symbol}"
            )

        # Calculate time to expiry
        exp_date = datetime.strptime(request.expiration_date, "%Y-%m-%d")
        days_to_expiry = (exp_date - datetime.now()).days
        time_to_expiry = days_to_expiry / 365.25

        # Get option chain
        chain = ticker.option_chain(request.expiration_date)

        strikes = []
        implied_vols = []

        # Combine calls and puts
        for _, row in chain.calls.iterrows():
            iv = safe_float(row.get("impliedVolatility"))
            if iv is not None and iv > 0:
                strikes.append(float(row["strike"]))
                implied_vols.append(iv)

        for _, row in chain.puts.iterrows():
            iv = safe_float(row.get("impliedVolatility"))
            if iv is not None and iv > 0:
                strike = float(row["strike"])
                if strike not in strikes:  # Avoid duplicates
                    strikes.append(strike)
                    implied_vols.append(iv)

        # Sort by strike
        sorted_data = sorted(zip(strikes, implied_vols))
        strikes, implied_vols = zip(*sorted_data) if sorted_data else ([], [])

        return VolSmileResponse(
            symbol=request.symbol,
            expiration_date=request.expiration_date,
            time_to_expiry=time_to_expiry,
            spot_price=spot_price,
            strikes=list(strikes),
            implied_vols=list(implied_vols),
        )
