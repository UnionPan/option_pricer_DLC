"""Volatility smile comparison service."""
import yfinance as yf
import numpy as np
from datetime import datetime
from typing import List
from backend.schemas.smile import VolSmileRequest, VolSmileComparisonResponse, VolSmileDataPoint
from backend.services.advanced_pricing import (
    HestonModel,
    SABRModel,
    MertonJumpDiffusion,
    implied_volatility_from_price,
)
from options_desk.pricing.black_scholes import black_scholes_price
from options_desk.core.option import OptionType


class SmileService:
    """Service for volatility smile analysis with model comparison."""

    @staticmethod
    def get_volatility_smile_comparison(request: VolSmileRequest) -> VolSmileComparisonResponse:
        """Get volatility smile with market IV vs calculated IV from different models."""
        ticker = yf.Ticker(request.symbol)

        # Get spot price
        info = ticker.info
        spot_price = info.get("currentPrice") or info.get("regularMarketPrice", 0.0)

        if spot_price == 0:
            raise ValueError(f"Could not fetch spot price for {request.symbol}")

        # Verify expiration exists
        if request.expiration_date not in ticker.options:
            raise ValueError(
                f"Expiration {request.expiration_date} not available for {request.symbol}"
            )

        # Calculate time to expiry
        exp_date = datetime.strptime(request.expiration_date, "%Y-%m-%d")
        days_to_expiry = (exp_date - datetime.now()).days
        time_to_expiry = max(days_to_expiry / 365.25, 1/365.25)  # At least 1 day

        # Get option chain
        chain = ticker.option_chain(request.expiration_date)

        # Use treasury rate as risk-free rate (simplified)
        risk_free_rate = 0.05  # 5% default

        data_points: List[VolSmileDataPoint] = []

        # Process calls
        for _, row in chain.calls.iterrows():
            strike = float(row["strike"])
            market_iv = SmileService._safe_float(row.get("impliedVolatility"))
            last_price = SmileService._safe_float(row.get("lastPrice"))

            if market_iv is None or market_iv <= 0 or last_price is None:
                continue

            moneyness = strike / spot_price
            calculated_ivs = {}

            # Calculate IV for each requested model
            for model in request.models:
                try:
                    if model == "black_scholes":
                        # For BS, market IV is the calculated IV
                        calculated_ivs[model] = market_iv

                    elif model == "heston":
                        # Calculate Heston price and back out IV
                        heston_price = HestonModel.price(
                            spot=spot_price,
                            strike=strike,
                            time_to_expiry=time_to_expiry,
                            rate=risk_free_rate,
                            v0=request.heston_v0,
                            theta=request.heston_theta,
                            kappa=request.heston_kappa,
                            sigma_v=request.heston_sigma_v,
                            rho=request.heston_rho,
                            option_type="call",
                        )
                        # Back out IV using BS formula
                        calculated_iv = implied_volatility_from_price(
                            heston_price, spot_price, strike, time_to_expiry,
                            risk_free_rate, "call"
                        )
                        calculated_ivs[model] = calculated_iv

                    elif model == "sabr":
                        # SABR directly gives IV
                        sabr_iv = SABRModel.implied_volatility(
                            forward=spot_price * np.exp(risk_free_rate * time_to_expiry),
                            strike=strike,
                            time_to_expiry=time_to_expiry,
                            alpha=request.sabr_alpha,
                            beta=request.sabr_beta,
                            rho=request.sabr_rho,
                            nu=request.sabr_nu,
                        )
                        calculated_ivs[model] = sabr_iv

                    elif model == "merton":
                        # Calculate Merton price and back out IV
                        merton_price = MertonJumpDiffusion.price(
                            spot=spot_price,
                            strike=strike,
                            time_to_expiry=time_to_expiry,
                            rate=risk_free_rate,
                            sigma=np.sqrt(request.heston_v0),  # Use base volatility
                            lambda_j=request.merton_lambda,
                            mu_j=request.merton_mu_j,
                            sigma_j=request.merton_sigma_j,
                            option_type="call",
                        )
                        calculated_iv = implied_volatility_from_price(
                            merton_price, spot_price, strike, time_to_expiry,
                            risk_free_rate, "call"
                        )
                        calculated_ivs[model] = calculated_iv

                except Exception as e:
                    print(f"Error calculating {model} IV for strike {strike}: {e}")
                    calculated_ivs[model] = market_iv  # Fallback to market IV

            data_points.append(
                VolSmileDataPoint(
                    strike=strike,
                    moneyness=moneyness,
                    market_iv=market_iv,
                    calculated_ivs=calculated_ivs,
                )
            )

        # Process puts (optional - can add if needed)
        # Similar logic for puts...

        # Sort by strike
        data_points.sort(key=lambda x: x.strike)

        return VolSmileComparisonResponse(
            symbol=request.symbol,
            expiration_date=request.expiration_date,
            time_to_expiry=time_to_expiry,
            spot_price=spot_price,
            risk_free_rate=risk_free_rate,
            data_points=data_points,
            models_used=request.models,
        )

    @staticmethod
    def _safe_float(value):
        """Convert to float, handling NaN."""
        import pandas as pd
        if pd.isna(value):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
