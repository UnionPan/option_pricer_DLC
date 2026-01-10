"""Pricing and Greeks calculation service."""
import numpy as np
from scipy.stats import norm

# Use new pricer module
from options_desk.pricer import BlackScholesPricer
from options_desk.derivatives import EuropeanCall, EuropeanPut
from options_desk.processes import GBM

from backend.schemas.pricing import (
    PricingRequest,
    PricingResponse,
    GreeksRequest,
    GreeksResponse,
)
from backend.schemas.common import OptionType


class PricingService:
    """Service for option pricing and Greeks calculations."""

    @staticmethod
    def calculate_price(request: PricingRequest) -> PricingResponse:
        """Calculate option price using Black-Scholes model."""
        # Create derivative based on option type
        if request.option_type == OptionType.CALL:
            option = EuropeanCall(
                strike=request.strike,
                maturity=request.time_to_expiry,
            )
        else:
            option = EuropeanPut(
                strike=request.strike,
                maturity=request.time_to_expiry,
            )

        # Create GBM process (mu doesn't affect BS price, only sigma matters)
        process = GBM(mu=request.risk_free_rate, sigma=request.volatility)

        # Create pricer
        pricer = BlackScholesPricer(
            risk_free_rate=request.risk_free_rate,
            dividend_yield=request.dividend_yield,
        )

        # Price the option
        result = pricer.price(
            derivative=option,
            process=process,
            X0=request.spot,
            compute_greeks=False,
        )

        return PricingResponse(price=float(result.price), model="black_scholes")

    @staticmethod
    def calculate_greeks(request: GreeksRequest) -> GreeksResponse:
        """Calculate all Greeks for an option."""
        # Create derivative based on option type
        if request.option_type == OptionType.CALL:
            option = EuropeanCall(
                strike=request.strike,
                maturity=request.time_to_expiry,
            )
        else:
            option = EuropeanPut(
                strike=request.strike,
                maturity=request.time_to_expiry,
            )

        # Create GBM process
        process = GBM(mu=request.risk_free_rate, sigma=request.volatility)

        # Create pricer
        pricer = BlackScholesPricer(
            risk_free_rate=request.risk_free_rate,
            dividend_yield=request.dividend_yield,
        )

        # Price with Greeks
        result = pricer.price(
            derivative=option,
            process=process,
            X0=request.spot,
            compute_greeks=True,
        )

        # Extract Greeks from result
        greeks = result.greeks or {}

        return GreeksResponse(
            delta=float(greeks.get('delta', 0.0)),
            gamma=float(greeks.get('gamma', 0.0)),
            vega=float(greeks.get('vega', 0.0)),
            theta=float(greeks.get('theta', 0.0)),
            rho=float(greeks.get('rho', 0.0)),
        )
