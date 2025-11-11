"""Pricing and Greeks calculation service."""
import numpy as np
from scipy.stats import norm
from options_desk.pricing.black_scholes import (
    black_scholes_price,
    black_scholes_delta,
    black_scholes_gamma,
    black_scholes_vega,
    black_scholes_theta,
    _d1,
    _d2,
)
from options_desk.core.option import OptionType as CoreOptionType
from backend.schemas.pricing import (
    PricingRequest,
    PricingResponse,
    GreeksRequest,
    GreeksResponse,
)
from backend.schemas.common import OptionType


def _convert_option_type(option_type: OptionType) -> CoreOptionType:
    """Convert API option type to core option type."""
    return CoreOptionType.CALL if option_type == OptionType.CALL else CoreOptionType.PUT


def _calculate_rho(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: CoreOptionType,
    dividend: float = 0.0,
) -> float:
    """
    Calculate option rho using Black-Scholes formula.

    Rho measures the sensitivity of option price to interest rate changes.
    """
    if time_to_expiry <= 0:
        return 0.0

    d_1 = _d1(spot, strike, time_to_expiry, rate, volatility, dividend)
    d_2 = _d2(d_1, volatility, time_to_expiry)

    if option_type == CoreOptionType.CALL:
        # Rho per 1% change in interest rate
        return strike * time_to_expiry * np.exp(-rate * time_to_expiry) * norm.cdf(d_2) / 100
    else:  # PUT
        return -strike * time_to_expiry * np.exp(-rate * time_to_expiry) * norm.cdf(-d_2) / 100


class PricingService:
    """Service for option pricing and Greeks calculations."""

    @staticmethod
    def calculate_price(request: PricingRequest) -> PricingResponse:
        """Calculate option price using Black-Scholes model."""
        core_option_type = _convert_option_type(request.option_type)

        price = black_scholes_price(
            spot=request.spot,
            strike=request.strike,
            time_to_expiry=request.time_to_expiry,
            rate=request.risk_free_rate,
            volatility=request.volatility,
            option_type=core_option_type,
            dividend=request.dividend_yield,
        )

        return PricingResponse(price=float(price), model="black_scholes")

    @staticmethod
    def calculate_greeks(request: GreeksRequest) -> GreeksResponse:
        """Calculate all Greeks for an option."""
        core_option_type = _convert_option_type(request.option_type)

        delta = black_scholes_delta(
            spot=request.spot,
            strike=request.strike,
            time_to_expiry=request.time_to_expiry,
            rate=request.risk_free_rate,
            volatility=request.volatility,
            option_type=core_option_type,
            dividend=request.dividend_yield,
        )

        gamma = black_scholes_gamma(
            spot=request.spot,
            strike=request.strike,
            time_to_expiry=request.time_to_expiry,
            rate=request.risk_free_rate,
            volatility=request.volatility,
            dividend=request.dividend_yield,
        )

        vega = black_scholes_vega(
            spot=request.spot,
            strike=request.strike,
            time_to_expiry=request.time_to_expiry,
            rate=request.risk_free_rate,
            volatility=request.volatility,
            dividend=request.dividend_yield,
        )

        theta = black_scholes_theta(
            spot=request.spot,
            strike=request.strike,
            time_to_expiry=request.time_to_expiry,
            rate=request.risk_free_rate,
            volatility=request.volatility,
            option_type=core_option_type,
            dividend=request.dividend_yield,
        )

        rho = _calculate_rho(
            spot=request.spot,
            strike=request.strike,
            time_to_expiry=request.time_to_expiry,
            rate=request.risk_free_rate,
            volatility=request.volatility,
            option_type=core_option_type,
            dividend=request.dividend_yield,
        )

        return GreeksResponse(
            delta=float(delta),
            gamma=float(gamma),
            vega=float(vega),
            theta=float(theta),
            rho=float(rho),
        )
