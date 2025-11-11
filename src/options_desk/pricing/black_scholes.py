"""
Black-Scholes-Merton option pricing model.
"""

import numpy as np
from scipy.stats import norm
from typing import Union

from ..core.option import OptionType


def _d1(
    spot: float, strike: float, time_to_expiry: float, rate: float, volatility: float, dividend: float = 0.0
) -> float:
    """Calculate d1 parameter in Black-Scholes formula."""
    return (
        np.log(spot / strike) + (rate - dividend + 0.5 * volatility**2) * time_to_expiry
    ) / (volatility * np.sqrt(time_to_expiry))


def _d2(d1: float, volatility: float, time_to_expiry: float) -> float:
    """Calculate d2 parameter in Black-Scholes formula."""
    return d1 - volatility * np.sqrt(time_to_expiry)


def black_scholes_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: OptionType,
    dividend: float = 0.0,
) -> float:
    """
    Calculate option price using Black-Scholes-Merton formula.

    Args:
        spot: Current price of underlying asset
        strike: Strike price
        time_to_expiry: Time to expiration (in years)
        rate: Risk-free interest rate (annual)
        volatility: Volatility (annual)
        option_type: CALL or PUT
        dividend: Continuous dividend yield (annual)

    Returns:
        Option price

    Note:
        This is for European-style options only.
    """
    if time_to_expiry <= 0:
        # At expiration
        if option_type == OptionType.CALL:
            return max(0, spot - strike)
        else:
            return max(0, strike - spot)

    d_1 = _d1(spot, strike, time_to_expiry, rate, volatility, dividend)
    d_2 = _d2(d_1, volatility, time_to_expiry)

    if option_type == OptionType.CALL:
        price = spot * np.exp(-dividend * time_to_expiry) * norm.cdf(d_1) - strike * np.exp(
            -rate * time_to_expiry
        ) * norm.cdf(d_2)
    else:  # PUT
        price = strike * np.exp(-rate * time_to_expiry) * norm.cdf(-d_2) - spot * np.exp(
            -dividend * time_to_expiry
        ) * norm.cdf(-d_1)

    return price


def black_scholes_delta(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: OptionType,
    dividend: float = 0.0,
) -> float:
    """
    Calculate option delta using Black-Scholes formula.

    Delta measures the rate of change of option price with respect to underlying price.

    Args:
        spot: Current price of underlying asset
        strike: Strike price
        time_to_expiry: Time to expiration (in years)
        rate: Risk-free interest rate (annual)
        volatility: Volatility (annual)
        option_type: CALL or PUT
        dividend: Continuous dividend yield (annual)

    Returns:
        Option delta
    """
    if time_to_expiry <= 0:
        if option_type == OptionType.CALL:
            return 1.0 if spot > strike else 0.0
        else:
            return -1.0 if spot < strike else 0.0

    d_1 = _d1(spot, strike, time_to_expiry, rate, volatility, dividend)

    if option_type == OptionType.CALL:
        return np.exp(-dividend * time_to_expiry) * norm.cdf(d_1)
    else:  # PUT
        return -np.exp(-dividend * time_to_expiry) * norm.cdf(-d_1)


def black_scholes_gamma(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    dividend: float = 0.0,
) -> float:
    """
    Calculate option gamma using Black-Scholes formula.

    Gamma measures the rate of change of delta with respect to underlying price.

    Args:
        spot: Current price of underlying asset
        strike: Strike price
        time_to_expiry: Time to expiration (in years)
        rate: Risk-free interest rate (annual)
        volatility: Volatility (annual)
        dividend: Continuous dividend yield (annual)

    Returns:
        Option gamma (same for calls and puts)
    """
    if time_to_expiry <= 0:
        return 0.0

    d_1 = _d1(spot, strike, time_to_expiry, rate, volatility, dividend)

    return (
        np.exp(-dividend * time_to_expiry)
        * norm.pdf(d_1)
        / (spot * volatility * np.sqrt(time_to_expiry))
    )


def black_scholes_vega(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    dividend: float = 0.0,
) -> float:
    """
    Calculate option vega using Black-Scholes formula.

    Vega measures the sensitivity of option price to volatility.

    Args:
        spot: Current price of underlying asset
        strike: Strike price
        time_to_expiry: Time to expiration (in years)
        rate: Risk-free interest rate (annual)
        volatility: Volatility (annual)
        dividend: Continuous dividend yield (annual)

    Returns:
        Option vega (per 1% change in volatility)
    """
    if time_to_expiry <= 0:
        return 0.0

    d_1 = _d1(spot, strike, time_to_expiry, rate, volatility, dividend)

    # Vega per 1% change in volatility
    return spot * np.exp(-dividend * time_to_expiry) * norm.pdf(d_1) * np.sqrt(time_to_expiry) / 100


def black_scholes_theta(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: OptionType,
    dividend: float = 0.0,
) -> float:
    """
    Calculate option theta using Black-Scholes formula.

    Theta measures the rate of change of option price with respect to time.

    Args:
        spot: Current price of underlying asset
        strike: Strike price
        time_to_expiry: Time to expiration (in years)
        rate: Risk-free interest rate (annual)
        volatility: Volatility (annual)
        option_type: CALL or PUT
        dividend: Continuous dividend yield (annual)

    Returns:
        Option theta (per day)
    """
    if time_to_expiry <= 0:
        return 0.0

    d_1 = _d1(spot, strike, time_to_expiry, rate, volatility, dividend)
    d_2 = _d2(d_1, volatility, time_to_expiry)

    term1 = (
        -spot
        * np.exp(-dividend * time_to_expiry)
        * norm.pdf(d_1)
        * volatility
        / (2 * np.sqrt(time_to_expiry))
    )

    if option_type == OptionType.CALL:
        term2 = dividend * spot * np.exp(-dividend * time_to_expiry) * norm.cdf(d_1)
        term3 = -rate * strike * np.exp(-rate * time_to_expiry) * norm.cdf(d_2)
    else:  # PUT
        term2 = -dividend * spot * np.exp(-dividend * time_to_expiry) * norm.cdf(-d_1)
        term3 = rate * strike * np.exp(-rate * time_to_expiry) * norm.cdf(-d_2)

    # Convert to per-day theta
    return (term1 + term2 + term3) / 365.25
