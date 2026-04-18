"""
Bachelier (Normal) Option Pricer

Closed-form pricing for European options under the Bachelier model
where the underlying follows arithmetic Brownian motion:

    dS_t = mu * dt + sigma_n * dW_t

The Bachelier formula uses normal (absolute) volatility and allows
negative underlying values, making it the standard for interest-rate
options, swaptions, and spread options.

author: Yunian Pan
email: yp1170@nyu.edu
"""

import numpy as np
from scipy.stats import norm


# ============================================================================
# Core Bachelier pricing
# ============================================================================

def bachelier_call(forward: float,
                   strike: float,
                   sigma_n: float,
                   ttm: float,
                   discfactor: float = 1.0) -> float:
    """
    Bachelier call price.

    C = D * [(F - K) * N(d) + sigma_n * sqrt(T) * n(d)]

    where d = (F - K) / (sigma_n * sqrt(T))

    Args:
        forward: Forward price F = S * exp((r - q) * T)
        strike: Strike price
        sigma_n: Normal (absolute) volatility
        ttm: Time to maturity
        discfactor: Discount factor exp(-r * T)

    Returns:
        Call option price
    """
    if ttm <= 0.0:
        return max(forward - strike, 0.0) * discfactor

    if sigma_n <= 0.0:
        return max(forward - strike, 0.0) * discfactor

    sqrt_t = np.sqrt(ttm)
    total_vol = sigma_n * sqrt_t
    d = (forward - strike) / total_vol

    price = discfactor * ((forward - strike) * norm.cdf(d) + total_vol * norm.pdf(d))
    return float(price)


def bachelier_put(forward: float,
                  strike: float,
                  sigma_n: float,
                  ttm: float,
                  discfactor: float = 1.0) -> float:
    """
    Bachelier put price.

    P = D * [(K - F) * N(-d) + sigma_n * sqrt(T) * n(d)]

    where d = (F - K) / (sigma_n * sqrt(T))

    Args:
        forward: Forward price F = S * exp((r - q) * T)
        strike: Strike price
        sigma_n: Normal (absolute) volatility
        ttm: Time to maturity
        discfactor: Discount factor exp(-r * T)

    Returns:
        Put option price
    """
    if ttm <= 0.0:
        return max(strike - forward, 0.0) * discfactor

    if sigma_n <= 0.0:
        return max(strike - forward, 0.0) * discfactor

    sqrt_t = np.sqrt(ttm)
    total_vol = sigma_n * sqrt_t
    d = (forward - strike) / total_vol

    price = discfactor * ((strike - forward) * norm.cdf(-d) + total_vol * norm.pdf(d))
    return float(price)


# ============================================================================
# Public API matching MGF pricer pattern
# ============================================================================

def bachelier_price_vanilla(S: float,
                            K: float,
                            T: float,
                            r: float,
                            q: float,
                            sigma_n: float,
                            is_call: bool) -> float:
    """
    Price a single vanilla option using the Bachelier (normal) model.

    Args:
        S: Spot price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        q: Dividend yield
        sigma_n: Normal (absolute) volatility
        is_call: True for call, False for put

    Returns:
        Option price
    """
    if T <= 0:
        intrinsic = max(S - K, 0) if is_call else max(K - S, 0)
        return intrinsic

    forward = S * np.exp((r - q) * T)
    discfactor = np.exp(-r * T)

    if is_call:
        return bachelier_call(forward, K, sigma_n, T, discfactor)
    else:
        return bachelier_put(forward, K, sigma_n, T, discfactor)


def bachelier_price_slice(S: float,
                          strikes: np.ndarray,
                          T: float,
                          r: float,
                          q: float,
                          sigma_n: float,
                          option_types: np.ndarray) -> np.ndarray:
    """
    Price multiple options at the same maturity.

    Args:
        S: Spot price
        strikes: Array of strike prices
        T: Time to maturity (years)
        r: Risk-free rate
        q: Dividend yield
        sigma_n: Normal (absolute) volatility
        option_types: Array of 'call' or 'put'

    Returns:
        Array of option prices
    """
    if T <= 0:
        prices = np.zeros(len(strikes))
        for i, (K, opt_type) in enumerate(zip(strikes, option_types)):
            intrinsic = max(S - K, 0) if opt_type == 'call' else max(K - S, 0)
            prices[i] = intrinsic
        return prices

    forward = S * np.exp((r - q) * T)
    discfactor = np.exp(-r * T)

    prices = np.zeros(len(strikes))
    for i, (K, opt_type) in enumerate(zip(strikes, option_types)):
        is_call = (opt_type == 'call')
        if is_call:
            prices[i] = bachelier_call(forward, K, sigma_n, T, discfactor)
        else:
            prices[i] = bachelier_put(forward, K, sigma_n, T, discfactor)

    return prices


# ============================================================================
# Bachelier Greeks
# ============================================================================

def bachelier_delta(forward: float,
                    strike: float,
                    sigma_n: float,
                    ttm: float,
                    discfactor: float = 1.0,
                    is_call: bool = True) -> float:
    """
    Bachelier delta: dC/dF or dP/dF.

    Call delta = D * N(d)
    Put  delta = -D * N(-d)

    Args:
        forward: Forward price
        strike: Strike price
        sigma_n: Normal volatility
        ttm: Time to maturity
        discfactor: Discount factor
        is_call: True for call delta

    Returns:
        Delta
    """
    if ttm <= 0.0 or sigma_n <= 0.0:
        if is_call:
            return discfactor if forward > strike else 0.0
        else:
            return -discfactor if forward < strike else 0.0

    total_vol = sigma_n * np.sqrt(ttm)
    d = (forward - strike) / total_vol

    if is_call:
        return discfactor * norm.cdf(d)
    else:
        return -discfactor * norm.cdf(-d)


def bachelier_gamma(forward: float,
                    strike: float,
                    sigma_n: float,
                    ttm: float,
                    discfactor: float = 1.0) -> float:
    """
    Bachelier gamma: d^2C/dF^2 (same for call and put).

    Gamma = D * n(d) / (sigma_n * sqrt(T))

    Args:
        forward: Forward price
        strike: Strike price
        sigma_n: Normal volatility
        ttm: Time to maturity
        discfactor: Discount factor

    Returns:
        Gamma
    """
    if ttm <= 0.0 or sigma_n <= 0.0:
        return 0.0

    total_vol = sigma_n * np.sqrt(ttm)
    d = (forward - strike) / total_vol

    return discfactor * norm.pdf(d) / total_vol


def bachelier_vega(forward: float,
                   strike: float,
                   sigma_n: float,
                   ttm: float,
                   discfactor: float = 1.0) -> float:
    """
    Bachelier vega: dC/d(sigma_n) (same for call and put).

    Vega = D * sqrt(T) * n(d)

    Args:
        forward: Forward price
        strike: Strike price
        sigma_n: Normal volatility
        ttm: Time to maturity
        discfactor: Discount factor

    Returns:
        Vega
    """
    if ttm <= 0.0 or sigma_n <= 0.0:
        return 0.0

    sqrt_t = np.sqrt(ttm)
    total_vol = sigma_n * sqrt_t
    d = (forward - strike) / total_vol

    return discfactor * sqrt_t * norm.pdf(d)


def bachelier_theta(forward: float,
                    strike: float,
                    sigma_n: float,
                    ttm: float,
                    discfactor: float = 1.0,
                    is_call: bool = True) -> float:
    """
    Bachelier theta: -dC/dT (time decay per year).

    Theta = -D * sigma_n * n(d) / (2 * sqrt(T))

    Note: This is the theta from the time-value component only.

    Args:
        forward: Forward price
        strike: Strike price
        sigma_n: Normal volatility
        ttm: Time to maturity
        discfactor: Discount factor
        is_call: True for call theta

    Returns:
        Theta (negative for long options)
    """
    if ttm <= 0.0 or sigma_n <= 0.0:
        return 0.0

    sqrt_t = np.sqrt(ttm)
    total_vol = sigma_n * sqrt_t
    d = (forward - strike) / total_vol

    return -discfactor * sigma_n * norm.pdf(d) / (2.0 * sqrt_t)


# ============================================================================
# Bachelier implied volatility (Newton-Raphson inversion)
# ============================================================================

def bachelier_implied_vol(price: float,
                          forward: float,
                          strike: float,
                          ttm: float,
                          discfactor: float = 1.0,
                          is_call: bool = True,
                          max_iter: int = 50,
                          tol: float = 1e-10) -> float:
    """
    Invert the Bachelier formula to recover normal implied volatility.

    Uses Newton-Raphson iteration with vega as the derivative.

    Args:
        price: Observed market price (undiscounted is fine if discfactor=1)
        forward: Forward price
        strike: Strike price
        ttm: Time to maturity
        discfactor: Discount factor
        is_call: True if the price is a call
        max_iter: Maximum Newton iterations
        tol: Convergence tolerance (absolute price error)

    Returns:
        Normal implied volatility sigma_n

    Raises:
        ValueError: If price is below intrinsic or iteration fails to converge
    """
    if ttm <= 0.0:
        raise ValueError("Cannot compute implied vol for zero or negative maturity")

    # Undiscounted price
    target = price / discfactor

    # Intrinsic
    intrinsic = max(forward - strike, 0.0) if is_call else max(strike - forward, 0.0)
    if target < intrinsic - tol:
        raise ValueError(
            f"Price {price} is below intrinsic value {intrinsic * discfactor}"
        )

    # Time value to solve for
    time_value = target - intrinsic

    # If time value is essentially zero, vol is zero
    if time_value < tol:
        return 0.0

    sqrt_t = np.sqrt(ttm)

    # Initial guess: Brenner-Subrahmanyam style for normal vol
    # sigma_n ~ price / (discfactor * sqrt(T / (2*pi)))
    sigma_init = target / (sqrt_t * np.sqrt(1.0 / (2.0 * np.pi)))
    sigma_n = max(sigma_init, 1e-8)

    for _ in range(max_iter):
        # Model price at current sigma_n
        if is_call:
            model_price = bachelier_call(forward, strike, sigma_n, ttm, 1.0)
        else:
            model_price = bachelier_put(forward, strike, sigma_n, ttm, 1.0)

        err = model_price - target

        if abs(err) < tol:
            return sigma_n

        # Vega (undiscounted)
        vega = bachelier_vega(forward, strike, sigma_n, ttm, 1.0)

        if vega < 1e-15:
            break

        sigma_n = sigma_n - err / vega
        sigma_n = max(sigma_n, 1e-12)

    return sigma_n
