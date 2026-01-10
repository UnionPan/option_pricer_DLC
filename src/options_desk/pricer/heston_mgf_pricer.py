"""
Heston MGF Grid-Based Pricer

Adapted from StochVolModels (https://github.com/ArturSepp/StochVolModels)
Pure NumPy implementation without numba dependency.

Uses MGF grid with Simpson's rule for stable numerical integration,
particularly important for short-dated options.
"""

import numpy as np
from typing import Tuple


def get_phi_grid(vol_scaler: float = 0.28,
                 max_phi: int = 1000,
                 is_spot_measure: bool = True,
                 real_part: float = None,
                 imag_mult: float = 5.6) -> np.ndarray:
    """
    Create complex phi grid for Fourier transform.

    Args:
        vol_scaler: sigma_0 * sqrt(ttm) for adaptive grid sizing
        max_phi: Number of grid points
        is_spot_measure: Use spot measure (True) vs forward measure
        real_part: Optional real part of phi (default -0.5 for spot measure, +0.5 for forward)
        imag_mult: Multiplier for imaginary grid extent (aligned with Sepp/Lewis grids)

    Returns:
        Complex phi grid: real_part + 1j * imaginary_part
    """
    # Imaginary part: linearly spaced up to imag_mult/vol_scaler
    p = np.linspace(0, imag_mult / vol_scaler, max_phi)

    # Real part: -0.5 for spot measure (dampening factor)
    if real_part is None:
        real_p = -0.5 if is_spot_measure else 0.5
    else:
        real_p = real_part

    phi_grid = real_p + 1j * p
    return phi_grid


def compute_simpson_weights(phi_grid: np.ndarray) -> np.ndarray:
    """
    Compute Simpson's rule integration weights.

    More stable than adaptive quadrature for oscillatory integrands.

    Args:
        phi_grid: Complex grid points

    Returns:
        Integration weights for Simpson's rule
    """
    p = np.imag(phi_grid)
    n = len(p)

    # Simpson's rule: 1, 4, 2, 4, 2, ..., 4, 1
    weights = 2.0 * np.ones(n)
    weights[0] = 1.0
    weights[-1] = 1.0
    weights[1::2] = 4.0  # Odd indices get weight 4

    # Scale by step size / 3
    dp = (p[1] - p[0]) / 3.0
    weights = weights * dp

    return weights


def compute_heston_mgf_grid(v0: float,
                           theta: float,
                           kappa: float,
                           volvol: float,
                           rho: float,
                           ttm: float,
                           phi_grid: np.ndarray) -> np.ndarray:
    """
    Compute Heston log-MGF on the phi grid.

    Uses formula from Sepp (2007), "Variance swaps under no conditions", Risk.
    More stable formulation than direct characteristic function.

    Args:
        v0: Initial variance
        theta: Long-run variance
        kappa: Mean reversion speed
        volvol: Vol-of-vol (xi)
        rho: Correlation
        ttm: Time to maturity
        phi_grid: Complex frequency grid

    Returns:
        log_mgf_grid: Log MGF values on grid
    """
    volvol2 = volvol * volvol

    # Coefficients
    b1 = kappa + rho * volvol * phi_grid
    b0 = 0.5 * phi_grid * (phi_grid + 1.0)

    # Discriminant and its square root
    zeta = np.sqrt(b1*b1 - 2.0*b0*volvol2)

    exp_zeta_t = np.exp(-zeta * ttm)

    # Psi coefficients
    psi_p = -b1 + zeta
    psi_m = b1 + zeta

    # C coefficients
    c_p = psi_p / (2.0 * zeta)
    c_m = psi_m / (2.0 * zeta)

    # MGF exponents
    # b_t = -(-psi_m*c_p*exp(-zeta*t) + psi_p*c_m) / (volvol2*(c_p*exp(-zeta*t) + c_m))
    b_t = -(-psi_m * c_p * exp_zeta_t + psi_p * c_m) / (volvol2 * (c_p * exp_zeta_t + c_m))

    # a_t = -(theta*kappa/volvol2) * (psi_p*t + 2*log(c_p*exp(-zeta*t) + c_m))
    a_t = -(theta * kappa / volvol2) * (psi_p * ttm + 2.0 * np.log(c_p * exp_zeta_t + c_m))

    # Log MGF = a_t + b_t * v0
    log_mgf_grid = a_t + b_t * v0

    return log_mgf_grid


def vanilla_option_price_from_mgf(log_mgf_grid: np.ndarray,
                                  phi_grid: np.ndarray,
                                  forward: float,
                                  strike: float,
                                  is_call: bool,
                                  discfactor: float = 1.0) -> float:
    """
    Price vanilla option using MGF grid via Fourier inversion.

    Uses Lewis (2001) formulation with real part = -0.5 dampening.

    Args:
        log_mgf_grid: Log MGF values on phi grid
        phi_grid: Complex frequency grid
        forward: Forward price (spot * exp((r-q)*T))
        strike: Strike price
        is_call: True for call, False for put
        discfactor: Discount factor exp(-r*T)

    Returns:
        Option price
    """
    # Compute Simpson weights
    weights = compute_simpson_weights(phi_grid)

    # Optimized payoff transform for phi = -0.5 + i*p
    # This assumes real part of phi_grid is -0.5 (spot measure)
    p = np.imag(phi_grid)
    p_payoff = (weights / np.pi) / (p * p + 0.25)

    # Log moneyness
    log_moneyness = np.log(forward / strike)

    # Fourier inversion integral using Simpson's rule
    # Integrand: exp(-x*phi + log_mgf) * payoff_transform
    integrand = np.exp(-log_moneyness * phi_grid + log_mgf_grid) * p_payoff
    capped_price = np.nansum(np.real(integrand))

    # Convert capped option to call/put
    # For phi = -0.5 (spot measure):
    # Call = discount * (forward - strike * capped_price)
    # Put = discount * (strike - strike * capped_price) = discount * strike * (1 - capped_price)
    if is_call:
        option_price = discfactor * (forward - strike * capped_price)
    else:
        option_price = discfactor * (strike - strike * capped_price)

    # Ensure price respects intrinsic value bounds (critical for extreme strikes)
    # Discounted intrinsic value
    if is_call:
        intrinsic = max(forward * discfactor - strike * discfactor, 0)
    else:
        intrinsic = max(strike * discfactor - forward * discfactor, 0)

    option_price = max(option_price, intrinsic)

    return option_price


def heston_price_vanilla(S: float,
                         K: float,
                         T: float,
                         r: float,
                         q: float,
                         v0: float,
                         theta: float,
                         kappa: float,
                         volvol: float,
                         rho: float,
                         is_call: bool,
                         max_phi: int = 1000,
                         imag_mult: float = 5.6) -> float:
    """
    Price single vanilla option using Heston model with MGF grid.

    Args:
        S: Spot price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        q: Dividend yield
        v0: Initial variance
        theta: Long-run variance
        kappa: Mean reversion speed
        volvol: Vol-of-vol
        rho: Correlation
        is_call: True for call, False for put

    Returns:
        Option price
    """
    if T <= 0:
        # Handle zero maturity
        intrinsic = max(S - K, 0) if is_call else max(K - S, 0)
        return intrinsic

    # Forward and discount factor
    forward = S * np.exp((r - q) * T)
    discfactor = np.exp(-r * T)

    # Adaptive phi grid based on volatility and maturity
    vol_scaler = min(0.3, np.sqrt(v0 * T))
    phi_grid = get_phi_grid(vol_scaler=vol_scaler, max_phi=max_phi, imag_mult=imag_mult)

    # Compute MGF on grid
    log_mgf_grid = compute_heston_mgf_grid(
        v0=v0,
        theta=theta,
        kappa=kappa,
        volvol=volvol,
        rho=rho,
        ttm=T,
        phi_grid=phi_grid
    )

    # Price option via Fourier inversion
    price = vanilla_option_price_from_mgf(
        log_mgf_grid=log_mgf_grid,
        phi_grid=phi_grid,
        forward=forward,
        strike=K,
        is_call=is_call,
        discfactor=discfactor
    )

    # Ensure price respects bounds
    intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0) if is_call else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
    price = max(price, intrinsic)

    return price


def heston_price_slice(S: float,
                       strikes: np.ndarray,
                       T: float,
                       r: float,
                       q: float,
                       v0: float,
                       theta: float,
                       kappa: float,
                       volvol: float,
                       rho: float,
                       option_types: np.ndarray,
                       max_phi: int = 1000,
                       imag_mult: float = 5.6) -> np.ndarray:
    """
    Price multiple options at same maturity efficiently.

    Computes MGF grid once, then prices all strikes.

    Args:
        S: Spot price
        strikes: Array of strike prices
        T: Time to maturity (years)
        r: Risk-free rate
        q: Dividend yield
        v0: Initial variance
        theta: Long-run variance
        kappa: Mean reversion speed
        volvol: Vol-of-vol
        rho: Correlation
        option_types: Array of 'call' or 'put'

    Returns:
        Array of option prices
    """
    if T <= 0:
        # Handle zero maturity
        prices = np.zeros(len(strikes))
        for i, (K, opt_type) in enumerate(zip(strikes, option_types)):
            intrinsic = max(S - K, 0) if opt_type == 'call' else max(K - S, 0)
            prices[i] = intrinsic
        return prices

    # Forward and discount factor
    forward = S * np.exp((r - q) * T)
    discfactor = np.exp(-r * T)

    # Adaptive phi grid
    vol_scaler = min(0.3, np.sqrt(v0 * T))
    phi_grid = get_phi_grid(vol_scaler=vol_scaler, max_phi=max_phi, imag_mult=imag_mult)

    # Compute MGF once for all strikes
    log_mgf_grid = compute_heston_mgf_grid(
        v0=v0,
        theta=theta,
        kappa=kappa,
        volvol=volvol,
        rho=rho,
        ttm=T,
        phi_grid=phi_grid
    )

    # Price each strike
    prices = np.zeros(len(strikes))
    for i, (K, opt_type) in enumerate(zip(strikes, option_types)):
        is_call = (opt_type == 'call')

        price = vanilla_option_price_from_mgf(
            log_mgf_grid=log_mgf_grid,
            phi_grid=phi_grid,
            forward=forward,
            strike=K,
            is_call=is_call,
            discfactor=discfactor
        )

        # Ensure price respects bounds
        intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0) if is_call else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
        prices[i] = max(price, intrinsic)

    return prices