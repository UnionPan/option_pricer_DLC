"""
Bates MGF Grid-Based Pricer

Extends the Heston MGF pricer with Merton jump component.
Uses the same phi grid / Simpson's rule infrastructure.

The Bates log-MGF is:
    log_mgf_Bates(phi) = log_mgf_Heston(phi) + log_mgf_jumps(phi)

where:
    log_mgf_jumps(phi) = lambda * T * (exp(mu_J*phi + 0.5*sigma_J^2*phi^2) - 1)

with phi being the frequency variable (phi = -0.5 + i*p under spot measure).

author: Yunian Pan
email: yp1170@nyu.edu
"""

import numpy as np
from typing import Tuple

from .heston_mgf_pricer import (
    get_phi_grid,
    compute_simpson_weights,
    compute_heston_mgf_grid,
    vanilla_option_price_from_mgf,
)


def compute_bates_mgf_grid(
    v0: float,
    theta: float,
    kappa: float,
    volvol: float,
    rho: float,
    lambda_j: float,
    mu_J: float,
    sigma_J: float,
    ttm: float,
    phi_grid: np.ndarray,
) -> np.ndarray:
    """
    Compute Bates log-MGF on the phi grid.

    Bates = Heston + Merton jumps:
        log_mgf_Bates = log_mgf_Heston + lambda*T*(exp(mu_J*phi + 0.5*sigma_J^2*phi^2) - 1)

    The jump compensator is absorbed into the drift, consistent with
    the Heston MGF formulation which uses the spot measure (phi=-0.5+i*p).

    Args:
        v0: Initial variance
        theta: Long-run variance
        kappa: Mean reversion speed
        volvol: Vol-of-vol (xi)
        rho: Correlation
        lambda_j: Jump intensity
        mu_J: Mean of log-jump size
        sigma_J: Std of log-jump size
        ttm: Time to maturity
        phi_grid: Complex frequency grid

    Returns:
        log_mgf_grid: Log MGF values on grid
    """
    # Heston component
    log_mgf_heston = compute_heston_mgf_grid(
        v0=v0, theta=theta, kappa=kappa, volvol=volvol, rho=rho,
        ttm=ttm, phi_grid=phi_grid,
    )

    # Jump component in Sepp convention (phi -> -phi of standard):
    #   Sepp uses MGF of -log(S_T/F), so exp(-phi*mu_J) and +k*phi
    k_comp = np.exp(mu_J + 0.5 * sigma_J ** 2) - 1.0
    log_mgf_jumps = lambda_j * ttm * (
        np.exp(-mu_J * phi_grid + 0.5 * sigma_J ** 2 * phi_grid ** 2) - 1.0
        + k_comp * phi_grid
    )

    return log_mgf_heston + log_mgf_jumps


def bates_price_vanilla(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    v0: float,
    theta: float,
    kappa: float,
    volvol: float,
    rho: float,
    lambda_j: float,
    mu_J: float,
    sigma_J: float,
    is_call: bool,
    max_phi: int = 1000,
    imag_mult: float = 5.6,
) -> float:
    """
    Price single vanilla option using Bates model with MGF grid.

    Args:
        S: Spot price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        q: Dividend yield
        v0: Initial variance
        theta: Long-run variance
        kappa: Mean reversion speed
        volvol: Vol-of-vol
        rho: Correlation
        lambda_j: Jump intensity
        mu_J: Mean of log-jump size
        sigma_J: Std of log-jump size
        is_call: True for call
        max_phi: Phi grid size
        imag_mult: Grid extent multiplier

    Returns:
        Option price
    """
    if T <= 0:
        intrinsic = max(S - K, 0) if is_call else max(K - S, 0)
        return intrinsic

    forward = S * np.exp((r - q) * T)
    discfactor = np.exp(-r * T)

    vol_scaler = min(0.3, np.sqrt(v0 * T))
    phi_grid = get_phi_grid(vol_scaler=vol_scaler, max_phi=max_phi, imag_mult=imag_mult)

    log_mgf_grid = compute_bates_mgf_grid(
        v0=v0, theta=theta, kappa=kappa, volvol=volvol, rho=rho,
        lambda_j=lambda_j, mu_J=mu_J, sigma_J=sigma_J,
        ttm=T, phi_grid=phi_grid,
    )

    price = vanilla_option_price_from_mgf(
        log_mgf_grid=log_mgf_grid, phi_grid=phi_grid,
        forward=forward, strike=K, is_call=is_call, discfactor=discfactor,
    )

    intrinsic = max(forward * discfactor - K * discfactor, 0) if is_call \
        else max(K * discfactor - forward * discfactor, 0)
    return max(price, intrinsic)


def bates_price_slice(
    S: float,
    strikes: np.ndarray,
    T: float,
    r: float,
    q: float,
    v0: float,
    theta: float,
    kappa: float,
    volvol: float,
    rho: float,
    lambda_j: float,
    mu_J: float,
    sigma_J: float,
    option_types: np.ndarray,
    max_phi: int = 1000,
    imag_mult: float = 5.6,
) -> np.ndarray:
    """
    Price multiple options at same maturity efficiently.

    Computes MGF grid once, then prices all strikes.

    Args:
        S: Spot price
        strikes: Array of strike prices
        T: Time to maturity
        r, q: Rates
        v0, theta, kappa, volvol, rho: Heston params
        lambda_j, mu_J, sigma_J: Jump params
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

    vol_scaler = min(0.3, np.sqrt(v0 * T))
    phi_grid = get_phi_grid(vol_scaler=vol_scaler, max_phi=max_phi, imag_mult=imag_mult)

    log_mgf_grid = compute_bates_mgf_grid(
        v0=v0, theta=theta, kappa=kappa, volvol=volvol, rho=rho,
        lambda_j=lambda_j, mu_J=mu_J, sigma_J=sigma_J,
        ttm=T, phi_grid=phi_grid,
    )

    prices = np.zeros(len(strikes))
    for i, (K, opt_type) in enumerate(zip(strikes, option_types)):
        is_call = (opt_type == 'call')
        price = vanilla_option_price_from_mgf(
            log_mgf_grid=log_mgf_grid, phi_grid=phi_grid,
            forward=forward, strike=K, is_call=is_call, discfactor=discfactor,
        )
        intrinsic = max(forward * discfactor - K * discfactor, 0) if is_call \
            else max(K * discfactor - forward * discfactor, 0)
        prices[i] = max(price, intrinsic)

    return prices
