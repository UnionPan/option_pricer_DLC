"""
Merton Jump-Diffusion MGF Pricer

Uses a log-MGF grid and Lewis-style Fourier inversion to price vanilla options.
"""

import numpy as np

from .heston_mgf_pricer import get_phi_grid, vanilla_option_price_from_mgf


def compute_merton_mgf_grid(
    sigma: float,
    lambda_jump: float,
    mu_J: float,
    sigma_J: float,
    ttm: float,
    phi_grid: np.ndarray,
) -> np.ndarray:
    """
    Compute log-MGF for log-forward returns under Merton model.

    X = log(S_T / F_T) follows diffusion + compound Poisson.
    """
    if ttm <= 0:
        return np.zeros_like(phi_grid, dtype=np.complex128)

    kappa = np.exp(mu_J + 0.5 * sigma_J**2) - 1.0
    drift = -0.5 * sigma**2 * ttm - lambda_jump * kappa * ttm
    diffusion = 0.5 * sigma**2 * ttm * phi_grid * phi_grid
    jump = lambda_jump * ttm * (np.exp(phi_grid * mu_J + 0.5 * sigma_J**2 * phi_grid**2) - 1.0)

    log_mgf = phi_grid * drift + diffusion + jump
    return log_mgf


def merton_price_vanilla(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    lambda_jump: float,
    mu_J: float,
    sigma_J: float,
    is_call: bool,
    max_phi: int = 1000,
    imag_mult: float = 5.6,
) -> float:
    if T <= 0:
        intrinsic = max(S - K, 0) if is_call else max(K - S, 0)
        return intrinsic

    forward = S * np.exp((r - q) * T)
    discfactor = np.exp(-r * T)

    vol_scaler = min(0.3, np.sqrt(max(sigma**2, 1e-8) * T))
    phi_grid = get_phi_grid(vol_scaler=vol_scaler, max_phi=max_phi, imag_mult=imag_mult)

    log_mgf_grid = compute_merton_mgf_grid(
        sigma=sigma,
        lambda_jump=lambda_jump,
        mu_J=mu_J,
        sigma_J=sigma_J,
        ttm=T,
        phi_grid=phi_grid,
    )

    price = vanilla_option_price_from_mgf(
        log_mgf_grid=log_mgf_grid,
        phi_grid=phi_grid,
        forward=forward,
        strike=K,
        is_call=is_call,
        discfactor=discfactor,
    )

    intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0) if is_call else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
    return max(price, intrinsic)


def merton_price_slice(
    S: float,
    strikes: np.ndarray,
    T: float,
    r: float,
    q: float,
    sigma: float,
    lambda_jump: float,
    mu_J: float,
    sigma_J: float,
    option_types: np.ndarray,
    max_phi: int = 1000,
    imag_mult: float = 5.6,
) -> np.ndarray:
    if T <= 0:
        prices = np.zeros(len(strikes))
        for i, (K, opt_type) in enumerate(zip(strikes, option_types)):
            intrinsic = max(S - K, 0) if opt_type == 'call' else max(K - S, 0)
            prices[i] = intrinsic
        return prices

    forward = S * np.exp((r - q) * T)
    discfactor = np.exp(-r * T)

    vol_scaler = min(0.3, np.sqrt(max(sigma**2, 1e-8) * T))
    phi_grid = get_phi_grid(vol_scaler=vol_scaler, max_phi=max_phi, imag_mult=imag_mult)

    log_mgf_grid = compute_merton_mgf_grid(
        sigma=sigma,
        lambda_jump=lambda_jump,
        mu_J=mu_J,
        sigma_J=sigma_J,
        ttm=T,
        phi_grid=phi_grid,
    )

    prices = np.zeros(len(strikes))
    for i, (K, opt_type) in enumerate(zip(strikes, option_types)):
        prices[i] = vanilla_option_price_from_mgf(
            log_mgf_grid=log_mgf_grid,
            phi_grid=phi_grid,
            forward=forward,
            strike=K,
            is_call=opt_type == 'call',
            discfactor=discfactor,
        )

    return prices
