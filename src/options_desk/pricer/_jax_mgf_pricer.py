"""
JAX-accelerated MGF-based option pricer.

Fully vectorized across strikes — no Python loops.
JIT-compiled for CPU/GPU. Supports Heston, Merton, Bates, and any
model that provides a log-MGF function.

The key speedup over the NumPy version:
  - Pricing N strikes is a single (N, M) matmul instead of N serial integrations
  - JIT compilation eliminates all Python overhead
  - Same code runs on GPU for massive parallelism

author: Yunian Pan
email: yp1170@nyu.edu
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple

from ..processes._jax_utils import to_numpy


# ============================================================================
# Phi grid and Simpson weights (JAX)
# ============================================================================
def make_phi_grid(
    vol_scaler: float,
    max_phi: int = 1000,
    real_part: float = -0.5,
    imag_mult: float = 5.6,
) -> jnp.ndarray:
    """Create complex phi grid for Fourier inversion."""
    p = jnp.linspace(0.0, imag_mult / vol_scaler, max_phi)
    return real_part + 1j * p


def make_simpson_weights(phi_grid: jnp.ndarray) -> jnp.ndarray:
    """Simpson's rule weights for the phi grid."""
    p = jnp.imag(phi_grid)
    n = len(p)
    dp = (p[1] - p[0]) / 3.0

    # Simpson: 1, 4, 2, 4, 2, ..., 4, 1
    weights = jnp.ones(n) * 2.0
    weights = weights.at[0].set(1.0)
    weights = weights.at[-1].set(1.0)
    # Odd indices get weight 4
    odd_mask = jnp.arange(n) % 2 == 1
    weights = jnp.where(odd_mask, 4.0, weights)

    return weights * dp


# ============================================================================
# Core: vectorized pricing across strikes (no Python loop)
# ============================================================================
def price_slice_from_mgf(
    log_mgf_grid: jnp.ndarray,
    phi_grid: jnp.ndarray,
    forward: float,
    strikes: jnp.ndarray,
    is_call: jnp.ndarray,
    discfactor: float,
) -> jnp.ndarray:
    """
    Price multiple vanilla options from a log-MGF grid in one vectorized call.

    This replaces the Python for-loop in the NumPy version with a single
    (n_strikes, n_phi) broadcast + sum.

    Args:
        log_mgf_grid: Log MGF values, shape (n_phi,)
        phi_grid: Complex phi grid, shape (n_phi,)
        forward: Forward price (scalar)
        strikes: Strike prices, shape (n_strikes,)
        is_call: Boolean array, shape (n_strikes,)
        discfactor: Discount factor (scalar)

    Returns:
        Option prices, shape (n_strikes,)
    """
    weights = make_simpson_weights(phi_grid)
    p = jnp.imag(phi_grid)
    p_payoff = (weights / jnp.pi) / (p * p + 0.25)

    # Log moneyness for each strike: shape (n_strikes,)
    log_moneyness = jnp.log(forward / strikes)

    # Vectorized integrand: (n_strikes, n_phi)
    # integrand[i, j] = exp(-log_moneyness[i] * phi[j] + log_mgf[j]) * p_payoff[j]
    integrand = (
        jnp.exp(-log_moneyness[:, None] * phi_grid[None, :] + log_mgf_grid[None, :])
        * p_payoff[None, :]
    )

    # Sum over phi dimension: shape (n_strikes,)
    capped_prices = jnp.sum(jnp.real(integrand), axis=1)

    # Convert to call/put prices
    call_prices = discfactor * (forward - strikes * capped_prices)
    put_prices = discfactor * (strikes - strikes * capped_prices)
    prices = jnp.where(is_call, call_prices, put_prices)

    # Floor at intrinsic value
    call_intrinsic = jnp.maximum(forward * discfactor - strikes * discfactor, 0.0)
    put_intrinsic = jnp.maximum(strikes * discfactor - forward * discfactor, 0.0)
    intrinsic = jnp.where(is_call, call_intrinsic, put_intrinsic)

    return jnp.maximum(prices, intrinsic)


# ============================================================================
# Heston log-MGF (JAX)
# ============================================================================
class HestonMGFParams(NamedTuple):
    v0: float
    theta: float
    kappa: float
    volvol: float
    rho: float


def heston_log_mgf(params: HestonMGFParams, ttm: float, phi_grid: jnp.ndarray) -> jnp.ndarray:
    """Compute Heston log-MGF on phi grid (Sepp formulation)."""
    volvol2 = params.volvol * params.volvol

    b1 = params.kappa + params.rho * params.volvol * phi_grid
    b0 = 0.5 * phi_grid * (phi_grid + 1.0)

    zeta = jnp.sqrt(b1 * b1 - 2.0 * b0 * volvol2)
    exp_zeta_t = jnp.exp(-zeta * ttm)

    psi_p = -b1 + zeta
    psi_m = b1 + zeta

    c_p = psi_p / (2.0 * zeta)
    c_m = psi_m / (2.0 * zeta)

    b_t = -(-psi_m * c_p * exp_zeta_t + psi_p * c_m) / (volvol2 * (c_p * exp_zeta_t + c_m))
    a_t = -(params.theta * params.kappa / volvol2) * (
        psi_p * ttm + 2.0 * jnp.log(c_p * exp_zeta_t + c_m)
    )

    return a_t + b_t * params.v0


# ============================================================================
# Merton log-MGF (JAX)
# ============================================================================
class MertonMGFParams(NamedTuple):
    sigma: float
    lambda_j: float
    mu_J: float
    sigma_J: float


def merton_log_mgf(params: MertonMGFParams, ttm: float, phi_grid: jnp.ndarray) -> jnp.ndarray:
    """Compute Merton jump-diffusion log-MGF on phi grid (Sepp convention).

    Sepp convention: MGF of -log(S_T/F), so b0 = 0.5*phi*(phi+1)
    and jump uses exp(-phi*mu_J) with +k*phi compensator.
    """
    kappa = jnp.exp(params.mu_J + 0.5 * params.sigma_J ** 2) - 1.0
    # Diffusion: Sepp convention 0.5*sigma^2*T*phi*(phi+1)
    diffusion = 0.5 * params.sigma ** 2 * ttm * phi_grid * (phi_grid + 1.0)
    # Jump: Sepp convention (phi -> -phi of standard)
    jump = params.lambda_j * ttm * (
        jnp.exp(-phi_grid * params.mu_J + 0.5 * params.sigma_J ** 2 * phi_grid ** 2) - 1.0
        + kappa * phi_grid
    )
    return diffusion + jump


# ============================================================================
# Bates log-MGF (JAX) = Heston + Merton jumps
# ============================================================================
class BatesMGFParams(NamedTuple):
    v0: float
    theta: float
    kappa: float
    volvol: float
    rho: float
    lambda_j: float
    mu_J: float
    sigma_J: float


def bates_log_mgf(params: BatesMGFParams, ttm: float, phi_grid: jnp.ndarray) -> jnp.ndarray:
    """Compute Bates log-MGF = Heston + jump component (Sepp convention)."""
    heston_params = HestonMGFParams(
        v0=params.v0, theta=params.theta, kappa=params.kappa,
        volvol=params.volvol, rho=params.rho,
    )
    log_mgf_h = heston_log_mgf(heston_params, ttm, phi_grid)

    # Jump component in Sepp convention (phi -> -phi of standard):
    #   Sepp uses MGF of -log(S_T/F), so exp(-phi*mu_J) and +k*phi
    kappa_j = jnp.exp(params.mu_J + 0.5 * params.sigma_J ** 2) - 1.0
    log_mgf_jump = params.lambda_j * ttm * (
        jnp.exp(-phi_grid * params.mu_J + 0.5 * params.sigma_J ** 2 * phi_grid ** 2) - 1.0
        + kappa_j * phi_grid
    )

    return log_mgf_h + log_mgf_jump


# ============================================================================
# Black-Scholes log-MGF (JAX)
# ============================================================================
def bs_log_mgf(sigma: float, ttm: float, phi_grid: jnp.ndarray) -> jnp.ndarray:
    """Black-Scholes log-MGF (GBM with constant vol, Sepp convention)."""
    # Sepp convention: 0.5*sigma^2*T*phi*(phi+1)
    return 0.5 * sigma ** 2 * ttm * phi_grid * (phi_grid + 1.0)


# ============================================================================
# High-level JIT-compiled pricing functions
# ============================================================================
@jax.jit(static_argnums=(11,))
def heston_price_grid_jax(
    S: float, strikes: jnp.ndarray, is_call: jnp.ndarray,
    T: float, r: float, q: float,
    v0: float, theta: float, kappa: float, volvol: float, rho: float,
    max_phi: int = 1000,
) -> jnp.ndarray:
    """
    Price a grid of vanilla options under Heston. Fully JIT-compiled.

    Args:
        S: Spot price
        strikes: Shape (n_strikes,)
        is_call: Boolean array, shape (n_strikes,)
        T: Time to maturity
        r, q: Risk-free rate, dividend yield
        v0, theta, kappa, volvol, rho: Heston parameters
        max_phi: Phi grid size

    Returns:
        Option prices, shape (n_strikes,)
    """
    forward = S * jnp.exp((r - q) * T)
    discfactor = jnp.exp(-r * T)
    vol_scaler = jnp.minimum(0.3, jnp.sqrt(v0 * T))
    phi_grid = make_phi_grid(vol_scaler, max_phi)

    params = HestonMGFParams(v0=v0, theta=theta, kappa=kappa, volvol=volvol, rho=rho)
    log_mgf = heston_log_mgf(params, T, phi_grid)

    return price_slice_from_mgf(log_mgf, phi_grid, forward, strikes, is_call, discfactor)


@jax.jit(static_argnums=(10,))
def merton_price_grid_jax(
    S: float, strikes: jnp.ndarray, is_call: jnp.ndarray,
    T: float, r: float, q: float,
    sigma: float, lambda_j: float, mu_J: float, sigma_J: float,
    max_phi: int = 1000,
) -> jnp.ndarray:
    """Price a grid of vanilla options under Merton JD. Fully JIT-compiled."""
    forward = S * jnp.exp((r - q) * T)
    discfactor = jnp.exp(-r * T)
    vol_scaler = jnp.minimum(0.3, jnp.sqrt(sigma ** 2 * T))
    phi_grid = make_phi_grid(vol_scaler, max_phi)

    params = MertonMGFParams(sigma=sigma, lambda_j=lambda_j, mu_J=mu_J, sigma_J=sigma_J)
    log_mgf = merton_log_mgf(params, T, phi_grid)

    return price_slice_from_mgf(log_mgf, phi_grid, forward, strikes, is_call, discfactor)


@jax.jit(static_argnums=(14,))
def bates_price_grid_jax(
    S: float, strikes: jnp.ndarray, is_call: jnp.ndarray,
    T: float, r: float, q: float,
    v0: float, theta: float, kappa: float, volvol: float, rho: float,
    lambda_j: float, mu_J: float, sigma_J: float,
    max_phi: int = 1000,
) -> jnp.ndarray:
    """Price a grid of vanilla options under Bates. Fully JIT-compiled."""
    forward = S * jnp.exp((r - q) * T)
    discfactor = jnp.exp(-r * T)
    vol_scaler = jnp.minimum(0.3, jnp.sqrt(v0 * T))
    phi_grid = make_phi_grid(vol_scaler, max_phi)

    params = BatesMGFParams(
        v0=v0, theta=theta, kappa=kappa, volvol=volvol, rho=rho,
        lambda_j=lambda_j, mu_J=mu_J, sigma_J=sigma_J,
    )
    log_mgf = bates_log_mgf(params, T, phi_grid)

    return price_slice_from_mgf(log_mgf, phi_grid, forward, strikes, is_call, discfactor)


@jax.jit(static_argnums=(7,))
def bs_price_grid_jax(
    S: float, strikes: jnp.ndarray, is_call: jnp.ndarray,
    T: float, r: float, q: float, sigma: float,
    max_phi: int = 1000,
) -> jnp.ndarray:
    """Price a grid of vanilla options under Black-Scholes via MGF. Fully JIT-compiled."""
    forward = S * jnp.exp((r - q) * T)
    discfactor = jnp.exp(-r * T)
    vol_scaler = jnp.minimum(0.3, jnp.sqrt(sigma ** 2 * T))
    phi_grid = make_phi_grid(vol_scaler, max_phi)

    log_mgf = bs_log_mgf(sigma, T, phi_grid)

    return price_slice_from_mgf(log_mgf, phi_grid, forward, strikes, is_call, discfactor)


# ============================================================================
# Numpy-returning convenience wrappers (drop-in replacements)
# ============================================================================
def heston_price_slice_fast(
    S, strikes, T, r, q, v0, theta, kappa, volvol, rho,
    option_types, max_phi=1000,
):
    """Drop-in replacement for heston_price_slice. Returns numpy."""
    strikes_j = jnp.array(strikes, dtype=jnp.float64)
    is_call = jnp.array([t == 'call' for t in option_types], dtype=bool)
    prices = heston_price_grid_jax(
        float(S), strikes_j, is_call, float(T), float(r), float(q),
        float(v0), float(theta), float(kappa), float(volvol), float(rho),
        max_phi,
    )
    return to_numpy(prices)


def merton_price_slice_fast(
    S, strikes, T, r, q, sigma, lambda_j, mu_J, sigma_J,
    option_types, max_phi=1000,
):
    """Drop-in replacement for merton_price_slice. Returns numpy."""
    strikes_j = jnp.array(strikes, dtype=jnp.float64)
    is_call = jnp.array([t == 'call' for t in option_types], dtype=bool)
    prices = merton_price_grid_jax(
        float(S), strikes_j, is_call, float(T), float(r), float(q),
        float(sigma), float(lambda_j), float(mu_J), float(sigma_J),
        max_phi,
    )
    return to_numpy(prices)


def bates_price_slice_fast(
    S, strikes, T, r, q, v0, theta, kappa, volvol, rho,
    lambda_j, mu_J, sigma_J, option_types, max_phi=1000,
):
    """Drop-in replacement for bates_price_slice. Returns numpy."""
    strikes_j = jnp.array(strikes, dtype=jnp.float64)
    is_call = jnp.array([t == 'call' for t in option_types], dtype=bool)
    prices = bates_price_grid_jax(
        float(S), strikes_j, is_call, float(T), float(r), float(q),
        float(v0), float(theta), float(kappa), float(volvol), float(rho),
        float(lambda_j), float(mu_J), float(sigma_J),
        max_phi,
    )
    return to_numpy(prices)
