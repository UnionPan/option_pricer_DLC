"""
JAX-accelerated Fourier-based option pricers.

COS method (Fang & Oosterlee 2008) and Carr-Madan FFT (Carr & Madan 1999),
with vectorized characteristic functions for 7 stochastic models.

All CF evaluations are fully vectorized over the frequency grid —
no Python for-loops. Multi-strike pricing broadcasts over a strike
dimension, giving a single JIT-compiled kernel for N strikes.

author: Yunian Pan
email: yp1170@nyu.edu
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple

from ..processes._jax_backend import configure_jax_runtime
from ..processes._jax_utils import to_numpy

configure_jax_runtime()


# ============================================================================
# Characteristic function parameter containers
# ============================================================================

class GBMCFParams(NamedTuple):
    mu: float       # drift (risk-neutral: r - q)
    sigma: float


class HestonCFParams(NamedTuple):
    v0: float       # initial variance
    kappa: float    # mean reversion speed
    theta: float    # long-run variance
    sigma_v: float  # vol-of-vol
    rho: float      # correlation
    mu: float       # drift (risk-neutral: r - q)


class MertonCFParams(NamedTuple):
    mu: float       # drift (risk-neutral: r - q)
    sigma: float    # diffusion vol
    lambda_j: float # jump intensity
    mu_J: float     # mean of log-jump
    sigma_J: float  # std of log-jump


class BatesCFParams(NamedTuple):
    v0: float
    kappa: float
    theta: float
    sigma_v: float
    rho: float
    mu: float       # drift (risk-neutral: r - q)
    lambda_j: float
    mu_J: float
    sigma_J: float


class KouCFParams(NamedTuple):
    mu: float       # drift (risk-neutral: r - q)
    sigma: float
    lambda_j: float # jump intensity
    p: float        # prob of upward jump
    eta_up: float   # rate of upward exponential
    eta_down: float # rate of downward exponential


class VGCFParams(NamedTuple):
    theta_vg: float  # drift in subordinated BM (skewness)
    sigma: float     # vol in subordinated BM
    nu: float        # variance rate (kurtosis)
    mu: float        # risk-neutral drift


class NIGCFParams(NamedTuple):
    alpha: float     # tail heaviness
    beta: float      # asymmetry
    delta: float     # scale
    mu: float        # location/drift


# ============================================================================
# JAX Characteristic Functions — vectorized over u
# ============================================================================

def jax_gbm_cf(
    params: GBMCFParams, u: jnp.ndarray, log_S0: float, T: float,
) -> jnp.ndarray:
    """GBM characteristic function for log(S_T).  E[exp(iu log S_T)]."""
    drift = (params.mu - 0.5 * params.sigma**2) * T
    variance = params.sigma**2 * T
    return jnp.exp(1j * u * (log_S0 + drift) - 0.5 * u**2 * variance)


def jax_heston_cf(
    params: HestonCFParams, u: jnp.ndarray, log_S0: float, v0: float, T: float,
) -> jnp.ndarray:
    """Heston characteristic function for log(S_T).

    Uses the formulation from Heston (1993) / Albrecher et al. with the
    'little Heston trap' stable form (g ratio).
    """
    d = jnp.sqrt(
        (params.rho * params.sigma_v * 1j * u - params.kappa)**2
        + params.sigma_v**2 * (1j * u + u**2)
    )
    g = (
        (params.kappa - params.rho * params.sigma_v * 1j * u - d)
        / (params.kappa - params.rho * params.sigma_v * 1j * u + d)
    )
    exp_dt = jnp.exp(-d * T)

    C = (
        params.mu * 1j * u * T
        + (params.kappa * params.theta / params.sigma_v**2)
        * (
            (params.kappa - params.rho * params.sigma_v * 1j * u - d) * T
            - 2.0 * jnp.log((1.0 - g * exp_dt) / (1.0 - g))
        )
    )
    D = (
        (params.kappa - params.rho * params.sigma_v * 1j * u - d)
        / params.sigma_v**2
    ) * ((1.0 - exp_dt) / (1.0 - g * exp_dt))

    return jnp.exp(C + D * v0 + 1j * u * log_S0)


def jax_merton_cf(
    params: MertonCFParams, u: jnp.ndarray, log_S0: float, T: float,
) -> jnp.ndarray:
    """Merton jump-diffusion CF for log(S_T)."""
    k = jnp.exp(params.mu_J + 0.5 * params.sigma_J**2) - 1.0
    drift_term = 1j * u * (params.mu - 0.5 * params.sigma**2 - params.lambda_j * k)
    diffusion_term = -0.5 * params.sigma**2 * u**2
    jump_term = params.lambda_j * (
        jnp.exp(1j * u * params.mu_J - 0.5 * u**2 * params.sigma_J**2) - 1.0
    )
    psi = drift_term + diffusion_term + jump_term
    return jnp.exp(1j * u * log_S0 + T * psi)


def jax_bates_cf(
    params: BatesCFParams, u: jnp.ndarray, log_S0: float, v0: float, T: float,
) -> jnp.ndarray:
    """Bates (Heston + Merton jumps) CF for log(S_T)."""
    jump_mean = jnp.exp(params.mu_J + 0.5 * params.sigma_J**2) - 1.0
    mu_adj = params.mu - params.lambda_j * jump_mean

    d = jnp.sqrt(
        (params.rho * params.sigma_v * 1j * u - params.kappa)**2
        + params.sigma_v**2 * (1j * u + u**2)
    )
    g = (
        (params.kappa - params.rho * params.sigma_v * 1j * u - d)
        / (params.kappa - params.rho * params.sigma_v * 1j * u + d)
    )
    exp_dt = jnp.exp(-d * T)

    C = (
        mu_adj * 1j * u * T
        + (params.kappa * params.theta / params.sigma_v**2)
        * (
            (params.kappa - params.rho * params.sigma_v * 1j * u - d) * T
            - 2.0 * jnp.log((1.0 - g * exp_dt) / (1.0 - g))
        )
    )
    D = (
        (params.kappa - params.rho * params.sigma_v * 1j * u - d)
        / params.sigma_v**2
    ) * ((1.0 - exp_dt) / (1.0 - g * exp_dt))

    phi_heston = jnp.exp(C + D * v0 + 1j * u * log_S0)

    phi_jumps = jnp.exp(
        params.lambda_j * T * (
            jnp.exp(1j * u * params.mu_J - 0.5 * u**2 * params.sigma_J**2) - 1.0
        )
    )
    return phi_heston * phi_jumps


def jax_kou_cf(
    params: KouCFParams, u: jnp.ndarray, log_S0: float, T: float,
) -> jnp.ndarray:
    """Kou double-exponential jump-diffusion CF for log(S_T)."""
    # E[e^Y] - 1 for Kou double-exponential (risk-neutral compensator)
    k = (params.p * params.eta_up / (params.eta_up - 1.0)
         + (1.0 - params.p) * params.eta_down / (params.eta_down + 1.0) - 1.0)
    phi_Y = (
        params.p * params.eta_up / (params.eta_up - 1j * u)
        + (1.0 - params.p) * params.eta_down / (params.eta_down + 1j * u)
    )
    drift_term = 1j * u * (params.mu - 0.5 * params.sigma**2 - params.lambda_j * k)
    diffusion_term = -0.5 * params.sigma**2 * u**2
    jump_term = params.lambda_j * (phi_Y - 1.0)
    psi = drift_term + diffusion_term + jump_term
    return jnp.exp(1j * u * log_S0 + T * psi)


def jax_vg_cf(
    params: VGCFParams, u: jnp.ndarray, log_S0: float, T: float,
) -> jnp.ndarray:
    """Variance Gamma CF for log(S_T).

    psi_VG(u) = -(1/nu) * ln(1 - i*theta*nu*u + 0.5*sigma^2*nu*u^2)
    Full CF = exp(i*u*log_S0 + i*u*omega*T + T*psi_VG(u))
    where omega = (1/nu)*ln(1 - theta*nu - 0.5*sigma^2*nu) is the
    risk-neutral drift correction.
    """
    psi_vg = -(1.0 / params.nu) * jnp.log(
        1.0 - 1j * params.theta_vg * params.nu * u
        + 0.5 * params.sigma**2 * params.nu * u**2
    )
    omega = (1.0 / params.nu) * jnp.log(
        1.0 - params.theta_vg * params.nu - 0.5 * params.sigma**2 * params.nu
    )
    return jnp.exp(1j * u * log_S0 + 1j * u * (params.mu + omega) * T + T * psi_vg)


def jax_nig_cf(
    params: NIGCFParams, u: jnp.ndarray, log_S0: float, T: float,
) -> jnp.ndarray:
    """Normal Inverse Gaussian CF for log(S_T).

    psi_NIG(u) = i*mu*u + delta*(gamma - sqrt(alpha^2 - (beta + i*u)^2))
    where gamma = sqrt(alpha^2 - beta^2).
    Full CF includes risk-neutral drift correction.
    """
    gamma = jnp.sqrt(params.alpha**2 - params.beta**2)
    # Pure NIG exponent (no drift — mu is handled separately)
    psi_nig = params.delta * (gamma - jnp.sqrt(params.alpha**2 - (params.beta + 1j * u)**2))
    # Risk-neutral drift correction: omega = -psi_NIG(-i) = -delta*(gamma - sqrt(alpha^2 - (beta+1)^2))
    omega = -params.delta * (
        gamma - jnp.sqrt(params.alpha**2 - (params.beta + 1.0)**2)
    )
    drift = params.mu + omega
    return jnp.exp(1j * u * log_S0 + 1j * u * drift * T + T * psi_nig)


# ============================================================================
# COS method — core functions (Fang & Oosterlee 2008)
#
# Payoff coefficients derived from definite integrals:
#   chi_k(c,d) = integral_c^d exp(x) cos(k*pi*(x-a)/(b-a)) dx
#   psi_k(c,d) = integral_c^d cos(k*pi*(x-a)/(b-a)) dx
#
# Call V_k = 2/(b-a) * [chi_k(log_K, b) - K * psi_k(log_K, b)]
# Put  V_k = 2/(b-a) * [-chi_k(a, log_K) + K * psi_k(a, log_K)]
# Price = exp(-rT) * SUM_k' Re{phi(u_k)*exp(-i*u_k*a)} * V_k
# ============================================================================

def _chi_k(
    k: jnp.ndarray, a: float, b: float, c: float, d: float,
) -> jnp.ndarray:
    """Definite integral of exp(x)*cos(k*pi*(x-a)/(b-a)) dx from c to d.

    Antiderivative: F(x) = exp(x)/(1+w^2) * (cos(w(x-a)) + w*sin(w(x-a)))
    where w = k*pi/(b-a).
    """
    ba = b - a
    w = k * jnp.pi / ba

    def _F(x):
        """Antiderivative of exp(x)*cos(w*(x-a))."""
        return jnp.exp(x) / (1.0 + w**2) * (
            jnp.cos(w * (x - a)) + w * jnp.sin(w * (x - a))
        )

    result = _F(d) - _F(c)
    # k=0: integral of exp(x) from c to d = exp(d) - exp(c)
    return jnp.where(k == 0, jnp.exp(d) - jnp.exp(c), result)


def _psi_k(
    k: jnp.ndarray, a: float, b: float, c: float, d: float,
) -> jnp.ndarray:
    """Definite integral of cos(k*pi*(x-a)/(b-a)) dx from c to d.

    Antiderivative: G(x) = (b-a)/(k*pi) * sin(k*pi*(x-a)/(b-a))
    """
    ba = b - a
    w = k * jnp.pi / ba
    result = (ba / (k * jnp.pi)) * (
        jnp.sin(w * (d - a)) - jnp.sin(w * (c - a))
    )
    # k=0: integral of 1 from c to d = d - c
    return jnp.where(k == 0, d - c, result)


@jax.jit
def _jax_cos_price_single(
    cf_values: jnp.ndarray,
    K: float,
    T: float,
    r: float,
    a: float,
    b: float,
    is_call: bool,
) -> float:
    """COS price for a single strike. cf_values already evaluated on k-grid."""
    N = cf_values.shape[0]
    k = jnp.arange(N)
    ba = b - a
    u_k = k * jnp.pi / ba

    # Fourier coefficients of the density
    F_k = jnp.real(cf_values * jnp.exp(-1j * u_k * a))

    log_K = jnp.log(K)

    # Payoff coefficients V_k (Fang-Oosterlee Eq. 22-23)
    chi_call = _chi_k(k, a, b, log_K, b)
    psi_call = _psi_k(k, a, b, log_K, b)
    V_call = 2.0 / ba * (chi_call - K * psi_call)

    chi_put = _chi_k(k, a, b, a, log_K)
    psi_put = _psi_k(k, a, b, a, log_K)
    V_put = 2.0 / ba * (-chi_put + K * psi_put)

    V_k = jnp.where(is_call, V_call, V_put)

    # k=0 term gets half weight (multiplicative mask — MPS-safe, no .at[].set)
    half_weight = jnp.where(k == 0, 0.5, 1.0)

    # Price = exp(-rT) * SUM(F_k * V_k)
    price = jnp.exp(-r * T) * jnp.sum(F_k * V_k * half_weight)
    return price


@jax.jit
def _jax_cos_price_multi(
    cf_values: jnp.ndarray,
    strikes: jnp.ndarray,
    T: float,
    r: float,
    a: float,
    b: float,
    is_call: jnp.ndarray,
) -> jnp.ndarray:
    """COS price for multiple strikes. Broadcasting over strike dimension.

    Uses explicit (M, N) broadcasting instead of vmap for MPS compatibility.
    The MPS backend crashes on vmap + scatter ops (.at[].set).

    cf_values: shape (N,) — CF evaluated on k-grid (shared across strikes)
    strikes: shape (M,)
    is_call: shape (M,) boolean
    Returns: shape (M,)
    """
    N = cf_values.shape[0]
    k = jnp.arange(N)                          # (N,)
    ba = b - a
    u_k = k * jnp.pi / ba                      # (N,)

    F_k = jnp.real(cf_values * jnp.exp(-1j * u_k * a))  # (N,)

    log_K = jnp.log(strikes)                    # (M,)
    w = k * jnp.pi / ba                         # (N,)

    # chi_k broadcast: (M, N)
    def _F_mat(x_col):
        x = x_col[:, None]                      # (M, 1)
        return jnp.exp(x) / (1.0 + w[None, :]**2) * (
            jnp.cos(w[None, :] * (x - a))
            + w[None, :] * jnp.sin(w[None, :] * (x - a))
        )

    M = strikes.shape[0]
    b_vec = jnp.full(M, b)                      # (M,)

    # Call payoff coefficients: chi(log_K, b), psi(log_K, b)  → (M, N)
    chi_call = _F_mat(b_vec) - _F_mat(log_K)
    safe_k = jnp.where(k == 0, 1.0, k)          # (N,)
    psi_call = (ba / (safe_k[None, :] * jnp.pi)) * (
        jnp.sin(w[None, :] * (b - a))
        - jnp.sin(w[None, :] * (log_K[:, None] - a))
    )
    psi_call = jnp.where(k[None, :] == 0, (b - log_K)[:, None], psi_call)
    V_call = 2.0 / ba * (chi_call - strikes[:, None] * psi_call)  # (M, N)

    # Put payoff coefficients: chi(a, log_K), psi(a, log_K)  → (M, N)
    a_vec = jnp.full(M, a)
    chi_put = _F_mat(log_K) - _F_mat(a_vec)
    psi_put = (ba / (safe_k[None, :] * jnp.pi)) * (
        jnp.sin(w[None, :] * (log_K[:, None] - a))
        - jnp.sin(w[None, :] * (a - a))     # = 0 for all k
    )
    psi_put = jnp.where(k[None, :] == 0, (log_K - a)[:, None], psi_put)
    V_put = 2.0 / ba * (-chi_put + strikes[:, None] * psi_put)    # (M, N)

    # Select call or put per strike
    V_k = jnp.where(is_call[:, None], V_call, V_put)  # (M, N)

    # k=0 half weight (multiplicative mask — MPS-safe)
    half_weight = jnp.where(k == 0, 0.5, 1.0)   # (N,)

    # Sum over k dimension
    prices = jnp.exp(-r * T) * jnp.sum(
        F_k[None, :] * V_k * half_weight[None, :], axis=-1
    )
    return prices


def _cos_truncation_range(
    log_S0: float, drift: float, sigma_approx: float, T: float, L: float,
) -> tuple[float, float]:
    """Compute [a, b] truncation range for the COS method.

    Args:
        log_S0: log of spot price
        drift: risk-neutral drift rate per unit time (e.g. r - q - 0.5*sigma^2)
        sigma_approx: approximate volatility for range estimation
        T: time to maturity
        L: truncation parameter (typically 8-12)
    """
    c1 = log_S0 + drift * T
    c2 = max(sigma_approx**2 * T, 1e-12)
    a = c1 - L * np.sqrt(c2)
    b = c1 + L * np.sqrt(c2)
    return float(a), float(b)


# ============================================================================
# Carr-Madan FFT — core
# ============================================================================

@jax.jit
def _jax_carr_madan_fft(
    cf_values: jnp.ndarray,
    T: float,
    r: float,
    alpha: float,
    eta: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Carr-Madan FFT pricing (Carr & Madan 1999).

    cf_values: shape (N,) — CF of log(S_T) evaluated at v - (alpha+1)*1j
    Returns: (log_strikes, call_prices) arrays of shape (N,)

    The log-strike grid is centered: k_n = -N*lambda/2 + n*lambda
    The centering factor (-1)^j handles the shift in the FFT.
    """
    N = cf_values.shape[0]
    v = jnp.arange(N) * eta
    lambda_val = 2.0 * jnp.pi / (N * eta)
    k_u = -N * lambda_val / 2.0 + jnp.arange(N) * lambda_val

    # Modified characteristic function (Carr-Madan integrand)
    psi_vals = (
        jnp.exp(-r * T) * cf_values
        / (alpha**2 + alpha - v**2 + 1j * v * (2.0 * alpha + 1.0))
    )

    # Simpson weights
    weights = jnp.ones(N) * 2.0
    weights = weights.at[0].set(1.0)
    weights = weights.at[-1].set(1.0)
    odd_mask = jnp.arange(N) % 2 == 1
    weights = jnp.where(odd_mask, 4.0, weights)
    weights = weights * eta / 3.0

    # Centering factor: (-1)^j shifts the strike grid to be centered at 0
    centering = jnp.where(jnp.arange(N) % 2 == 0, 1.0, -1.0)

    # FFT
    fft_input = centering * psi_vals * weights
    fft_output = jnp.fft.fft(fft_input)

    # Call prices on the log-strike grid
    call_prices = jnp.real(fft_output * jnp.exp(-alpha * k_u)) / jnp.pi

    return k_u, call_prices


# ============================================================================
# Convenience wrappers — NumPy-in, NumPy-out
# ============================================================================

# ---------- GBM / Black-Scholes ----------

def jax_cos_price_gbm(
    S0: float, K: float, T: float, r: float, q: float,
    sigma: float, is_call: bool = True, N: int = 128, L: float = 10.0,
) -> float:
    """COS price for GBM (Black-Scholes)."""
    mu = r - q
    log_S0 = np.log(S0)
    a, b = _cos_truncation_range(log_S0, mu - 0.5 * sigma**2, sigma, T, L)
    k = jnp.arange(N)
    u = k * jnp.pi / (b - a)
    params = GBMCFParams(mu=mu, sigma=sigma)
    cf_vals = jax_gbm_cf(params, u, log_S0, T)
    price = _jax_cos_price_single(cf_vals, K, T, r, a, b, is_call)
    return float(price)


def jax_cos_price_gbm_multi(
    S0: float, strikes: np.ndarray, T: float, r: float, q: float,
    sigma: float, is_call: np.ndarray, N: int = 128, L: float = 10.0,
) -> np.ndarray:
    """COS prices for GBM across multiple strikes."""
    mu = r - q
    log_S0 = np.log(S0)
    a, b = _cos_truncation_range(log_S0, mu - 0.5 * sigma**2, sigma, T, L)
    k = jnp.arange(N)
    u = k * jnp.pi / (b - a)
    params = GBMCFParams(mu=mu, sigma=sigma)
    cf_vals = jax_gbm_cf(params, u, log_S0, T)
    prices = _jax_cos_price_multi(
        cf_vals, jnp.asarray(strikes), T, r, a, b, jnp.asarray(is_call),
    )
    return to_numpy(prices)


# ---------- Heston ----------

def jax_cos_price_heston(
    S0: float, K: float, T: float, r: float, q: float,
    v0: float, kappa: float, theta: float, sigma_v: float, rho: float,
    is_call: bool = True, N: int = 256, L: float = 12.0,
) -> float:
    """COS price for Heston model."""
    mu = r - q
    log_S0 = np.log(S0)
    sigma_approx = np.sqrt(v0)
    a, b = _cos_truncation_range(log_S0, mu - 0.5 * v0, sigma_approx, T, L)
    k = jnp.arange(N)
    u = k * jnp.pi / (b - a)
    params = HestonCFParams(v0=v0, kappa=kappa, theta=theta,
                            sigma_v=sigma_v, rho=rho, mu=mu)
    cf_vals = jax_heston_cf(params, u, log_S0, v0, T)
    price = _jax_cos_price_single(cf_vals, K, T, r, a, b, is_call)
    return float(price)


def jax_cos_price_heston_multi(
    S0: float, strikes: np.ndarray, T: float, r: float, q: float,
    v0: float, kappa: float, theta: float, sigma_v: float, rho: float,
    is_call: np.ndarray, N: int = 256, L: float = 12.0,
) -> np.ndarray:
    """COS prices for Heston across multiple strikes."""
    mu = r - q
    log_S0 = np.log(S0)
    sigma_approx = np.sqrt(v0)
    a, b = _cos_truncation_range(log_S0, mu - 0.5 * v0, sigma_approx, T, L)
    k = jnp.arange(N)
    u = k * jnp.pi / (b - a)
    params = HestonCFParams(v0=v0, kappa=kappa, theta=theta,
                            sigma_v=sigma_v, rho=rho, mu=mu)
    cf_vals = jax_heston_cf(params, u, log_S0, v0, T)
    prices = _jax_cos_price_multi(
        cf_vals, jnp.asarray(strikes), T, r, a, b, jnp.asarray(is_call),
    )
    return to_numpy(prices)


# ---------- Merton ----------

def jax_cos_price_merton(
    S0: float, K: float, T: float, r: float, q: float,
    sigma: float, lambda_j: float, mu_J: float, sigma_J: float,
    is_call: bool = True, N: int = 256, L: float = 12.0,
) -> float:
    """COS price for Merton jump-diffusion."""
    mu = r - q
    log_S0 = np.log(S0)
    total_var = sigma**2 + lambda_j * (mu_J**2 + sigma_J**2)
    sigma_approx = np.sqrt(total_var)
    a, b = _cos_truncation_range(log_S0, mu - 0.5 * sigma**2, sigma_approx, T, L)
    k = jnp.arange(N)
    u = k * jnp.pi / (b - a)
    params = MertonCFParams(mu=mu, sigma=sigma, lambda_j=lambda_j,
                            mu_J=mu_J, sigma_J=sigma_J)
    cf_vals = jax_merton_cf(params, u, log_S0, T)
    price = _jax_cos_price_single(cf_vals, K, T, r, a, b, is_call)
    return float(price)


def jax_cos_price_merton_multi(
    S0: float, strikes: np.ndarray, T: float, r: float, q: float,
    sigma: float, lambda_j: float, mu_J: float, sigma_J: float,
    is_call: np.ndarray, N: int = 256, L: float = 12.0,
) -> np.ndarray:
    """COS prices for Merton across multiple strikes."""
    mu = r - q
    log_S0 = np.log(S0)
    total_var = sigma**2 + lambda_j * (mu_J**2 + sigma_J**2)
    sigma_approx = np.sqrt(total_var)
    a, b = _cos_truncation_range(log_S0, mu - 0.5 * sigma**2, sigma_approx, T, L)
    k = jnp.arange(N)
    u = k * jnp.pi / (b - a)
    params = MertonCFParams(mu=mu, sigma=sigma, lambda_j=lambda_j,
                            mu_J=mu_J, sigma_J=sigma_J)
    cf_vals = jax_merton_cf(params, u, log_S0, T)
    prices = _jax_cos_price_multi(
        cf_vals, jnp.asarray(strikes), T, r, a, b, jnp.asarray(is_call),
    )
    return to_numpy(prices)


# ---------- Bates ----------

def jax_cos_price_bates(
    S0: float, K: float, T: float, r: float, q: float,
    v0: float, kappa: float, theta: float, sigma_v: float, rho: float,
    lambda_j: float, mu_J: float, sigma_J: float,
    is_call: bool = True, N: int = 256, L: float = 12.0,
) -> float:
    """COS price for Bates model."""
    mu = r - q
    log_S0 = np.log(S0)
    sigma_approx = np.sqrt(v0)
    a, b = _cos_truncation_range(log_S0, mu - 0.5 * v0, sigma_approx, T, L)
    k = jnp.arange(N)
    u = k * jnp.pi / (b - a)
    params = BatesCFParams(v0=v0, kappa=kappa, theta=theta, sigma_v=sigma_v,
                           rho=rho, mu=mu, lambda_j=lambda_j,
                           mu_J=mu_J, sigma_J=sigma_J)
    cf_vals = jax_bates_cf(params, u, log_S0, v0, T)
    price = _jax_cos_price_single(cf_vals, K, T, r, a, b, is_call)
    return float(price)


def jax_cos_price_bates_multi(
    S0: float, strikes: np.ndarray, T: float, r: float, q: float,
    v0: float, kappa: float, theta: float, sigma_v: float, rho: float,
    lambda_j: float, mu_J: float, sigma_J: float,
    is_call: np.ndarray, N: int = 256, L: float = 12.0,
) -> np.ndarray:
    """COS prices for Bates across multiple strikes."""
    mu = r - q
    log_S0 = np.log(S0)
    sigma_approx = np.sqrt(v0)
    a, b = _cos_truncation_range(log_S0, mu - 0.5 * v0, sigma_approx, T, L)
    k = jnp.arange(N)
    u = k * jnp.pi / (b - a)
    params = BatesCFParams(v0=v0, kappa=kappa, theta=theta, sigma_v=sigma_v,
                           rho=rho, mu=mu, lambda_j=lambda_j,
                           mu_J=mu_J, sigma_J=sigma_J)
    cf_vals = jax_bates_cf(params, u, log_S0, v0, T)
    prices = _jax_cos_price_multi(
        cf_vals, jnp.asarray(strikes), T, r, a, b, jnp.asarray(is_call),
    )
    return to_numpy(prices)


# ---------- Kou ----------

def jax_cos_price_kou(
    S0: float, K: float, T: float, r: float, q: float,
    sigma: float, lambda_j: float, p: float, eta_up: float, eta_down: float,
    is_call: bool = True, N: int = 256, L: float = 12.0,
) -> float:
    """COS price for Kou double-exponential jump-diffusion."""
    mu = r - q
    log_S0 = np.log(S0)
    a, b = _cos_truncation_range(log_S0, mu - 0.5 * sigma**2, sigma, T, L)
    k = jnp.arange(N)
    u = k * jnp.pi / (b - a)
    params = KouCFParams(mu=mu, sigma=sigma, lambda_j=lambda_j,
                         p=p, eta_up=eta_up, eta_down=eta_down)
    cf_vals = jax_kou_cf(params, u, log_S0, T)
    price = _jax_cos_price_single(cf_vals, K, T, r, a, b, is_call)
    return float(price)


def jax_cos_price_kou_multi(
    S0: float, strikes: np.ndarray, T: float, r: float, q: float,
    sigma: float, lambda_j: float, p: float, eta_up: float, eta_down: float,
    is_call: np.ndarray, N: int = 256, L: float = 12.0,
) -> np.ndarray:
    """COS prices for Kou across multiple strikes."""
    mu = r - q
    log_S0 = np.log(S0)
    a, b = _cos_truncation_range(log_S0, mu - 0.5 * sigma**2, sigma, T, L)
    k = jnp.arange(N)
    u = k * jnp.pi / (b - a)
    params = KouCFParams(mu=mu, sigma=sigma, lambda_j=lambda_j,
                         p=p, eta_up=eta_up, eta_down=eta_down)
    cf_vals = jax_kou_cf(params, u, log_S0, T)
    prices = _jax_cos_price_multi(
        cf_vals, jnp.asarray(strikes), T, r, a, b, jnp.asarray(is_call),
    )
    return to_numpy(prices)


# ---------- Variance Gamma ----------

def jax_cos_price_vg(
    S0: float, K: float, T: float, r: float, q: float,
    theta_vg: float, sigma: float, nu: float,
    is_call: bool = True, N: int = 256, L: float = 12.0,
) -> float:
    """COS price for Variance Gamma."""
    mu = r - q
    log_S0 = np.log(S0)
    total_var = sigma**2 + nu * theta_vg**2
    sigma_approx = np.sqrt(total_var)
    a, b = _cos_truncation_range(log_S0, mu, sigma_approx, T, L)
    k = jnp.arange(N)
    u = k * jnp.pi / (b - a)
    params = VGCFParams(theta_vg=theta_vg, sigma=sigma, nu=nu, mu=mu)
    cf_vals = jax_vg_cf(params, u, log_S0, T)
    price = _jax_cos_price_single(cf_vals, K, T, r, a, b, is_call)
    return float(price)


def jax_cos_price_vg_multi(
    S0: float, strikes: np.ndarray, T: float, r: float, q: float,
    theta_vg: float, sigma: float, nu: float,
    is_call: np.ndarray, N: int = 256, L: float = 12.0,
) -> np.ndarray:
    """COS prices for Variance Gamma across multiple strikes."""
    mu = r - q
    log_S0 = np.log(S0)
    total_var = sigma**2 + nu * theta_vg**2
    sigma_approx = np.sqrt(total_var)
    a, b = _cos_truncation_range(log_S0, mu, sigma_approx, T, L)
    k = jnp.arange(N)
    u = k * jnp.pi / (b - a)
    params = VGCFParams(theta_vg=theta_vg, sigma=sigma, nu=nu, mu=mu)
    cf_vals = jax_vg_cf(params, u, log_S0, T)
    prices = _jax_cos_price_multi(
        cf_vals, jnp.asarray(strikes), T, r, a, b, jnp.asarray(is_call),
    )
    return to_numpy(prices)


# ---------- NIG ----------

def jax_cos_price_nig(
    S0: float, K: float, T: float, r: float, q: float,
    alpha: float, beta: float, delta: float,
    is_call: bool = True, N: int = 256, L: float = 12.0,
) -> float:
    """COS price for Normal Inverse Gaussian."""
    mu = r - q
    log_S0 = np.log(S0)
    gamma_nig = np.sqrt(alpha**2 - beta**2)
    nig_var = delta * alpha**2 / gamma_nig**3
    sigma_approx = np.sqrt(nig_var)
    a, b = _cos_truncation_range(log_S0, mu, sigma_approx, T, L)
    k = jnp.arange(N)
    u = k * jnp.pi / (b - a)
    params = NIGCFParams(alpha=alpha, beta=beta, delta=delta, mu=mu)
    cf_vals = jax_nig_cf(params, u, log_S0, T)
    price = _jax_cos_price_single(cf_vals, K, T, r, a, b, is_call)
    return float(price)


def jax_cos_price_nig_multi(
    S0: float, strikes: np.ndarray, T: float, r: float, q: float,
    alpha: float, beta: float, delta: float,
    is_call: np.ndarray, N: int = 256, L: float = 12.0,
) -> np.ndarray:
    """COS prices for NIG across multiple strikes."""
    mu = r - q
    log_S0 = np.log(S0)
    gamma_nig = np.sqrt(alpha**2 - beta**2)
    nig_var = delta * alpha**2 / gamma_nig**3
    sigma_approx = np.sqrt(nig_var)
    a, b = _cos_truncation_range(log_S0, mu, sigma_approx, T, L)
    k = jnp.arange(N)
    u = k * jnp.pi / (b - a)
    params = NIGCFParams(alpha=alpha, beta=beta, delta=delta, mu=mu)
    cf_vals = jax_nig_cf(params, u, log_S0, T)
    prices = _jax_cos_price_multi(
        cf_vals, jnp.asarray(strikes), T, r, a, b, jnp.asarray(is_call),
    )
    return to_numpy(prices)


# ---------- Carr-Madan wrappers ----------

def _carr_madan_interp(
    k_u: jnp.ndarray, call_prices: jnp.ndarray,
    K: float, S0: float, T: float, r: float, q: float, is_call: bool,
) -> float:
    """Interpolate Carr-Madan FFT output to a specific strike."""
    log_K = np.log(K)
    price = float(jnp.interp(log_K, k_u, call_prices))
    if not is_call:
        price = price - S0 * np.exp(-q * T) + K * np.exp(-r * T)
    return price


def jax_carr_madan_price_gbm(
    S0: float, K: float, T: float, r: float, q: float,
    sigma: float, is_call: bool = True,
    N: int = 4096, alpha: float = 1.5, eta: float = 0.25,
) -> float:
    """Carr-Madan FFT price for GBM."""
    mu = r - q
    log_S0 = np.log(S0)
    v = jnp.arange(N) * eta
    u_cm = v - (alpha + 1.0) * 1j
    params = GBMCFParams(mu=mu, sigma=sigma)
    cf_vals = jax_gbm_cf(params, u_cm, log_S0, T)
    k_u, call_prices = _jax_carr_madan_fft(cf_vals, T, r, alpha, eta)
    return _carr_madan_interp(k_u, call_prices, K, S0, T, r, q, is_call)


def jax_carr_madan_price_heston(
    S0: float, K: float, T: float, r: float, q: float,
    v0: float, kappa: float, theta: float, sigma_v: float, rho: float,
    is_call: bool = True, N: int = 4096, alpha: float = 1.5, eta: float = 0.25,
) -> float:
    """Carr-Madan FFT price for Heston."""
    mu = r - q
    log_S0 = np.log(S0)
    v = jnp.arange(N) * eta
    u_cm = v - (alpha + 1.0) * 1j
    params = HestonCFParams(v0=v0, kappa=kappa, theta=theta,
                            sigma_v=sigma_v, rho=rho, mu=mu)
    cf_vals = jax_heston_cf(params, u_cm, log_S0, v0, T)
    k_u, call_prices = _jax_carr_madan_fft(cf_vals, T, r, alpha, eta)
    return _carr_madan_interp(k_u, call_prices, K, S0, T, r, q, is_call)


def jax_carr_madan_price_merton(
    S0: float, K: float, T: float, r: float, q: float,
    sigma: float, lambda_j: float, mu_J: float, sigma_J: float,
    is_call: bool = True, N: int = 4096, alpha: float = 1.5, eta: float = 0.25,
) -> float:
    """Carr-Madan FFT price for Merton jump-diffusion."""
    mu = r - q
    log_S0 = np.log(S0)
    v = jnp.arange(N) * eta
    u_cm = v - (alpha + 1.0) * 1j
    params = MertonCFParams(mu=mu, sigma=sigma, lambda_j=lambda_j,
                            mu_J=mu_J, sigma_J=sigma_J)
    cf_vals = jax_merton_cf(params, u_cm, log_S0, T)
    k_u, call_prices = _jax_carr_madan_fft(cf_vals, T, r, alpha, eta)
    return _carr_madan_interp(k_u, call_prices, K, S0, T, r, q, is_call)
