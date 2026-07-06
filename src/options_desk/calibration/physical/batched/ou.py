"""
Batched Ornstein-Uhlenbeck AR(1) discretization MLE calibration via JAX.

Matches conventions of physical/ou_calibrator.py (scipy version, method='discretization').
"""
import jax
import jax.numpy as jnp
import numpy as np
from .common import pad_returns as pad_levels  # Alias for clarity


@jax.jit
def _fit(levels, mask, dt):
    """
    Batched OU exact MLE estimation via AR(1) closed-form regression.

    The OU process discretization is AR(1):
        X_{t+1} = a + b*X_t + ε
    where:
        b = exp(-κΔt)  → κ = -log(b)/Δt
        a = θ*(1 - b)  → θ = a/(1 - b)
        Var(ε) = σ²*(1 - exp(-2κΔt))/(2κ) → σ

    Args:
        levels: (N, T) level series
        mask: (N, T) binary mask
        dt: scalar time increment

    Returns:
        kappa, theta, sigma, log_likelihood, n (all shape (N,))
    """
    # AR(1) regression: X_t = a + b*X_{t-1} + ε
    # X_t corresponds to levels[:, 1:], X_{t-1} to levels[:, :-1]
    X_t = levels[:, 1:]
    X_lag = levels[:, :-1]
    mask_t = mask[:, 1:]
    mask_lag = mask[:, :-1]
    # Effective mask: both current and lagged must be valid
    eff_mask = mask_t * mask_lag

    # Compute n = number of valid pairs
    n = jnp.sum(eff_mask, axis=-1)

    # Masked OLS: regress X_t on [1, X_lag]
    # Normal equations: [sum(1*1), sum(1*X_lag)] [a]   [sum(1*X_t)]
    #                   [sum(X_lag*1), sum(X_lag*X_lag)] [b] = [sum(X_lag*X_t)]

    sum_1 = jnp.sum(eff_mask, axis=-1)  # Same as n
    sum_X_lag = jnp.sum(X_lag * eff_mask, axis=-1)
    sum_X_t = jnp.sum(X_t * eff_mask, axis=-1)
    sum_X_lag_sq = jnp.sum(X_lag * X_lag * eff_mask, axis=-1)
    sum_X_lag_X_t = jnp.sum(X_lag * X_t * eff_mask, axis=-1)

    # Solve 2x2 system
    det = sum_1 * sum_X_lag_sq - sum_X_lag * sum_X_lag
    a = (sum_X_lag_sq * sum_X_t - sum_X_lag * sum_X_lag_X_t) / det
    b = (sum_1 * sum_X_lag_X_t - sum_X_lag * sum_X_t) / det

    # Residuals
    predictions = a[:, None] + b[:, None] * X_lag
    residuals = X_t - predictions
    # Residual variance (ddof=2: we estimated 2 parameters a, b)
    sum_sq_resid = jnp.sum(residuals * residuals * eff_mask, axis=-1)
    residual_var = sum_sq_resid / (n - 2)

    # Convert AR(1) coefficients to OU parameters
    # b = exp(-κΔt), clamp to avoid log(0) or log(negative)
    b_safe = jnp.clip(b, 1e-10, 0.9999999)
    kappa = -jnp.log(b_safe) / dt

    # θ = a/(1-b), handle division by zero
    theta = jnp.where(
        jnp.abs(1 - b) > 1e-10,
        a / (1 - b),
        # Fallback: mean of the series
        jnp.sum(levels * mask, axis=-1) / jnp.sum(mask, axis=-1)
    )

    # σ from residual variance
    # Var(ε) = σ²*(1 - exp(-2κΔt))/(2κ)
    # → σ² = Var(ε) * 2κ / (1 - exp(-2κΔt))
    exp_minus_2kdt = jnp.exp(-2 * kappa * dt)
    denominator = 1 - exp_minus_2kdt
    # Guard against division by zero
    sigma_squared = jnp.where(
        denominator > 1e-10,
        residual_var * 2 * kappa / denominator,
        residual_var  # Fallback
    )
    sigma = jnp.sqrt(jnp.maximum(sigma_squared, 0.0))

    # Log-likelihood (Gaussian AR(1))
    # LL = -0.5 * n * (log(2π) + log(s²)) - 0.5 * sum(residuals²) / s²
    # Since sum(residuals²) = n * residual_var, this simplifies:
    # LL = -0.5 * n * (log(2π) + log(residual_var) + 1)
    log_likelihood = -0.5 * n * (jnp.log(2 * jnp.pi) + jnp.log(residual_var))
    log_likelihood -= 0.5 * sum_sq_resid / residual_var

    return kappa, theta, sigma, log_likelihood, n


def fit_batch(levels, mask, dt):
    """
    Calibrate OU parameters for a batch of assets.

    Args:
        levels: (N, T) array of level series
        mask: (N, T) binary mask (1.0 = valid, 0.0 = padding)
        dt: Time increment in years (e.g., 1/252 for daily)

    Returns:
        Dictionary with keys:
            kappa: (N,) mean reversion speed
            theta: (N,) long-term mean
            sigma: (N,) volatility
            log_likelihood: (N,) log-likelihood values
            half_life: (N,) half-life values (log(2)/kappa)
            n_observations: (N,) number of valid pairs per asset
            converged: (N,) boolean array, True if n >= 10
    """
    kappa, theta, sigma, ll, n = _fit(
        jnp.asarray(levels), jnp.asarray(mask), dt
    )

    half_life = jnp.log(2.0) / kappa

    return {
        "kappa": np.asarray(kappa, np.float64),
        "theta": np.asarray(theta, np.float64),
        "sigma": np.asarray(sigma, np.float64),
        "log_likelihood": np.asarray(ll, np.float64),
        "half_life": np.asarray(half_life, np.float64),
        "n_observations": np.asarray(n, np.int64),
        "converged": np.asarray(n >= 10),
    }
