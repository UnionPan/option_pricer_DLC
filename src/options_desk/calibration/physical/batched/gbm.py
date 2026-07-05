"""
Batched GBM calibration via JAX.

Matches conventions of physical/gbm_calibrator.py (scipy version).
"""
import jax
import jax.numpy as jnp
import numpy as np
from .common import masked_mean


def _masked_var_ddof1(x, mask, axis=-1):
    """
    Compute variance with Bessel's correction (ddof=1), ignoring masked entries.

    This matches scipy's np.var(ddof=1) behavior, which is the unbiased estimator.
    """
    n = jnp.sum(mask, axis=axis)
    mean = jnp.expand_dims(masked_mean(x, mask, axis=axis), axis=axis)
    sq_dev = (x - mean) ** 2
    sum_sq_dev = jnp.sum(sq_dev * mask, axis=axis)
    # Bessel's correction: divide by (n-1) instead of n
    return sum_sq_dev / (n - 1)


@jax.jit
def _fit(returns, mask, dt):
    """
    Batched GBM MLE estimation.

    Args:
        returns: (N, T) log-returns
        mask: (N, T) binary mask
        dt: scalar time increment

    Returns:
        mu, sigma, log_likelihood, n (all shape (N,))
    """
    n = mask.sum(-1)
    rbar = masked_mean(returns, mask)
    # Use ddof=1 to match scipy GBMCalibrator
    r_var = _masked_var_ddof1(returns, mask)

    sigma_squared = r_var / dt
    sigma = jnp.sqrt(sigma_squared)
    mu = rbar / dt + 0.5 * sigma_squared

    # Log-likelihood formula from scipy (uses r_var with ddof=1)
    log_likelihood = -0.5 * n * (jnp.log(2 * jnp.pi) + jnp.log(r_var))
    log_likelihood -= 0.5 * n

    return mu, sigma, log_likelihood, n


def fit_batch(returns, mask, dt):
    """
    Calibrate GBM parameters for a batch of assets.

    Args:
        returns: (N, T) array of log-returns
        mask: (N, T) binary mask (1.0 = valid, 0.0 = padding)
        dt: Time increment in years (e.g., 1/252 for daily)

    Returns:
        Dictionary with keys:
            mu: (N,) drift parameters
            sigma: (N,) volatility parameters
            log_likelihood: (N,) log-likelihood values
            n_observations: (N,) number of observations per asset
            converged: (N,) boolean array, True if n >= 2
    """
    mu, sigma, ll, n = _fit(jnp.asarray(returns), jnp.asarray(mask), dt)

    return {
        "mu": np.asarray(mu, np.float64),
        "sigma": np.asarray(sigma, np.float64),
        "log_likelihood": np.asarray(ll, np.float64),
        "n_observations": np.asarray(n, np.int64),
        "converged": np.asarray(n >= 2),
    }
