"""
NPE prior, JAX Heston simulator, and summary features.

Implements:
- Uniform prior over Heston parameters θ = (kappa, theta, sigma_v, rho, mu, v0)
- Unconstrained ↔ natural parameter transforms
- Euler-discretized Heston path simulator (full truncation)
- 16-dimensional summary features for NPE training/inference
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from jax import lax


# ────────────────────────────────────────────────────────────────────────────
# Prior specification
# ────────────────────────────────────────────────────────────────────────────
PRIOR_LOW = {
    "kappa": 0.5,
    "theta": 0.005,
    "sigma_v": 0.1,
    "rho": -0.95,
    "mu": -0.1,
    "v0": 0.005,
}

PRIOR_HIGH = {
    "kappa": 15.0,
    "theta": 0.25,
    "sigma_v": 1.5,
    "rho": 0.1,
    "mu": 0.3,
    "v0": 0.25,
}


# ────────────────────────────────────────────────────────────────────────────
# Parameter transforms
# ────────────────────────────────────────────────────────────────────────────
def to_unconstrained(thetas):
    """
    Transform natural parameters to unconstrained space.

    Transforms:
    - kappa, theta, sigma_v, v0: log
    - rho: atanh(rho / 0.99)
    - mu: identity

    Args:
        thetas: (n, 6) array [kappa, theta, sigma_v, rho, mu, v0]

    Returns:
        (n, 6) unconstrained parameters
    """
    kappa, theta, sigma_v, rho, mu, v0 = jnp.split(thetas, 6, axis=-1)

    kappa_unc = jnp.log(kappa)
    theta_unc = jnp.log(theta)
    sigma_v_unc = jnp.log(sigma_v)
    rho_unc = jnp.arctanh(rho / 0.99)
    mu_unc = mu  # Identity
    v0_unc = jnp.log(v0)

    return jnp.concatenate([kappa_unc, theta_unc, sigma_v_unc, rho_unc, mu_unc, v0_unc], axis=-1)


def to_natural(z):
    """
    Transform unconstrained parameters to natural space.

    Inverse of to_unconstrained.

    Args:
        z: (n, 6) unconstrained parameters

    Returns:
        (n, 6) natural parameters [kappa, theta, sigma_v, rho, mu, v0]
    """
    z0, z1, z2, z3, z4, z5 = jnp.split(z, 6, axis=-1)

    kappa = jnp.exp(z0)
    theta = jnp.exp(z1)
    sigma_v = jnp.exp(z2)
    rho = 0.99 * jnp.tanh(z3)
    mu = z4  # Identity
    v0 = jnp.exp(z5)

    return jnp.concatenate([kappa, theta, sigma_v, rho, mu, v0], axis=-1)


def sample_prior(key, n):
    """
    Sample n parameter vectors from the uniform prior.

    Args:
        key: JAX PRNGKey
        n: number of samples

    Returns:
        (n, 6) float32 array of natural-scale parameters
    """
    # Sample uniform [0, 1)
    u = jax.random.uniform(key, shape=(n, 6), dtype=jnp.float32)

    # Map to prior bounds
    param_names = ["kappa", "theta", "sigma_v", "rho", "mu", "v0"]
    low = jnp.array([PRIOR_LOW[name] for name in param_names], dtype=jnp.float32)
    high = jnp.array([PRIOR_HIGH[name] for name in param_names], dtype=jnp.float32)

    return low + u * (high - low)


# ────────────────────────────────────────────────────────────────────────────
# Heston simulator
# ────────────────────────────────────────────────────────────────────────────
def _variance_step(v_t, kappa, theta, sigma_v, dt, z_v):
    """
    Full-truncation Euler variance update.

    v_plus = max(v_t, 0) is used in BOTH the drift and the diffusion:
        v_{t+1} = v_t + kappa (theta - v_plus) dt + sigma_v sqrt(v_plus dt) z_v

    Args:
        v_t: current variance (may be negative)
        kappa, theta, sigma_v: Heston variance parameters
        dt: time increment
        z_v: standard normal variance shock

    Returns:
        v_{t+1} (un-truncated; truncation is applied at the next use)
    """
    v_plus = jnp.maximum(v_t, 0.0)
    return v_t + kappa * (theta - v_plus) * dt + sigma_v * jnp.sqrt(v_plus * dt) * z_v


def simulate_heston_paths(key, thetas, T, dt=1/252):
    """
    Simulate Heston model log-returns via Euler discretization with full truncation.

    Uses the SDE:
        dS_t / S_t = mu dt + sqrt(v_t) dW_t^S
        dv_t = kappa (theta - v_t) dt + sigma_v sqrt(v_t) dW_t^v
        dW_t^S dW_t^v = rho dt

    Euler scheme (full truncation):
        v_plus = max(v, 0)  # truncate before use (in BOTH sqrt and drift)
        r_t = (mu - 0.5 * v_plus) dt + sqrt(v_plus * dt) * (rho z_v + sqrt(1 - rho²) z_perp)
        v_{t+1} = v_t + kappa (theta - v_plus) dt + sigma_v sqrt(v_plus dt) z_v

    Args:
        key: JAX PRNGKey
        thetas: (n, 6) parameter array [kappa, theta, sigma_v, rho, mu, v0]
        T: number of time steps
        dt: time increment in years (default 1/252 for daily)

    Returns:
        (n, T) float32 log-returns
    """
    n = thetas.shape[0]

    # Unpack parameters
    kappa = thetas[:, 0]
    theta = thetas[:, 1]
    sigma_v = thetas[:, 2]
    rho = thetas[:, 3]
    mu = thetas[:, 4]
    v0 = thetas[:, 5]

    sqrt_1m_rho2 = jnp.sqrt(1.0 - rho**2)

    # Generate random shocks: (n, T, 2) for [z_v, z_perp]
    key, subkey = jax.random.split(key)
    z = jax.random.normal(subkey, shape=(n, T, 2), dtype=jnp.float32)
    z_v = z[:, :, 0]  # variance shocks
    z_perp = z[:, :, 1]  # orthogonal spot shocks

    def step_fn(v_t, t):
        """Single time step for variance and return."""
        # Full truncation
        v_plus = jnp.maximum(v_t, 0.0)

        # Spot return (correlated with variance shock)
        z_v_t = z_v[:, t]
        z_perp_t = z_perp[:, t]
        spot_shock = rho * z_v_t + sqrt_1m_rho2 * z_perp_t
        r_t = (mu - 0.5 * v_plus) * dt + jnp.sqrt(v_plus * dt) * spot_shock

        # Variance update (full truncation: v_plus in both drift and diffusion)
        v_tp1 = _variance_step(v_t, kappa, theta, sigma_v, dt, z_v_t)

        return v_tp1, r_t

    # Scan over time steps
    _, returns = lax.scan(step_fn, v0, jnp.arange(T))

    # returns is (T, n), transpose to (n, T)
    returns = returns.T

    return returns


# ────────────────────────────────────────────────────────────────────────────
# Summary features
# ────────────────────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "std_returns",
    "skew_returns",
    "excess_kurt_returns",
    "acf_r2_lag1",
    "acf_r2_lag5",
    "acf_r2_lag10",
    "acf_r2_lag21",
    "acf_abs_r_lag1",
    "acf_abs_r_lag5",
    "log_mean_rv21",
    "log_std_rv21",
    "corr_r_drv21",
    "corr_r_r2_lag1",
    "mean_abs_r",
    "frac_r_gt_2std",
    "log_std_rv63",
]


def _masked_mean(x, mask, eps=1e-12):
    """Compute masked mean, guarded against empty mask."""
    count = jnp.sum(mask, axis=-1, keepdims=True)
    count = jnp.maximum(count, eps)
    return jnp.sum(x * mask, axis=-1, keepdims=True) / count


def _masked_std(x, mask, eps=1e-12):
    """Compute masked standard deviation."""
    mean = _masked_mean(x, mask, eps)
    var = _masked_mean((x - mean)**2, mask, eps)
    return jnp.sqrt(jnp.maximum(var, eps))


def _masked_corr(x, y, mask, eps=1e-12):
    """Compute masked correlation coefficient."""
    mean_x = _masked_mean(x, mask, eps)
    mean_y = _masked_mean(y, mask, eps)

    dx = x - mean_x
    dy = y - mean_y

    cov = _masked_mean(dx * dy, mask, eps)
    std_x = jnp.sqrt(_masked_mean(dx**2, mask, eps))
    std_y = jnp.sqrt(_masked_mean(dy**2, mask, eps))

    denom = std_x * std_y
    denom = jnp.maximum(denom, eps)

    return cov / denom


def _masked_acf(x, mask, lag, eps=1e-12):
    """Compute masked autocorrelation at given lag."""
    # Align sequences
    x_t = x[:, :-lag]
    x_tlag = x[:, lag:]
    mask_t = mask[:, :-lag]
    mask_tlag = mask[:, lag:]

    # Both must be valid
    pair_mask = mask_t * mask_tlag

    return _masked_corr(x_t, x_tlag, pair_mask, eps)


def _rolling_mean(x, mask, window):
    """Compute rolling mean with masking."""
    N, T = x.shape
    if T < window:
        return jnp.zeros((N, 0)), jnp.zeros((N, 0))

    def convolve_row(x_row, mask_row):
        kernel = jnp.ones(window)
        sum_vals = jnp.convolve(x_row * mask_row, kernel, mode='valid')
        count = jnp.convolve(mask_row, kernel, mode='valid')
        mean = jnp.where(count > 0, sum_vals / count, 0.0)
        out_mask = (count == window).astype(jnp.float32)
        return mean, out_mask

    rolled, rolled_mask = jax.vmap(convolve_row)(x, mask)
    return rolled, rolled_mask


def summary_features(returns, mask):
    """
    Compute 16-dimensional summary features from log-returns.

    Features:
    1. std(returns)
    2. skew(returns)
    3. excess kurtosis(returns)
    4-7. acf(r²) at lags 1, 5, 10, 21
    8-9. acf(|r|) at lags 1, 5
    10. log mean RV(21)
    11. log std RV(21)
    12. corr(r_t, RV21_{t+1} - RV21_t)
    13. corr(r_t, r²_{t+1})
    14. mean |r|
    15. fraction |r| > 2σ
    16. log std RV(63)

    All masked, guarded against division by zero, finite on constant paths.

    Args:
        returns: (N, T) log-returns
        mask: (N, T) binary mask

    Returns:
        (N, 16) float32 feature array
    """
    N, T = returns.shape
    eps = 1e-12

    features = []

    # 1. Standard deviation
    std = _masked_std(returns, mask, eps)
    features.append(std[:, 0])

    # 2. Skewness
    mean = _masked_mean(returns, mask, eps)
    centered = returns - mean
    m3 = _masked_mean(centered**3, mask, eps)
    skew = m3 / jnp.maximum(std**3, eps)
    features.append(skew[:, 0])

    # 3. Excess kurtosis
    m4 = _masked_mean(centered**4, mask, eps)
    kurt = m4 / jnp.maximum(std**4, eps) - 3.0
    features.append(kurt[:, 0])

    # Squared returns and absolute returns
    r2 = returns**2
    abs_r = jnp.abs(returns)

    # 4-7. ACF of r² at lags 1, 5, 10, 21
    for lag in [1, 5, 10, 21]:
        acf = _masked_acf(r2, mask, lag, eps)
        features.append(acf[:, 0])

    # 8-9. ACF of |r| at lags 1, 5
    for lag in [1, 5]:
        acf = _masked_acf(abs_r, mask, lag, eps)
        features.append(acf[:, 0])

    # 10-11. log mean RV(21), log std RV(21)
    rv21, rv21_mask = _rolling_mean(r2, mask, 21)
    if rv21.shape[1] > 0:
        mean_rv21 = _masked_mean(rv21, rv21_mask, eps)
        std_rv21 = _masked_std(rv21, rv21_mask, eps)
        features.append(jnp.log(jnp.maximum(mean_rv21[:, 0], eps)))
        features.append(jnp.log(jnp.maximum(std_rv21[:, 0], eps)))
    else:
        # Not enough data for RV(21)
        features.append(jnp.full(N, jnp.log(eps)))
        features.append(jnp.full(N, jnp.log(eps)))

    # 12. corr(r_t, RV21_{t+1} - RV21_t)
    if rv21.shape[1] > 1:
        drv21 = rv21[:, 1:] - rv21[:, :-1]
        drv21_mask = rv21_mask[:, 1:] * rv21_mask[:, :-1]
        # Align with returns: need returns at same time points
        # rv21 starts at t=20 (0-indexed), so r starts at t=20
        # drv21 is rv21[t+1] - rv21[t], so we need r at t
        r_for_drv = returns[:, 20:20+drv21.shape[1]]
        r_mask_for_drv = mask[:, 20:20+drv21.shape[1]]
        pair_mask = r_mask_for_drv * drv21_mask
        corr_r_drv = _masked_corr(r_for_drv, drv21, pair_mask, eps)
        features.append(corr_r_drv[:, 0])
    else:
        features.append(jnp.zeros(N))

    # 13. corr(r_t, r²_{t+1})
    if T > 1:
        r_t = returns[:, :-1]
        r2_tp1 = r2[:, 1:]
        mask_t = mask[:, :-1]
        mask_tp1 = mask[:, 1:]
        pair_mask = mask_t * mask_tp1
        corr_r_r2 = _masked_corr(r_t, r2_tp1, pair_mask, eps)
        features.append(corr_r_r2[:, 0])
    else:
        features.append(jnp.zeros(N))

    # 14. mean |r|
    mean_abs = _masked_mean(abs_r, mask, eps)
    features.append(mean_abs[:, 0])

    # 15. fraction |r| > 2σ
    threshold = 2.0 * std
    outliers = (abs_r > threshold).astype(jnp.float32)
    frac_outliers = _masked_mean(outliers, mask, eps)
    features.append(frac_outliers[:, 0])

    # 16. log std RV(63)
    rv63, rv63_mask = _rolling_mean(r2, mask, 63)
    if rv63.shape[1] > 0:
        std_rv63 = _masked_std(rv63, rv63_mask, eps)
        features.append(jnp.log(jnp.maximum(std_rv63[:, 0], eps)))
    else:
        features.append(jnp.full(N, jnp.log(eps)))

    # Stack into (N, 16)
    return jnp.stack(features, axis=-1).astype(jnp.float32)
