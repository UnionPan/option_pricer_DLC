"""
Batched Heston QMLE calibration via JAX.

Matches conventions of physical/heston_qmle.py (scipy reference, method-of-moments).
Includes Garman-Klass OHLC proxy variant for higher-fidelity variance estimation.
"""
import jax
import jax.numpy as jnp
import numpy as np


def _rolling_mean_masked(x, mask, window):
    """
    Compute causal rolling mean with masking via convolution.

    Matches scipy's np.convolve approach: applies a uniform kernel to both
    the data and the mask, then divides to get the mean.

    Args:
        x: (N, T) input array
        mask: (N, T) binary mask
        window: int, rolling window size

    Returns:
        rolled: (N, T-window+1) rolling mean
        rolled_mask: (N, T-window+1) mask (1.0 if all entries in window valid)
    """
    N, T = x.shape
    if window > T:
        # Return empty arrays
        return jnp.zeros((N, 0)), jnp.zeros((N, 0))

    # Kernel for rolling window
    kernel = jnp.ones(window)

    # We'll process each asset separately using vmap
    def process_asset(x_row, mask_row):
        # Convolve data with kernel
        masked_data = x_row * mask_row
        sum_vals = jnp.convolve(masked_data, kernel, mode='valid')

        # Convolve mask with kernel to count valid entries
        count = jnp.convolve(mask_row, kernel, mode='valid')

        # Compute mean
        mean = jnp.where(count > 0, sum_vals / count, 0.0)

        # Output mask: 1.0 if window is fully valid
        out_mask = (count == window).astype(jnp.float32)

        return mean, out_mask

    rolled, rolled_mask = jax.vmap(process_asset)(x, mask)

    return rolled, rolled_mask


def _garman_klass_variance(open_, high, low, close, dt):
    """
    Compute Garman-Klass variance estimator from OHLC data.

    σ²_GK = 0.5·ln(H/L)² - (2ln2-1)·ln(C/O)²

    Divided by dt to annualize.

    Args:
        open_, high, low, close: (N, T) arrays
        dt: time increment in years

    Returns:
        gk_var: (N, T) annualized variance estimates
    """
    # Ensure positive prices
    open_ = jnp.maximum(open_, 1e-10)
    high = jnp.maximum(high, 1e-10)
    low = jnp.maximum(low, 1e-10)
    close = jnp.maximum(close, 1e-10)

    # Garman-Klass formula
    hl_term = 0.5 * jnp.log(high / low) ** 2
    co_term = (2 * jnp.log(2.0) - 1) * jnp.log(close / open_) ** 2
    gk_var = (hl_term - co_term) / dt

    # Ensure non-negative
    gk_var = jnp.maximum(gk_var, 1e-10)

    return gk_var


@jax.jit
def _fit_heston_qmle(returns_full, mask_full, returns_aligned, mask_aligned, dt, effective_dt, variance_proxy, var_mask):
    """
    Core Heston QMLE estimation matching the scipy reference pipeline.

    Args:
        returns_full: (N, T) full log-returns (for mu estimation)
        mask_full: (N, T) mask for full returns
        returns_aligned: (N, T_v) returns aligned with variance proxy (for rho)
        mask_aligned: (N, T_v) mask for aligned returns
        dt: scalar time increment (original)
        effective_dt: scalar effective time increment (window * dt)
        variance_proxy: (N, T_v) realized variance proxy (already smoothed)
        var_mask: (N, T_v) mask for variance proxy

    Returns:
        Dictionary with estimated parameters (all shape (N,))
    """
    N = returns_full.shape[0]

    # Effective dt for smoothed variance (window * dt)
    # We'll extract this from the alignment below
    # For now, assume window=1 case (will be passed correctly from caller)

    # ────────────────────────────────────────────────────────────────
    # Step 1: Variance process parameters via AR(1) MLE
    # ────────────────────────────────────────────────────────────────
    v_t = variance_proxy[:, :-1]
    v_tp1 = variance_proxy[:, 1:]
    mask_t = var_mask[:, :-1]
    mask_tp1 = var_mask[:, 1:]
    eff_mask_v = mask_t * mask_tp1

    n_obs_v = jnp.sum(eff_mask_v, axis=-1)

    # theta from sample mean
    theta = jnp.sum(variance_proxy * var_mask, axis=-1) / jnp.sum(var_mask, axis=-1)

    # kappa from AR(1) coefficient: phi = cov(v_t, v_tp1) / var(v_t)
    # E[v_{t+1} | v_t] = v_t + kappa*(theta - v_t)*dt  →  phi = 1 - kappa*dt
    mean_v_t = jnp.sum(v_t * eff_mask_v, axis=-1) / jnp.maximum(n_obs_v, 1.0)
    mean_v_tp1 = jnp.sum(v_tp1 * eff_mask_v, axis=-1) / jnp.maximum(n_obs_v, 1.0)

    cov_num = jnp.sum((v_t - mean_v_t[:, None]) * (v_tp1 - mean_v_tp1[:, None]) * eff_mask_v, axis=-1)
    var_v_t = jnp.sum((v_t - mean_v_t[:, None]) ** 2 * eff_mask_v, axis=-1)

    phi = jnp.where(var_v_t > 1e-12, cov_num / var_v_t, 0.5)
    phi = jnp.clip(phi, 0.001, 0.999)

    kappa = jnp.maximum(-jnp.log(phi) / effective_dt, 1e-3)

    # sigma_v from stationary variance: Var[v] = sigma_v^2 * theta / (2*kappa)
    var_V = jnp.sum((variance_proxy - theta[:, None]) ** 2 * var_mask, axis=-1) / jnp.sum(var_mask, axis=-1)
    sigma_v_sq = 2.0 * kappa * var_V / jnp.maximum(theta, 1e-8)
    sigma_v = jnp.sqrt(jnp.maximum(sigma_v_sq, 0.0))

    # ────────────────────────────────────────────────────────────────
    # Step 2: rho from corr(r_t, Δv_t)
    # ────────────────────────────────────────────────────────────────
    dv = v_tp1 - v_t

    # Use aligned returns passed from caller
    # r_aligned = returns_aligned[:-1] to match dv
    T_v = variance_proxy.shape[1]
    r_aligned = returns_aligned[:, :T_v-1]
    mask_r_aligned = mask_aligned[:, :T_v-1]

    # Effective mask: both return and variance change must be valid
    eff_mask_rho = mask_r_aligned * eff_mask_v

    n_obs_rho = jnp.sum(eff_mask_rho, axis=-1)

    # Structural moment: Cov(r, dv) ≈ rho * sigma_v * sqrt(theta) * effective_dt
    mean_r = jnp.sum(r_aligned * eff_mask_rho, axis=-1) / jnp.maximum(n_obs_rho, 1.0)
    mean_dv = jnp.sum(dv * eff_mask_rho, axis=-1) / jnp.maximum(n_obs_rho, 1.0)

    cov_r_dv = jnp.sum((r_aligned - mean_r[:, None]) * (dv - mean_dv[:, None]) * eff_mask_rho, axis=-1)
    cov_r_dv = cov_r_dv / jnp.maximum(n_obs_rho - 1, 1.0)  # ddof=1

    denom = sigma_v * jnp.sqrt(jnp.maximum(theta, 1e-8)) * effective_dt
    rho_raw = cov_r_dv / jnp.maximum(denom, 1e-12)
    rho = jnp.clip(rho_raw, -0.99, 0.99)

    # ────────────────────────────────────────────────────────────────
    # Step 3: mu and v0
    # ────────────────────────────────────────────────────────────────
    # Use FULL returns for mu estimation (not aligned)
    n_obs_r = jnp.sum(mask_full, axis=-1)
    mean_return = jnp.sum(returns_full * mask_full, axis=-1) / jnp.maximum(n_obs_r, 1.0)
    mu = mean_return / dt + 0.5 * theta  # Itô correction

    # v0 = last realized variance (last valid value in variance_proxy)
    # Find the last valid entry for each asset
    def get_last_valid_variance(v_row, m_row, theta_val):
        # Find max index where mask is 1
        indices = jnp.arange(m_row.shape[0])
        valid_mask = m_row > 0
        valid_indices = jnp.where(valid_mask, indices, -1)
        last_idx = jnp.max(valid_indices)
        # If no valid entries (last_idx == -1), use theta
        return jnp.where(last_idx >= 0, v_row[last_idx], theta_val)

    v0 = jax.vmap(get_last_valid_variance)(variance_proxy, var_mask, theta)

    # ────────────────────────────────────────────────────────────────
    # Diagnostics
    # ────────────────────────────────────────────────────────────────
    # Log-likelihood of AR(1) step
    v_hat = v_t + kappa[:, None] * (theta[:, None] - v_t) * effective_dt
    ss_var = (sigma_v[:, None] ** 2) * jnp.maximum(v_t, 1e-10) * effective_dt
    ss_var = jnp.maximum(ss_var, 1e-12)

    ll_terms = -0.5 * (jnp.log(2 * jnp.pi * ss_var) + (v_tp1 - v_hat) ** 2 / ss_var)
    log_likelihood = jnp.sum(ll_terms * eff_mask_v, axis=-1)

    # R^2 of AR(1) fit
    ss_res = jnp.sum((v_tp1 - v_hat) ** 2 * eff_mask_v, axis=-1)
    ss_tot = jnp.sum((v_tp1 - mean_v_tp1[:, None]) ** 2 * eff_mask_v, axis=-1)
    variance_proxy_r2 = jnp.where(ss_tot > 1e-12, 1.0 - ss_res / ss_tot, 0.0)

    # Feller condition
    feller_ratio = 2.0 * kappa * theta / jnp.maximum(sigma_v ** 2, 1e-12)
    converged = jnp.ones(N, dtype=bool)  # Method-of-moments always converges

    n_observations = n_obs_r  # Total number of returns

    return {
        "kappa": kappa,
        "theta": theta,
        "sigma_v": sigma_v,
        "rho": rho,
        "mu": mu,
        "v0": v0,
        "log_likelihood": log_likelihood,
        "feller_ratio": feller_ratio,
        "variance_proxy_r2": variance_proxy_r2,
        "converged": converged,
        "n_observations": n_observations,
    }


def fit_batch(returns, mask, dt, smooth_window=10):
    """
    Calibrate Heston QMLE parameters for a batch of assets using close-close returns.

    Args:
        returns: (N, T) array of log-returns
        mask: (N, T) binary mask (1.0 = valid, 0.0 = padding)
        dt: Time increment in years (e.g., 1/252 for daily)
        smooth_window: Rolling window for variance proxy smoothing (default 10)

    Returns:
        Dictionary with keys:
            kappa: (N,) mean reversion speed of variance
            theta: (N,) long-run variance
            sigma_v: (N,) volatility of variance
            rho: (N,) spot-vol correlation
            mu: (N,) drift
            v0: (N,) initial variance (current realized variance estimate)
            log_likelihood: (N,) log-likelihood values
            feller_ratio: (N,) Feller condition ratio (>1 = OK)
            variance_proxy_r2: (N,) R^2 of variance AR(1) fit
            converged: (N,) boolean array (always True for method-of-moments)
            n_observations: (N,) number of valid returns per asset
    """
    returns_jax = jnp.asarray(returns, dtype=jnp.float32)
    mask_jax = jnp.asarray(mask, dtype=jnp.float32)

    # Compute squared returns (variance proxy before smoothing)
    sq_returns = returns_jax ** 2
    sq_returns_annualized = sq_returns / dt

    # Apply rolling mean smoothing
    if smooth_window > 1:
        V_all, V_mask = _rolling_mean_masked(sq_returns_annualized, mask_jax, smooth_window)
        # Align returns with smoothed variance: r_for_corr = returns[window-1:]
        r_for_corr = returns_jax[:, smooth_window - 1:]
        r_mask = mask_jax[:, smooth_window - 1:]
        # Trim to match V_all length
        T_v = V_all.shape[1]
        r_for_corr = r_for_corr[:, :T_v]
        r_mask = r_mask[:, :T_v]
        effective_dt = dt * smooth_window
    else:
        V_all = sq_returns_annualized
        V_mask = mask_jax
        r_for_corr = returns_jax
        r_mask = mask_jax
        effective_dt = dt

    # Ensure positive variance
    V_all = jnp.maximum(V_all, 1e-10)

    # Call core estimation
    result = _fit_heston_qmle(
        returns_jax, mask_jax,  # Full returns for mu
        r_for_corr, r_mask,     # Aligned returns for rho
        dt, effective_dt,        # Original and effective dt
        V_all, V_mask           # Variance proxy
    )

    # Convert to numpy
    return {k: np.asarray(v, dtype=np.float64 if v.dtype != bool else bool) for k, v in result.items()}


def fit_batch_ohlc(open_, high, low, close, mask, dt, smooth_window=10):
    """
    Calibrate Heston QMLE parameters using Garman-Klass OHLC variance proxy.

    Args:
        open_: (N, T) array of daily open prices
        high: (N, T) array of daily high prices
        low: (N, T) array of daily low prices
        close: (N, T) array of daily close prices
        mask: (N, T) binary mask (1.0 = valid, 0.0 = padding)
        dt: Time increment in years (e.g., 1/252 for daily)
        smooth_window: Rolling window for variance proxy smoothing (default 10)

    Returns:
        Dictionary with Heston parameters (same keys as fit_batch)
    """
    open_jax = jnp.asarray(open_, dtype=jnp.float32)
    high_jax = jnp.asarray(high, dtype=jnp.float32)
    low_jax = jnp.asarray(low, dtype=jnp.float32)
    close_jax = jnp.asarray(close, dtype=jnp.float32)
    mask_jax = jnp.asarray(mask, dtype=jnp.float32)

    # Compute Garman-Klass variance
    gk_var = _garman_klass_variance(open_jax, high_jax, low_jax, close_jax, dt)

    # Compute returns from close prices
    log_prices = jnp.log(jnp.maximum(close_jax, 1e-10))
    returns = jnp.diff(log_prices, axis=-1)
    returns_mask = mask_jax[:, 1:] * mask_jax[:, :-1]  # Both t and t-1 must be valid

    # Apply rolling mean smoothing
    if smooth_window > 1:
        V_all, V_mask = _rolling_mean_masked(gk_var, mask_jax, smooth_window)
        # Align returns with smoothed variance: r_for_corr = returns[window-1:]
        r_for_corr = returns[:, smooth_window - 1:]
        r_mask = returns_mask[:, smooth_window - 1:]
        # Trim to match V_all length
        T_v = V_all.shape[1]
        r_for_corr = r_for_corr[:, :T_v]
        r_mask = r_mask[:, :T_v]
        effective_dt = dt * smooth_window
    else:
        V_all = gk_var
        V_mask = mask_jax
        r_for_corr = returns
        r_mask = returns_mask
        effective_dt = dt

    # Ensure positive variance
    V_all = jnp.maximum(V_all, 1e-10)

    # Call core estimation
    result = _fit_heston_qmle(
        returns, returns_mask,  # Full returns for mu
        r_for_corr, r_mask,     # Aligned returns for rho
        dt, effective_dt,        # Original and effective dt
        V_all, V_mask           # Variance proxy
    )

    # Convert to numpy
    return {k: np.asarray(v, dtype=np.float64 if v.dtype != bool else bool) for k, v in result.items()}
