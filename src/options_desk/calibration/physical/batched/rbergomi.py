"""
Batched rough Bergomi variogram estimation via JAX.

Matches conventions of physical/rough_bergomi_calibrator.py (scipy reference).
Estimates Hurst parameter H and vol-of-vol eta from log-variance variogram.
"""
import jax
import jax.numpy as jnp
import numpy as np
from .common import masked_mean


def _rolling_mean_masked(x, mask, window):
    """
    Compute causal rolling mean with masking via convolution.

    Args:
        x: (N, T) input array
        mask: (N, T) binary mask
        window: int, rolling window size

    Returns:
        rolled: (N, T-window+1) rolling mean
        rolled_mask: (N, T-window+1) mask (1.0 if all entries in window valid)
    """
    N, T = x.shape

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


def _fit_rbergomi_variogram(returns, mask, dt, window, max_lag):
    """
    Core rough Bergomi variogram estimation.

    Pipeline (matching scipy reference):
    1. Rolling realized variance (window) -> variance proxy
    2. Log-variance series
    3. Variogram: E[(log v_{t+lag} - log v_t)^2] for lags 1..max_lag
    4. OLS regression: log(variogram) = intercept + slope * log(lag)
       where slope = 2*H, intercept -> eta
    5. xi0 from first realized variance

    Args:
        returns: (N, T) log-returns
        mask: (N, T) binary mask
        dt: scalar time increment
        window: rolling window for realized variance
        max_lag: maximum lag for variogram

    Returns:
        Dictionary with estimated parameters (all shape (N,))
    """
    N, T = returns.shape

    # Step 1: Rolling realized variance
    sq_returns = returns ** 2
    sq_returns_annualized = sq_returns / dt

    rv, rv_mask = _rolling_mean_masked(sq_returns_annualized, mask, window)

    # Step 2: Log-variance series (with safeguard)
    log_var = jnp.log(jnp.maximum(rv, 1e-10))

    # Step 3: Compute variogram for each lag
    # We need to handle this without dynamic slicing in vmap
    # Instead, we'll manually loop over lags and stack results

    T_log_var = log_var.shape[1]

    def compute_asset_variogram_all_lags(log_v_row, mask_row):
        """Compute variogram for all lags for a single asset."""
        variograms_asset = []
        has_data_asset = []

        for lag in range(1, max_lag + 1):
            # Extract lagged differences (static slicing)
            if lag < T_log_var:
                diff = log_v_row[lag:] - log_v_row[:-lag]
                # Mask: both t and t+lag must be valid
                mask_t = mask_row[:-lag]
                mask_tp_lag = mask_row[lag:]
                eff_mask = mask_t * mask_tp_lag

                # Compute variance with ddof=1 (Bessel correction)
                n_valid = jnp.sum(eff_mask)
                mean_diff = jnp.sum(diff * eff_mask) / jnp.maximum(n_valid, 1.0)
                sq_dev = (diff - mean_diff) ** 2
                var = jnp.sum(sq_dev * eff_mask) / jnp.maximum(n_valid - 1, 1.0)

                # Whether we have enough data
                has_data = n_valid >= 2
            else:
                var = 0.0
                has_data = False

            variograms_asset.append(var)
            has_data_asset.append(has_data)

        return jnp.array(variograms_asset), jnp.array(has_data_asset)

    # Process all assets
    variograms, has_data = jax.vmap(compute_asset_variogram_all_lags)(log_var, rv_mask)
    # Shape: (N, max_lag)

    # Step 4: OLS regression of log(variogram) on log(lag)
    # x = log(lag * dt), y = log(variogram)

    # Pre-compute x values (log of lag times)
    lags_arr = jnp.arange(1, max_lag + 1, dtype=jnp.float32)  # Shape: (max_lag,)
    lags_dt = lags_arr * dt
    x_all = jnp.log(lags_dt)  # Pre-compute log lags

    def fit_ols_single_asset(variogram_row, has_data_row):
        """Fit OLS for a single asset."""
        # Filter valid variogram points
        valid = has_data_row & (variogram_row > 0)
        n_valid = jnp.sum(valid)

        # Compute y
        x = x_all  # Use pre-computed x
        y = jnp.log(jnp.maximum(variogram_row, 1e-10))

        # Apply valid mask
        x_valid = x * valid
        y_valid = y * valid

        # OLS: y = intercept + slope * x
        # Normal equations: [sum(x^2), sum(x); sum(x), n] [slope; intercept] = [sum(x*y); sum(y)]

        # Compute sums (masked)
        sum_x = jnp.sum(x_valid)
        sum_y = jnp.sum(y_valid)
        sum_xx = jnp.sum(x_valid * x_valid)
        sum_xy = jnp.sum(x_valid * y_valid)

        # Solve 2x2 system
        # A = [[sum_xx, sum_x], [sum_x, n]]
        # b = [sum_xy, sum_y]
        denom = sum_xx * n_valid - sum_x * sum_x
        slope = jnp.where(
            (n_valid >= 2) & (jnp.abs(denom) > 1e-12),
            (n_valid * sum_xy - sum_x * sum_y) / denom,
            0.2  # Default slope corresponding to H ~ 0.1
        )
        intercept = jnp.where(
            (n_valid >= 2) & (jnp.abs(denom) > 1e-12),
            (sum_xx * sum_y - sum_x * sum_xy) / denom,
            0.0
        )

        # Extract parameters
        # H = slope / 2 (clamped to [0.01, 0.49])
        H = jnp.clip(0.5 * slope, 0.01, 0.49)

        # eta = exp(0.5 * intercept)
        eta = jnp.exp(0.5 * intercept)

        # Convergence: we have at least 2 valid lags
        converged = n_valid >= 2

        return H, eta, converged

    H, eta, converged = jax.vmap(fit_ols_single_asset)(variograms, has_data)

    # Step 5: xi0 from first realized variance (first valid entry)
    def get_first_valid_rv(rv_row, mask_row):
        """Get first valid realized variance for xi0."""
        # Find first index where mask is 1
        indices = jnp.arange(rv_row.shape[0])
        valid_mask = mask_row > 0
        valid_indices = jnp.where(valid_mask, indices, rv_row.shape[0])
        first_idx = jnp.min(valid_indices)
        # If no valid entries, return default (will be handled by converged=False)
        return jnp.where(first_idx < rv_row.shape[0], rv_row[first_idx], 0.04)

    xi0 = jax.vmap(get_first_valid_rv)(rv, rv_mask)

    # Count observations (number of valid returns)
    n_observations = jnp.sum(mask, axis=-1)

    return {
        "H": H,
        "eta": eta,
        "xi0": xi0,
        "converged": converged,
        "n_observations": n_observations,
    }


def fit_batch(returns, mask, dt, window=20, max_lag=10):
    """
    Calibrate rough Bergomi parameters for a batch of assets.

    Estimates H (Hurst) and eta (vol-of-vol) from log-variance variogram,
    matching the scipy reference (physical/rough_bergomi_calibrator.py).

    Pipeline:
    1. Rolling realized variance (window) -> variance proxy
    2. Log-variance series
    3. Variogram: Var[log v_{t+lag} - log v_t] for lags 1..max_lag
    4. OLS: log(variogram) ~ slope * log(lag) + intercept
       -> H = slope/2, eta from intercept
    5. xi0 from first realized variance

    Args:
        returns: (N, T) array of log-returns
        mask: (N, T) binary mask (1.0 = valid, 0.0 = padding)
        dt: Time increment in years (e.g., 1/252 for daily)
        window: Rolling window for realized variance (default 20)
        max_lag: Maximum lag for variogram (default 10)

    Returns:
        Dictionary with keys:
            H: (N,) Hurst parameter
            eta: (N,) volatility of volatility
            xi0: (N,) initial variance level
            converged: (N,) boolean array (True if at least 2 valid lags)
            n_observations: (N,) number of valid returns per asset
    """
    returns_jax = jnp.asarray(returns, dtype=jnp.float32)
    mask_jax = jnp.asarray(mask, dtype=jnp.float32)

    # Call core estimation
    result = _fit_rbergomi_variogram(
        returns_jax, mask_jax, dt, window, max_lag
    )

    # Convert to numpy
    return {k: np.asarray(v, dtype=np.float64 if v.dtype != bool else bool) for k, v in result.items()}
