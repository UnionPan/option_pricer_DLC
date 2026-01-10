"""
Data Preprocessing and Quality Checks for Historical Calibration

Utilities for cleaning time series data before calibration:
- Return computation (log vs arithmetic)
- Outlier detection
- Stationarity testing
- Mean reversion testing

author: Yunian Pan
email: yp1170@nyu.edu
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from scipy import stats
import warnings


def compute_returns(
    prices: np.ndarray,
    method: str = 'log',
    remove_nan: bool = True,
) -> np.ndarray:
    """
    Compute returns from price series.

    Args:
        prices: Price time series
        method: 'log' for log returns, 'simple' for arithmetic returns
        remove_nan: Remove NaN values from result

    Returns:
        Return series (length = len(prices) - 1)

    Notes:
        - Log returns: r_t = log(S_t / S_{t-1})
          Advantages: Time-additive, symmetric, better for long horizons
        - Simple returns: r_t = (S_t - S_{t-1}) / S_{t-1}
          Advantages: Portfolio aggregation, intuitive

        For daily data, log ≈ simple returns when |r| < 5%
    """
    prices = np.asarray(prices).flatten()

    if len(prices) < 2:
        raise ValueError("Need at least 2 prices to compute returns")

    if method == 'log':
        returns = np.diff(np.log(prices))
    elif method == 'simple':
        returns = np.diff(prices) / prices[:-1]
    else:
        raise ValueError(f"Unknown method: {method}. Use 'log' or 'simple'")

    if remove_nan:
        returns = returns[~np.isnan(returns)]

    return returns


def detect_outliers(
    returns: np.ndarray,
    method: str = 'iqr',
    threshold: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect outliers in return series.

    Args:
        returns: Return series
        method: Detection method ('iqr', 'zscore', 'mad')
        threshold: Threshold for outlier detection
                  - IQR: multiple of IQR (default 3.0)
                  - Z-score: number of std devs (default 3.0)
                  - MAD: multiple of MAD (default 3.0)

    Returns:
        Tuple of (outlier_mask, cleaned_returns)
        outlier_mask: Boolean array (True = outlier)
        cleaned_returns: Returns with outliers removed

    Notes:
        Outliers can be:
        - Flash crashes / fat finger trades
        - Data errors (bad ticks)
        - True extreme events (2008 crisis, COVID crash)

        Be careful: Removing true tail events biases volatility estimates!
    """
    returns = np.asarray(returns)

    if method == 'iqr':
        # Interquartile Range method (robust to outliers)
        q1, q3 = np.percentile(returns, [25, 75])
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        outliers = (returns < lower) | (returns > upper)

    elif method == 'zscore':
        # Z-score method (assumes normality)
        z_scores = np.abs(stats.zscore(returns))
        outliers = z_scores > threshold

    elif method == 'mad':
        # Median Absolute Deviation (very robust)
        median = np.median(returns)
        mad = np.median(np.abs(returns - median))
        # Scale factor to match std dev under normality
        mad_scaled = 1.4826 * mad
        z_scores = np.abs(returns - median) / mad_scaled
        outliers = z_scores > threshold

    else:
        raise ValueError(f"Unknown method: {method}")

    cleaned_returns = returns[~outliers]

    if np.sum(outliers) > 0:
        pct_outliers = 100.0 * np.sum(outliers) / len(returns)
        warnings.warn(
            f"Detected {np.sum(outliers)} outliers ({pct_outliers:.1f}%) "
            f"using {method} method with threshold={threshold}"
        )

    return outliers, cleaned_returns


def test_stationarity(
    series: np.ndarray,
    method: str = 'adf',
    alpha: float = 0.05,
) -> Dict:
    """
    Test for stationarity (constant mean and variance).

    Args:
        series: Time series to test
        method: 'adf' (Augmented Dickey-Fuller)
        alpha: Significance level

    Returns:
        Dictionary with test results

    Notes:
        Stationarity is required for many calibration methods:
        - Non-stationary: Stock prices (random walk)
        - Stationary: Stock returns, mean-reverting spreads

        ADF test:
        H0: Series has unit root (non-stationary)
        H1: Series is stationary

        If p-value < alpha: Reject H0 → stationary
    """
    series = np.asarray(series)

    if method != 'adf':
        raise ValueError("Only 'adf' method currently supported")

    # Simple ADF test implementation
    # For production, use statsmodels.tsa.stattools.adfuller

    # Compute first differences
    diff = np.diff(series)
    lagged = series[:-1]

    # Run regression: Δy_t = α + β*y_{t-1} + ε_t
    # If β < 0 and significant → mean-reverting (stationary)
    X = np.column_stack([np.ones(len(lagged)), lagged])
    y = diff

    # OLS
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ beta
    se = np.sqrt(np.sum(residuals**2) / (len(y) - 2))

    # Test statistic
    t_stat = beta[1] / (se / np.sqrt(np.sum((lagged - lagged.mean())**2)))

    # Critical values (approximate, -3 regime)
    # For exact values, use statsmodels
    critical_values = {
        0.01: -3.43,
        0.05: -2.86,
        0.10: -2.57,
    }

    # P-value approximation (very rough!)
    # For production, use statsmodels
    if t_stat < -3.5:
        p_value = 0.001
    elif t_stat < -2.9:
        p_value = 0.04
    elif t_stat < -2.6:
        p_value = 0.09
    else:
        p_value = 0.20

    is_stationary = p_value < alpha

    return {
        'test_statistic': t_stat,
        'p_value': p_value,
        'critical_values': critical_values,
        'is_stationary': is_stationary,
        'conclusion': 'Stationary' if is_stationary else 'Non-stationary (unit root)',
    }


def estimate_half_life(
    series: np.ndarray,
    method: str = 'ar1',
) -> float:
    """
    Estimate half-life of mean reversion.

    Args:
        series: Time series (should be mean-reverting)
        method: 'ar1' (AR(1) regression method)

    Returns:
        Half-life in same units as series (e.g., days for daily data)

    Notes:
        Half-life = time for deviation to decay by 50%

        For OU process: dX = κ(θ - X)dt + σ dW
        Half-life = log(2) / κ

        Estimate κ from AR(1): X_t = a + b*X_{t-1} + ε
        Then: κ = -log(b) / Δt

        Interpretation:
        - Half-life = 5 days: Fast mean reversion (high-freq pairs trading)
        - Half-life = 100 days: Slow mean reversion (interest rates)
        - Half-life = ∞: No mean reversion (random walk)
    """
    series = np.asarray(series)

    if method != 'ar1':
        raise ValueError("Only 'ar1' method currently supported")

    if len(series) < 10:
        raise ValueError("Need at least 10 observations for half-life estimation")

    # AR(1) regression: X_t = a + b*X_{t-1} + ε
    X_t = series[1:]
    X_lag = series[:-1]

    # Add intercept
    X_design = np.column_stack([np.ones(len(X_lag)), X_lag])

    # OLS
    beta = np.linalg.lstsq(X_design, X_t, rcond=None)[0]
    b = beta[1]

    # Check for mean reversion
    if b >= 1.0:
        warnings.warn(
            f"No mean reversion detected (AR(1) coeff = {b:.4f} >= 1). "
            "Series may be random walk or explosive."
        )
        return np.inf

    if b <= 0:
        warnings.warn(
            f"Negative autocorrelation detected (AR(1) coeff = {b:.4f}). "
            "Half-life estimation may be unreliable."
        )
        return np.nan

    # Convert to mean reversion speed
    # Assuming Δt = 1 (daily data)
    kappa = -np.log(b)

    # Half-life
    half_life = np.log(2) / kappa

    return half_life


def clean_price_series(
    prices: np.ndarray,
    remove_zeros: bool = True,
    remove_duplicates: bool = True,
    interpolate_gaps: bool = False,
) -> np.ndarray:
    """
    Clean price series for calibration.

    Args:
        prices: Raw price series
        remove_zeros: Remove zero prices (data errors)
        remove_duplicates: Remove consecutive duplicate prices (stale quotes)
        interpolate_gaps: Interpolate missing values (use with caution!)

    Returns:
        Cleaned price series

    Notes:
        Common data quality issues:
        - Zeros: Exchange outages, data errors
        - Duplicates: Stale quotes, low liquidity
        - Gaps: Missing data, holidays
        - Spikes: Fat finger trades, flash crashes
    """
    prices = np.asarray(prices).copy()

    # Remove zeros
    if remove_zeros:
        valid = prices > 0
        prices = prices[valid]

    # Remove consecutive duplicates
    if remove_duplicates:
        unique_idx = np.concatenate([[True], prices[1:] != prices[:-1]])
        prices = prices[unique_idx]

    # Interpolate gaps (optional, use with caution!)
    if interpolate_gaps:
        # Simple linear interpolation
        # For production, use more sophisticated methods
        nans = np.isnan(prices)
        if np.any(nans):
            x = np.arange(len(prices))
            prices[nans] = np.interp(x[nans], x[~nans], prices[~nans])

    return prices


def compute_summary_statistics(returns: np.ndarray, annualize: bool = True) -> Dict:
    """
    Compute summary statistics for returns.

    Useful for diagnostics and model validation.

    Args:
        returns: Return series
        annualize: Annualize mean and volatility (assumes daily data)

    Returns:
        Dictionary with statistics
    """
    returns = np.asarray(returns)

    factor = 252 if annualize else 1

    stats_dict = {
        # Location
        'mean': np.mean(returns) * factor,
        'median': np.median(returns) * factor,

        # Dispersion
        'std': np.std(returns, ddof=1) * np.sqrt(factor),
        'min': np.min(returns),
        'max': np.max(returns),

        # Shape
        'skewness': stats.skew(returns),
        'kurtosis': stats.kurtosis(returns),  # Excess kurtosis

        # Tests
        'jarque_bera_stat': stats.jarque_bera(returns)[0],
        'jarque_bera_pvalue': stats.jarque_bera(returns)[1],
        'is_normal': stats.jarque_bera(returns)[1] > 0.05,

        # Autocorrelation
        'autocorr_lag1': np.corrcoef(returns[:-1], returns[1:])[0, 1],

        # Sample size
        'n_observations': len(returns),
    }

    return stats_dict


def split_train_test(
    series: np.ndarray,
    train_fraction: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split time series into train/test sets.

    Args:
        series: Time series data
        train_fraction: Fraction for training (default 80%)

    Returns:
        Tuple of (train, test)

    Notes:
        Time series split is DIFFERENT from random split:
        - Must respect temporal order (no shuffling!)
        - Test set comes AFTER train set
        - Used for out-of-sample validation
    """
    n = len(series)
    split_idx = int(n * train_fraction)

    train = series[:split_idx]
    test = series[split_idx:]

    return train, test
