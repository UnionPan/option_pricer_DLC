"""
Geometric Brownian Motion (GBM) Calibration

Calibrate GBM parameters from historical price data:
    dS_t = μ * S_t * dt + σ * S_t * dW_t

This is the foundation of Black-Scholes and most equity modeling.

Key outputs:
- μ (drift): Historical return (physical measure)
- σ (volatility): Historical volatility

Usage for simulation:
>>> calibrator = GBMCalibrator()
>>> result = calibrator.fit(prices)
>>> # Create GBM process with calibrated parameters
>>> from processes import GBM
>>> process = GBM(mu=result.mu, sigma=result.sigma)

author: Yunian Pan
email: yp1170@nyu.edu
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Optional, Dict
import warnings


@dataclass
class GBMCalibrationResult:
    """
    Result of GBM calibration from historical data.

    Parameters are in PHYSICAL measure (P-measure).
    For RISK-NEUTRAL pricing (Q-measure), replace μ with risk-free rate r.
    """

    # Calibrated parameters
    mu: float              # Drift (annualized expected return)
    sigma: float           # Volatility (annualized standard deviation)

    # Standard errors (statistical uncertainty)
    mu_stderr: float
    sigma_stderr: float

    # Goodness of fit
    log_likelihood: float
    aic: float             # Akaike Information Criterion
    bic: float             # Bayesian Information Criterion

    # Diagnostics
    n_observations: int
    time_horizon: float    # In years
    jarque_bera_pvalue: float  # Test for normality

    # Raw data statistics
    mean_return: float     # Arithmetic mean of returns
    median_return: float
    skewness: float
    kurtosis: float        # Excess kurtosis

    # Confidence intervals (95%)
    mu_ci: tuple
    sigma_ci: tuple

    @property
    def is_normal(self) -> bool:
        """Returns True if returns pass normality test (JB test, α=0.05)."""
        return self.jarque_bera_pvalue > 0.05

    @property
    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Sharpe ratio = (μ - r_f) / σ

        Note: This is ex-post (historical) Sharpe ratio.
        """
        return (self.mu - risk_free_rate) / self.sigma if self.sigma > 0 else 0.0

    def summary(self) -> str:
        """Pretty-printed summary of calibration results."""
        lines = [
            "=" * 60,
            "GBM Calibration Result",
            "=" * 60,
            "",
            "Model: dS = μ*S*dt + σ*S*dW",
            "",
            "Calibrated Parameters (Physical Measure):",
            f"  μ (drift)      = {self.mu:8.4f} ± {self.mu_stderr:.4f}  ({self.mu*100:.2f}% annualized)",
            f"  σ (volatility) = {self.sigma:8.4f} ± {self.sigma_stderr:.4f}  ({self.sigma*100:.2f}% annualized)",
            "",
            f"  95% CI for μ:  [{self.mu_ci[0]:.4f}, {self.mu_ci[1]:.4f}]",
            f"  95% CI for σ:  [{self.sigma_ci[0]:.4f}, {self.sigma_ci[1]:.4f}]",
            "",
            "Model Fit:",
            f"  Log-Likelihood = {self.log_likelihood:.2f}",
            f"  AIC            = {self.aic:.2f}",
            f"  BIC            = {self.bic:.2f}",
            "",
            "Diagnostics:",
            f"  Sample size    = {self.n_observations}",
            f"  Time horizon   = {self.time_horizon:.2f} years",
            f"  JB test p-val  = {self.jarque_bera_pvalue:.4f}  {'✓ Normal' if self.is_normal else '✗ Non-normal'}",
            f"  Skewness       = {self.skewness:.4f}  (0 for GBM)",
            f"  Excess Kurt    = {self.kurtosis:.4f}  (0 for GBM)",
            "",
            "⚠ Note: These are PHYSICAL measure parameters (historical).",
            "   For pricing, use RISK-NEUTRAL measure (replace μ with r).",
            "=" * 60,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"GBMCalibrationResult(μ={self.mu:.4f}, σ={self.sigma:.4f}, "
            f"n={self.n_observations})"
        )


class GBMCalibrator:
    """
    Calibrate GBM from historical prices using Maximum Likelihood Estimation.

    GBM Model:
        dS_t = μ * S_t * dt + σ * S_t * dW_t

    Equivalent to:
        log(S_t / S_0) ~ N((μ - σ²/2)*t, σ²*t)

    Log-returns:
        r_t = log(S_t / S_{t-1}) ~ N((μ - σ²/2)*Δt, σ²*Δt)

    MLE estimates (closed-form):
        σ̂² = (1/(n*Δt)) * Σ(r_t - r̄)²
        μ̂  = r̄/Δt + σ̂²/2

    Example:
        >>> calibrator = GBMCalibrator()
        >>> prices = np.array([100, 102, 101, 103, 105])
        >>> result = calibrator.fit(prices, dt=1/252)  # Daily prices
        >>> print(result.summary())
        >>> # Use for simulation
        >>> from processes import GBM
        >>> gbm = GBM(mu=result.mu, sigma=result.sigma)
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
    ):
        """
        Initialize GBM calibrator.

        Args:
            confidence_level: Confidence level for parameter intervals (default 95%)
        """
        self.confidence_level = confidence_level

    def fit(
        self,
        prices: np.ndarray,
        dt: float = 1.0 / 252.0,
        periods_per_year: int = 252,
    ) -> GBMCalibrationResult:
        """
        Calibrate GBM parameters via MLE.

        Args:
            prices: Price time series (shape: [n_observations])
            dt: Time increment between observations (in years)
                Default: 1/252 (daily data)
            periods_per_year: Number of periods per year for annualization
                             (252 for daily, 12 for monthly, 4 for quarterly)

        Returns:
            GBMCalibrationResult with fitted parameters and diagnostics

        Raises:
            ValueError: If insufficient data or invalid prices
        """
        prices = np.asarray(prices).flatten()

        # Validation
        if len(prices) < 2:
            raise ValueError("Need at least 2 price observations")

        if np.any(prices <= 0):
            raise ValueError("All prices must be positive (GBM assumption)")

        # Compute log returns
        log_prices = np.log(prices)
        log_returns = np.diff(log_prices)
        n = len(log_returns)

        # MLE estimates
        r_mean = np.mean(log_returns)
        r_var = np.var(log_returns, ddof=1)  # Unbiased estimator

        # Annualized parameters
        sigma_squared = r_var / dt
        sigma = np.sqrt(sigma_squared)
        mu = r_mean / dt + 0.5 * sigma_squared

        # Standard errors (from Fisher information matrix)
        # For GBM, the MLE has known asymptotic distribution
        mu_stderr = sigma / np.sqrt(n)
        sigma_stderr = sigma / np.sqrt(2 * n)

        # Confidence intervals (assuming normal distribution of MLE)
        z_score = stats.norm.ppf(0.5 + self.confidence_level / 2)
        mu_ci = (
            mu - z_score * mu_stderr,
            mu + z_score * mu_stderr,
        )
        sigma_ci = (
            max(0, sigma - z_score * sigma_stderr),  # Volatility must be positive
            sigma + z_score * sigma_stderr,
        )

        # Log-likelihood
        # L = -0.5 * n * [log(2π) + log(σ²*Δt)] - (1/(2σ²*Δt)) * Σ(r_t - (μ-σ²/2)*Δt)²
        log_likelihood = -0.5 * n * (np.log(2 * np.pi) + np.log(r_var))
        log_likelihood -= 0.5 * n  # From sum of squared deviations

        # Information criteria
        k = 2  # Number of parameters (μ, σ)
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood

        # Diagnostics
        jb_stat, jb_pvalue = stats.jarque_bera(log_returns)
        skewness = stats.skew(log_returns)
        kurtosis = stats.kurtosis(log_returns)  # Excess kurtosis

        # Check for violations of GBM assumptions
        if not (jb_pvalue > 0.01):
            warnings.warn(
                f"Returns significantly non-normal (JB test p={jb_pvalue:.4f}). "
                "Consider jump-diffusion or stochastic volatility models."
            )

        if abs(skewness) > 0.5:
            warnings.warn(
                f"Returns have significant skewness ({skewness:.2f}). "
                "GBM assumes zero skewness."
            )

        if kurtosis > 2.0:
            warnings.warn(
                f"Returns have fat tails (excess kurtosis = {kurtosis:.2f}). "
                "Consider jump-diffusion models."
            )

        # Annualized statistics
        mean_return = np.mean(log_returns) * periods_per_year
        median_return = np.median(log_returns) * periods_per_year
        time_horizon = n * dt

        return GBMCalibrationResult(
            mu=mu,
            sigma=sigma,
            mu_stderr=mu_stderr,
            sigma_stderr=sigma_stderr,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            n_observations=n,
            time_horizon=time_horizon,
            jarque_bera_pvalue=jb_pvalue,
            mean_return=mean_return,
            median_return=median_return,
            skewness=skewness,
            kurtosis=kurtosis,
            mu_ci=mu_ci,
            sigma_ci=sigma_ci,
        )

    def fit_with_dividends(
        self,
        prices: np.ndarray,
        dividends: np.ndarray,
        dt: float = 1.0 / 252.0,
    ) -> GBMCalibrationResult:
        """
        Calibrate GBM with discrete dividend adjustments.

        Dividends reduce the stock price, so we need to adjust returns:
            r_t = log((S_t + D_t) / S_{t-1})

        Args:
            prices: Price time series
            dividends: Dividend amounts (0 for most days)
            dt: Time increment

        Returns:
            GBMCalibrationResult

        Note:
            For continuous dividend yield q, use GBM with modified drift:
            dS = (μ - q)*S*dt + σ*S*dW
        """
        prices = np.asarray(prices).flatten()
        dividends = np.asarray(dividends).flatten()

        if len(prices) != len(dividends):
            raise ValueError("Prices and dividends must have same length")

        # Adjust prices for dividends
        adjusted_prices = prices.copy()
        adjusted_prices[1:] += dividends[1:]

        # Fit to adjusted prices
        return self.fit(adjusted_prices, dt=dt)

    def rolling_calibration(
        self,
        prices: np.ndarray,
        window: int = 252,
        dt: float = 1.0 / 252.0,
    ) -> Dict[str, np.ndarray]:
        """
        Perform rolling window calibration.

        Useful for:
        - Tracking parameter stability over time
        - Detecting regime changes
        - Out-of-sample validation

        Args:
            prices: Price time series
            window: Rolling window size (default 252 = 1 year of daily data)
            dt: Time increment

        Returns:
            Dictionary with arrays of:
            - 'mu': Rolling drift estimates
            - 'sigma': Rolling volatility estimates
            - 'dates': Window end indices
        """
        prices = np.asarray(prices)
        n = len(prices)

        if n < window + 1:
            raise ValueError(f"Need at least {window+1} observations for rolling window")

        mu_series = []
        sigma_series = []
        dates = []

        for i in range(window, n):
            window_prices = prices[i - window : i + 1]
            result = self.fit(window_prices, dt=dt)
            mu_series.append(result.mu)
            sigma_series.append(result.sigma)
            dates.append(i)

        return {
            'mu': np.array(mu_series),
            'sigma': np.array(sigma_series),
            'dates': np.array(dates),
        }
