"""
Ornstein-Uhlenbeck (OU) Process Calibration

Calibrate mean-reverting processes from historical data:
    dX_t = κ(θ - X_t)dt + σ dW_t

Where:
- κ (kappa): Mean reversion speed
- θ (theta): Long-term mean
- σ (sigma): Volatility

Used for:
- Interest rates (Vasicek model)
- Commodity prices
- Pairs trading spreads
- Volatility indices (VIX)

Key insight: OU discretization is AR(1) model!

author: Yunian Pan
email: yp1170@nyu.edu
"""

import numpy as np
from scipy import stats, optimize
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import warnings


@dataclass
class OUCalibrationResult:
    """Result of OU process calibration."""

    # Calibrated parameters
    kappa: float      # Mean reversion speed (1/time)
    theta: float      # Long-term mean
    sigma: float      # Volatility

    # Standard errors
    kappa_stderr: float
    theta_stderr: float
    sigma_stderr: float

    # Goodness of fit
    log_likelihood: float
    r_squared: float
    rmse: float

    # Diagnostics
    n_observations: int
    half_life: float           # Time for 50% mean reversion
    stationarity_test_pvalue: float  # ADF test

    # Confidence intervals
    kappa_ci: tuple
    theta_ci: tuple
    sigma_ci: tuple

    @property
    def is_mean_reverting(self) -> bool:
        """True if process exhibits mean reversion (κ > 0)."""
        return self.kappa > 0

    @property
    def is_stationary(self) -> bool:
        """True if process is stationary (passes ADF test)."""
        return self.stationarity_test_pvalue < 0.05

    @property
    def long_run_variance(self) -> float:
        """Stationary variance: Var[X_∞] = σ²/(2κ)."""
        if self.kappa > 0:
            return self.sigma**2 / (2 * self.kappa)
        return np.inf

    def summary(self) -> str:
        """Pretty-printed summary."""
        lines = [
            "=" * 60,
            "Ornstein-Uhlenbeck (OU) Calibration Result",
            "=" * 60,
            "",
            "Model: dX = κ(θ - X)dt + σ dW",
            "",
            "Calibrated Parameters:",
            f"  κ (kappa)  = {self.kappa:8.4f} ± {self.kappa_stderr:.4f}  (mean reversion speed)",
            f"  θ (theta)  = {self.theta:8.4f} ± {self.theta_stderr:.4f}  (long-term mean)",
            f"  σ (sigma)  = {self.sigma:8.4f} ± {self.sigma_stderr:.4f}  (volatility)",
            "",
            f"  Half-life  = {self.half_life:.2f} time units",
            f"  Long-run σ = {np.sqrt(self.long_run_variance):.4f}",
            "",
            "Model Fit:",
            f"  Log-Likelihood = {self.log_likelihood:.2f}",
            f"  R²             = {self.r_squared:.4f}",
            f"  RMSE           = {self.rmse:.6f}",
            "",
            "Diagnostics:",
            f"  Mean-reverting? {('Yes (κ > 0)' if self.is_mean_reverting else 'No (κ ≤ 0)'):<20}",
            f"  Stationary?     {('Yes (ADF p < 0.05)' if self.is_stationary else 'No (ADF p ≥ 0.05)'):<20}",
            f"  Sample size     = {self.n_observations}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"OUCalibrationResult(κ={self.kappa:.4f}, θ={self.theta:.4f}, "
            f"σ={self.sigma:.4f}, half_life={self.half_life:.2f})"
        )


class OUCalibrator:
    """
    Calibrate Ornstein-Uhlenbeck process from time series.

    Methods:
    1. Discretization method (fast, approximate)
    2. Exact MLE (slower, more accurate)

    The discretization method exploits that OU → AR(1):

    Exact discretization:
        X_{t+Δt} = θ + (X_t - θ)*exp(-κΔt) + noise

    This is equivalent to:
        X_{t+1} = a + b*X_t + ε

    where:
        b = exp(-κΔt)  → κ = -log(b)/Δt
        a = θ*(1 - b)  → θ = a/(1 - b)
        Var(ε) = σ²*(1 - exp(-2κΔt))/(2κ) → σ

    Example:
        >>> calibrator = OUCalibrator(method='discretization')
        >>> time_series = np.random.randn(1000)  # Example data
        >>> result = calibrator.fit(time_series, dt=1.0)
        >>> print(result.summary())
        >>> # Use for simulation
        >>> from processes import OrnsteinUhlenbeck
        >>> ou = OrnsteinUhlenbeck(
        ...     kappa=result.kappa,
        ...     theta=result.theta,
        ...     sigma=result.sigma
        ... )
    """

    def __init__(
        self,
        method: str = 'discretization',
        confidence_level: float = 0.95,
    ):
        """
        Initialize OU calibrator.

        Args:
            method: 'discretization' (fast) or 'exact_mle' (accurate)
            confidence_level: Confidence level for intervals
        """
        self.method = method.lower()
        self.confidence_level = confidence_level

        if self.method not in ['discretization', 'exact_mle']:
            raise ValueError("Method must be 'discretization' or 'exact_mle'")

    def fit(
        self,
        series: np.ndarray,
        dt: float = 1.0,
    ) -> OUCalibrationResult:
        """
        Calibrate OU parameters from time series.

        Args:
            series: Time series data (e.g., interest rates, spreads)
            dt: Time increment between observations (in same units as κ)

        Returns:
            OUCalibrationResult with fitted parameters

        Raises:
            ValueError: If insufficient data
        """
        series = np.asarray(series).flatten()

        if len(series) < 10:
            raise ValueError("Need at least 10 observations for OU calibration")

        if self.method == 'discretization':
            return self._fit_discretization(series, dt)
        elif self.method == 'exact_mle':
            return self._fit_exact_mle(series, dt)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _fit_discretization(
        self,
        series: np.ndarray,
        dt: float,
    ) -> OUCalibrationResult:
        """
        Fit OU via AR(1) regression (discretization method).

        Fast and works well for small dt.
        """
        n = len(series)

        # AR(1) regression: X_t = a + b*X_{t-1} + ε
        X_t = series[1:]
        X_lag = series[:-1]

        # Design matrix [1, X_{t-1}]
        X_design = np.column_stack([np.ones(len(X_lag)), X_lag])

        # OLS: β = (X'X)^{-1}(X'y)
        beta = np.linalg.lstsq(X_design, X_t, rcond=None)[0]
        a, b = beta

        # Residuals
        predictions = X_design @ beta
        residuals = X_t - predictions
        residual_var = np.var(residuals, ddof=2)  # ddof=2 (2 parameters)

        # Convert to OU parameters
        if b >= 1.0:
            warnings.warn(
                f"No mean reversion detected (AR coeff b={b:.4f} >= 1). "
                "Series may be non-stationary."
            )
            kappa = 0.0
            theta = np.mean(series)
        else:
            kappa = -np.log(max(b, 1e-10)) / dt  # Avoid log(0)
            theta = a / (1 - b) if abs(1 - b) > 1e-10 else np.mean(series)

        # Volatility from residual variance
        # Var(ε) = σ²*(1 - exp(-2κΔt))/(2κ)
        if kappa > 1e-10:
            numerator = 2 * kappa * residual_var
            denominator = (1 - np.exp(-2 * kappa * dt))
            sigma = np.sqrt(numerator / denominator) if denominator > 0 else np.sqrt(residual_var)
        else:
            sigma = np.sqrt(residual_var)

        # Standard errors (from OLS)
        # Covariance matrix: σ²_ε * (X'X)^{-1}
        XtX_inv = np.linalg.inv(X_design.T @ X_design)
        beta_cov = residual_var * XtX_inv
        a_stderr = np.sqrt(beta_cov[0, 0])
        b_stderr = np.sqrt(beta_cov[1, 1])

        # Delta method for parameter standard errors
        # This is approximate - for exact SE, use numerical Hessian
        if b < 1.0 and b > 0:
            kappa_stderr = b_stderr / (b * dt)  # d(kappa)/db = -1/(b*dt)
            theta_stderr = a_stderr / (1 - b)   # Approximate
        else:
            kappa_stderr = np.nan
            theta_stderr = np.std(series) / np.sqrt(n)

        sigma_stderr = sigma / np.sqrt(2 * (n - 2))  # Approximate

        # Confidence intervals
        z = stats.norm.ppf(0.5 + self.confidence_level / 2)
        kappa_ci = (
            max(0, kappa - z * kappa_stderr),
            kappa + z * kappa_stderr,
        )
        theta_ci = (
            theta - z * theta_stderr,
            theta + z * theta_stderr,
        )
        sigma_ci = (
            max(0, sigma - z * sigma_stderr),
            sigma + z * sigma_stderr,
        )

        # Goodness of fit
        r_squared = 1 - (np.sum(residuals**2) / np.sum((X_t - np.mean(X_t))**2))
        rmse = np.sqrt(np.mean(residuals**2))

        # Log-likelihood (Gaussian AR(1))
        log_likelihood = -0.5 * (n - 1) * (np.log(2 * np.pi) + np.log(residual_var))
        log_likelihood -= 0.5 * np.sum(residuals**2) / residual_var

        # Half-life
        if kappa > 0:
            half_life = np.log(2) / kappa
        else:
            half_life = np.inf

        # Stationarity test (simplified ADF)
        stationarity_pvalue = self._adf_test_simple(series)

        return OUCalibrationResult(
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            kappa_stderr=kappa_stderr,
            theta_stderr=theta_stderr,
            sigma_stderr=sigma_stderr,
            log_likelihood=log_likelihood,
            r_squared=r_squared,
            rmse=rmse,
            n_observations=n - 1,
            half_life=half_life,
            stationarity_test_pvalue=stationarity_pvalue,
            kappa_ci=kappa_ci,
            theta_ci=theta_ci,
            sigma_ci=sigma_ci,
        )

    def _fit_exact_mle(
        self,
        series: np.ndarray,
        dt: float,
    ) -> OUCalibrationResult:
        """
        Fit OU via exact Maximum Likelihood.

        Uses the exact transition density of OU process:
            X_{t+Δt} | X_t ~ N(μ_t, v_t)
        where:
            μ_t = θ + (X_t - θ)*exp(-κΔt)
            v_t = σ²*(1 - exp(-2κΔt))/(2κ)

        More accurate than discretization, especially for large dt.
        """
        n = len(series)

        # Negative log-likelihood function
        def neg_log_likelihood(params):
            kappa, theta, sigma = params

            if kappa <= 0 or sigma <= 0:
                return 1e10  # Invalid parameters

            # Compute conditional means and variances
            exp_term = np.exp(-kappa * dt)
            mu = theta + (series[:-1] - theta) * exp_term
            var = (sigma**2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * dt))

            if var <= 0:
                return 1e10

            # Log-likelihood
            ll = -0.5 * np.sum(
                np.log(2 * np.pi * var) + (series[1:] - mu)**2 / var
            )

            return -ll

        # Initial guess from discretization method
        quick_result = self._fit_discretization(series, dt)
        x0 = [quick_result.kappa, quick_result.theta, quick_result.sigma]

        # Bounds
        bounds = [
            (1e-6, None),  # kappa > 0
            (None, None),  # theta unconstrained
            (1e-6, None),  # sigma > 0
        ]

        # Optimize
        result = optimize.minimize(
            neg_log_likelihood,
            x0=x0,
            method='L-BFGS-B',
            bounds=bounds,
        )

        if not result.success:
            warnings.warn(f"MLE optimization did not converge: {result.message}")

        kappa, theta, sigma = result.x

        # Compute standard errors from Hessian (numerical)
        # For production, use more robust Hessian computation
        try:
            # Finite difference Hessian
            hess = optimize.approx_fprime(result.x, neg_log_likelihood, epsilon=1e-5)
            stderr = np.sqrt(np.abs(hess))
        except:
            # Fallback to discretization estimates
            stderr = [quick_result.kappa_stderr, quick_result.theta_stderr, quick_result.sigma_stderr]

        kappa_stderr, theta_stderr, sigma_stderr = stderr

        # Confidence intervals
        z = stats.norm.ppf(0.5 + self.confidence_level / 2)
        kappa_ci = (max(0, kappa - z * kappa_stderr), kappa + z * kappa_stderr)
        theta_ci = (theta - z * theta_stderr, theta + z * theta_stderr)
        sigma_ci = (max(0, sigma - z * sigma_stderr), sigma + z * sigma_stderr)

        # Goodness of fit
        exp_term = np.exp(-kappa * dt)
        mu = theta + (series[:-1] - theta) * exp_term
        predictions = mu
        residuals = series[1:] - predictions
        rmse = np.sqrt(np.mean(residuals**2))
        r_squared = 1 - (np.sum(residuals**2) / np.sum((series[1:] - np.mean(series[1:]))**2))

        log_likelihood = -result.fun

        # Half-life
        half_life = np.log(2) / kappa if kappa > 0 else np.inf

        # Stationarity test
        stationarity_pvalue = self._adf_test_simple(series)

        return OUCalibrationResult(
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            kappa_stderr=kappa_stderr,
            theta_stderr=theta_stderr,
            sigma_stderr=sigma_stderr,
            log_likelihood=log_likelihood,
            r_squared=r_squared,
            rmse=rmse,
            n_observations=n - 1,
            half_life=half_life,
            stationarity_test_pvalue=stationarity_pvalue,
            kappa_ci=kappa_ci,
            theta_ci=theta_ci,
            sigma_ci=sigma_ci,
        )

    def _adf_test_simple(self, series: np.ndarray) -> float:
        """
        Simplified Augmented Dickey-Fuller test.

        For production, use statsmodels.tsa.stattools.adfuller
        """
        n = len(series)
        diff = np.diff(series)
        lagged = series[:-1]

        # Regression: Δy_t = α + β*y_{t-1} + ε
        X = np.column_stack([np.ones(len(lagged)), lagged])
        beta = np.linalg.lstsq(X, diff, rcond=None)[0]
        residuals = diff - X @ beta
        se = np.sqrt(np.sum(residuals**2) / (len(diff) - 2))

        # Test statistic
        t_stat = beta[1] / (se / np.sqrt(np.sum((lagged - lagged.mean())**2)))

        # Approximate p-value (very rough!)
        if t_stat < -3.5:
            return 0.001
        elif t_stat < -2.9:
            return 0.04
        elif t_stat < -2.6:
            return 0.09
        else:
            return 0.20
