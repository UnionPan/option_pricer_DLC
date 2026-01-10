"""
Volatility Estimation from Historical Data

Implements industry-standard volatility estimators:

Close-based:
- Simple historical volatility (close-to-close)
- EWMA (RiskMetrics)
- GARCH(1,1) (Engle-Bollerslev)

OHLC-based (more efficient):
- Parkinson (High-Low range estimator)
- Garman-Klass (OHLC estimator)
- Rogers-Satchell (drift-independent OHLC)
- Yang-Zhang (open-to-close + overnight, most efficient)

These are used for:
1. Risk management (VaR, volatility targeting)
2. Parameter initialization for calibration
3. Validation of implied volatility

References:
- Parkinson (1980): "The Extreme Value Method for Estimating the Variance of the Rate of Return"
- Garman & Klass (1980): "On the Estimation of Security Price Volatilities from Historical Data"
- Rogers & Satchell (1991): "Estimating Variance From High, Low and Closing Prices"
- Yang & Zhang (2000): "Drift-Independent Volatility Estimation Based on High, Low, Open, and Close Prices"

author: Yunian Pan
email: yp1170@nyu.edu
"""

import numpy as np
import pandas as pd
from scipy import optimize
from typing import Optional, Tuple, Dict, Union
from dataclasses import dataclass
import warnings


@dataclass
class VolatilityEstimate:
    """Container for volatility estimation results."""

    # Volatility time series (annualized)
    volatility: np.ndarray

    # Metadata
    method: str
    annualization_factor: float

    # Parameters (method-dependent)
    params: Optional[Dict] = None

    @property
    def current_vol(self) -> float:
        """Most recent volatility estimate."""
        return self.volatility[-1]

    @property
    def average_vol(self) -> float:
        """Average volatility over the period."""
        return np.mean(self.volatility)

    def __repr__(self) -> str:
        return (
            f"VolatilityEstimate(method={self.method}, "
            f"current={self.current_vol:.2%}, "
            f"average={self.average_vol:.2%})"
        )


class VolatilityEstimator:
    """
    Historical volatility estimation using various methods.

    Example:
        >>> estimator = VolatilityEstimator(method='ewma', lambda_=0.94)
        >>> returns = np.random.randn(252) * 0.01
        >>> result = estimator.estimate(returns, annualize=True)
        >>> print(f"Current vol: {result.current_vol:.2%}")
    """

    def __init__(
        self,
        method: str = 'simple',
        window: int = 252,
        lambda_: float = 0.94,
        min_periods: int = 20,
    ):
        """
        Initialize volatility estimator.

        Args:
            method: Estimation method ('simple', 'ewma', 'garch')
            window: Rolling window size for simple method (default: 252 days = 1 year)
            lambda_: Decay factor for EWMA (default: 0.94, RiskMetrics standard)
            min_periods: Minimum observations required
        """
        self.method = method.lower()
        self.window = window
        self.lambda_ = lambda_
        self.min_periods = min_periods

        if self.method not in ['simple', 'ewma', 'garch']:
            raise ValueError(f"Unknown method: {method}. Use 'simple', 'ewma', or 'garch'")

    def estimate(
        self,
        returns: np.ndarray,
        annualize: bool = True,
        periods_per_year: int = 252,
    ) -> VolatilityEstimate:
        """
        Estimate volatility from return series.

        Args:
            returns: Return series (NOT prices - use compute_returns() first)
            annualize: Whether to annualize volatility
            periods_per_year: Number of periods per year (252 for daily, 12 for monthly)

        Returns:
            VolatilityEstimate with volatility time series
        """
        returns = np.asarray(returns).flatten()

        if len(returns) < self.min_periods:
            raise ValueError(
                f"Insufficient data: {len(returns)} < {self.min_periods} required"
            )

        # Compute volatility based on method
        if self.method == 'simple':
            vol = self._simple_vol(returns)
        elif self.method == 'ewma':
            vol = self._ewma_vol(returns)
        elif self.method == 'garch':
            raise ValueError("Use GARCHEstimator class for GARCH estimation")
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Annualize if requested
        annualization_factor = np.sqrt(periods_per_year) if annualize else 1.0
        vol = vol * annualization_factor

        params = {
            'window': self.window,
            'lambda': self.lambda_,
        }

        return VolatilityEstimate(
            volatility=vol,
            method=self.method,
            annualization_factor=annualization_factor,
            params=params,
        )

    def _simple_vol(self, returns: np.ndarray) -> np.ndarray:
        """
        Simple rolling standard deviation.

        σ_t = sqrt(1/(n-1) * Σ(r_i - r_mean)²) over window [t-n+1, t]

        Pros: Easy to understand, unbiased estimator
        Cons: Equal weight to all observations, sensitive to window size
        """
        n = len(returns)
        vol = np.full(n, np.nan)

        for i in range(self.window - 1, n):
            window_returns = returns[i - self.window + 1 : i + 1]
            vol[i] = np.std(window_returns, ddof=1)

        return vol

    def _ewma_vol(self, returns: np.ndarray) -> np.ndarray:
        """
        Exponentially Weighted Moving Average (RiskMetrics).

        Recursive formula:
        σ²_t = λ * σ²_{t-1} + (1-λ) * r²_t

        where λ = decay factor (0.94 for daily data)

        Decay half-life: log(0.5) / log(λ) ≈ 11.4 days for λ=0.94

        Pros: Adaptive, no window parameter, recent data weighted more
        Cons: One parameter to tune (but 0.94 is industry standard)
        """
        n = len(returns)
        variance = np.zeros(n)

        # Initialize with simple variance over first window
        init_window = min(self.min_periods, len(returns))
        variance[0] = np.var(returns[:init_window], ddof=1)

        # Recursive update
        for t in range(1, n):
            variance[t] = self.lambda_ * variance[t-1] + (1 - self.lambda_) * returns[t]**2

        return np.sqrt(variance)


class GARCHEstimator:
    """
    GARCH(1,1) Volatility Estimation

    Industry standard for volatility modeling.

    Model:
        r_t = σ_t * ε_t,  ε_t ~ N(0, 1)
        σ²_t = ω + α*r²_{t-1} + β*σ²_{t-1}

    Constraints:
        ω > 0, α ≥ 0, β ≥ 0
        α + β < 1 (stationarity)

    Long-run variance: σ²_LR = ω / (1 - α - β)
    Persistence: α + β (close to 1 → high persistence)

    Example:
        >>> estimator = GARCHEstimator()
        >>> result = estimator.estimate(returns)
        >>> print(f"ω={result.params['omega']:.6f}")
        >>> print(f"α={result.params['alpha']:.4f}")
        >>> print(f"β={result.params['beta']:.4f}")
        >>> print(f"Persistence: {result.persistence:.4f}")
    """

    def __init__(
        self,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ):
        """
        Initialize GARCH estimator.

        Args:
            max_iter: Maximum iterations for MLE optimization
            tol: Convergence tolerance
        """
        self.max_iter = max_iter
        self.tol = tol

    def estimate(
        self,
        returns: np.ndarray,
        annualize: bool = True,
        periods_per_year: int = 252,
    ) -> VolatilityEstimate:
        """
        Estimate GARCH(1,1) model via Maximum Likelihood.

        Args:
            returns: Return series
            annualize: Whether to annualize volatility
            periods_per_year: Number of periods per year

        Returns:
            VolatilityEstimate with fitted volatility and parameters
        """
        returns = np.asarray(returns).flatten()

        if len(returns) < 50:
            raise ValueError("GARCH requires at least 50 observations")

        # Initial parameter guess
        # Common starting values based on unconditional moments
        uncond_var = np.var(returns, ddof=1)

        # Initial guess: ω=0.01*var, α=0.1, β=0.85 (typical values)
        x0 = np.array([
            0.01 * uncond_var,  # omega
            0.10,                # alpha
            0.85,                # beta
        ])

        # Parameter bounds
        bounds = [
            (1e-6, None),  # omega > 0
            (0.0, 1.0),    # 0 ≤ alpha < 1
            (0.0, 1.0),    # 0 ≤ beta < 1
        ]

        # Constraint: alpha + beta < 1 (stationarity)
        constraints = {
            'type': 'ineq',
            'fun': lambda x: 0.9999 - (x[1] + x[2])  # alpha + beta < 1
        }

        # Optimize
        result = optimize.minimize(
            fun=self._negative_log_likelihood,
            x0=x0,
            args=(returns,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iter, 'ftol': self.tol},
        )

        if not result.success:
            warnings.warn(f"GARCH optimization did not converge: {result.message}")

        omega, alpha, beta = result.x

        # Compute fitted conditional volatility
        conditional_var = self._compute_conditional_variance(returns, omega, alpha, beta)
        conditional_vol = np.sqrt(conditional_var)

        # Annualize if requested
        annualization_factor = np.sqrt(periods_per_year) if annualize else 1.0
        conditional_vol = conditional_vol * annualization_factor

        # Compute diagnostics
        long_run_var = omega / (1 - alpha - beta)
        persistence = alpha + beta
        half_life = np.log(0.5) / np.log(persistence) if persistence > 0 else np.inf

        params = {
            'omega': omega,
            'alpha': alpha,
            'beta': beta,
            'long_run_vol': np.sqrt(long_run_var) * annualization_factor,
            'persistence': persistence,
            'half_life': half_life,
            'log_likelihood': -result.fun,
            'converged': result.success,
        }

        return VolatilityEstimate(
            volatility=conditional_vol,
            method='garch',
            annualization_factor=annualization_factor,
            params=params,
        )

    def _negative_log_likelihood(
        self,
        params: np.ndarray,
        returns: np.ndarray,
    ) -> float:
        """
        Negative log-likelihood for GARCH(1,1).

        L = -0.5 * Σ[log(2π) + log(σ²_t) + r²_t/σ²_t]

        Minimize negative LL = maximize LL
        """
        omega, alpha, beta = params

        # Compute conditional variances
        conditional_var = self._compute_conditional_variance(returns, omega, alpha, beta)

        # Log-likelihood (ignoring constant term)
        # LL = -0.5 * Σ[log(σ²_t) + r²_t/σ²_t]
        log_likelihood = -0.5 * np.sum(
            np.log(conditional_var) + returns**2 / conditional_var
        )

        return -log_likelihood  # Return negative for minimization

    def _compute_conditional_variance(
        self,
        returns: np.ndarray,
        omega: float,
        alpha: float,
        beta: float,
    ) -> np.ndarray:
        """
        Compute conditional variance series σ²_t.

        σ²_t = ω + α*r²_{t-1} + β*σ²_{t-1}
        """
        n = len(returns)
        variance = np.zeros(n)

        # Initialize with unconditional variance
        # E[σ²] = ω / (1 - α - β)
        if alpha + beta < 1:
            variance[0] = omega / (1 - alpha - beta)
        else:
            variance[0] = np.var(returns, ddof=1)

        # Recursive update
        for t in range(1, n):
            variance[t] = omega + alpha * returns[t-1]**2 + beta * variance[t-1]

        return variance


def compare_volatility_methods(
    returns: np.ndarray,
    window: int = 252,
    lambda_: float = 0.94,
    annualize: bool = True,
) -> Dict[str, VolatilityEstimate]:
    """
    Compare different volatility estimation methods.

    Useful for understanding method differences and choosing the best one.

    Args:
        returns: Return series
        window: Window for simple method
        lambda_: Decay factor for EWMA
        annualize: Annualize results

    Returns:
        Dictionary with results from each method
    """
    results = {}

    # Simple
    simple_est = VolatilityEstimator(method='simple', window=window)
    results['simple'] = simple_est.estimate(returns, annualize=annualize)

    # EWMA
    ewma_est = VolatilityEstimator(method='ewma', lambda_=lambda_)
    results['ewma'] = ewma_est.estimate(returns, annualize=annualize)

    # GARCH
    try:
        garch_est = GARCHEstimator()
        results['garch'] = garch_est.estimate(returns, annualize=annualize)
    except Exception as e:
        warnings.warn(f"GARCH estimation failed: {e}")
        results['garch'] = None

    return results


# =============================================================================
# OHLC-based Volatility Estimators
# =============================================================================

def parkinson_volatility(
    high: np.ndarray,
    low: np.ndarray,
    window: int = 30,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> np.ndarray:
    """
    Parkinson (1980) High-Low Range Volatility Estimator.

    Formula:
        σ²_P = (1 / (4 * ln(2))) * E[(ln(H/L))²]
             ≈ 0.361 * E[(ln(H/L))²]

    where H = high price, L = low price

    Efficiency: ~5x more efficient than close-to-close (uses intraday range)

    Assumptions:
    - Continuous trading (no overnight jumps)
    - No drift
    - Prices follow GBM

    Args:
        high: High prices
        low: Low prices
        window: Rolling window size
        annualize: Whether to annualize volatility
        periods_per_year: Annualization factor

    Returns:
        Array of volatility estimates

    Example:
        >>> vol = parkinson_volatility(df['High'], df['Low'], window=30)
    """
    high = np.asarray(high)
    low = np.asarray(low)

    if len(high) != len(low):
        raise ValueError("High and Low must have same length")

    # Log range squared
    log_hl = np.log(high / low)
    log_hl_sq = log_hl ** 2

    # Parkinson constant
    k = 1.0 / (4 * np.log(2))

    # Rolling variance
    n = len(high)
    variance = np.full(n, np.nan)

    for i in range(window - 1, n):
        variance[i] = k * np.mean(log_hl_sq[i - window + 1 : i + 1])

    vol = np.sqrt(variance)

    if annualize:
        vol = vol * np.sqrt(periods_per_year)

    return vol


def garman_klass_volatility(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int = 30,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> np.ndarray:
    """
    Garman-Klass (1980) OHLC Volatility Estimator.

    Formula:
        σ²_GK = 0.5 * (ln(H/L))² - (2*ln(2) - 1) * (ln(C/O))²

    Combines high-low range with open-close information.

    Efficiency: ~8x more efficient than close-to-close

    Assumptions:
    - Continuous trading (no overnight jumps)
    - Zero drift
    - Prices follow GBM

    Args:
        open_: Open prices
        high: High prices
        low: Low prices
        close: Close prices
        window: Rolling window size
        annualize: Whether to annualize volatility
        periods_per_year: Annualization factor

    Returns:
        Array of volatility estimates

    Example:
        >>> vol = garman_klass_volatility(
        ...     df['Open'], df['High'], df['Low'], df['Close'], window=30
        ... )
    """
    open_ = np.asarray(open_)
    high = np.asarray(high)
    low = np.asarray(low)
    close = np.asarray(close)

    # Log ratios
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)

    # GK variance components
    term1 = 0.5 * log_hl ** 2
    term2 = (2 * np.log(2) - 1) * log_co ** 2

    gk_variance_components = term1 - term2

    # Rolling variance
    n = len(high)
    variance = np.full(n, np.nan)

    for i in range(window - 1, n):
        variance[i] = np.mean(gk_variance_components[i - window + 1 : i + 1])

    vol = np.sqrt(variance)

    if annualize:
        vol = vol * np.sqrt(periods_per_year)

    return vol


def rogers_satchell_volatility(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int = 30,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> np.ndarray:
    """
    Rogers-Satchell (1991) Drift-Independent OHLC Volatility Estimator.

    Formula:
        σ²_RS = E[ln(H/C) * ln(H/O) + ln(L/C) * ln(L/O)]

    Key advantage: Handles non-zero drift (trending markets)

    Efficiency: Similar to Garman-Klass but works with drift

    Assumptions:
    - Continuous trading (no overnight jumps)
    - Allows non-zero drift

    Args:
        open_: Open prices
        high: High prices
        low: Low prices
        close: Close prices
        window: Rolling window size
        annualize: Whether to annualize volatility
        periods_per_year: Annualization factor

    Returns:
        Array of volatility estimates

    Example:
        >>> vol = rogers_satchell_volatility(
        ...     df['Open'], df['High'], df['Low'], df['Close'], window=30
        ... )
    """
    open_ = np.asarray(open_)
    high = np.asarray(high)
    low = np.asarray(low)
    close = np.asarray(close)

    # Log ratios
    log_ho = np.log(high / open_)
    log_hc = np.log(high / close)
    log_lo = np.log(low / open_)
    log_lc = np.log(low / close)

    # RS variance components
    rs_variance_components = log_hc * log_ho + log_lc * log_lo

    # Rolling variance
    n = len(high)
    variance = np.full(n, np.nan)

    for i in range(window - 1, n):
        variance[i] = np.mean(rs_variance_components[i - window + 1 : i + 1])

    vol = np.sqrt(variance)

    if annualize:
        vol = vol * np.sqrt(periods_per_year)

    return vol


def yang_zhang_volatility(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int = 30,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> np.ndarray:
    """
    Yang-Zhang (2000) Volatility Estimator - Most Efficient OHLC Estimator.

    Combines:
    - Overnight volatility (close-to-open)
    - Open-to-close volatility (Rogers-Satchell)
    - Drift correction

    Formula:
        σ²_YZ = σ²_overnight + k * σ²_open_close + (1-k) * σ²_RS

    where k is chosen to minimize variance of the estimator.

    Efficiency: ~14x more efficient than close-to-close (best OHLC estimator)

    Advantages:
    - Handles overnight jumps
    - Drift-independent
    - Minimum variance among OHLC estimators

    Args:
        open_: Open prices
        high: High prices
        low: Low prices
        close: Close prices
        window: Rolling window size
        annualize: Whether to annualize volatility
        periods_per_year: Annualization factor

    Returns:
        Array of volatility estimates

    Example:
        >>> vol = yang_zhang_volatility(
        ...     df['Open'], df['High'], df['Low'], df['Close'], window=30
        ... )

    Reference:
        Yang, D., & Zhang, Q. (2000). Drift-independent volatility estimation
        based on high, low, open, and close prices. Journal of Business, 73(3), 477-492.
    """
    open_ = np.asarray(open_)
    high = np.asarray(high)
    low = np.asarray(low)
    close = np.asarray(close)

    n = len(close)
    variance = np.full(n, np.nan)

    # Constant k (from Yang-Zhang paper)
    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    for i in range(window, n):
        window_open = open_[i - window : i]
        window_high = high[i - window : i]
        window_low = low[i - window : i]
        window_close = close[i - window : i]

        # Overnight volatility (close[t-1] to open[t])
        co_returns = np.log(window_open[1:] / window_close[:-1])
        overnight_var = np.var(co_returns, ddof=1)

        # Open-to-close volatility
        oc_returns = np.log(window_close / window_open)
        open_close_var = np.var(oc_returns, ddof=1)

        # Rogers-Satchell component
        log_ho = np.log(window_high / window_open)
        log_hc = np.log(window_high / window_close)
        log_lo = np.log(window_low / window_open)
        log_lc = np.log(window_low / window_close)

        rs_var = np.mean(log_hc * log_ho + log_lc * log_lo)

        # Yang-Zhang estimator
        variance[i] = overnight_var + k * open_close_var + (1 - k) * rs_var

    vol = np.sqrt(variance)

    if annualize:
        vol = vol * np.sqrt(periods_per_year)

    return vol


def compare_ohlc_estimators(
    ohlc_data: Union[pd.DataFrame, Dict[str, np.ndarray]],
    window: int = 30,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> Dict[str, np.ndarray]:
    """
    Compare all OHLC-based volatility estimators.

    Args:
        ohlc_data: DataFrame with columns ['Open', 'High', 'Low', 'Close']
                   or dict with these keys
        window: Rolling window size
        annualize: Whether to annualize
        periods_per_year: Annualization factor

    Returns:
        Dictionary with results from each estimator

    Example:
        >>> results = compare_ohlc_estimators(df, window=30)
        >>> for method, vol in results.items():
        ...     print(f"{method}: {vol[-1]:.2%}")
    """
    # Extract OHLC arrays
    if isinstance(ohlc_data, pd.DataFrame):
        open_ = ohlc_data['Open'].values
        high = ohlc_data['High'].values
        low = ohlc_data['Low'].values
        close = ohlc_data['Close'].values
    else:
        open_ = ohlc_data['Open']
        high = ohlc_data['High']
        low = ohlc_data['Low']
        close = ohlc_data['Close']

    results = {}

    # Parkinson (HL only)
    results['parkinson'] = parkinson_volatility(
        high, low, window, annualize, periods_per_year
    )

    # Garman-Klass (OHLC, no drift)
    results['garman_klass'] = garman_klass_volatility(
        open_, high, low, close, window, annualize, periods_per_year
    )

    # Rogers-Satchell (OHLC, with drift)
    results['rogers_satchell'] = rogers_satchell_volatility(
        open_, high, low, close, window, annualize, periods_per_year
    )

    # Yang-Zhang (OHLC + overnight, best)
    results['yang_zhang'] = yang_zhang_volatility(
        open_, high, low, close, window, annualize, periods_per_year
    )

    # Close-to-close for comparison
    returns = np.diff(np.log(close))
    results['close_to_close'] = VolatilityEstimator(
        method='simple', window=window
    ).estimate(returns, annualize, periods_per_year).volatility

    return results
