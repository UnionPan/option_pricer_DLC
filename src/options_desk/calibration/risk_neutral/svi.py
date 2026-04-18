"""
SVI and SSVI implied volatility surface parameterization.

Raw SVI (Gatheral 2004):
    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

where w = sigma_BS^2 * T is total implied variance and k = log(K/F) is log-moneyness.

SSVI (Gatheral & Jacquier 2014):
    w(k, theta_t) = theta_t/2 * (1 + rho*phi(theta_t)*k
                     + sqrt((phi(theta_t)*k + rho)^2 + 1 - rho^2))

where theta_t = sigma_ATM^2 * T is ATM total variance.

References:
  - Gatheral (2004), "A parsimonious arbitrage-free implied volatility parameterization"
  - Gatheral & Jacquier (2014), "Arbitrage-free SVI volatility surfaces"

author: Yunian Pan
email: yp1170@nyu.edu
"""

import numpy as np
from scipy import optimize
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple


# ============================================================================
# Raw SVI parameterization
# ============================================================================
@dataclass
class SVIParams:
    """Raw SVI parameters for a single expiry slice."""
    a: float      # vertical shift (level)
    b: float      # slope (controls wings)
    rho: float    # rotation (-1 < rho < 1, controls skew)
    m: float      # horizontal shift (translation)
    sigma: float  # smoothing (curvature at vertex)
    T: float      # time to expiry (for context, not part of the 5 SVI params)

    def total_variance(self, k: np.ndarray) -> np.ndarray:
        """
        Compute total implied variance w(k) = sigma_BS^2 * T.

        Args:
            k: Log-moneyness k = log(K/F), array

        Returns:
            Total implied variance w(k)
        """
        return self.a + self.b * (
            self.rho * (k - self.m)
            + np.sqrt((k - self.m) ** 2 + self.sigma ** 2)
        )

    def implied_vol(self, k: np.ndarray) -> np.ndarray:
        """
        Compute Black-Scholes implied volatility from SVI.

        Args:
            k: Log-moneyness

        Returns:
            Implied volatility sigma_BS(k)
        """
        w = self.total_variance(k)
        w = np.maximum(w, 1e-10)
        return np.sqrt(w / self.T)

    @property
    def atm_variance(self) -> float:
        """ATM total variance w(0)."""
        return self.total_variance(np.array([0.0]))[0]

    @property
    def atm_vol(self) -> float:
        """ATM implied volatility."""
        return np.sqrt(self.atm_variance / self.T)

    def __repr__(self) -> str:
        return (
            f"SVIParams(a={self.a:.6f}, b={self.b:.6f}, rho={self.rho:.4f}, "
            f"m={self.m:.6f}, sigma={self.sigma:.6f}, T={self.T:.4f})"
        )


# ============================================================================
# SSVI parameterization
# ============================================================================
@dataclass
class SSVIParams:
    """
    SSVI surface parameters (Gatheral-Jacquier).

    The SSVI surface is defined by:
      w(k, theta) = theta/2 * (1 + rho*phi(theta)*k
                     + sqrt((phi(theta)*k + rho)^2 + 1 - rho^2))

    where theta = ATM total variance for each slice, and phi(theta) is a
    function controlling the wing behavior.

    Power-law phi: phi(theta) = eta / (theta^gamma * (1 + theta)^(1-gamma))
    """
    rho: float      # global skew parameter (-1 < rho < 1)
    eta: float      # wing steepness (eta > 0)
    gamma: float    # power-law exponent (0 < gamma < 1)

    def phi(self, theta: np.ndarray) -> np.ndarray:
        """Power-law phi function."""
        theta = np.asarray(theta)
        return self.eta / (theta ** self.gamma * (1.0 + theta) ** (1.0 - self.gamma))

    def total_variance(self, k: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        SSVI total variance surface.

        Args:
            k: Log-moneyness, shape (n_strikes,) or (n_strikes, n_expiries)
            theta: ATM total variance per expiry, shape (n_expiries,)

        Returns:
            Total variance w(k, theta)
        """
        k = np.asarray(k)
        theta = np.asarray(theta)
        p = self.phi(theta)

        return (theta / 2.0) * (
            1.0 + self.rho * p * k
            + np.sqrt((p * k + self.rho) ** 2 + 1.0 - self.rho ** 2)
        )

    def implied_vol(self, k: np.ndarray, theta: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Implied volatility from SSVI.

        Args:
            k: Log-moneyness
            theta: ATM total variance
            T: Time to expiry

        Returns:
            Implied volatility
        """
        w = self.total_variance(k, theta)
        w = np.maximum(w, 1e-10)
        return np.sqrt(w / T)

    def __repr__(self) -> str:
        return f"SSVIParams(rho={self.rho:.4f}, eta={self.eta:.4f}, gamma={self.gamma:.4f})"


# ============================================================================
# Arbitrage checks
# ============================================================================
def check_butterfly_arbitrage(svi: SVIParams, k_grid: np.ndarray = None) -> Dict:
    """
    Check for butterfly (static) arbitrage in an SVI slice.

    Butterfly arbitrage occurs when the risk-neutral density g(k) < 0.
    The density is related to the second derivative of call prices, which
    in terms of total variance w(k) requires:

        g(k) = (1 - k*w'/(2*w))^2 - w'^2/4*(1/w + 1/4) + w''/2  >= 0

    Args:
        svi: SVIParams for a single slice
        k_grid: Grid of log-moneyness to check (default: -2 to 2)

    Returns:
        Dict with 'is_arbitrage_free', 'violations', 'min_density'
    """
    if k_grid is None:
        k_grid = np.linspace(-2.0, 2.0, 500)

    w = svi.total_variance(k_grid)
    dk = k_grid[1] - k_grid[0]

    # First and second derivatives via finite differences
    w_prime = np.gradient(w, dk)
    w_double_prime = np.gradient(w_prime, dk)

    # Risk-neutral density (Breeden-Litzenberger in total variance form)
    # g(k) = (1 - k*w'/(2*w))^2 - (w'^2/4)*(1/w + 1/4) + w''/2
    with np.errstate(divide='ignore', invalid='ignore'):
        term1 = (1.0 - k_grid * w_prime / (2.0 * w)) ** 2
        term2 = (w_prime ** 2 / 4.0) * (1.0 / w + 0.25)
        term3 = w_double_prime / 2.0
        g = term1 - term2 + term3

    # Also check w >= 0 (negative total variance is obviously wrong)
    w_negative = np.any(w < -1e-10)

    violations = np.sum(g < -1e-8)
    min_density = float(np.nanmin(g))

    return {
        'is_arbitrage_free': violations == 0 and not w_negative,
        'violations': int(violations),
        'min_density': min_density,
        'negative_variance': bool(w_negative),
    }


def check_calendar_arbitrage(
    svi_slices: List[SVIParams],
    k_grid: np.ndarray = None,
) -> Dict:
    """
    Check for calendar spread arbitrage across expiry slices.

    Calendar arbitrage occurs when total variance decreases with maturity:
        w(k, T1) > w(k, T2) for T1 < T2

    This violates the monotonicity of total variance in T.

    Args:
        svi_slices: List of SVIParams sorted by increasing T
        k_grid: Grid of log-moneyness (default: -2 to 2)

    Returns:
        Dict with 'is_arbitrage_free', 'violations'
    """
    if k_grid is None:
        k_grid = np.linspace(-2.0, 2.0, 200)

    # Sort by maturity
    slices = sorted(svi_slices, key=lambda s: s.T)

    violations = 0
    for i in range(len(slices) - 1):
        w_short = slices[i].total_variance(k_grid)
        w_long = slices[i + 1].total_variance(k_grid)
        # Total variance must be non-decreasing in T
        n_violations = np.sum(w_short > w_long + 1e-8)
        violations += n_violations

    return {
        'is_arbitrage_free': violations == 0,
        'violations': int(violations),
    }


# ============================================================================
# SVI fitting (single slice)
# ============================================================================
@dataclass
class SVIFitResult:
    """Result of SVI calibration to a single expiry slice."""
    params: SVIParams
    rmse_iv: float
    max_error_iv: float
    n_points: int
    arbitrage_check: Dict
    success: bool

    def __repr__(self) -> str:
        arb = "✓" if self.arbitrage_check['is_arbitrage_free'] else "✗"
        return (
            f"SVIFitResult(rmse_iv={self.rmse_iv*100:.2f}%, "
            f"max_err={self.max_error_iv*100:.2f}%, "
            f"arb_free={arb}, n={self.n_points})\n"
            f"  {self.params}"
        )


def fit_svi_slice(
    k: np.ndarray,
    iv: np.ndarray,
    T: float,
    weights: np.ndarray = None,
    initial_guess: np.ndarray = None,
) -> SVIFitResult:
    """
    Fit raw SVI to a single expiry slice of implied volatilities.

    Args:
        k: Log-moneyness k = log(K/F), shape (n,)
        iv: Implied volatilities, shape (n,)
        T: Time to expiry
        weights: Optional weights for each point (default: uniform)
        initial_guess: Optional [a, b, rho, m, sigma] initial guess

    Returns:
        SVIFitResult with calibrated parameters
    """
    w_market = iv ** 2 * T  # Total variance

    if weights is None:
        weights = np.ones_like(k)
    weights = weights / weights.sum()

    def objective(params):
        a, b, rho, m, sigma = params
        svi = SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma, T=T)
        w_model = svi.total_variance(k)
        residuals = (w_model - w_market) * np.sqrt(weights)
        return residuals

    # Initial guess
    if initial_guess is None:
        w_atm = np.interp(0.0, k, w_market)
        initial_guess = np.array([
            w_atm,                  # a: ATM level
            0.1,                    # b: slope
            -0.3,                   # rho: skew (negative for equity)
            0.0,                    # m: centered
            max(0.1, w_atm * 0.5),  # sigma: smoothing
        ])

    # Bounds: a∈R, b>0, -1<rho<1, m∈R, sigma>0
    bounds = (
        [-np.inf, 1e-6, -0.999, -2.0, 1e-6],
        [np.inf, 5.0, 0.999, 2.0, 5.0],
    )

    result = optimize.least_squares(
        objective, initial_guess, bounds=bounds, method='trf',
        max_nfev=2000, ftol=1e-12, xtol=1e-12, gtol=1e-12,
    )

    a, b, rho, m, sigma = result.x
    svi = SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma, T=T)

    # Compute fit quality in IV terms
    iv_model = svi.implied_vol(k)
    iv_errors = iv_model - iv
    rmse_iv = float(np.sqrt(np.mean(iv_errors ** 2)))
    max_error_iv = float(np.max(np.abs(iv_errors)))

    arb = check_butterfly_arbitrage(svi)

    return SVIFitResult(
        params=svi,
        rmse_iv=rmse_iv,
        max_error_iv=max_error_iv,
        n_points=len(k),
        arbitrage_check=arb,
        success=result.success,
    )


# ============================================================================
# SSVI fitting (global surface)
# ============================================================================
@dataclass
class SSVIFitResult:
    """Result of SSVI calibration to the full surface."""
    params: SSVIParams
    theta_values: np.ndarray    # ATM total variance per expiry
    expiries: np.ndarray        # Expiry times
    rmse_iv: float
    max_error_iv: float
    n_slices: int
    n_points: int
    butterfly_check: Dict
    calendar_check: Dict
    success: bool

    def __repr__(self) -> str:
        b_arb = "✓" if self.butterfly_check['is_arbitrage_free'] else "✗"
        c_arb = "✓" if self.calendar_check['is_arbitrage_free'] else "✗"
        return (
            f"SSVIFitResult(rmse_iv={self.rmse_iv*100:.2f}%, "
            f"max_err={self.max_error_iv*100:.2f}%, "
            f"n_slices={self.n_slices}, n_pts={self.n_points})\n"
            f"  butterfly_arb_free={b_arb}, calendar_arb_free={c_arb}\n"
            f"  {self.params}"
        )


def fit_ssvi_surface(
    slices: List[Dict],
    initial_guess: np.ndarray = None,
) -> SSVIFitResult:
    """
    Fit SSVI to a multi-expiry implied volatility surface.

    Two-stage approach:
      1. Extract ATM total variance theta_t for each slice
      2. Fit global (rho, eta, gamma) to minimize IV error across all slices

    Args:
        slices: List of dicts, each with:
            - 'k': log-moneyness array
            - 'iv': implied vol array
            - 'T': time to expiry
            - 'weights': optional weight array
        initial_guess: Optional [rho, eta, gamma]

    Returns:
        SSVIFitResult with calibrated SSVI parameters
    """
    # Stage 1: Extract ATM total variance for each slice
    expiries = np.array([s['T'] for s in slices])
    theta_values = np.zeros(len(slices))

    for i, s in enumerate(slices):
        k, iv, T = s['k'], s['iv'], s['T']
        # Interpolate ATM IV
        atm_iv = float(np.interp(0.0, k, iv))
        theta_values[i] = atm_iv ** 2 * T

    # Stage 2: Fit global SSVI parameters
    all_k = []
    all_w_market = []
    all_theta = []
    all_T = []
    all_weights = []

    for i, s in enumerate(slices):
        k, iv, T = s['k'], s['iv'], s['T']
        w = iv ** 2 * T
        wt = s.get('weights', np.ones_like(k))
        all_k.append(k)
        all_w_market.append(w)
        all_theta.append(np.full_like(k, theta_values[i]))
        all_T.append(np.full_like(k, T))
        all_weights.append(wt)

    all_k = np.concatenate(all_k)
    all_w_market = np.concatenate(all_w_market)
    all_theta = np.concatenate(all_theta)
    all_T = np.concatenate(all_T)
    all_weights = np.concatenate(all_weights)
    all_weights = all_weights / all_weights.sum()

    n_total = len(all_k)

    def objective(params):
        rho, eta, gamma = params
        ssvi = SSVIParams(rho=rho, eta=eta, gamma=gamma)
        w_model = ssvi.total_variance(all_k, all_theta)
        residuals = (w_model - all_w_market) * np.sqrt(all_weights)
        return residuals

    if initial_guess is None:
        initial_guess = np.array([-0.3, 1.0, 0.5])

    bounds = (
        [-0.999, 0.01, 0.01],
        [0.999, 5.0, 0.99],
    )

    result = optimize.least_squares(
        objective, initial_guess, bounds=bounds, method='trf',
        max_nfev=2000, ftol=1e-12, xtol=1e-12, gtol=1e-12,
    )

    rho, eta, gamma = result.x
    ssvi = SSVIParams(rho=rho, eta=eta, gamma=gamma)

    # Compute fit quality
    all_iv_market = np.sqrt(all_w_market / all_T)
    w_model = ssvi.total_variance(all_k, all_theta)
    all_iv_model = np.sqrt(np.maximum(w_model, 1e-10) / all_T)
    iv_errors = all_iv_model - all_iv_market
    rmse_iv = float(np.sqrt(np.mean(iv_errors ** 2)))
    max_error_iv = float(np.max(np.abs(iv_errors)))

    # Arbitrage checks via per-slice SVI approximation
    svi_slices = []
    for i in range(len(slices)):
        T = slices[i]['T']
        theta = theta_values[i]
        phi_val = ssvi.phi(theta)
        # Reconstruct raw SVI from SSVI for this slice
        svi_slice = SVIParams(
            a=theta / 2.0 * (1.0 + rho * np.sqrt(1.0 - rho ** 2)
                              - np.sqrt(1.0 - rho ** 2)),
            b=theta / 2.0 * phi_val,
            rho=rho,
            m=0.0,
            sigma=np.sqrt(1.0 - rho ** 2) / phi_val if phi_val > 1e-8 else 1.0,
            T=T,
        )
        svi_slices.append(svi_slice)

    butterfly_violations = 0
    for svi_s in svi_slices:
        check = check_butterfly_arbitrage(svi_s)
        butterfly_violations += check['violations']

    butterfly_check = {
        'is_arbitrage_free': butterfly_violations == 0,
        'violations': butterfly_violations,
    }

    calendar_check = check_calendar_arbitrage(svi_slices)

    return SSVIFitResult(
        params=ssvi,
        theta_values=theta_values,
        expiries=expiries,
        rmse_iv=rmse_iv,
        max_error_iv=max_error_iv,
        n_slices=len(slices),
        n_points=n_total,
        butterfly_check=butterfly_check,
        calendar_check=calendar_check,
        success=result.success,
    )


# ============================================================================
# Convenience: fit from OptionChain
# ============================================================================
def fit_svi_from_chain(
    chain,
    forward: float = None,
) -> List[SVIFitResult]:
    """
    Fit SVI to each expiry slice in an OptionChain.

    Args:
        chain: OptionChain object (from calibration.data.data_provider)
        forward: Forward price. If None, uses spot * exp((r-q)*T)

    Returns:
        List of SVIFitResult, one per expiry
    """
    df = chain.to_dataframe()
    if df.empty:
        return []

    spot = chain.spot_price
    r = chain.risk_free_rate
    q = chain.dividend_yield

    results = []
    for T, group in df.groupby('time_to_expiry'):
        if T < 1e-6:
            continue

        F = forward if forward is not None else spot * np.exp((r - q) * T)
        k = np.log(group['strike'].values / F)
        iv = group['implied_volatility'].values

        # Filter out NaN/zero IVs
        valid = np.isfinite(iv) & (iv > 0.001)
        if valid.sum() < 3:
            continue

        fit = fit_svi_slice(k[valid], iv[valid], T)
        results.append(fit)

    return results
