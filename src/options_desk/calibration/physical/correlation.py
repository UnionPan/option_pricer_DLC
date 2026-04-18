"""
Multi-Asset Correlation Calibration

Robust estimation of correlation and covariance matrices from
historical return data.  Methods include sample correlation, EWMA,
Ledoit-Wolf shrinkage, and Marchenko-Pastur random-matrix cleaning.

author: Yunian Pan
email: yp1170@nyu.edu
"""

import numpy as np
from scipy import linalg
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Result of correlation estimation."""

    correlation: np.ndarray      # (n, n) correlation matrix
    covariance: np.ndarray       # (n, n) covariance matrix
    volatilities: np.ndarray     # (n,) annualized vols
    asset_names: List[str]

    method: str
    n_observations: int
    effective_observations: float

    # Diagnostics
    is_positive_definite: bool
    condition_number: float
    eigenvalues: np.ndarray

    computation_time: float

    def __repr__(self) -> str:
        n = len(self.asset_names)
        lines = [
            f"CorrelationResult ({self.method}):",
            f"  Assets: {n} ({', '.join(self.asset_names[:5])}"
            + ("..." if n > 5 else "") + ")",
            f"  Observations: {self.n_observations} "
            f"(effective: {self.effective_observations:.0f})",
            f"  PD: {'yes' if self.is_positive_definite else 'NO'}",
            f"  Condition number: {self.condition_number:.1f}",
            f"  Eigenvalue range: [{self.eigenvalues.min():.4f}, "
            f"{self.eigenvalues.max():.4f}]",
            f"  Time: {self.computation_time:.4f}s",
        ]
        return "\n".join(lines)


class CorrelationCalibrator:
    """
    Estimate correlation matrices from historical returns.

    Supports multiple methods:
        - 'sample': Standard Pearson sample correlation
        - 'ewma': Exponentially weighted moving average
        - 'shrinkage': Ledoit-Wolf shrinkage toward identity
        - 'rmt': Random matrix theory (Marchenko-Pastur) cleaning

    Example::

        calibrator = CorrelationCalibrator()
        result = calibrator.fit(
            returns,
            asset_names=['SPY', 'AAPL', 'MSFT'],
            method='shrinkage',
        )
        corr_matrix = result.correlation
    """

    def __init__(
        self,
        frequency: str = 'daily',
        annualization_factor: int = 252,
    ):
        """
        Args:
            frequency: Return frequency ('daily', 'weekly', 'monthly').
            annualization_factor: Factor for annualising volatility.
        """
        self.frequency = frequency
        self.annualization_factor = annualization_factor

    def fit(
        self,
        returns: np.ndarray,
        asset_names: Optional[List[str]] = None,
        method: str = 'shrinkage',
        **kwargs,
    ) -> CorrelationResult:
        """
        Estimate correlation matrix.

        Args:
            returns: Log-returns matrix, shape (T, n_assets).
            asset_names: Optional list of asset names.
            method: One of 'sample', 'ewma', 'shrinkage', 'rmt'.
            **kwargs: Method-specific parameters.

        Returns:
            CorrelationResult with estimated matrices and diagnostics.
        """
        start_time = time.time()
        returns = np.asarray(returns, dtype=np.float64)
        T, n = returns.shape

        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(n)]

        dispatch = {
            'sample': self._sample_correlation,
            'ewma': self._ewma_correlation,
            'shrinkage': self._ledoit_wolf_shrinkage,
            'rmt': self._random_matrix_cleaning,
        }
        if method not in dispatch:
            raise ValueError(f"Unknown method '{method}'. Use: {list(dispatch)}")

        corr, cov = dispatch[method](returns, **kwargs)

        # Diagnostics
        eigvals = np.linalg.eigvalsh(corr)
        is_pd = bool(eigvals.min() > 0)
        cond = float(eigvals.max() / max(eigvals.min(), 1e-15))

        # Effective observations (Newey-West-style autocorrelation adjustment)
        eff_obs = self._effective_observations(returns)

        vols = np.sqrt(np.diag(cov)) * np.sqrt(self.annualization_factor)

        computation_time = time.time() - start_time

        return CorrelationResult(
            correlation=corr,
            covariance=cov,
            volatilities=vols,
            asset_names=asset_names,
            method=method,
            n_observations=T,
            effective_observations=eff_obs,
            is_positive_definite=is_pd,
            condition_number=cond,
            eigenvalues=eigvals,
            computation_time=computation_time,
        )

    # ------------------------------------------------------------------
    # Method implementations
    # ------------------------------------------------------------------

    def _sample_correlation(
        self, returns: np.ndarray, **kwargs
    ) -> tuple:
        """Standard Pearson sample correlation."""
        cov = np.cov(returns, rowvar=False)
        std = np.sqrt(np.diag(cov))
        std_safe = np.where(std > 0, std, 1e-12)
        corr = cov / np.outer(std_safe, std_safe)
        np.fill_diagonal(corr, 1.0)
        return corr, cov

    def _ewma_correlation(
        self, returns: np.ndarray, halflife: int = 60, **kwargs
    ) -> tuple:
        """
        Exponentially weighted moving average correlation.

        Args:
            returns: (T, n) returns matrix.
            halflife: Halflife in periods for the decay factor.
        """
        decay = np.exp(-np.log(2) / halflife)
        T, n = returns.shape

        # Demean
        mu = returns.mean(axis=0)
        centered = returns - mu

        # Recursive EWMA covariance
        cov = np.outer(centered[0], centered[0])
        for t in range(1, T):
            cov = decay * cov + (1 - decay) * np.outer(centered[t], centered[t])

        std = np.sqrt(np.diag(cov))
        std_safe = np.where(std > 0, std, 1e-12)
        corr = cov / np.outer(std_safe, std_safe)
        np.fill_diagonal(corr, 1.0)

        return corr, cov

    def _ledoit_wolf_shrinkage(
        self, returns: np.ndarray, **kwargs
    ) -> tuple:
        """
        Ledoit-Wolf shrinkage estimator.

        Shrinks the sample covariance toward a structured target
        (scaled identity) using the analytically optimal shrinkage
        intensity from Ledoit & Wolf (2004).
        """
        T, n = returns.shape
        mu = returns.mean(axis=0)
        centered = returns - mu

        # Sample covariance
        S = centered.T @ centered / T

        # Target: mu * I  (where mu = trace(S)/n)
        trace_S = np.trace(S)
        mu_target = trace_S / n
        F = mu_target * np.eye(n)

        # Optimal shrinkage intensity (Ledoit-Wolf 2004, Eq. 2)
        delta = S - F

        # sum of squared off-diagonals of S
        sum_sq = np.sum(delta ** 2)

        # Estimate of pi (sum of asymptotic variances)
        # pi_hat = (1/T^2) * sum_t sum_{ij} (x_{ti}*x_{tj} - s_{ij})^2
        pi_hat = 0.0
        for t in range(T):
            outer_t = np.outer(centered[t], centered[t])
            pi_hat += np.sum((outer_t - S) ** 2)
        pi_hat /= T ** 2

        # Shrinkage intensity
        alpha = min(pi_hat / max(sum_sq, 1e-12), 1.0)
        alpha = max(alpha, 0.0)

        cov_shrunk = alpha * F + (1 - alpha) * S

        std = np.sqrt(np.diag(cov_shrunk))
        std_safe = np.where(std > 0, std, 1e-12)
        corr = cov_shrunk / np.outer(std_safe, std_safe)
        np.fill_diagonal(corr, 1.0)

        logger.info("Ledoit-Wolf shrinkage intensity: %.4f", alpha)
        return corr, cov_shrunk

    def _random_matrix_cleaning(
        self, returns: np.ndarray, q_ratio: Optional[float] = None, **kwargs
    ) -> tuple:
        """
        Marchenko-Pastur random-matrix cleaning.

        Eigenvalues of the sample correlation that fall within the
        Marchenko-Pastur bulk (consistent with pure noise) are replaced
        with their average, preserving the trace.  The correlation
        matrix is then reconstructed from the cleaned spectrum.

        Args:
            returns: (T, n) returns matrix.
            q_ratio: T/n ratio. If None, computed from data.
        """
        T, n = returns.shape

        # Standardise to unit variance
        std = returns.std(axis=0)
        std_safe = np.where(std > 0, std, 1e-12)
        standardised = returns / std_safe

        # Sample correlation of standardised returns
        corr_sample = np.corrcoef(standardised, rowvar=False)
        np.fill_diagonal(corr_sample, 1.0)

        q = q_ratio if q_ratio is not None else T / n
        if q < 1:
            logger.warning("q = T/n = %.2f < 1: more assets than observations, "
                           "RMT cleaning may be unreliable", q)

        # Marchenko-Pastur bounds
        lambda_plus = (1 + 1 / np.sqrt(q)) ** 2
        lambda_minus = (1 - 1 / np.sqrt(q)) ** 2

        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(corr_sample)

        # Identify noise eigenvalues (within MP bulk)
        noise_mask = eigvals <= lambda_plus
        signal_mask = ~noise_mask

        n_noise = noise_mask.sum()
        n_signal = signal_mask.sum()
        logger.info("RMT cleaning: %d signal, %d noise eigenvalues "
                     "(MP upper bound: %.4f)", n_signal, n_noise, lambda_plus)

        # Replace noise eigenvalues with their average
        if n_noise > 0:
            noise_avg = eigvals[noise_mask].mean()
            eigvals_clean = eigvals.copy()
            eigvals_clean[noise_mask] = noise_avg

            # Rescale to preserve trace = n
            eigvals_clean *= n / eigvals_clean.sum()
        else:
            eigvals_clean = eigvals.copy()

        # Ensure positivity
        eigvals_clean = np.maximum(eigvals_clean, 1e-8)

        # Reconstruct correlation
        corr_clean = eigvecs @ np.diag(eigvals_clean) @ eigvecs.T

        # Force exact unit diagonal
        d = np.sqrt(np.diag(corr_clean))
        corr_clean = corr_clean / np.outer(d, d)
        np.fill_diagonal(corr_clean, 1.0)

        # Reconstruct covariance
        cov_sample = np.cov(returns, rowvar=False)
        std_sample = np.sqrt(np.diag(cov_sample))
        cov_clean = corr_clean * np.outer(std_sample, std_sample)

        return corr_clean, cov_clean

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _effective_observations(self, returns: np.ndarray) -> float:
        """
        Estimate effective number of observations accounting for
        first-order autocorrelation (Newey-West-style).
        """
        T, n = returns.shape
        # Average first-order autocorrelation across assets
        rho1_sum = 0.0
        for j in range(n):
            r = returns[:, j]
            mu = r.mean()
            c0 = np.sum((r - mu) ** 2) / T
            c1 = np.sum((r[1:] - mu) * (r[:-1] - mu)) / T
            rho1 = c1 / max(c0, 1e-12)
            rho1_sum += rho1
        rho1_avg = rho1_sum / n

        # Effective sample size adjustment
        rho1_clip = np.clip(rho1_avg, -0.99, 0.99)
        return T * (1 - rho1_clip) / (1 + rho1_clip)

    @staticmethod
    def nearest_positive_definite(A: np.ndarray) -> np.ndarray:
        """
        Find the nearest positive-definite matrix to A.

        Uses the Higham (2002) alternating projections algorithm.

        Args:
            A: Symmetric matrix that may not be PD.

        Returns:
            Nearest PD matrix in Frobenius norm.
        """
        B = (A + A.T) / 2.0
        _, s, V = np.linalg.svd(B)
        H = V.T @ np.diag(s) @ V
        A_pd = (B + H) / 2.0
        A_pd = (A_pd + A_pd.T) / 2.0

        # Iteratively ensure PD
        spacing = np.spacing(np.linalg.norm(A))
        k = 0
        while not _is_pd(A_pd):
            eigvals = np.linalg.eigvalsh(A_pd)
            min_eig = eigvals.min()
            A_pd += np.eye(A.shape[0]) * (-min_eig * (1 + spacing) + spacing)
            k += 1
            if k > 100:
                break

        return A_pd


def _is_pd(A: np.ndarray) -> bool:
    """Check if matrix is positive definite via Cholesky."""
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


# ============================================================================
# Utility functions
# ============================================================================

def rolling_correlation(
    returns: np.ndarray,
    window: int = 60,
) -> np.ndarray:
    """
    Compute rolling correlation matrices.

    Args:
        returns: (T, n) log-returns matrix.
        window: Rolling window size.

    Returns:
        Array of shape (T - window + 1, n, n) with rolling correlations.
    """
    T, n = returns.shape
    n_windows = T - window + 1

    if n_windows <= 0:
        raise ValueError(f"Window {window} exceeds data length {T}")

    result = np.zeros((n_windows, n, n))
    for i in range(n_windows):
        chunk = returns[i: i + window]
        corr = np.corrcoef(chunk, rowvar=False)
        np.fill_diagonal(corr, 1.0)
        result[i] = corr

    return result


def correlation_stability(
    returns: np.ndarray,
    windows: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Assess correlation stability across different estimation windows.

    Computes the mean absolute difference in off-diagonal correlations
    between each pair of window sizes.

    Args:
        returns: (T, n) log-returns matrix.
        windows: List of window sizes. Defaults to [20, 60, 120, 252].

    Returns:
        Dict with 'mean_abs_diff' (average instability), 'max_abs_diff',
        and per-pair differences.
    """
    if windows is None:
        windows = [20, 60, 120, 252]

    T = returns.shape[0]
    windows = [w for w in windows if w <= T]

    if len(windows) < 2:
        return {'mean_abs_diff': 0.0, 'max_abs_diff': 0.0}

    corrs = {}
    for w in windows:
        chunk = returns[-w:]
        c = np.corrcoef(chunk, rowvar=False)
        np.fill_diagonal(c, 0.0)
        corrs[w] = c

    diffs = {}
    all_diffs = []
    for i, w1 in enumerate(windows):
        for w2 in windows[i + 1:]:
            diff = np.abs(corrs[w1] - corrs[w2])
            mask = np.triu(np.ones_like(diff, dtype=bool), k=1)
            mad = diff[mask].mean()
            diffs[f"{w1}_vs_{w2}"] = mad
            all_diffs.append(diff[mask])

    all_diffs_flat = np.concatenate(all_diffs)

    return {
        'mean_abs_diff': float(all_diffs_flat.mean()),
        'max_abs_diff': float(all_diffs_flat.max()),
        **diffs,
    }
