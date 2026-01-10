"""
Rough Bergomi (rBergomi) Calibration - Physical Measure

Estimate rBergomi parameters from historical price data.

This is a pragmatic calibration approach using:
- log returns for drift
- rolling realized variance for variance level
- log-variance variogram to estimate H and eta
- leverage proxy for rho

author: Yunian Pan
email: yp1170@nyu.edu
"""

from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np


@dataclass
class RoughBergomiCalibrationResult:
    """Calibration result for rBergomi (physical measure)."""

    mu: float
    xi0: float
    eta: float
    rho: float
    H: float

    n_observations: int
    dt: float
    window: int
    max_lag: int

    diagnostics: Dict[str, float]

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "Rough Bergomi Calibration Result (Physical Measure)",
            "=" * 60,
            "",
            "Parameters:",
            f"  mu   = {self.mu:8.4f}",
            f"  xi0  = {self.xi0:8.6f}",
            f"  eta  = {self.eta:8.4f}",
            f"  rho  = {self.rho:8.4f}",
            f"  H    = {self.H:8.4f}",
            "",
            "Diagnostics:",
            f"  n_obs       = {self.n_observations}",
            f"  dt          = {self.dt:.6f}",
            f"  window      = {self.window}",
            f"  max_lag     = {self.max_lag}",
        ]
        if self.diagnostics:
            lines.append("")
            lines.append("  variogram_r2 = {:.4f}".format(self.diagnostics.get('variogram_r2', float('nan'))))
            lines.append("  mean_return  = {:.6f}".format(self.diagnostics.get('mean_return', float('nan'))))
            lines.append("  mean_var     = {:.6f}".format(self.diagnostics.get('mean_var', float('nan'))))
        lines.append("=" * 60)
        return "\n".join(lines)


class RoughBergomiCalibrator:
    """
    Calibrate rBergomi parameters from historical prices.

    Notes:
    - This is a lightweight estimator meant for initialization.
    - For production use, consider more robust ML or Bayesian methods.
    """

    def __init__(
        self,
        window: int = 20,
        max_lag: int = 10,
    ):
        self.window = window
        self.max_lag = max_lag

    def fit(
        self,
        prices: np.ndarray,
        dt: float = 1.0 / 252.0,
    ) -> RoughBergomiCalibrationResult:
        prices = np.asarray(prices).flatten()
        if len(prices) < self.window + self.max_lag + 2:
            raise ValueError("Insufficient data for rough Bergomi calibration")
        if np.any(prices <= 0):
            raise ValueError("All prices must be positive")

        log_prices = np.log(prices)
        log_returns = np.diff(log_prices)
        n_obs = len(log_returns)

        # Drift estimate
        mu = np.mean(log_returns) / dt

        # Realized variance proxy
        rv = self._rolling_realized_variance(log_returns, dt, self.window)
        rv = rv[~np.isnan(rv)]
        if len(rv) < self.max_lag + 2:
            raise ValueError("Insufficient variance data after rolling window")

        # Log-variance series
        log_var = np.log(rv)

        # Variogram regression to estimate H and eta
        H, eta, r2 = self._estimate_rough_parameters(log_var, dt, self.max_lag)

        # Leverage effect proxy for rho
        delta_log_var = np.diff(log_var)
        aligned_returns = log_returns[-len(delta_log_var):]
        if len(aligned_returns) == 0:
            rho = 0.0
        else:
            rho = np.corrcoef(aligned_returns, delta_log_var)[0, 1]
            if np.isnan(rho):
                rho = 0.0

        xi0 = float(rv[0])

        diagnostics = {
            'variogram_r2': r2,
            'mean_return': float(np.mean(log_returns)),
            'mean_var': float(np.mean(rv)),
        }

        return RoughBergomiCalibrationResult(
            mu=mu,
            xi0=xi0,
            eta=eta,
            rho=rho,
            H=H,
            n_observations=n_obs,
            dt=dt,
            window=self.window,
            max_lag=self.max_lag,
            diagnostics=diagnostics,
        )

    def _rolling_realized_variance(
        self,
        returns: np.ndarray,
        dt: float,
        window: int,
    ) -> np.ndarray:
        rv = np.full(len(returns), np.nan)
        for i in range(window - 1, len(returns)):
            window_returns = returns[i - window + 1 : i + 1]
            rv[i] = np.mean(window_returns**2) / dt
        return rv

    def _estimate_rough_parameters(
        self,
        log_var: np.ndarray,
        dt: float,
        max_lag: int,
    ) -> tuple:
        lags = []
        variograms = []

        for lag in range(1, max_lag + 1):
            diffs = log_var[lag:] - log_var[:-lag]
            if len(diffs) < 2:
                continue
            var = np.var(diffs, ddof=1)
            if var <= 0:
                continue
            lags.append(lag * dt)
            variograms.append(var)

        if len(lags) < 2:
            return 0.1, 1.0, float('nan')

        x = np.log(np.array(lags))
        y = np.log(np.array(variograms))

        A = np.vstack([x, np.ones_like(x)]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        slope, intercept = coeffs

        H = max(0.01, min(0.49, 0.5 * slope))
        eta = float(np.exp(0.5 * intercept))

        y_pred = A @ coeffs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

        return H, eta, r2
