"""
GARCH(1,1) Calibration (Physical Measure)

Model:
    r_t = mu + eps_t
    eps_t = sigma_t z_t, z_t ~ N(0,1)
    sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2

author: Yunian Pan
email: yp1170@nyu.edu
"""

from dataclasses import dataclass
import numpy as np
from scipy import optimize


@dataclass
class GARCHCalibrationResult:
    """Result of GARCH(1,1) calibration."""

    mu: float
    omega: float
    alpha: float
    beta: float

    log_likelihood: float
    aic: float
    bic: float
    n_observations: int
    dt: float

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "GARCH(1,1) Calibration Result",
            "=" * 60,
            f"mu    = {self.mu:.6f}",
            f"omega = {self.omega:.6f}",
            f"alpha = {self.alpha:.6f}",
            f"beta  = {self.beta:.6f}",
            "",
            f"logL = {self.log_likelihood:.2f}",
            f"AIC  = {self.aic:.2f}",
            f"BIC  = {self.bic:.2f}",
            "=" * 60,
        ]
        return "\n".join(lines)


class GARCHCalibrator:
    """
    QMLE calibration for GARCH(1,1) using Gaussian likelihood.
    """

    def fit(self, prices: np.ndarray, dt: float = 1.0 / 252.0) -> GARCHCalibrationResult:
        prices = np.asarray(prices).flatten()
        if len(prices) < 3:
            raise ValueError("Need at least 3 price observations")
        if np.any(prices <= 0):
            raise ValueError("All prices must be positive")

        returns = np.diff(np.log(prices))
        n = len(returns)

        mu_init = np.mean(returns)
        var_init = np.var(returns, ddof=1)
        omega_init = 0.1 * var_init
        alpha_init = 0.05
        beta_init = 0.9

        x0 = np.array([mu_init, omega_init, alpha_init, beta_init])

        bounds = [
            (None, None),      # mu
            (1e-12, None),     # omega
            (1e-6, 1.0),       # alpha
            (1e-6, 1.0),       # beta
        ]

        def neg_log_likelihood(params: np.ndarray) -> float:
            mu, omega, alpha, beta = params
            if omega <= 0 or alpha <= 0 or beta <= 0 or alpha + beta >= 0.999:
                return 1e12

            eps = returns - mu
            sigma2 = np.zeros(n)
            sigma2[0] = var_init
            for t in range(1, n):
                sigma2[t] = omega + alpha * eps[t - 1]**2 + beta * sigma2[t - 1]
                sigma2[t] = max(sigma2[t], 1e-12)

            ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + eps**2 / sigma2)
            return -ll

        result = optimize.minimize(
            neg_log_likelihood,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
        )

        if not result.success:
            params = x0
            log_likelihood = -neg_log_likelihood(params)
        else:
            params = result.x
            log_likelihood = -result.fun

        mu, omega, alpha, beta = params
        k = 4
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood

        return GARCHCalibrationResult(
            mu=float(mu),
            omega=float(omega),
            alpha=float(alpha),
            beta=float(beta),
            log_likelihood=float(log_likelihood),
            aic=float(aic),
            bic=float(bic),
            n_observations=n,
            dt=dt,
        )
