"""
Merton Jump-Diffusion Calibration (Physical Measure)

Log return model (Merton, 1976):
    r_t = log(S_t / S_{t-1}) ~ sum of diffusion + jump components

Diffusion:
    dS_t / S_t = mu dt + sigma dW_t

Jumps:
    N_t ~ Poisson(lambda dt)
    J ~ Normal(mu_j, sigma_j^2)

Log-return distribution is a Poisson mixture of normals.

author: Yunian Pan
email: yp1170@nyu.edu
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy import optimize, stats


@dataclass
class MertonCalibrationResult:
    """Result of Merton jump-diffusion calibration."""

    mu: float
    sigma: float
    lambda_: float
    mu_j: float
    sigma_j: float

    log_likelihood: float
    aic: float
    bic: float
    n_observations: int
    dt: float

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "Merton Jump-Diffusion Calibration Result",
            "=" * 60,
            f"mu      = {self.mu:.6f}",
            f"sigma   = {self.sigma:.6f}",
            f"lambda  = {self.lambda_:.6f}",
            f"mu_j    = {self.mu_j:.6f}",
            f"sigma_j = {self.sigma_j:.6f}",
            "",
            f"logL = {self.log_likelihood:.2f}",
            f"AIC  = {self.aic:.2f}",
            f"BIC  = {self.bic:.2f}",
            "=" * 60,
        ]
        return "\n".join(lines)


class MertonJumpCalibrator:
    """
    MLE calibration for the Merton jump-diffusion model.
    """

    def __init__(self, k_max: int = 5):
        """
        Args:
            k_max: Truncation level for Poisson mixture
        """
        self.k_max = k_max

    def fit(self, prices: np.ndarray, dt: float = 1.0 / 252.0) -> MertonCalibrationResult:
        prices = np.asarray(prices).flatten()
        if len(prices) < 2:
            raise ValueError("Need at least 2 price observations")
        if np.any(prices <= 0):
            raise ValueError("All prices must be positive")

        returns = np.diff(np.log(prices))
        n = len(returns)

        mu_init = np.mean(returns) / dt
        sigma_init = np.std(returns, ddof=1) / np.sqrt(dt)
        lambda_init = 0.1
        mu_j_init = 0.0
        sigma_j_init = 0.05

        x0 = np.array([mu_init, sigma_init, lambda_init, mu_j_init, sigma_j_init])

        bounds = [
            (None, None),      # mu
            (1e-6, None),      # sigma
            (1e-6, None),      # lambda
            (None, None),      # mu_j
            (1e-6, None),      # sigma_j
        ]

        def neg_log_likelihood(params: np.ndarray) -> float:
            mu, sigma, lambda_, mu_j, sigma_j = params
            if sigma <= 0 or lambda_ <= 0 or sigma_j <= 0:
                return 1e12

            kappa = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
            base_mean = (mu - 0.5 * sigma**2 - lambda_ * kappa) * dt
            base_var = sigma**2 * dt

            loglik = 0.0
            for r in returns:
                mix_prob = 0.0
                for k in range(self.k_max + 1):
                    weight = stats.poisson.pmf(k, lambda_ * dt)
                    mean_k = base_mean + k * mu_j
                    var_k = base_var + k * sigma_j**2
                    mix_prob += weight * stats.norm.pdf(r, loc=mean_k, scale=np.sqrt(var_k))
                mix_prob = max(mix_prob, 1e-300)
                loglik += np.log(mix_prob)
            return -loglik

        result = optimize.minimize(
            neg_log_likelihood,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
        )

        if not result.success:
            # Fall back to initial guesses if optimization fails
            params = x0
            log_likelihood = -neg_log_likelihood(params)
        else:
            params = result.x
            log_likelihood = -result.fun

        mu, sigma, lambda_, mu_j, sigma_j = params
        k = 5
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood

        return MertonCalibrationResult(
            mu=float(mu),
            sigma=float(sigma),
            lambda_=float(lambda_),
            mu_j=float(mu_j),
            sigma_j=float(sigma_j),
            log_likelihood=float(log_likelihood),
            aic=float(aic),
            bic=float(bic),
            n_observations=n,
            dt=dt,
        )
