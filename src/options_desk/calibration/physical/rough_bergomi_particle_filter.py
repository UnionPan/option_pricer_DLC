"""
Particle Filter for Rough Bergomi Model (Physical Measure)

State:
    W_t^H (fractional Brownian motion path via Riemann-Liouville kernel)
    v_t = xi0 * exp(eta * W_t^H - 0.5 * eta^2 * t^(2H))
Observation:
    r_t = log(S_t / S_{t-1}) ~ N((mu - 0.5 v_t) dt, v_t dt)

author: Yunian Pan
email: yp1170@nyu.edu
"""

from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np


@dataclass
class RoughBergomiParticleFilterResult:
    """Results of rBergomi particle filtering."""

    filtered_variance: np.ndarray
    log_likelihood: float
    effective_sample_size: np.ndarray

    params: Dict[str, float]
    n_particles: int
    dt: float


class RoughBergomiParticleFilter:
    """
    Bootstrap particle filter for rBergomi using RL-fBm approximation.
    """

    def __init__(
        self,
        n_particles: int = 2000,
        resample_threshold: float = 0.5,
        variance_floor: float = 1e-8,
    ):
        self.n_particles = n_particles
        self.resample_threshold = resample_threshold
        self.variance_floor = variance_floor

    def filter(
        self,
        prices: np.ndarray,
        dt: float,
        mu: float,
        xi0: float,
        eta: float,
        H: float,
        rho: float = 0.0,
        random_seed: Optional[int] = None,
    ) -> RoughBergomiParticleFilterResult:
        prices = np.asarray(prices).flatten()
        if len(prices) < 2:
            raise ValueError("Need at least 2 prices to filter")
        if np.any(prices <= 0):
            raise ValueError("All prices must be positive")

        if random_seed is not None:
            np.random.seed(random_seed)

        returns = np.diff(np.log(prices))
        n_steps = len(returns)
        t_grid = np.linspace(0.0, n_steps * dt, n_steps + 1)

        kernel = self._rl_kernel(t_grid, H)
        dW1_hist = np.zeros((self.n_particles, n_steps))

        filtered_variance = np.zeros(n_steps)
        ess_series = np.zeros(n_steps)
        log_likelihood = 0.0

        for t in range(n_steps):
            # Sample new Brownian increments for the variance driver
            dW1_hist[:, t] = np.random.normal(0.0, np.sqrt(dt), size=self.n_particles)

            # Compute W_H at time t+1 for each particle
            W_H = dW1_hist[:, : t + 1] @ kernel[t + 1, : t + 1]

            t_pow = t_grid[t + 1] ** (2.0 * H)
            v_t = xi0 * np.exp(eta * W_H - 0.5 * eta * eta * t_pow)
            v_t = np.maximum(v_t, self.variance_floor)

            # Observation likelihood
            mean = (mu - 0.5 * v_t) * dt
            var = np.maximum(v_t * dt, self.variance_floor)
            diff = returns[t] - mean
            log_w = -0.5 * (np.log(2.0 * np.pi * var) + (diff * diff) / var)

            max_log_w = np.max(log_w)
            w = np.exp(log_w - max_log_w)
            mean_w = np.mean(w)
            log_likelihood += np.log(mean_w) + max_log_w

            w_sum = np.sum(w)
            w_norm = w / w_sum
            ess = 1.0 / np.sum(w_norm**2)
            ess_series[t] = ess

            filtered_variance[t] = np.sum(w_norm * v_t)

            if ess < self.resample_threshold * self.n_particles:
                idx = self._systematic_resample(w_norm)
                dW1_hist = dW1_hist[idx]

        return RoughBergomiParticleFilterResult(
            filtered_variance=filtered_variance,
            log_likelihood=log_likelihood,
            effective_sample_size=ess_series,
            params={
                'mu': mu,
                'xi0': xi0,
                'eta': eta,
                'H': H,
                'rho': rho,
            },
            n_particles=self.n_particles,
            dt=dt,
        )

    def _rl_kernel(self, t_grid: np.ndarray, H: float) -> np.ndarray:
        n_steps = len(t_grid) - 1
        kernel = np.zeros((n_steps + 1, n_steps))
        coef = np.sqrt(2.0 * H)

        for i in range(1, n_steps + 1):
            t_i = t_grid[i]
            t_j = t_grid[:i]
            kernel[i, :i] = coef * np.power(t_i - t_j, H - 0.5)

        return kernel

    def _systematic_resample(self, weights: np.ndarray) -> np.ndarray:
        n = len(weights)
        positions = (np.random.rand() + np.arange(n)) / n
        cumulative_sum = np.cumsum(weights)
        idx = np.searchsorted(cumulative_sum, positions)
        return idx
