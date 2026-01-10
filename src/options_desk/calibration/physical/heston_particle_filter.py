"""
Particle Filter for Heston Model (Physical Measure)

State:
    v_t (variance)
Observation:
    r_t = log(S_t / S_{t-1}) ~ N((mu - 0.5 v_t) dt, v_t dt)

author: Yunian Pan
email: yp1170@nyu.edu
"""

from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np


@dataclass
class HestonParticleFilterResult:
    """Results of Heston particle filtering."""

    filtered_variance: np.ndarray
    log_likelihood: float
    effective_sample_size: np.ndarray

    params: Dict[str, float]
    n_particles: int
    dt: float


class HestonParticleFilter:
    """
    Bootstrap particle filter for the Heston stochastic volatility model.
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
        kappa: float,
        theta: float,
        xi: float,
        v0: Optional[float] = None,
        random_seed: Optional[int] = None,
    ) -> HestonParticleFilterResult:
        prices = np.asarray(prices).flatten()
        if len(prices) < 2:
            raise ValueError("Need at least 2 prices to filter")
        if np.any(prices <= 0):
            raise ValueError("All prices must be positive")

        if random_seed is not None:
            np.random.seed(random_seed)

        returns = np.diff(np.log(prices))
        n_steps = len(returns)

        v_init = float(theta if v0 is None else v0)
        v_particles = np.full(self.n_particles, v_init, dtype=float)

        filtered_variance = np.zeros(n_steps)
        ess_series = np.zeros(n_steps)
        log_likelihood = 0.0

        for t in range(n_steps):
            # Propagate variance particles
            z = np.random.normal(0.0, 1.0, size=self.n_particles)
            v_particles = (
                v_particles
                + kappa * (theta - v_particles) * dt
                + xi * np.sqrt(np.maximum(v_particles, 0.0)) * np.sqrt(dt) * z
            )
            v_particles = np.maximum(v_particles, self.variance_floor)

            # Observation likelihood
            mean = (mu - 0.5 * v_particles) * dt
            var = np.maximum(v_particles * dt, self.variance_floor)
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

            # Filtered variance (posterior mean)
            filtered_variance[t] = np.sum(w_norm * v_particles)

            # Resample if needed
            if ess < self.resample_threshold * self.n_particles:
                idx = self._systematic_resample(w_norm)
                v_particles = v_particles[idx]

        return HestonParticleFilterResult(
            filtered_variance=filtered_variance,
            log_likelihood=log_likelihood,
            effective_sample_size=ess_series,
            params={
                'mu': mu,
                'kappa': kappa,
                'theta': theta,
                'xi': xi,
                'v0': v_init,
            },
            n_particles=self.n_particles,
            dt=dt,
        )

    def _systematic_resample(self, weights: np.ndarray) -> np.ndarray:
        n = len(weights)
        positions = (np.random.rand() + np.arange(n)) / n
        cumulative_sum = np.cumsum(weights)
        idx = np.searchsorted(cumulative_sum, positions)
        return idx
