"""
Rough Bergomi Model

Stochastic volatility model with rough (fractional) volatility dynamics.

Reference:
  Bayer, Friz, Gatheral (2016) - Pricing under rough volatility

author: Yunian Pan
email: yp1170@nyu.edu
"""

from typing import Tuple
import numpy as np

from .base import MultiFactorProcess, SimulationConfig


class RoughBergomi(MultiFactorProcess):
    """
    Rough Bergomi model (rBergomi).

    Variance process:
        v_t = xi0 * exp(eta * W_t^H - 0.5 * eta^2 * t^(2H))

    Asset process:
        dS_t = mu * S_t dt + sqrt(v_t) * S_t dW_t^S

    where dW^S is correlated with the Brownian motion that drives W^H.
    This implementation uses a Riemann-Liouville fractional Brownian motion
    approximation for W^H.
    """

    def __init__(
        self,
        mu: float,
        xi0: float,
        eta: float,
        rho: float,
        H: float,
        name: str = "RoughBergomi",
    ):
        super().__init__(dim=2, name=name)
        self.mu = float(mu)
        self.xi0 = float(xi0)
        self.eta = float(eta)
        self.rho = float(rho)
        self.H = float(H)

        self.params['mu'] = self.mu
        self.params['xi0'] = self.xi0
        self.params['eta'] = self.eta
        self.params['rho'] = self.rho
        self.params['H'] = self.H

    def drift(self, X: np.ndarray, t: float) -> np.ndarray:
        """Not used in rBergomi simulation."""
        return np.zeros_like(X)

    def diffusion(self, X: np.ndarray, t: float) -> np.ndarray:
        """Not used in rBergomi simulation."""
        return np.zeros_like(X)

    def simulate(
        self,
        X0: np.ndarray,
        T: float,
        config: SimulationConfig,
        scheme: str = "euler",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate rBergomi paths using a kernel-based RL-fBm approximation.

        Returns:
            t_grid, paths with shape (n_steps+1, n_paths, 2) for [S, v]
        """
        if config.random_seed is not None:
            np.random.seed(config.random_seed)

        n_steps = config.n_steps
        n_paths = config.n_paths
        dt = T / n_steps
        t_grid = np.linspace(0.0, T, n_steps + 1)

        # Initial state
        X0_arr = np.atleast_2d(X0)
        S0 = float(X0_arr[0, 0])
        v0 = self.xi0

        # Brownian increments
        dW1 = np.random.normal(0.0, np.sqrt(dt), size=(n_steps, n_paths))
        dW2 = np.random.normal(0.0, np.sqrt(dt), size=(n_steps, n_paths))

        # Fractional Brownian motion approximation (Riemann-Liouville)
        kernel = self._rl_kernel(t_grid, dt)
        W_H = kernel @ dW1

        # Variance path
        t_pow = t_grid**(2 * self.H)
        v_path = self.xi0 * np.exp(self.eta * W_H - 0.5 * self.eta**2 * t_pow[:, None])

        # Correlated Brownian increments for spot
        dW_S = self.rho * dW1 + np.sqrt(1.0 - self.rho**2) * dW2

        # Simulate log-spot
        log_S = np.zeros((n_steps + 1, n_paths))
        log_S[0] = np.log(S0)

        for i in range(n_steps):
            v_t = v_path[i]
            drift = (self.mu - 0.5 * v_t) * dt
            diffusion = np.sqrt(np.maximum(v_t, 0.0)) * dW_S[i]
            log_S[i + 1] = log_S[i] + drift + diffusion

        S_path = np.exp(log_S)

        # Combine into paths array [S, v]
        paths = np.zeros((n_steps + 1, n_paths, 2))
        paths[:, :, 0] = S_path
        paths[:, :, 1] = v_path

        return t_grid, paths

    def _rl_kernel(self, t_grid: np.ndarray, dt: float) -> np.ndarray:
        """
        Build Riemann-Liouville kernel matrix for fractional Brownian motion.

        W_t^H ≈ sqrt(2H) * sum_{j<=i} (t_i - t_j)^(H-1/2) * dW_j
        """
        n_steps = len(t_grid) - 1
        kernel = np.zeros((n_steps + 1, n_steps))

        coef = np.sqrt(2.0 * self.H)
        for i in range(1, n_steps + 1):
            t_i = t_grid[i]
            t_j = t_grid[:i]
            kernel[i, :i] = coef * np.power(t_i - t_j, self.H - 0.5)

        kernel *= 1.0
        return kernel
