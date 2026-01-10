"""
Geometric Brownian Motion (Black-Scholes model)

Single-factor diffusion process for modeling stock prices

author: Yunian Pan
email: yp1170@nyu.edu
"""
import numpy as np
from .base import DriftDiffusionProcess


class GBM(DriftDiffusionProcess):
    """
    Geometric Brownian Motion

    dS_t = mu * S_t dt + sigma * S_t dW_t

    The classic Black-Scholes model for stock prices.
    """

    def __init__(self, mu: float, sigma: float, name: str = "GBM"):
        """
        Initialize GBM

        Args:
            mu: Drift (expected return)
            sigma: Volatility
            name: Model name
        """
        super().__init__(name=name)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.params['mu'] = self.mu
        self.params['sigma'] = self.sigma

    def drift(self, X: np.ndarray, t: float) -> np.ndarray:
        """
        Drift term: mu * S_t

        Args:
            X: Current state, shape (n_paths, 1)
            t: Current time

        Returns:
            Drift coefficient, shape (n_paths, 1)
        """
        return self.mu * X

    def diffusion(self, X: np.ndarray, t: float) -> np.ndarray:
        """
        Diffusion term: sigma * S_t

        Args:
            X: Current state, shape (n_paths, 1)
            t: Current time

        Returns:
            Diffusion coefficient, shape (n_paths, 1)
        """
        return self.sigma * X

    def _exact_simulation(self, X0, T, dt, t_grid, config):
        """
        Exact simulation using analytical solution

        S_t = S_0 * exp((mu - sigma^2/2)*t + sigma*W_t)
        """
        n_paths = config.n_paths
        if config.antithetic:
            n_paths = n_paths // 2

        paths = np.zeros((len(t_grid), n_paths, self.dim))
        paths[0] = X0

        for i in range(1, len(t_grid)):
            t = t_grid[i]
            Z = np.random.normal(0, 1, size=(n_paths, self.dim))

            drift_term = (self.mu - 0.5 * self.sigma**2) * t
            diffusion_term = self.sigma * np.sqrt(t) * Z

            paths[i] = X0 * np.exp(drift_term + diffusion_term)

        if config.antithetic:
            paths_anti = np.zeros((len(t_grid), n_paths, self.dim))
            paths_anti[0] = X0

            if config.random_seed is not None:
                np.random.seed(config.random_seed)

            for i in range(1, len(t_grid)):
                t = t_grid[i]
                Z = -np.random.normal(0, 1, size=(n_paths, self.dim))

                drift_term = (self.mu - 0.5 * self.sigma**2) * t
                diffusion_term = self.sigma * np.sqrt(t) * Z

                paths_anti[i] = X0 * np.exp(drift_term + diffusion_term)

            paths = np.concatenate([paths, paths_anti], axis=1)

        return t_grid, paths

    def expectation(self, X0: np.ndarray, t: float) -> np.ndarray:
        """E[S_t | S_0] = S_0 * exp(mu * t)"""
        X0 = np.atleast_1d(X0)
        return X0 * np.exp(self.mu * t)

    def variance(self, X0: np.ndarray, t: float) -> np.ndarray:
        """Var[S_t | S_0] = S_0^2 * exp(2*mu*t) * (exp(sigma^2*t) - 1)"""
        X0 = np.atleast_1d(X0)
        return X0**2 * np.exp(2 * self.mu * t) * (np.exp(self.sigma**2 * t) - 1)

    def characteristic_function(self, u: complex, X0: np.ndarray, t: float) -> complex:
        """Characteristic function for log(S_t)"""
        X0_val = np.atleast_1d(X0)[0]
        log_S0 = np.log(X0_val)
        drift = (self.mu - 0.5 * self.sigma**2) * t
        variance = self.sigma**2 * t

        return np.exp(1j * u * (log_S0 + drift) - 0.5 * u**2 * variance)
