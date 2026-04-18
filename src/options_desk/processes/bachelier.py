"""
Bachelier (Arithmetic/Normal) Model

Additive diffusion process for rates, spreads, and assets that can go negative

    dS_t = mu * dt + sigma * dW_t

Unlike GBM, the Bachelier model uses absolute (normal) volatility, not
relative (lognormal) volatility.  The process is Gaussian and unbounded
below, making it the natural choice for interest-rate spreads, basis
trades, and any underlying that can take negative values.

author: Yunian Pan
email: yp1170@nyu.edu
"""
import numpy as np
from .base import DriftDiffusionProcess


class Bachelier(DriftDiffusionProcess):
    """
    Bachelier (Arithmetic Brownian Motion) Process

    dS_t = mu * dt + sigma * dW_t

    where:
        - S_t: state variable (rate, spread, price)
        - mu: constant drift
        - sigma: normal (absolute) volatility (sigma > 0)

    Properties:
        - Gaussian: X_t | X_0 ~ N(X_0 + mu*t, sigma^2 * t)
        - Can go negative (suitable for rates, spreads)
        - Constant volatility in absolute terms
        - Linear expectation in time

    Uses:
        - Interest rate modeling
        - Credit spread modeling
        - Basis and calendar spread trading
        - Normal implied volatility quoting
    """

    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 1.0,
        name: str = "Bachelier"
    ):
        """
        Initialize Bachelier process

        Args:
            mu: Constant drift
            sigma: Normal (absolute) volatility (sigma > 0)
            name: Model name
        """
        super().__init__(name=name)
        self.mu = float(mu)
        self.sigma = float(sigma)

        self.params['mu'] = self.mu
        self.params['sigma'] = self.sigma

    def _build_jax_spec(self):
        from ._process_defs import BachelierParams, bachelier_drift, bachelier_diffusion
        return {
            'drift_fn': bachelier_drift,
            'diffusion_fn': bachelier_diffusion,
            'params': BachelierParams(mu=self.mu, sigma=self.sigma),
            'dim': 1,
        }

    def drift(self, X: np.ndarray, t: float) -> np.ndarray:
        """
        Drift term: mu (constant, NOT proportional to X)

        Args:
            X: Current state, shape (n_paths, 1)
            t: Current time

        Returns:
            Drift coefficient, shape (n_paths, 1)
        """
        return self.mu * np.ones_like(X)

    def diffusion(self, X: np.ndarray, t: float) -> np.ndarray:
        """
        Diffusion term: sigma (constant, NOT proportional to X)

        Args:
            X: Current state, shape (n_paths, 1)
            t: Current time

        Returns:
            Diffusion coefficient, shape (n_paths, 1)
        """
        return self.sigma * np.ones_like(X)

    def diffusion_derivative(self, X: np.ndarray, t: float) -> np.ndarray:
        """
        Derivative of diffusion w.r.t. X: d(sigma)/dX = 0

        Milstein correction vanishes for additive noise.

        Args:
            X: Current state, shape (n_paths, 1)
            t: Current time

        Returns:
            Zero array, shape (n_paths, 1)
        """
        return np.zeros_like(X)

    def _exact_simulation(self, X0, T, dt, t_grid, config):
        """
        Exact simulation using analytical solution

        X_t = X_0 + mu * t + sigma * W_t

        where W_t ~ N(0, t)
        """
        n_paths = config.n_paths
        if config.antithetic:
            n_paths = n_paths // 2

        paths = np.zeros((len(t_grid), n_paths, self.dim))
        paths[0] = X0

        for i in range(1, len(t_grid)):
            t = t_grid[i]
            Z = np.random.normal(0, 1, size=(n_paths, self.dim))

            drift_term = self.mu * t
            diffusion_term = self.sigma * np.sqrt(t) * Z

            paths[i] = X0 + drift_term + diffusion_term

        if config.antithetic:
            paths_anti = np.zeros((len(t_grid), n_paths, self.dim))
            paths_anti[0] = X0

            if config.random_seed is not None:
                np.random.seed(config.random_seed)

            for i in range(1, len(t_grid)):
                t = t_grid[i]
                Z = -np.random.normal(0, 1, size=(n_paths, self.dim))

                drift_term = self.mu * t
                diffusion_term = self.sigma * np.sqrt(t) * Z

                paths_anti[i] = X0 + drift_term + diffusion_term

            paths = np.concatenate([paths, paths_anti], axis=1)

        return t_grid, paths

    def expectation(self, X0: np.ndarray, t: float) -> np.ndarray:
        """
        Expected value E[X_t | X_0]

        E[X_t] = X_0 + mu * t

        Args:
            X0: Initial value
            t: Time

        Returns:
            Expected value
        """
        X0 = np.atleast_1d(X0)
        return X0 + self.mu * t

    def variance(self, X0: np.ndarray, t: float) -> np.ndarray:
        """
        Variance Var[X_t | X_0]

        Var[X_t] = sigma^2 * t

        Note: Variance does not depend on X_0.

        Args:
            X0: Initial value (unused, kept for API consistency)
            t: Time

        Returns:
            Variance
        """
        X0 = np.atleast_1d(X0)
        return self.sigma**2 * t * np.ones_like(X0)

    def characteristic_function(self, u: complex, X0: np.ndarray, t: float) -> complex:
        """
        Characteristic function phi(u) = E[exp(i*u*X_t) | X_0]

        For Bachelier (Gaussian) process:
            phi(u) = exp(i*u*(X_0 + mu*t) - 0.5*u^2*sigma^2*t)

        Args:
            u: Frequency parameter
            X0: Initial value
            t: Time

        Returns:
            Characteristic function value
        """
        X0_val = np.atleast_1d(X0)[0]
        mean = X0_val + self.mu * t
        var = self.sigma**2 * t

        return np.exp(1j * u * mean - 0.5 * u**2 * var)
