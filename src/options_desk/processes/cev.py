"""
Constant Elasticity of Variance (CEV) Model

Generalization of GBM with state-dependent volatility

author: Yunian Pan
email: yp1170@nyu.edu
"""
import numpy as np
from .base import DriftDiffusionProcess


class CEV(DriftDiffusionProcess):
    """
    Constant Elasticity of Variance (CEV) Model

    dS_t = mu * S_t dt + sigma * S_t^beta * dW_t

    where:
        - S_t: asset price
        - mu: drift (expected return)
        - sigma: volatility coefficient
        - beta: elasticity parameter (controls how vol depends on price)
          * beta = 0: Bachelier (normal) model, constant volatility
          * beta = 0.5: Popular choice, volatility ~ sqrt(S)
          * beta = 1: GBM (lognormal) model, volatility ~ S

    Properties:
        - Generalizes GBM by allowing elasticity in volatility
        - beta < 1: Leverage effect (vol increases when price falls)
        - beta > 1: Inverse leverage (vol increases when price rises)
        - beta = 1: Reduces to standard GBM

    Uses:
        - Equity options (captures volatility smile/skew)
        - Leverage effect modeling
        - Alternative to stochastic volatility models
    """

    def __init__(
        self,
        mu: float,
        sigma: float,
        beta: float,
        name: str = "CEV"
    ):
        """
        Initialize CEV model

        Args:
            mu: Drift (expected return)
            sigma: Volatility coefficient (sigma > 0)
            beta: Elasticity parameter (typically 0 <= beta <= 1)
            name: Model name
        """
        super().__init__(name=name)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.beta = float(beta)

        self.params['mu'] = self.mu
        self.params['sigma'] = self.sigma
        self.params['beta'] = self.beta

        # Special cases
        if abs(self.beta) < 1e-10:
            self.params['model_type'] = 'Bachelier (normal)'
        elif abs(self.beta - 1.0) < 1e-10:
            self.params['model_type'] = 'GBM (lognormal)'
        else:
            self.params['model_type'] = 'CEV'

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
        Diffusion term: sigma * S_t^beta

        Args:
            X: Current state, shape (n_paths, 1)
            t: Current time

        Returns:
            Diffusion coefficient, shape (n_paths, 1)
        """
        if abs(self.beta) < 1e-10:
            # beta = 0: constant volatility
            return self.sigma * np.ones_like(X)
        elif abs(self.beta - 1.0) < 1e-10:
            # beta = 1: GBM
            return self.sigma * X
        else:
            # General case: handle potential negative values for beta < 1
            return self.sigma * np.sign(X) * np.abs(X)**self.beta

    def _milstein(self, X0, T, dt, t_grid, config):
        """
        Milstein scheme for CEV with correction term

        For CEV: sigma * S^beta * d(sigma * S^beta)/dS * (dW^2 - dt)
               = sigma^2 * beta * S^(2*beta - 1) * (dW^2 - dt)
        """
        n_paths = config.n_paths
        if config.antithetic:
            n_paths = n_paths // 2

        paths = np.zeros((len(t_grid), n_paths, self.dim))
        paths[0] = X0
        sqrt_dt = np.sqrt(dt)

        for i, t in enumerate(t_grid[:-1]):
            X_current = paths[i]

            # Generate Brownian increments
            dW = np.random.normal(0, sqrt_dt, size=(n_paths, self.dim))

            # Euler terms
            drift_term = self.drift(X_current, t) * dt
            sigma = self.diffusion(X_current, t)
            diffusion_term = sigma * dW

            # Milstein correction
            if abs(self.beta) < 1e-10 or abs(self.beta - 1.0) < 1e-10:
                # beta = 0 or beta = 1: no Milstein correction needed
                milstein_correction = np.zeros((n_paths, self.dim))
            else:
                # General case: sigma^2 * beta * S^(2*beta - 1) * (dW^2 - dt)
                X_power = np.sign(X_current) * np.abs(X_current)**(2 * self.beta - 1)
                milstein_correction = self.sigma**2 * self.beta * X_power * (dW**2 - dt)

            paths[i + 1] = X_current + drift_term + diffusion_term + milstein_correction

        # Antithetic paths
        if config.antithetic:
            paths_anti = np.zeros((len(t_grid), n_paths, self.dim))
            paths_anti[0] = X0

            if config.random_seed is not None:
                np.random.seed(config.random_seed)

            for i, t in enumerate(t_grid[:-1]):
                X_current = paths_anti[i]

                dW = -np.random.normal(0, sqrt_dt, size=(n_paths, self.dim))

                drift_term = self.drift(X_current, t) * dt
                sigma = self.diffusion(X_current, t)
                diffusion_term = sigma * dW

                if abs(self.beta) < 1e-10 or abs(self.beta - 1.0) < 1e-10:
                    milstein_correction = np.zeros((n_paths, self.dim))
                else:
                    X_power = np.sign(X_current) * np.abs(X_current)**(2 * self.beta - 1)
                    milstein_correction = self.sigma**2 * self.beta * X_power * (dW**2 - dt)

                paths_anti[i + 1] = X_current + drift_term + diffusion_term + milstein_correction

            paths = np.concatenate([paths, paths_anti], axis=1)

        return t_grid, paths

    def _exact_simulation(self, X0, T, dt, t_grid, config):
        """
        Exact simulation for special case beta = 1 (GBM)

        For beta = 1: S_t = S_0 * exp((mu - 0.5*sigma^2)*t + sigma*W_t)
        """
        if abs(self.beta - 1.0) < 1e-10:
            # Beta = 1: Use GBM exact simulation
            n_paths = config.n_paths
            if config.antithetic:
                n_paths = n_paths // 2

            paths = np.zeros((len(t_grid), n_paths, self.dim))
            paths[0] = X0

            drift_adjusted = self.mu - 0.5 * self.sigma**2

            for i, t in enumerate(t_grid[1:], 1):
                sqrt_t = np.sqrt(t)
                Z = np.random.normal(0, 1, (n_paths, self.dim))
                W_t = sqrt_t * Z

                exponent = drift_adjusted * t + self.sigma * W_t
                paths[i] = X0 * np.exp(exponent)

            # Antithetic variates
            if config.antithetic:
                paths_anti = np.zeros((len(t_grid), n_paths, self.dim))
                paths_anti[0] = X0

                if config.random_seed is not None:
                    np.random.seed(config.random_seed)

                for i, t in enumerate(t_grid[1:], 1):
                    sqrt_t = np.sqrt(t)
                    Z = np.random.normal(0, 1, (n_paths, self.dim))
                    W_t = -sqrt_t * Z  # Antithetic

                    exponent = drift_adjusted * t + self.sigma * W_t
                    paths_anti[i] = X0 * np.exp(exponent)

                paths = np.concatenate([paths, paths_anti], axis=1)

            return t_grid, paths
        else:
            # For other beta values, no closed-form solution
            raise NotImplementedError(
                f"Exact simulation only available for beta=1 (GBM). Current beta={self.beta}"
            )

    def expectation(self, X0: np.ndarray, t: float) -> np.ndarray:
        """
        Expected value E[S_t | S_0]

        For beta = 1 (GBM): E[S_t] = S_0 * exp(mu * t)
        For general beta: no closed form

        Args:
            X0: Initial value
            t: Time

        Returns:
            Expected value (only for beta=1)
        """
        if abs(self.beta - 1.0) < 1e-10:
            return X0 * np.exp(self.mu * t)
        else:
            raise NotImplementedError(
                f"Analytical expectation not available for beta={self.beta}"
            )

    def variance(self, X0: np.ndarray, t: float) -> np.ndarray:
        """
        Variance Var[S_t]

        For beta = 1 (GBM): Var[S_t] = S_0^2 * exp(2*mu*t) * (exp(sigma^2*t) - 1)
        For general beta: no closed form

        Args:
            X0: Initial value
            t: Time

        Returns:
            Variance (only for beta=1)
        """
        if abs(self.beta - 1.0) < 1e-10:
            return X0**2 * np.exp(2 * self.mu * t) * (np.exp(self.sigma**2 * t) - 1.0)
        else:
            raise NotImplementedError(
                f"Analytical variance not available for beta={self.beta}"
            )
