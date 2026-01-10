"""
Ornstein-Uhlenbeck Process

Mean-reverting process for interest rates, volatilities, and spreads

author: Yunian Pan
email: yp1170@nyu.edu
"""
import numpy as np
from .base import DriftDiffusionProcess


class OrnsteinUhlenbeck(DriftDiffusionProcess):
    """
    Ornstein-Uhlenbeck (OU) Process

    dX_t = theta * (mu - X_t) dt + sigma * dW_t

    where:
        - X_t: state variable (interest rate, spread, etc.)
        - theta: mean reversion speed (theta > 0)
        - mu: long-term mean (equilibrium level)
        - sigma: volatility (sigma > 0)

    Properties:
        - Mean-reverting: pulls toward mu at rate theta
        - Gaussian: can go negative (suitable for rates, spreads)
        - Stationary distribution: N(mu, sigma^2 / (2*theta))
        - Half-life of mean reversion: ln(2) / theta

    Uses:
        - Interest rate models (Vasicek)
        - Volatility models
        - Pair trading (spread modeling)
        - Commodity prices
    """

    def __init__(
        self,
        theta: float,
        mu: float,
        sigma: float,
        name: str = "OrnsteinUhlenbeck"
    ):
        """
        Initialize Ornstein-Uhlenbeck process

        Args:
            theta: Mean reversion speed (theta > 0)
            mu: Long-term mean
            sigma: Volatility (sigma > 0)
            name: Model name
        """
        super().__init__(name=name)
        self.theta = float(theta)
        self.mu = float(mu)
        self.sigma = float(sigma)

        self.params['theta'] = self.theta
        self.params['mu'] = self.mu
        self.params['sigma'] = self.sigma

        # Half-life of mean reversion
        self.half_life = np.log(2) / self.theta
        self.params['half_life'] = self.half_life

    def drift(self, X: np.ndarray, t: float) -> np.ndarray:
        """
        Drift term: theta * (mu - X_t)

        Args:
            X: Current state, shape (n_paths, 1)
            t: Current time

        Returns:
            Drift coefficient, shape (n_paths, 1)
        """
        return self.theta * (self.mu - X)

    def diffusion(self, X: np.ndarray, t: float) -> np.ndarray:
        """
        Diffusion term: sigma (constant)

        Args:
            X: Current state, shape (n_paths, 1)
            t: Current time

        Returns:
            Diffusion coefficient, shape (n_paths, 1)
        """
        n_paths = X.shape[0]
        return self.sigma * np.ones((n_paths, self.dim))

    def _exact_simulation(self, X0, T, dt, t_grid, config):
        """
        Exact simulation using analytical solution

        X_t = X_0 * exp(-theta*t) + mu*(1 - exp(-theta*t)) + sigma * Z

        where Z ~ N(0, sigma^2/(2*theta) * (1 - exp(-2*theta*t)))
        """
        n_paths = config.n_paths
        if config.antithetic:
            n_paths = n_paths // 2

        paths = np.zeros((len(t_grid), n_paths, self.dim))
        paths[0] = X0

        for i, t in enumerate(t_grid[1:], 1):
            # Mean reversion component
            exp_theta_t = np.exp(-self.theta * t)
            mean = X0 * exp_theta_t + self.mu * (1.0 - exp_theta_t)

            # Stochastic component variance
            variance = (self.sigma**2 / (2 * self.theta)) * (1.0 - np.exp(-2 * self.theta * t))
            std = np.sqrt(variance)

            # Generate Gaussian noise
            Z = np.random.normal(0, 1, (n_paths, self.dim))

            paths[i] = mean + std * Z

        # Antithetic variates
        if config.antithetic:
            paths_anti = np.zeros((len(t_grid), n_paths, self.dim))
            paths_anti[0] = X0

            if config.random_seed is not None:
                np.random.seed(config.random_seed)

            for i, t in enumerate(t_grid[1:], 1):
                exp_theta_t = np.exp(-self.theta * t)
                mean = X0 * exp_theta_t + self.mu * (1.0 - exp_theta_t)
                variance = (self.sigma**2 / (2 * self.theta)) * (1.0 - np.exp(-2 * self.theta * t))
                std = np.sqrt(variance)

                Z = np.random.normal(0, 1, (n_paths, self.dim))
                paths_anti[i] = mean - std * Z  # Antithetic

            paths = np.concatenate([paths, paths_anti], axis=1)

        return t_grid, paths

    def expectation(self, X0: np.ndarray, t: float) -> np.ndarray:
        """
        Expected value E[X_t | X_0]

        E[X_t] = X_0 * exp(-theta*t) + mu * (1 - exp(-theta*t))

        Args:
            X0: Initial value
            t: Time

        Returns:
            Expected value
        """
        exp_theta_t = np.exp(-self.theta * t)
        return X0 * exp_theta_t + self.mu * (1.0 - exp_theta_t)

    def variance(self, t: float) -> float:
        """
        Variance Var[X_t]

        Var[X_t] = (sigma^2 / (2*theta)) * (1 - exp(-2*theta*t))

        As t -> infinity: Var[X_t] -> sigma^2 / (2*theta) (stationary variance)

        Args:
            t: Time

        Returns:
            Variance
        """
        return (self.sigma**2 / (2 * self.theta)) * (1.0 - np.exp(-2 * self.theta * t))

    def stationary_distribution(self) -> tuple:
        """
        Stationary distribution as t -> infinity

        X_t ~ N(mu, sigma^2 / (2*theta))

        Returns:
            Tuple of (mean, variance)
        """
        mean = self.mu
        variance = self.sigma**2 / (2 * self.theta)
        return (mean, variance)

    def characteristic_function(self, u: complex, X0: np.ndarray, t: float) -> complex:
        """
        Characteristic function phi(u) = E[exp(i*u*X_t) | X_0]

        For OU process:
        phi(u) = exp(i*u*E[X_t] - 0.5*u^2*Var[X_t])

        Args:
            u: Frequency parameter
            X0: Initial value
            t: Time

        Returns:
            Characteristic function value
        """
        X0_val = np.atleast_1d(X0)[0]
        mean = self.expectation(np.array([X0_val]), t)[0]
        var = self.variance(t)

        return np.exp(1j * u * mean - 0.5 * u**2 * var)
