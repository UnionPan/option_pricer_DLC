"""
Merton Jump-Diffusion Model

GBM with log-normal jumps in stock price

author: Yunian Pan
email: yp1170@nyu.edu
"""
import numpy as np
from .jump_diffusion import JumpDiffusionProcess


class MertonJD(JumpDiffusionProcess):
    """
    Merton Jump-Diffusion Model

    dS_t = mu * S_t dt + sigma * S_t dW_t + S_{t-} dJ_t

    where J_t is a compound Poisson process with:
    - Jump intensity: lambda (expected jumps per unit time)
    - Jump sizes: log(1+Y) ~ N(mu_J, sigma_J^2)

    When a jump occurs, price multiplies by (1+Y) where Y = exp(Z)-1, Z ~ N(mu_J, sigma_J^2)
    """

    def __init__(
        self,
        mu: float,
        sigma: float,
        lambda_jump: float,
        mu_J: float,
        sigma_J: float,
        name: str = "MertonJD"
    ):
        """
        Initialize Merton Jump-Diffusion

        Args:
            mu: Drift (expected return excluding jumps)
            sigma: Diffusion volatility
            lambda_jump: Jump intensity (expected jumps per year)
            mu_J: Mean of log-jump size
            sigma_J: Std dev of log-jump size
            name: Model name
        """
        super().__init__(name=name)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.mu_J = float(mu_J)
        self.sigma_J = float(sigma_J)
        self.set_jump_intensity(lambda_jump)

        self.params['mu'] = self.mu
        self.params['sigma'] = self.sigma
        self.params['lambda'] = self.jump_intensity
        self.params['mu_J'] = self.mu_J
        self.params['sigma_J'] = self.sigma_J

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

    def jump_size_distribution(self, n_jumps: int) -> np.ndarray:
        """
        Generate log-normal jump sizes

        Jump sizes Y where log(1+Y) ~ N(mu_J, sigma_J^2)
        Equivalently: Y = exp(Z) - 1 where Z ~ N(mu_J, sigma_J^2)

        Args:
            n_jumps: Number of jumps to generate

        Returns:
            Array of shape (n_jumps, 1) with jump proportions Y
        """
        # Generate log-jumps: Z ~ N(mu_J, sigma_J^2)
        log_jumps = np.random.normal(self.mu_J, self.sigma_J, (n_jumps, self.dim))

        # Convert to proportional jumps: Y = exp(Z) - 1
        jump_proportions = np.exp(log_jumps) - 1.0

        return jump_proportions

    def jump_component(self, X: np.ndarray, t: float, dt: float) -> np.ndarray:
        """
        Multiplicative jump component for Merton model

        Overrides base class to handle multiplicative (not additive) jumps.
        When a jump Y occurs, price changes by S * Y (not just Y).

        Args:
            X: Current state, shape (n_paths, 1)
            t: Current time
            dt: Time increment

        Returns:
            Jump contribution S * sum(Y_i), shape (n_paths, 1)
        """
        n_paths = X.shape[0]
        jumps = np.zeros((n_paths, self.dim))

        # Sample number of jumps for each path
        n_jumps_per_path = np.random.poisson(self.jump_intensity * dt, n_paths)

        # Optimize by grouping paths with same number of jumps
        unique_jump_counts = np.unique(n_jumps_per_path)

        for n_jumps in unique_jump_counts:
            if n_jumps == 0:
                continue

            mask = (n_jumps_per_path == n_jumps)
            n_paths_with_jumps = np.sum(mask)

            # Generate jump proportions Y
            total_jumps = n_jumps * n_paths_with_jumps
            jump_proportions = self.jump_size_distribution(total_jumps)
            jump_proportions = jump_proportions.reshape(n_paths_with_jumps, n_jumps, self.dim)

            # Sum jump proportions for each path
            total_jump_proportion = jump_proportions.sum(axis=1)

            # Multiplicative: jump contribution is X * sum(Y_i)
            jumps[mask] = X[mask] * total_jump_proportion

        return jumps

    def characteristic_function(self, u: complex, X0: np.ndarray, t: float) -> complex:
        """
        Characteristic function for log(S_t) in Merton model

        phi(u) = exp(i*u*log(S_0) + t*psi(u))

        where psi(u) = i*u*(r - 0.5*sigma^2 - lambda*k) - 0.5*sigma^2*u^2
                       + lambda*(exp(i*u*mu_J - 0.5*u^2*sigma_J^2) - 1)

        and k = exp(mu_J + 0.5*sigma_J^2) - 1 is the expected jump size

        Args:
            u: Frequency parameter
            X0: Initial value
            t: Time

        Returns:
            Characteristic function value
        """
        X0_val = np.atleast_1d(X0)[0]
        log_S0 = np.log(X0_val)

        # Expected jump size
        k = np.exp(self.mu_J + 0.5 * self.sigma_J**2) - 1.0

        # Characteristic exponent
        drift_term = 1j * u * (self.mu - 0.5 * self.sigma**2 - self.jump_intensity * k)
        diffusion_term = -0.5 * self.sigma**2 * u**2
        jump_term = self.jump_intensity * (np.exp(1j * u * self.mu_J - 0.5 * u**2 * self.sigma_J**2) - 1)

        psi = drift_term + diffusion_term + jump_term

        return np.exp(1j * u * log_S0 + t * psi)
