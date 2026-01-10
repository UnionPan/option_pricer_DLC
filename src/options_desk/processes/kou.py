"""
Kou Jump-Diffusion Model

GBM with double exponential jumps in stock price

author: Yunian Pan
email: yp1170@nyu.edu
"""
import numpy as np
from .jump_diffusion import JumpDiffusionProcess


class KouJD(JumpDiffusionProcess):
    """
    Kou Jump-Diffusion Model

    dS_t = mu * S_t dt + sigma * S_t dW_t + S_{t-} dJ_t

    where J_t is a compound Poisson process with:
    - Jump intensity: lambda (expected jumps per unit time)
    - Jump sizes: Y with double exponential distribution
      * Y = +ξ with probability p, where ξ ~ Exp(eta_up)
      * Y = -ζ with probability 1-p, where ζ ~ Exp(eta_down)

    The asymmetric double exponential captures:
    - Upward jumps (rare large gains)
    - Downward jumps (frequent small crashes)
    """

    def __init__(
        self,
        mu: float,
        sigma: float,
        lambda_jump: float,
        p: float,
        eta_up: float,
        eta_down: float,
        name: str = "KouJD"
    ):
        """
        Initialize Kou Jump-Diffusion

        Args:
            mu: Drift (expected return excluding jumps)
            sigma: Diffusion volatility
            lambda_jump: Jump intensity (expected jumps per year)
            p: Probability of upward jump (0 < p < 1)
            eta_up: Rate parameter for upward exponential (eta_up > 0)
            eta_down: Rate parameter for downward exponential (eta_down > 0)
            name: Model name
        """
        super().__init__(name=name)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.p = float(p)
        self.eta_up = float(eta_up)
        self.eta_down = float(eta_down)
        self.set_jump_intensity(lambda_jump)

        self.params['mu'] = self.mu
        self.params['sigma'] = self.sigma
        self.params['lambda'] = self.jump_intensity
        self.params['p'] = self.p
        self.params['eta_up'] = self.eta_up
        self.params['eta_down'] = self.eta_down

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
        Generate double exponential jump sizes

        For each jump:
        - With probability p: Y = +ξ where ξ ~ Exp(eta_up)
        - With probability 1-p: Y = -ζ where ζ ~ Exp(eta_down)

        Args:
            n_jumps: Number of jumps to generate

        Returns:
            Array of shape (n_jumps, 1) with jump proportions Y
        """
        # Determine direction of each jump (up or down)
        is_upward = np.random.random((n_jumps, self.dim)) < self.p

        # Generate exponential random variables
        # Upward: ξ ~ Exp(eta_up), so ξ = -ln(U) / eta_up
        # Downward: ζ ~ Exp(eta_down), so ζ = -ln(U) / eta_down
        uniform_samples = np.random.random((n_jumps, self.dim))

        # Initialize jump sizes
        jump_proportions = np.zeros((n_jumps, self.dim))

        # Upward jumps: positive exponential
        jump_proportions[is_upward] = -np.log(uniform_samples[is_upward]) / self.eta_up

        # Downward jumps: negative exponential
        jump_proportions[~is_upward] = np.log(uniform_samples[~is_upward]) / self.eta_down

        return jump_proportions

    def jump_component(self, X: np.ndarray, t: float, dt: float) -> np.ndarray:
        """
        Multiplicative jump component for Kou model

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
        Characteristic function for log(S_t) in Kou model

        phi(u) = exp(i*u*log(S_0) + t*psi(u))

        where psi(u) = i*u*(r - 0.5*sigma^2 - lambda*k) - 0.5*sigma^2*u^2
                       + lambda*(phi_Y(u) - 1)

        and phi_Y(u) = p * eta_up/(eta_up - i*u) + (1-p) * eta_down/(eta_down + i*u)
        is the CF of the jump size Y

        and k = p/eta_up - (1-p)/eta_down is the expected jump size

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
        k = self.p / self.eta_up - (1.0 - self.p) / self.eta_down

        # Jump size characteristic function
        phi_Y = (self.p * self.eta_up / (self.eta_up - 1j * u) +
                 (1.0 - self.p) * self.eta_down / (self.eta_down + 1j * u))

        # Characteristic exponent
        drift_term = 1j * u * (self.mu - 0.5 * self.sigma**2 - self.jump_intensity * k)
        diffusion_term = -0.5 * self.sigma**2 * u**2
        jump_term = self.jump_intensity * (phi_Y - 1)

        psi = drift_term + diffusion_term + jump_term

        return np.exp(1j * u * log_S0 + t * psi)
