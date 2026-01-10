"""
Variance Gamma Process

Levy process obtained by time-changing Brownian motion with Gamma process

author: Yunian Pan
email: yp1170@nyu.edu
"""
import numpy as np
from .Levy import SubordinatedBrownianMotion


class VarianceGamma(SubordinatedBrownianMotion):
    """
    Variance Gamma (VG) Process

    X_t = theta * T_t + sigma * W_{T_t}

    where:
    - T_t is a Gamma process with rate 1/nu (subordinator)
    - W_t is standard Brownian motion
    - theta: drift in the subordinated Brownian motion
    - sigma: volatility in the subordinated Brownian motion
    - nu: variance rate (controls kurtosis/tail heaviness)

    The VG process has:
    - Symmetric jumps when theta = 0
    - Positive skew when theta > 0, negative skew when theta < 0
    - Heavy tails controlled by nu (larger nu = heavier tails)

    Characteristic exponent:
        psi(u) = -(1/nu) * ln(1 - i*theta*nu*u + 0.5*sigma^2*nu*u^2)
    """

    def __init__(
        self,
        theta: float,
        sigma: float,
        nu: float,
        name: str = "VarianceGamma"
    ):
        """
        Initialize Variance Gamma process

        Args:
            theta: Drift in subordinated BM (controls skewness)
            sigma: Volatility in subordinated BM (controls spread)
            nu: Variance rate (controls kurtosis, must be > 0)
            name: Process name
        """
        super().__init__(dim=1, name=name)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.nu = float(nu)

        self.params['theta'] = self.theta
        self.params['sigma'] = self.sigma
        self.params['nu'] = self.nu

    def _simulate_subordinator(self, dt: float, n_paths: int) -> np.ndarray:
        """
        Simulate Gamma subordinator increments

        T_t ~ Gamma(shape = t/nu, rate = 1/nu)
        which has mean E[T_t] = t and variance Var[T_t] = nu*t

        Args:
            dt: Time step
            n_paths: Number of paths

        Returns:
            Array of shape (n_paths,) with Gamma increments
        """
        # Gamma parameters for increment over dt
        shape = dt / self.nu  # shape parameter
        scale = self.nu       # scale parameter (NumPy uses scale = 1/rate)

        # Generate Gamma random variables
        dT = np.random.gamma(shape, scale, n_paths)

        return dT

    def characteristic_exponent(self, u: np.ndarray) -> np.ndarray:
        """
        Characteristic exponent for VG process

        psi(u) = -(1/nu) * ln(1 - i*theta*nu*u + 0.5*sigma^2*nu*u^2)

        Args:
            u: Frequency parameter(s)

        Returns:
            Characteristic exponent values
        """
        # Complex argument inside logarithm
        arg = 1.0 - 1j * self.theta * self.nu * u + 0.5 * self.sigma**2 * self.nu * u**2

        # Characteristic exponent
        psi = -(1.0 / self.nu) * np.log(arg)

        return psi

    def levy_triplet(self):
        """
        Return the Levy-Khintchine triplet for VG

        The VG Levy measure is:
            nu(dx) = (C / |x|) * exp(-lambda_+*x) * 1_{x>0} + exp(lambda_-*|x|) * 1_{x<0}

        where:
            lambda_+ = sqrt(theta^2 + 2*sigma^2/nu) + theta
            lambda_- = sqrt(theta^2 + 2*sigma^2/nu) - theta
            C = 1/nu

        Returns:
            tuple: (gamma, sigma_gaussian, levy_measure_description)
        """
        # Compute exponential rates
        discriminant = self.theta**2 + 2.0 * self.sigma**2 / self.nu
        sqrt_disc = np.sqrt(discriminant)

        lambda_plus = sqrt_disc + self.theta
        lambda_minus = sqrt_disc - self.theta

        levy_measure_desc = {
            'type': 'variance_gamma',
            'lambda_plus': lambda_plus,
            'lambda_minus': lambda_minus,
            'C': 1.0 / self.nu
        }

        # VG has no Gaussian component (sigma_gaussian = 0)
        # and gamma is determined by centering condition
        gamma = 0.0
        sigma_gaussian = 0.0

        return (gamma, sigma_gaussian, levy_measure_desc)

    def expectation(self, X0: np.ndarray, t: float) -> np.ndarray:
        """
        Expected value E[X_t | X_0]

        E[X_t] = X_0 + theta * t

        Args:
            X0: Initial value
            t: Time

        Returns:
            Expected value
        """
        return X0 + self.theta * t

    def variance(self, t: float) -> float:
        """
        Variance Var[X_t]

        Var[X_t] = (sigma^2 + nu * theta^2) * t

        Args:
            t: Time

        Returns:
            Variance
        """
        return (self.sigma**2 + self.nu * self.theta**2) * t

    def skewness(self, t: float) -> float:
        """
        Skewness of X_t

        Skewness = (2*nu*theta^3 + 3*sigma^2*nu*theta) / (sigma^2 + nu*theta^2)^(3/2) * t^(1/2)

        Args:
            t: Time

        Returns:
            Skewness coefficient
        """
        var = self.sigma**2 + self.nu * self.theta**2
        numerator = 2.0 * self.nu * self.theta**3 + 3.0 * self.sigma**2 * self.nu * self.theta
        skew = numerator / (var**(1.5)) * np.sqrt(t)
        return skew

    def kurtosis(self, t: float) -> float:
        """
        Excess kurtosis of X_t

        Excess Kurtosis = 3 * nu / t

        Args:
            t: Time

        Returns:
            Excess kurtosis (kurtosis - 3)
        """
        return 3.0 * self.nu / t
