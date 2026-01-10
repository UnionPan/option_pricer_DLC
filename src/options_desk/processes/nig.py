"""
Normal Inverse Gaussian (NIG) Process

Levy process obtained by time-changing Brownian motion with Inverse Gaussian process

author: Yunian Pan
email: yp1170@nyu.edu
"""
import numpy as np
from .Levy import SubordinatedBrownianMotion


class NIG(SubordinatedBrownianMotion):
    """
    Normal Inverse Gaussian (NIG) Process

    X_t = mu*t + beta*T_t + sqrt(T_t)*W_t

    where:
    - T_t is an Inverse Gaussian process with parameters (delta*t, gamma)
    - W_t is standard Brownian motion
    - gamma = sqrt(alpha^2 - beta^2)

    Parameters:
    - alpha > 0: tail heaviness (larger alpha = lighter tails)
    - beta: asymmetry (-alpha < beta < alpha)
      * beta > 0: positive skew
      * beta < 0: negative skew
    - delta > 0: scale parameter
    - mu: location/drift parameter

    The NIG distribution is widely used in finance for:
    - Modeling log-returns with heavy tails
    - Pricing derivatives
    - Risk management (captures both skewness and kurtosis)

    Characteristic exponent:
        psi(u) = i*mu*u + delta*(sqrt(alpha^2 - beta^2) - sqrt(alpha^2 - (beta + i*u)^2))
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        delta: float,
        mu: float = 0.0,
        name: str = "NIG"
    ):
        """
        Initialize NIG process

        Args:
            alpha: Tail parameter (alpha > 0, controls kurtosis)
            beta: Skewness parameter (|beta| < alpha)
            delta: Scale parameter (delta > 0)
            mu: Location/drift parameter
            name: Process name
        """
        super().__init__(dim=1, name=name)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.delta = float(delta)
        self.mu = float(mu)

        # Derived parameter for IG subordinator
        self.gamma = np.sqrt(alpha**2 - beta**2)

        # For subordinated BM representation: X_t = mu*t + beta*T_t + sqrt(T_t)*W_t
        # We use theta and sigma to match the subordinated BM framework
        self.theta = beta  # Drift in subordinated BM
        self.sigma = 1.0   # Unit volatility in subordinated BM

        self.params['alpha'] = self.alpha
        self.params['beta'] = self.beta
        self.params['delta'] = self.delta
        self.params['mu'] = self.mu
        self.params['gamma'] = self.gamma

    def _simulate_subordinator(self, dt: float, n_paths: int) -> np.ndarray:
        """
        Simulate Inverse Gaussian subordinator increments

        T_t ~ IG(delta*t, gamma)

        Uses the algorithm from Michael, Schucany, and Haas (1976)
        for generating Inverse Gaussian random variables.

        Args:
            dt: Time step
            n_paths: Number of paths

        Returns:
            Array of shape (n_paths,) with IG increments
        """
        # Parameters for IG(mu, lambda) where:
        # mu = delta*dt / gamma (mean)
        # lambda = (delta*dt)^2 (shape parameter)
        mean_ig = self.delta * dt / self.gamma
        lambda_ig = (self.delta * dt)**2

        # Generate IG using Michael-Schucany-Haas algorithm
        nu = np.random.normal(0, 1, n_paths)
        y = nu**2
        x = mean_ig + (mean_ig**2 * y) / (2 * lambda_ig) - \
            (mean_ig / (2 * lambda_ig)) * np.sqrt(4 * mean_ig * lambda_ig * y + mean_ig**2 * y**2)

        # Accept or use alternative
        z = np.random.uniform(0, 1, n_paths)
        mask = z <= mean_ig / (mean_ig + x)
        dT = np.where(mask, x, mean_ig**2 / x)

        return dT

    def _simulate_increments(self, dt: float, n_paths: int) -> np.ndarray:
        """
        Simulate NIG increments

        X_t = mu*dt + beta*T_t + sqrt(T_t)*W_t

        where T_t ~ IG(delta*dt, gamma)

        Args:
            dt: Time step
            n_paths: Number of paths

        Returns:
            Array of shape (n_paths, dim) with increments
        """
        # Simulate subordinator increments
        dT = self._simulate_subordinator(dt, n_paths)

        # Generate Brownian increments scaled by time change
        Z = np.random.normal(0, 1, (n_paths, self.dim))

        # NIG increment: mu*dt + beta*dT + sqrt(dT)*Z
        increments = self.mu * dt + self.beta * dT[:, np.newaxis] + np.sqrt(dT[:, np.newaxis]) * Z

        return increments

    def characteristic_exponent(self, u: np.ndarray) -> np.ndarray:
        """
        Characteristic exponent for NIG process

        psi(u) = i*mu*u + delta*(gamma - sqrt(alpha^2 - (beta + i*u)^2))

        where gamma = sqrt(alpha^2 - beta^2)

        Args:
            u: Frequency parameter(s)

        Returns:
            Characteristic exponent values
        """
        # Compute sqrt(alpha^2 - (beta + i*u)^2)
        arg = self.alpha**2 - (self.beta + 1j * u)**2
        sqrt_arg = np.sqrt(arg)

        # Characteristic exponent
        psi = 1j * self.mu * u + self.delta * (self.gamma - sqrt_arg)

        return psi

    def levy_triplet(self):
        """
        Return the Levy-Khintchine triplet for NIG

        The NIG Levy measure is:
            nu(dx) = (delta*alpha / (pi*|x|)) * exp(beta*x) * K_1(alpha*|x|)

        where K_1 is the modified Bessel function of the second kind.

        Returns:
            tuple: (gamma, sigma_gaussian, levy_measure_description)
        """
        levy_measure_desc = {
            'type': 'nig',
            'alpha': self.alpha,
            'beta': self.beta,
            'delta': self.delta
        }

        # NIG has no Gaussian component
        gamma = self.mu
        sigma_gaussian = 0.0

        return (gamma, sigma_gaussian, levy_measure_desc)

    def expectation(self, X0: np.ndarray, t: float) -> np.ndarray:
        """
        Expected value E[X_t | X_0]

        E[X_t] = X_0 + (mu + delta*beta/gamma) * t

        Args:
            X0: Initial value
            t: Time

        Returns:
            Expected value
        """
        drift = self.mu + self.delta * self.beta / self.gamma
        return X0 + drift * t

    def variance(self, t: float) -> float:
        """
        Variance Var[X_t]

        Var[X_t] = delta * alpha^2 / gamma^3 * t

        Args:
            t: Time

        Returns:
            Variance
        """
        return self.delta * self.alpha**2 / (self.gamma**3) * t

    def skewness(self, t: float) -> float:
        """
        Skewness of X_t

        Skewness = 3*beta / (alpha * sqrt(delta*t*gamma))

        Args:
            t: Time

        Returns:
            Skewness coefficient
        """
        return 3.0 * self.beta / (self.alpha * np.sqrt(self.delta * t * self.gamma))

    def kurtosis(self, t: float) -> float:
        """
        Excess kurtosis of X_t

        Excess Kurtosis = 3 * (1 + 4*beta^2/alpha^2) / (delta*t*gamma)

        Args:
            t: Time

        Returns:
            Excess kurtosis (kurtosis - 3)
        """
        return 3.0 * (1.0 + 4.0 * self.beta**2 / self.alpha**2) / (self.delta * t * self.gamma)
