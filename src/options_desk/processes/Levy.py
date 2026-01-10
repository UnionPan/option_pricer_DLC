"""
Levy process base class

For infinite activity jump processes (VG, NIG, CGMY, etc.)

author: Yunian Pan
email: yp1170@nyu.edu
"""
import numpy as np
from .base import StochasticProcess
from abc import abstractmethod


class LevyProcess(StochasticProcess):
    """
    Base class for Levy processes with infinite activity

    Levy processes are characterized by independent and stationary increments.
    Unlike jump-diffusion, they have infinite jumps in any time interval (but most are small).

    Common models:
        - Variance Gamma (VG): Time-changed Brownian motion with Gamma subordinator
        - Normal Inverse Gaussian (NIG): Time-changed Brownian motion with IG subordinator
        - CGMY: Generalization with tunable tail behavior

    Levy-Khintchine representation:
        E[exp(iu X_t)] = exp(t * psi(u))
        psi(u) = i*gamma*u - (sigma^2/2)*u^2 + integral[e^{iux} - 1 - iux*1_{|x|<1}] nu(dx)

    where (gamma, sigma^2, nu) is the Levy triplet.
    """

    def __init__(self, dim: int = 1, name: str = "LevyProcess"):
        super().__init__(dim, name)
        # Levy processes typically don't use standard drift/diffusion
        # They're characterized by their characteristic function

    @abstractmethod
    def levy_triplet(self):
        """
        Return the Levy-Khintchine triplet (gamma, sigma^2, nu)

        Returns:
            tuple: (drift, gaussian_variance, levy_measure)
                - drift: gamma (deterministic drift)
                - gaussian_variance: sigma^2 (Gaussian component variance)
                - levy_measure: nu (measure describing jump distribution)
        """
        pass

    @abstractmethod
    def characteristic_exponent(self, u: np.ndarray) -> np.ndarray:
        """
        Characteristic exponent psi(u) where E[exp(iu X_t)] = exp(t * psi(u))

        This is the key function defining the Levy process.

        Args:
            u: Frequency parameter(s)

        Returns:
            psi(u): Characteristic exponent
        """
        pass

    def characteristic_function(self, u: complex, X0: np.ndarray, t: float) -> complex:
        """
        Characteristic function phi(u) = E[exp(i*u*X_t) | X_0]

        For Levy processes: phi(u, t) = exp(i*u*X0 + t*psi(u))

        Args:
            u: Frequency parameter
            X0: Initial value
            t: Time

        Returns:
            Characteristic function value
        """
        psi = self.characteristic_exponent(np.array([u]))[0]
        return np.exp(1j * u * X0 + t * psi)

    def drift(self, X: np.ndarray, t: float) -> np.ndarray:
        """
        Levy processes don't have simple drift terms.
        Use subordinator-based or CF-based simulation instead.
        """
        raise NotImplementedError(
            "Levy processes should override simulation methods, not drift/diffusion"
        )

    def diffusion(self, X: np.ndarray, t: float) -> np.ndarray:
        """
        Levy processes don't have simple diffusion terms.
        Use subordinator-based or CF-based simulation instead.
        """
        raise NotImplementedError(
            "Levy processes should override simulation methods, not drift/diffusion"
        )

    @abstractmethod
    def _simulate_increments(self, dt: float, n_paths: int) -> np.ndarray:
        """
        Simulate increments dX_t over time step dt

        This is the core simulation method that each Levy process must implement.
        Common approaches:
            - Subordination: X_t = W_{T_t} where T_t is an increasing Levy process
            - Series representation: Approximate infinite jumps with finite series
            - Acceptance-rejection: Sample from Levy measure directly
            - Fourier inversion: Use characteristic function

        Args:
            dt: Time step
            n_paths: Number of paths

        Returns:
            Array of shape (n_paths, dim) with increments
        """
        pass

    def _euler_maruyama(self, X0, T, dt, t_grid, config):
        """
        Simulate Levy process by generating increments directly

        Unlike standard Euler-Maruyama, this uses the increment simulation
        specific to each Levy process.
        """
        n_paths = config.n_paths
        if config.antithetic:
            n_paths = n_paths // 2

        paths = np.zeros((len(t_grid), n_paths, self.dim))
        paths[0] = X0

        # Generate increments for each time step
        for i in range(len(t_grid) - 1):
            increments = self._simulate_increments(dt, n_paths)
            paths[i + 1] = paths[i] + increments

        # Antithetic variates
        if config.antithetic:
            paths_anti = np.zeros((len(t_grid), n_paths, self.dim))
            paths_anti[0] = X0

            # Reset seed for antithetic
            if config.random_seed is not None:
                np.random.seed(config.random_seed)

            for i in range(len(t_grid) - 1):
                # For Levy processes, antithetic means negating increments
                increments = -self._simulate_increments(dt, n_paths)
                paths_anti[i + 1] = paths_anti[i] + increments

            paths = np.concatenate([paths, paths_anti], axis=1)

        return t_grid, paths

    def _milstein(self, X0, T, dt, t_grid, config):
        """Milstein not applicable to Levy processes, falls back to increment simulation"""
        return self._euler_maruyama(X0, T, dt, t_grid, config)


class SubordinatedBrownianMotion(LevyProcess):
    """
    Base class for Levy processes defined as time-changed Brownian motion

    X_t = theta*T_t + sigma*W_{T_t}

    where:
        - W_t is standard Brownian motion
        - T_t is an increasing Levy process (subordinator)
        - theta is drift parameter
        - sigma is volatility parameter

    Examples: Variance Gamma, NIG
    """

    def __init__(self, dim: int = 1, name: str = "SubordinatedBM"):
        super().__init__(dim, name)
        self.theta = None  # Drift in subordinated BM
        self.sigma = None  # Vol in subordinated BM

    @abstractmethod
    def _simulate_subordinator(self, dt: float, n_paths: int) -> np.ndarray:
        """
        Simulate the subordinator T_t (increasing Levy process)

        Args:
            dt: Time step
            n_paths: Number of paths

        Returns:
            Array of shape (n_paths,) with subordinator increments
        """
        pass

    def _simulate_increments(self, dt: float, n_paths: int) -> np.ndarray:
        """
        Simulate increments via subordination:
            dX_t = theta*dT_t + sigma*sqrt{(dT_t)}*Z

        where Z ~ N(0,1) and dT_t is subordinator increment
        """
        # Simulate subordinator increments
        dT = self._simulate_subordinator(dt, n_paths)

        # Generate Brownian increments scaled by time change
        Z = np.random.normal(0, 1, (n_paths, self.dim))
        increments = self.theta * dT[:, np.newaxis] + self.sigma * np.sqrt(dT[:, np.newaxis]) * Z

        return increments
