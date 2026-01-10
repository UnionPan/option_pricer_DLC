"""
SABR Stochastic Volatility Model

Stochastic Alpha Beta Rho model for interest rate derivatives

author: Yunian Pan
email: yp1170@nyu.edu
"""
import numpy as np
from .base import MultiFactorProcess


class SABR(MultiFactorProcess):
    """
    SABR Stochastic Volatility Model

    Two-dimensional system:
        dF_t = sigma_t * F_t^beta * dW_t^F
        dsigma_t = alpha * sigma_t * dW_t^sigma

    where:
        - F_t: forward rate/price
        - sigma_t: stochastic volatility
        - beta: CEV exponent (elasticity parameter)
          * beta = 0: Normal/Bachelier model
          * beta = 0.5: CIR-like dynamics
          * beta = 1: Lognormal/Black-Scholes
        - alpha: volatility of volatility (vol-of-vol)
        - rho: correlation between dW^F and dW^sigma

    State vector: X = [F, sigma]

    The SABR model is widely used for:
        - Interest rate derivatives (swaptions, caps/floors)
        - FX options
        - Implied volatility smile modeling
        - Has analytical approximation (Hagan formula)
    """

    def __init__(
        self,
        beta: float,
        alpha: float,
        rho: float,
        sigma0: float = None,
        name: str = "SABR"
    ):
        """
        Initialize SABR model

        Args:
            beta: CEV exponent (0 <= beta <= 1)
                  0 = Normal, 0.5 = popular choice, 1 = Lognormal
            alpha: Volatility of volatility (alpha > 0)
            rho: Correlation between forward and vol (-1 < rho < 1)
            sigma0: Initial volatility (optional, for reference)
            name: Model name
        """
        super().__init__(dim=2, name=name)

        self.beta = float(beta)
        self.alpha = float(alpha)
        self.rho = float(rho)
        self.sigma0 = float(sigma0) if sigma0 is not None else 0.2

        # Set correlation between the two Brownian motions
        corr_matrix = np.array([
            [1.0, rho],
            [rho, 1.0]
        ])
        self.set_correlation(corr_matrix)

        self.params['beta'] = self.beta
        self.params['alpha'] = self.alpha
        self.params['rho'] = self.rho
        self.params['sigma0'] = self.sigma0

    def drift(self, X: np.ndarray, t: float) -> np.ndarray:
        """
        Drift term for SABR model

        mu_F = 0 (martingale under forward measure)
        mu_sigma = 0 (martingale volatility)

        Args:
            X: Current state [F, sigma], shape (n_paths, 2)
            t: Current time

        Returns:
            Drift vector (zeros), shape (n_paths, 2)
        """
        n_paths = X.shape[0]
        # SABR has zero drift (under risk-neutral/forward measure)
        return np.zeros((n_paths, self.dim))

    def diffusion(self, X: np.ndarray, t: float) -> np.ndarray:
        """
        Diffusion term for SABR model

        sigma_F = sigma_t * F_t^beta
        sigma_sigma = alpha * sigma_t

        Args:
            X: Current state [F, sigma], shape (n_paths, 2)
            t: Current time

        Returns:
            Diffusion coefficient (diagonal), shape (n_paths, 2)
        """
        F = X[:, 0:1]      # Forward rate, shape (n_paths, 1)
        sigma = X[:, 1:2]  # Volatility, shape (n_paths, 1)

        # Handle potential negative F for beta < 1
        # For beta in (0,1), use abs(F)^beta to avoid complex numbers
        if self.beta != 0.0:
            F_power = np.sign(F) * np.abs(F)**self.beta
        else:
            F_power = np.ones_like(F)

        sigma_F = sigma * F_power
        sigma_sigma = self.alpha * sigma

        return np.concatenate([sigma_F, sigma_sigma], axis=1)

    def _euler_maruyama(self, X0, T, dt, t_grid, config):
        """
        Override Euler-Maruyama to handle volatility positivity

        After each step, ensure sigma_t > 0 (absorbing at zero).
        """
        # Call parent class method
        t_grid, paths = super()._euler_maruyama(X0, T, dt, t_grid, config)

        # Ensure volatility stays positive (absorption at zero)
        paths[:, :, 1] = np.maximum(paths[:, :, 1], 0.0)

        return t_grid, paths

    def _milstein(self, X0, T, dt, t_grid, config):
        """
        Milstein scheme for SABR with correction terms

        For F: 0.5 * sigma * F^beta * d(sigma * F^beta)/dF * (dW^2 - dt)
             = 0.5 * sigma^2 * beta * F^(2*beta - 1) * (dW_F^2 - dt)

        For sigma: 0.5 * alpha * sigma * d(alpha * sigma)/dsigma * (dW^2 - dt)
                 = 0.5 * alpha^2 * sigma * (dW_sigma^2 - dt)
        """
        n_paths = config.n_paths
        if config.antithetic:
            n_paths = n_paths // 2

        paths = np.zeros((len(t_grid), n_paths, self.dim))
        paths[0] = X0
        sqrt_dt = np.sqrt(dt)

        for i, t in enumerate(t_grid[:-1]):
            X_current = paths[i]
            F_current = X_current[:, 0:1]
            sigma_current = X_current[:, 1:2]

            # Generate correlated Brownian increments
            dW = np.random.normal(0, sqrt_dt, size=(n_paths, self.dim))
            if self.cholesky_decomp is not None:
                dW = dW @ self.cholesky_decomp.T

            # Euler terms
            drift_term = self.drift(X_current, t) * dt
            diffusion_coeff = self.diffusion(X_current, t)
            diffusion_term = self._apply_diffusion(diffusion_coeff, dW)

            # Milstein correction for F
            if self.beta != 0.0:
                F_power = np.sign(F_current) * np.abs(F_current)**(2*self.beta - 1)
                milstein_F = 0.5 * sigma_current**2 * self.beta * F_power * (dW[:, 0:1]**2 - dt)
            else:
                milstein_F = np.zeros((n_paths, 1))

            # Milstein correction for sigma
            milstein_sigma = 0.5 * self.alpha**2 * sigma_current * (dW[:, 1:2]**2 - dt)

            milstein_correction = np.concatenate([milstein_F, milstein_sigma], axis=1)

            paths[i + 1] = X_current + drift_term + diffusion_term + milstein_correction

        # Ensure volatility stays positive
        paths[:, :, 1] = np.maximum(paths[:, :, 1], 0.0)

        # Antithetic paths
        if config.antithetic:
            paths_anti = np.zeros((len(t_grid), n_paths, self.dim))
            paths_anti[0] = X0

            if config.random_seed is not None:
                np.random.seed(config.random_seed)

            for i, t in enumerate(t_grid[:-1]):
                X_current = paths_anti[i]
                F_current = X_current[:, 0:1]
                sigma_current = X_current[:, 1:2]

                dW = -np.random.normal(0, sqrt_dt, size=(n_paths, self.dim))
                if self.cholesky_decomp is not None:
                    dW = dW @ self.cholesky_decomp.T

                drift_term = self.drift(X_current, t) * dt
                diffusion_coeff = self.diffusion(X_current, t)
                diffusion_term = self._apply_diffusion(diffusion_coeff, dW)

                if self.beta != 0.0:
                    F_power = np.sign(F_current) * np.abs(F_current)**(2*self.beta - 1)
                    milstein_F = 0.5 * sigma_current**2 * self.beta * F_power * (dW[:, 0:1]**2 - dt)
                else:
                    milstein_F = np.zeros((n_paths, 1))

                milstein_sigma = 0.5 * self.alpha**2 * sigma_current * (dW[:, 1:2]**2 - dt)
                milstein_correction = np.concatenate([milstein_F, milstein_sigma], axis=1)

                paths_anti[i + 1] = X_current + drift_term + diffusion_term + milstein_correction

            paths_anti[:, :, 1] = np.maximum(paths_anti[:, :, 1], 0.0)
            paths = np.concatenate([paths, paths_anti], axis=1)

        return t_grid, paths

    def implied_volatility_hagan(self, F: float, K: float, T: float, sigma_atm: float = None) -> float:
        """
        Hagan's approximation for SABR implied volatility

        This is the famous approximate formula for European option pricing.

        Args:
            F: Forward price
            K: Strike price
            T: Time to maturity
            sigma_atm: ATM volatility (if None, uses sigma0)

        Returns:
            Implied Black volatility
        """
        if sigma_atm is None:
            sigma_atm = self.sigma0

        # ATM case (K = F)
        if abs(K - F) < 1e-10:
            # ATM approximation
            if self.beta == 1.0:
                F_mid = F
            else:
                F_mid = (F * K)**0.5

            if self.beta != 1.0:
                term1 = sigma_atm / (F_mid**(1 - self.beta))
            else:
                term1 = sigma_atm

            term2 = 1 + T * (
                ((1 - self.beta)**2 / 24) * (sigma_atm**2 / F_mid**(2 - 2*self.beta)) +
                0.25 * self.rho * self.beta * self.alpha * sigma_atm / F_mid**(1 - self.beta) +
                ((2 - 3*self.rho**2) / 24) * self.alpha**2
            )

            return term1 * term2

        # Non-ATM case
        log_FK = np.log(F / K)
        F_mid = (F * K)**0.5

        if self.beta != 1.0:
            z = (self.alpha / sigma_atm) * F_mid**(1 - self.beta) * log_FK
        else:
            z = (self.alpha / sigma_atm) * log_FK

        # x(z) function
        x_z = np.log((np.sqrt(1 - 2*self.rho*z + z**2) + z - self.rho) / (1 - self.rho))

        # Numerator
        if self.beta != 1.0:
            num = sigma_atm
        else:
            num = sigma_atm

        # Denominator
        denom_part1 = F_mid**(1 - self.beta) * (
            1 + ((1 - self.beta)**2 / 24) * log_FK**2 +
            ((1 - self.beta)**4 / 1920) * log_FK**4
        )

        # Correction factor
        if abs(z) > 1e-10:
            correction = z / x_z
        else:
            correction = 1.0

        # Time correction
        time_correction = 1 + T * (
            ((1 - self.beta)**2 / 24) * (sigma_atm**2 / F_mid**(2 - 2*self.beta)) +
            0.25 * self.rho * self.beta * self.alpha * sigma_atm / F_mid**(1 - self.beta) +
            ((2 - 3*self.rho**2) / 24) * self.alpha**2
        )

        return (num / denom_part1) * correction * time_correction
