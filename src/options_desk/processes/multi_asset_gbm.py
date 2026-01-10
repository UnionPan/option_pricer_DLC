"""
Multi-Asset Geometric Brownian Motion

N correlated asset prices following GBM dynamics

author: Yunian Pan
email: yp1170@nyu.edu
"""
import numpy as np
from .base import MultiFactorProcess


class MultiAssetGBM(MultiFactorProcess):
    """
    Multi-Asset Geometric Brownian Motion

    N-dimensional system of correlated GBMs:
        dS_i(t) = mu_i * S_i(t) dt + sigma_i * S_i(t) dW_i(t)

    where:
        - S_i(t): i-th asset price
        - mu_i: drift of i-th asset
        - sigma_i: volatility of i-th asset
        - W_i(t): Brownian motions with correlation matrix rho

    This is a natural extension of single-asset GBM to multiple correlated assets.
    Useful for:
        - Multi-asset option pricing (basket, spread, etc.)
        - Portfolio simulation
        - Risk management with correlated assets
    """

    def __init__(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        rho: np.ndarray = None,
        name: str = "MultiAssetGBM"
    ):
        """
        Initialize Multi-Asset GBM

        Args:
            mu: Drift vector, shape (n_assets,) or list
            sigma: Volatility vector, shape (n_assets,) or list
            rho: Correlation matrix, shape (n_assets, n_assets)
                 If None, assumes independent assets
            name: Model name
        """
        self.mu = np.atleast_1d(mu)
        self.sigma = np.atleast_1d(sigma)
        n_assets = len(self.mu)

        super().__init__(dim=n_assets, name=name)

        # Set correlation if provided
        if rho is not None:
            self.set_correlation(np.array(rho))
        else:
            # Independent assets (identity correlation)
            self.set_correlation(np.eye(n_assets))

        self.params['mu'] = self.mu
        self.params['sigma'] = self.sigma
        self.params['n_assets'] = n_assets

    def drift(self, X: np.ndarray, t: float) -> np.ndarray:
        """
        Drift term: mu_i * S_i for each asset

        Args:
            X: Current state (all asset prices), shape (n_paths, n_assets)
            t: Current time

        Returns:
            Drift vector, shape (n_paths, n_assets)
        """
        # Element-wise: mu * X
        return self.mu * X

    def diffusion(self, X: np.ndarray, t: float) -> np.ndarray:
        """
        Diffusion term: sigma_i * S_i for each asset

        Args:
            X: Current state (all asset prices), shape (n_paths, n_assets)
            t: Current time

        Returns:
            Diffusion coefficient (diagonal), shape (n_paths, n_assets)
        """
        # Element-wise: sigma * X
        return self.sigma * X

    def _exact_simulation(self, X0, T, dt, t_grid, config):
        """
        Exact simulation using analytical solution for each asset

        S_i(t) = S_i(0) * exp((mu_i - 0.5*sigma_i^2)*t + sigma_i*W_i(t))

        where W_i are correlated Brownian motions.
        """
        n_paths = config.n_paths
        if config.antithetic:
            n_paths = n_paths // 2

        paths = np.zeros((len(t_grid), n_paths, self.dim))
        paths[0] = X0

        # Pre-compute constants
        drift_adjusted = self.mu - 0.5 * self.sigma**2  # Shape (n_assets,)

        for i, t in enumerate(t_grid[1:], 1):
            # Generate correlated Brownian motion W(t)
            sqrt_t = np.sqrt(t)
            Z = np.random.normal(0, 1, size=(n_paths, self.dim))

            # Apply correlation
            if self.cholesky_decomp is not None:
                Z_corr = Z @ self.cholesky_decomp.T
            else:
                Z_corr = Z

            W_t = sqrt_t * Z_corr

            # Exact solution: S(t) = S(0) * exp((mu - 0.5*sigma^2)*t + sigma*W(t))
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
                Z = np.random.normal(0, 1, size=(n_paths, self.dim))

                if self.cholesky_decomp is not None:
                    Z_corr = Z @ self.cholesky_decomp.T
                else:
                    Z_corr = Z

                # Antithetic: negate Z
                W_t = -sqrt_t * Z_corr

                exponent = drift_adjusted * t + self.sigma * W_t
                paths_anti[i] = X0 * np.exp(exponent)

            paths = np.concatenate([paths, paths_anti], axis=1)

        return t_grid, paths

    def expectation(self, X0: np.ndarray, t: float) -> np.ndarray:
        """
        Expected value E[S_i(t) | S_i(0)] for each asset

        E[S_i(t)] = S_i(0) * exp(mu_i * t)

        Args:
            X0: Initial asset prices, shape (n_assets,)
            t: Time

        Returns:
            Expected prices, shape (n_assets,)
        """
        return X0 * np.exp(self.mu * t)

    def variance(self, X0: np.ndarray, t: float) -> np.ndarray:
        """
        Variance Var[S_i(t)] for each asset

        Var[S_i(t)] = S_i(0)^2 * exp(2*mu_i*t) * (exp(sigma_i^2*t) - 1)

        Args:
            X0: Initial asset prices, shape (n_assets,)
            t: Time

        Returns:
            Variance for each asset, shape (n_assets,)
        """
        return X0**2 * np.exp(2 * self.mu * t) * (np.exp(self.sigma**2 * t) - 1.0)

    def covariance(self, X0: np.ndarray, t: float) -> np.ndarray:
        """
        Covariance matrix Cov[S_i(t), S_j(t)]

        Cov[S_i(t), S_j(t)] = S_i(0)*S_j(0) * exp((mu_i + mu_j)*t) *
                              (exp(rho_ij*sigma_i*sigma_j*t) - 1)

        Args:
            X0: Initial asset prices, shape (n_assets,)
            t: Time

        Returns:
            Covariance matrix, shape (n_assets, n_assets)
        """
        n = self.dim

        # Correlation matrix from Cholesky (recover from L*L^T)
        if self.cholesky_decomp is not None:
            rho = self.cholesky_decomp @ self.cholesky_decomp.T
        else:
            rho = np.eye(n)

        # Compute covariance
        cov = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                cov[i, j] = (X0[i] * X0[j] *
                             np.exp((self.mu[i] + self.mu[j]) * t) *
                             (np.exp(rho[i, j] * self.sigma[i] * self.sigma[j] * t) - 1.0))

        return cov

    def characteristic_function(self, u: np.ndarray, X0: np.ndarray, t: float) -> complex:
        """
        Joint characteristic function for log-prices

        phi(u) = E[exp(i * u^T * log(S(t)))]

        For multi-asset GBM with correlation:
        log(S(t)) ~ N(log(S(0)) + (mu - 0.5*sigma^2)*t, Sigma*t)

        where Sigma is the covariance matrix of log-returns:
        Sigma_ij = rho_ij * sigma_i * sigma_j

        Args:
            u: Frequency vector, shape (n_assets,)
            X0: Initial asset prices, shape (n_assets,)
            t: Time

        Returns:
            Characteristic function value (complex)
        """
        u = np.atleast_1d(u)
        log_S0 = np.log(X0)

        # Mean of log(S(t))
        mean = log_S0 + (self.mu - 0.5 * self.sigma**2) * t

        # Covariance matrix of log-returns
        if self.cholesky_decomp is not None:
            rho = self.cholesky_decomp @ self.cholesky_decomp.T
        else:
            rho = np.eye(self.dim)

        Sigma = np.outer(self.sigma, self.sigma) * rho * t

        # CF: exp(i*u^T*mean - 0.5*u^T*Sigma*u)
        return np.exp(1j * np.dot(u, mean) - 0.5 * np.dot(u, Sigma @ u))
