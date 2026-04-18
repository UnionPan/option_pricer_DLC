"""
3/2 Stochastic Volatility Model

Two-factor model where variance dynamics have v_t^(3/2) diffusion

author: Yunian Pan
email: yp1170@nyu.edu
"""
import numpy as np
from .base import MultiFactorProcess


class ThreeHalf(MultiFactorProcess):
    """
    3/2 Stochastic Volatility Model

    Two-dimensional system:
        dS_t = mu * S_t dt + sqrt(v_t) * S_t dW_t^S
        dv_t = kappa * v_t * (theta - v_t) dt + sigma_v * v_t^(3/2) dW_t^v

    where:
        - S_t: asset price
        - v_t: instantaneous variance
        - mu: drift of asset
        - kappa: mean reversion speed of variance
        - theta: long-term mean of variance
        - sigma_v: volatility of volatility
        - rho: correlation between dW^S and dW^v

    State vector: X = [S, v]

    Key differences from Heston:
        - Mean reversion has v_t factor: kappa * v_t * (theta - v_t)
        - Diffusion scales as v_t^(3/2) instead of sqrt(v_t)
        - Volatility of variance is proportional to variance level
        - Captures inverse leverage effect more naturally
        - Variance process is the reciprocal of a CIR process
    """

    def __init__(
        self,
        mu: float,
        kappa: float,
        theta: float,
        sigma_v: float,
        rho: float,
        v0: float = None,
        variance_scheme: str = "truncation",
        name: str = "3/2 Model"
    ):
        """
        Initialize 3/2 model

        Args:
            mu: Drift of asset price
            kappa: Mean reversion speed (kappa > 0)
            theta: Long-term variance mean (theta > 0)
            sigma_v: Volatility of volatility (sigma_v > 0)
            rho: Correlation between price and vol Brownian motions (-1 < rho < 1)
            v0: Initial variance (optional, defaults to theta)
            variance_scheme: How to handle negative variance
                - 'truncation': Set v_t = max(v_t, 0) (default)
                - 'reflection': Reflect at zero |v_t|
                - 'absorption': Set drift/diffusion to 0 when v_t <= 0
            name: Model name
        """
        super().__init__(dim=2, name=name)

        self.mu = float(mu)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma_v = float(sigma_v)
        self.rho = float(rho)
        self.v0 = float(v0) if v0 is not None else theta
        self.variance_scheme = variance_scheme

        # Set correlation between the two Brownian motions
        corr_matrix = np.array([
            [1.0, rho],
            [rho, 1.0]
        ])
        self.set_correlation(corr_matrix)

        self.params['mu'] = self.mu
        self.params['kappa'] = self.kappa
        self.params['theta'] = self.theta
        self.params['sigma_v'] = self.sigma_v
        self.params['rho'] = self.rho
        self.params['v0'] = self.v0

    def _build_jax_spec(self):
        if self.variance_scheme != "truncation":
            return None
        from ._process_defs import (
            ThreeHalfParams, three_half_drift, three_half_diffusion,
            three_half_post_step, heston_cholesky,
        )
        return {
            'drift_fn': three_half_drift,
            'diffusion_fn': three_half_diffusion,
            'params': ThreeHalfParams(
                mu=self.mu, kappa=self.kappa, theta=self.theta,
                sigma_v=self.sigma_v, rho=self.rho,
            ),
            'dim': 2,
            'cholesky': heston_cholesky(self.rho),
            'post_step_fn': three_half_post_step,
        }

    def drift(self, X: np.ndarray, t: float) -> np.ndarray:
        """
        Drift term for 3/2 model

        mu_S = mu * S_t
        mu_v = kappa * v_t * (theta - v_t)

        Args:
            X: Current state [S, v], shape (n_paths, 2)
            t: Current time

        Returns:
            Drift vector, shape (n_paths, 2)
        """
        S = X[:, 0:1]  # Shape (n_paths, 1)
        v = X[:, 1:2]  # Shape (n_paths, 1)

        # Handle negative variance in drift
        if self.variance_scheme == "absorption":
            v_drift = np.where(v > 0, self.kappa * v * (self.theta - v), 0.0)
        else:
            v_drift = self.kappa * v * (self.theta - v)

        drift_S = self.mu * S
        drift_v = v_drift

        return np.concatenate([drift_S, drift_v], axis=1)

    def diffusion(self, X: np.ndarray, t: float) -> np.ndarray:
        """
        Diffusion term for 3/2 model

        sigma_S = sqrt(v_t) * S_t
        sigma_v = sigma_v * v_t^(3/2)

        Args:
            X: Current state [S, v], shape (n_paths, 2)
            t: Current time

        Returns:
            Diffusion coefficient (diagonal), shape (n_paths, 2)
        """
        S = X[:, 0:1]  # Shape (n_paths, 1)
        v = X[:, 1:2]  # Shape (n_paths, 1)

        # Handle negative variance
        if self.variance_scheme == "truncation":
            v_pos = np.maximum(v, 0.0)
        elif self.variance_scheme == "reflection":
            v_pos = np.abs(v)
        elif self.variance_scheme == "absorption":
            v_pos = np.where(v > 0, v, 0.0)
        else:
            v_pos = v

        sqrt_v = np.sqrt(v_pos)

        sigma_S = sqrt_v * S
        sigma_v = self.sigma_v * v_pos * sqrt_v  # v^(3/2) = v * sqrt(v)

        return np.concatenate([sigma_S, sigma_v], axis=1)

    def _euler_maruyama(self, X0, T, dt, t_grid, config):
        """
        Override Euler-Maruyama to handle variance positivity

        After each step, apply variance scheme to ensure v_t stays valid.
        """
        # Call parent class method
        t_grid, paths = super()._euler_maruyama(X0, T, dt, t_grid, config)

        # Apply variance scheme to all variance paths
        if self.variance_scheme == "truncation":
            paths[:, :, 1] = np.maximum(paths[:, :, 1], 0.0)
        elif self.variance_scheme == "reflection":
            paths[:, :, 1] = np.abs(paths[:, :, 1])
        # absorption scheme is handled in drift/diffusion

        return t_grid, paths

    def _milstein(self, X0, T, dt, t_grid, config):
        """
        Override Milstein to handle variance positivity and add correction terms

        For 3/2 model, Milstein adds:
            0.5 * sigma * sigma' * (dW^2 - dt)

        For S: same as Heston (no correction needed for diagonal S component)
        For v: d(sigma_v * v^(3/2))/dv = sigma_v * (3/2) * v^(1/2)
               correction = 0.5 * sigma_v * v^(3/2) * sigma_v * (3/2) * v^(1/2) * (dW^2 - dt)
                          = 0.75 * sigma_v^2 * v^2 * (dW^2 - dt)
        """
        n_paths = config.n_paths
        if config.antithetic:
            n_paths = n_paths // 2

        paths = np.zeros((len(t_grid), n_paths, self.dim))
        paths[0] = X0
        sqrt_dt = np.sqrt(dt)

        for i, t in enumerate(t_grid[:-1]):
            X_current = paths[i]

            # Generate correlated Brownian increments
            dW = np.random.normal(0, sqrt_dt, size=(n_paths, self.dim))
            if self.cholesky_decomp is not None:
                dW = dW @ self.cholesky_decomp.T

            # Euler terms
            drift_term = self.drift(X_current, t) * dt
            sigma = self.diffusion(X_current, t)
            diffusion_term = self._apply_diffusion(sigma, dW)

            # Milstein correction (only for variance process)
            v = X_current[:, 1:2]
            v_pos = np.maximum(v, 0.0)
            dW_v = dW[:, 1:2]
            milstein_correction_v = 0.75 * self.sigma_v**2 * v_pos**2 * (dW_v**2 - dt)
            milstein_correction = np.concatenate([
                np.zeros((n_paths, 1)),  # No correction for S
                milstein_correction_v
            ], axis=1)

            paths[i + 1] = X_current + drift_term + diffusion_term + milstein_correction

        # Apply variance scheme
        if self.variance_scheme == "truncation":
            paths[:, :, 1] = np.maximum(paths[:, :, 1], 0.0)
        elif self.variance_scheme == "reflection":
            paths[:, :, 1] = np.abs(paths[:, :, 1])

        # Antithetic paths
        if config.antithetic:
            paths_anti = np.zeros((len(t_grid), n_paths, self.dim))
            paths_anti[0] = X0

            if config.random_seed is not None:
                np.random.seed(config.random_seed)

            for i, t in enumerate(t_grid[:-1]):
                X_current = paths_anti[i]

                dW = -np.random.normal(0, sqrt_dt, size=(n_paths, self.dim))
                if self.cholesky_decomp is not None:
                    dW = dW @ self.cholesky_decomp.T

                drift_term = self.drift(X_current, t) * dt
                sigma = self.diffusion(X_current, t)
                diffusion_term = self._apply_diffusion(sigma, dW)

                v = X_current[:, 1:2]
                v_pos = np.maximum(v, 0.0)
                dW_v = dW[:, 1:2]
                milstein_correction_v = 0.75 * self.sigma_v**2 * v_pos**2 * (dW_v**2 - dt)
                milstein_correction = np.concatenate([
                    np.zeros((n_paths, 1)),
                    milstein_correction_v
                ], axis=1)

                paths_anti[i + 1] = X_current + drift_term + diffusion_term + milstein_correction

            # Apply variance scheme
            if self.variance_scheme == "truncation":
                paths_anti[:, :, 1] = np.maximum(paths_anti[:, :, 1], 0.0)
            elif self.variance_scheme == "reflection":
                paths_anti[:, :, 1] = np.abs(paths_anti[:, :, 1])

            paths = np.concatenate([paths, paths_anti], axis=1)

        return t_grid, paths

    def characteristic_function(self, u: complex, X0: np.ndarray, t: float) -> complex:
        """
        Characteristic function for log(S_t) in the 3/2 model

        The 3/2 model admits a semi-analytical characteristic function via the
        reciprocal CIR trick. Defining y_t = 1/v_t, y_t follows a CIR process:

            dy_t = [sigma_v^2 - kappa*theta + (kappa - rho*sigma_v*iu)*y_t] dt
                   - sigma_v * sqrt(y_t) dW_t^v

        The CF takes the form:
            phi(u) = exp(i*u*log(S_0) + i*u*mu*t) * G(y_0, t)

        where G involves the confluent hypergeometric function (Kummer's U).
        The full derivation requires careful treatment of complex-valued Kummer
        functions which is non-trivial to implement robustly.

        Args:
            u: Frequency parameter
            X0: Initial state [S_0, v_0]
            t: Time

        Returns:
            Characteristic function value

        Raises:
            NotImplementedError: The 3/2 CF involves confluent hypergeometric
                functions (Kummer U) with complex parameters. A robust
                implementation requires scipy.special or mpmath and careful
                branch-cut handling. Left for future work.
        """
        raise NotImplementedError(
            "Characteristic function for the 3/2 model requires confluent "
            "hypergeometric functions (Kummer U) with complex parameters. "
            "Use Monte Carlo simulation for pricing instead."
        )
