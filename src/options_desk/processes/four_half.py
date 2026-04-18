"""
4/2 Stochastic Volatility Model (Grasselli 2017)

Two-factor model combining sqrt(v) and 1/sqrt(v) in the spot volatility

author: Yunian Pan
email: yp1170@nyu.edu
"""
import numpy as np
from .base import MultiFactorProcess


class FourHalf(MultiFactorProcess):
    """
    4/2 Stochastic Volatility Model

    Two-dimensional system:
        dS_t = mu * S_t dt + (a*sqrt(v_t) + b/sqrt(v_t)) * S_t dW_t^S
        dv_t = kappa * (theta - v_t) dt + sigma_v * sqrt(v_t) dW_t^v

    where:
        - S_t: asset price
        - v_t: instantaneous variance (CIR process, same as Heston)
        - mu: drift of asset
        - kappa: mean reversion speed of variance
        - theta: long-term mean of variance
        - sigma_v: volatility of volatility
        - rho: correlation between dW^S and dW^v
        - a: weight on sqrt(v_t) component
        - b: weight on 1/sqrt(v_t) component

    State vector: X = [S, v]

    The "4/2" name comes from combining the 1/2-power (sqrt) and
    the -1/2-power (inverse sqrt) of v_t in the spot volatility,
    giving effective powers of 1/2 and 3/2 when combined with the
    CIR variance process.

    Special cases:
        - b=0: reduces to Heston model (with a scaling the vol)
        - a=0: pure inverse-sqrt volatility model

    The 4/2 model captures:
        - Stochastic volatility with mean reversion
        - Both leverage and inverse leverage effects
        - Volatility that increases when variance is both very high AND very low
        - Richer implied volatility surface than Heston
    """

    def __init__(
        self,
        mu: float,
        kappa: float,
        theta: float,
        sigma_v: float,
        rho: float,
        a: float,
        b: float,
        v0: float = None,
        variance_scheme: str = "truncation",
        name: str = "4/2 Model"
    ):
        """
        Initialize 4/2 model

        Args:
            mu: Drift of asset price
            kappa: Mean reversion speed (kappa > 0)
            theta: Long-term variance mean (theta > 0)
            sigma_v: Volatility of volatility (sigma_v > 0)
            rho: Correlation between price and vol Brownian motions (-1 < rho < 1)
            a: Weight on sqrt(v_t) component in spot volatility
            b: Weight on 1/sqrt(v_t) component in spot volatility
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
        self.a = float(a)
        self.b = float(b)
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
        self.params['a'] = self.a
        self.params['b'] = self.b
        self.params['v0'] = self.v0

        # Feller condition for the CIR variance process
        self.feller_condition = 2 * self.kappa * self.theta > self.sigma_v**2
        self.params['feller_satisfied'] = self.feller_condition

    def _build_jax_spec(self):
        if self.variance_scheme != "truncation":
            return None
        from ._process_defs import (
            FourHalfParams, four_half_drift, four_half_diffusion,
            four_half_post_step, heston_cholesky,
        )
        return {
            'drift_fn': four_half_drift,
            'diffusion_fn': four_half_diffusion,
            'params': FourHalfParams(
                mu=self.mu, kappa=self.kappa, theta=self.theta,
                sigma_v=self.sigma_v, rho=self.rho, a=self.a, b=self.b,
            ),
            'dim': 2,
            'cholesky': heston_cholesky(self.rho),
            'post_step_fn': four_half_post_step,
        }

    def drift(self, X: np.ndarray, t: float) -> np.ndarray:
        """
        Drift term for 4/2 model

        mu_S = mu * S_t
        mu_v = kappa * (theta - v_t)

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
            v_drift = np.where(v > 0, self.kappa * (self.theta - v), 0.0)
        else:
            v_drift = self.kappa * (self.theta - v)

        drift_S = self.mu * S
        drift_v = v_drift

        return np.concatenate([drift_S, drift_v], axis=1)

    def diffusion(self, X: np.ndarray, t: float) -> np.ndarray:
        """
        Diffusion term for 4/2 model

        sigma_S = (a*sqrt(v_t) + b/sqrt(v_t)) * S_t
        sigma_v = sigma_v * sqrt(v_t)

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

        # Floor to avoid division by zero in b/sqrt(v)
        v_safe = np.maximum(v_pos, 1e-12)
        sqrt_v = np.sqrt(v_safe)

        # 4/2 spot volatility: a*sqrt(v) + b/sqrt(v)
        spot_vol = self.a * sqrt_v + self.b / sqrt_v

        sigma_S = spot_vol * S
        sigma_v = self.sigma_v * sqrt_v

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

        For 4/2 model, Milstein correction for variance is same as Heston:
            d(sigma_v * sqrt(v))/dv = sigma_v / (2*sqrt(v))
            correction_v = 0.5 * sigma_v*sqrt(v) * sigma_v/(2*sqrt(v)) * (dW^2 - dt)
                         = 0.25 * sigma_v^2 * (dW^2 - dt)
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

            # Milstein correction (only for variance process, CIR dynamics)
            dW_v = dW[:, 1:2]
            milstein_correction_v = 0.25 * self.sigma_v**2 * (dW_v**2 - dt)
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

                dW_v = dW[:, 1:2]
                milstein_correction_v = 0.25 * self.sigma_v**2 * (dW_v**2 - dt)
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
