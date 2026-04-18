"""
Bates Stochastic Volatility Jump-Diffusion Model

Heston stochastic volatility + Merton log-normal jumps.

    dS_t = (mu - lambda*k) * S_t dt + sqrt(v_t) * S_t dW_t^S + S_{t-} dJ_t
    dv_t = kappa * (theta - v_t) dt + sigma_v * sqrt(v_t) dW_t^v

where:
    - J_t is compound Poisson with intensity lambda
    - Jump sizes: log(1+Y) ~ N(mu_J, sigma_J^2)
    - k = E[Y] = exp(mu_J + 0.5*sigma_J^2) - 1 (compensator)
    - corr(dW^S, dW^v) = rho

Reference:
    Bates (1996), "Jumps and Stochastic Volatility: Exchange Rate Processes
    Implicit in Deutsche Mark Options"

author: Yunian Pan
email: yp1170@nyu.edu
"""

import numpy as np
from .base import MultiFactorProcess


class Bates(MultiFactorProcess):
    """
    Bates Stochastic Volatility Jump-Diffusion Model.

    State vector: X = [S, v]

    Combines:
    - Heston stochastic variance dynamics for v_t
    - Merton-style log-normal jumps in the asset price S_t

    The characteristic function has a closed-form expression,
    enabling fast Fourier-based pricing (COS, Carr-Madan).
    """

    def __init__(
        self,
        mu: float,
        kappa: float,
        theta: float,
        sigma_v: float,
        rho: float,
        lambda_j: float,
        mu_J: float,
        sigma_J: float,
        v0: float = None,
        variance_scheme: str = "truncation",
        name: str = "Bates",
    ):
        """
        Initialize Bates model.

        Args:
            mu: Drift of asset price
            kappa: Mean reversion speed of variance
            theta: Long-term variance mean
            sigma_v: Volatility of volatility
            rho: Correlation between price and variance Brownian motions
            lambda_j: Jump intensity (expected jumps per year)
            mu_J: Mean of log-jump size
            sigma_J: Std dev of log-jump size
            v0: Initial variance (default: theta)
            variance_scheme: 'truncation', 'reflection', or 'absorption'
            name: Model name
        """
        super().__init__(dim=2, name=name)

        self.mu = float(mu)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma_v = float(sigma_v)
        self.rho = float(rho)
        self.lambda_j = float(lambda_j)
        self.mu_J = float(mu_J)
        self.sigma_J = float(sigma_J)
        self.v0 = float(v0) if v0 is not None else theta
        self.variance_scheme = variance_scheme

        # Expected jump size (compensator)
        self.jump_mean = np.exp(self.mu_J + 0.5 * self.sigma_J ** 2) - 1.0

        # Correlation between Brownian motions
        corr_matrix = np.array([[1.0, rho], [rho, 1.0]])
        self.set_correlation(corr_matrix)

        self.params['mu'] = self.mu
        self.params['kappa'] = self.kappa
        self.params['theta'] = self.theta
        self.params['sigma_v'] = self.sigma_v
        self.params['rho'] = self.rho
        self.params['lambda_j'] = self.lambda_j
        self.params['mu_J'] = self.mu_J
        self.params['sigma_J'] = self.sigma_J
        self.params['v0'] = self.v0

        # Feller condition for the variance process
        self.feller_condition = 2 * self.kappa * self.theta > self.sigma_v ** 2
        self.params['feller_satisfied'] = self.feller_condition

    def _build_jax_spec(self):
        if self.variance_scheme != "truncation":
            return None
        from ._process_defs import (
            BatesParams, bates_drift, bates_diffusion,
            bates_post_step, bates_jump_fn, heston_cholesky,
        )
        return {
            'drift_fn': bates_drift,
            'diffusion_fn': bates_diffusion,
            'params': BatesParams(
                mu=self.mu, kappa=self.kappa, theta=self.theta,
                sigma_v=self.sigma_v, rho=self.rho,
                lambda_j=self.lambda_j, mu_J=self.mu_J, sigma_J=self.sigma_J,
            ),
            'dim': 2,
            'cholesky': heston_cholesky(self.rho),
            'post_step_fn': bates_post_step,
            'jump_fn': bates_jump_fn,
        }

    def drift(self, X: np.ndarray, t: float) -> np.ndarray:
        """
        Drift term for Bates model.

        mu_S = (mu - lambda*k) * S_t   (jump-compensated drift)
        mu_v = kappa * (theta - v_t)

        Args:
            X: Current state [S, v], shape (n_paths, 2)
            t: Current time

        Returns:
            Drift vector, shape (n_paths, 2)
        """
        S = X[:, 0:1]
        v = X[:, 1:2]

        # Jump-compensated drift for asset
        drift_S = (self.mu - self.lambda_j * self.jump_mean) * S
        drift_v = self.kappa * (self.theta - v)

        return np.concatenate([drift_S, drift_v], axis=1)

    def diffusion(self, X: np.ndarray, t: float) -> np.ndarray:
        """
        Diffusion term for Bates model (same as Heston).

        sigma_S = sqrt(v_t) * S_t
        sigma_v = sigma_v * sqrt(v_t)

        Args:
            X: Current state [S, v], shape (n_paths, 2)
            t: Current time

        Returns:
            Diffusion coefficient, shape (n_paths, 2)
        """
        S = X[:, 0:1]
        v = X[:, 1:2]

        if self.variance_scheme == "truncation":
            v_pos = np.maximum(v, 0.0)
        elif self.variance_scheme == "reflection":
            v_pos = np.abs(v)
        else:
            v_pos = np.where(v > 0, v, 0.0)

        sqrt_v = np.sqrt(v_pos)
        sigma_S = sqrt_v * S
        sigma_v = self.sigma_v * sqrt_v

        return np.concatenate([sigma_S, sigma_v], axis=1)

    def jump_component(self, X: np.ndarray, t: float, dt: float) -> np.ndarray:
        """
        Multiplicative Merton-style jumps applied to asset price only.

        Args:
            X: Current state [S, v], shape (n_paths, 2)
            t: Current time
            dt: Time step

        Returns:
            Jump contribution, shape (n_paths, 2)
        """
        n_paths = X.shape[0]
        jumps = np.zeros((n_paths, self.dim))

        S = X[:, 0]
        n_jumps_per_path = np.random.poisson(self.lambda_j * dt, n_paths)
        unique_counts = np.unique(n_jumps_per_path)

        for n_j in unique_counts:
            if n_j == 0:
                continue
            mask = n_jumps_per_path == n_j
            n_with_jumps = np.sum(mask)

            # Log-normal jump sizes: Y = exp(Z) - 1, Z ~ N(mu_J, sigma_J)
            log_jumps = np.random.normal(
                self.mu_J, self.sigma_J, (n_with_jumps, n_j)
            )
            total_Y = (np.exp(log_jumps) - 1.0).sum(axis=1)

            jumps[mask, 0] = S[mask] * total_Y

        return jumps

    def _euler_maruyama(self, X0, T, dt, t_grid, config):
        """Override to handle variance positivity."""
        t_grid, paths = super()._euler_maruyama(X0, T, dt, t_grid, config)

        if self.variance_scheme == "truncation":
            paths[:, :, 1] = np.maximum(paths[:, :, 1], 0.0)
        elif self.variance_scheme == "reflection":
            paths[:, :, 1] = np.abs(paths[:, :, 1])

        return t_grid, paths

    def characteristic_function(self, u: complex, X0: np.ndarray, t: float) -> complex:
        """
        Characteristic function for log(S_t) in the Bates model.

        phi_Bates(u) = phi_Heston(u) * phi_jumps(u)

        where:
            phi_Heston is the standard Heston CF (with jump-compensated drift)
            phi_jumps = exp(lambda*t * (exp(i*u*mu_J - 0.5*u^2*sigma_J^2) - 1))

        Args:
            u: Frequency parameter
            X0: Initial state [S_0, v_0]
            t: Time

        Returns:
            Characteristic function value
        """
        S0, v0 = X0[0], X0[1]
        log_S0 = np.log(S0)

        # --- Heston CF component (with jump-compensated drift) ---
        mu_adj = self.mu - self.lambda_j * self.jump_mean

        d = np.sqrt(
            (self.rho * self.sigma_v * 1j * u - self.kappa) ** 2
            + self.sigma_v ** 2 * (1j * u + u ** 2)
        )

        g = (self.kappa - self.rho * self.sigma_v * 1j * u - d) / \
            (self.kappa - self.rho * self.sigma_v * 1j * u + d)

        exp_dt = np.exp(-d * t)
        C = (
            mu_adj * 1j * u * t
            + (self.kappa * self.theta / self.sigma_v ** 2)
            * (
                (self.kappa - self.rho * self.sigma_v * 1j * u - d) * t
                - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g))
            )
        )

        D = (
            (self.kappa - self.rho * self.sigma_v * 1j * u - d)
            / self.sigma_v ** 2
        ) * ((1.0 - exp_dt) / (1.0 - g * exp_dt))

        phi_heston = np.exp(C + D * v0 + 1j * u * log_S0)

        # --- Jump CF component ---
        phi_jumps = np.exp(
            self.lambda_j * t * (
                np.exp(1j * u * self.mu_J - 0.5 * u ** 2 * self.sigma_J ** 2)
                - 1.0
            )
        )

        return phi_heston * phi_jumps
