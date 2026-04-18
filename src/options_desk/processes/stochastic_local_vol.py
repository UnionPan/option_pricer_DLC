"""
Stochastic Local Volatility (SLV) Model

Combines Dupire local volatility with Heston-style stochastic volatility
via a leverage function L(S, t):

    dS_t = mu * S_t dt + L(S_t, t) * sqrt(v_t) * S_t dW_t^S
    dv_t = kappa * (theta - v_t) dt + sigma_v * sqrt(v_t) dW_t^v

The leverage function L is calibrated so that the SLV model exactly
reprices vanilla options (matching the Dupire local vol surface).

References:
    - Ren, Y., Madan, D., Qian, M.Q. (2007) "Calibrating and Pricing
      with Embedded Local Volatility Models"
    - Guyon, J., Henry-Labordere, P. (2012) "Being Particular About
      Calibration"

author: Yunian Pan
email: yp1170@nyu.edu
"""
import numpy as np
from scipy import interpolate
from typing import Callable, Dict, Optional, Tuple
from .base import MultiFactorProcess
from ._structured_inputs import StructuredGridSurface


class StochasticLocalVol(MultiFactorProcess):
    """
    Stochastic Local Volatility (SLV) Model

    Two-dimensional system:
        dS_t = mu * S_t dt + L(S_t, t) * sqrt(v_t) * S_t dW_t^S
        dv_t = kappa * (theta - v_t) dt + sigma_v * sqrt(v_t) dW_t^v

    where:
        - S_t: asset price
        - v_t: instantaneous variance (CIR dynamics, same as Heston)
        - L(S, t): leverage function connecting local vol to stochastic vol
        - kappa, theta, sigma_v, rho: Heston parameters

    The leverage function satisfies:
        sigma_LV(K, T)^2 = L(K, T)^2 * E[v_T | S_T = K]

    Special cases:
        - L(S, t) = 1: reduces to Heston
        - v_t = 1 constant: reduces to Dupire local vol

    State vector: X = [S, v]
    """

    def __init__(
        self,
        mu: float,
        kappa: float,
        theta: float,
        sigma_v: float,
        rho: float,
        leverage_fn: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
        spot_grid: Optional[np.ndarray] = None,
        time_grid: Optional[np.ndarray] = None,
        leverage_surface: Optional[np.ndarray] = None,
        v0: float = None,
        variance_scheme: str = "truncation",
        name: str = "SLV",
    ):
        """
        Initialize SLV model.

        Args:
            mu: Drift of asset price.
            kappa: Mean reversion speed (kappa > 0).
            theta: Long-term variance mean (theta > 0).
            sigma_v: Volatility of volatility (sigma_v > 0).
            rho: Correlation between price and vol Brownians.
            leverage_fn: Callable (S, t) -> array. The leverage function
                L(S, t) applied to the spot diffusion. Must accept
                vectorized S of shape (n_paths, 1) and scalar t.
            spot_grid: Optional spot grid for a JAX-compatible leverage surface.
            time_grid: Optional time grid for a JAX-compatible leverage surface.
            leverage_surface: Optional leverage values on (time_grid, spot_grid).
            v0: Initial variance (defaults to theta).
            variance_scheme: 'truncation', 'reflection', or 'absorption'.
            name: Model name.
        """
        super().__init__(dim=2, name=name)

        self.mu = float(mu)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma_v = float(sigma_v)
        self.rho = float(rho)
        self.v0 = float(v0) if v0 is not None else theta
        self.variance_scheme = variance_scheme
        self.leverage_surface_data = None
        if spot_grid is None and time_grid is None and leverage_surface is None:
            self.leverage_fn = leverage_fn
        else:
            if spot_grid is None or time_grid is None or leverage_surface is None:
                raise ValueError("spot_grid, time_grid, and leverage_surface must be provided together")
            self.leverage_surface_data = StructuredGridSurface(spot_grid, time_grid, leverage_surface)
            if leverage_fn is None:
                leverage_fn = self.leverage_surface_data.as_numpy_callable()
            self.leverage_fn = leverage_fn

        if self.leverage_fn is None:
            raise ValueError("Either leverage_fn or a structured leverage surface must be provided")

        self.spot_grid = None if self.leverage_surface_data is None else self.leverage_surface_data.x_grid
        self.time_grid = None if self.leverage_surface_data is None else self.leverage_surface_data.y_grid
        self.leverage_surface = None if self.leverage_surface_data is None else self.leverage_surface_data.values

        corr_matrix = np.array([
            [1.0, rho],
            [rho, 1.0],
        ])
        self.set_correlation(corr_matrix)

        self.params['mu'] = self.mu
        self.params['kappa'] = self.kappa
        self.params['theta'] = self.theta
        self.params['sigma_v'] = self.sigma_v
        self.params['rho'] = self.rho
        self.params['v0'] = self.v0
        if self.leverage_surface_data is not None:
            self.params['spot_grid'] = self.leverage_surface_data.x_grid
            self.params['time_grid'] = self.leverage_surface_data.y_grid
            self.params['leverage_surface'] = self.leverage_surface_data.values

        self.feller_condition = 2 * self.kappa * self.theta > self.sigma_v ** 2
        self.params['feller_satisfied'] = self.feller_condition

    def _build_jax_spec(self):
        if self.leverage_surface_data is None:
            return None

        from ._process_defs import (
            SLVParams,
            heston_cholesky,
            slv_drift,
            slv_drift_absorption,
            slv_diffusion,
            slv_diffusion_absorption,
            slv_diffusion_reflection,
            slv_post_step,
            slv_post_step_reflection,
        )

        params = SLVParams(
            mu=self.mu,
            kappa=self.kappa,
            theta=self.theta,
            sigma_v=self.sigma_v,
            rho=self.rho,
            v0=self.v0,
            spot_grid=self.leverage_surface_data.x_grid,
            time_grid=self.leverage_surface_data.y_grid,
            leverage_surface=self.leverage_surface_data.values,
        )

        if self.variance_scheme == "truncation":
            return {
                'drift_fn': slv_drift,
                'diffusion_fn': slv_diffusion,
                'params': params,
                'dim': 2,
                'cholesky': heston_cholesky(self.rho),
                'post_step_fn': slv_post_step,
            }
        if self.variance_scheme == "reflection":
            return {
                'drift_fn': slv_drift,
                'diffusion_fn': slv_diffusion_reflection,
                'params': params,
                'dim': 2,
                'cholesky': heston_cholesky(self.rho),
                'post_step_fn': slv_post_step_reflection,
            }
        if self.variance_scheme == "absorption":
            return {
                'drift_fn': slv_drift_absorption,
                'diffusion_fn': slv_diffusion_absorption,
                'params': params,
                'dim': 2,
                'cholesky': heston_cholesky(self.rho),
            }
        return None

    def drift(self, X: np.ndarray, t: float) -> np.ndarray:
        """
        Drift term (identical to Heston).

        mu_S = mu * S_t
        mu_v = kappa * (theta - v_t)

        Args:
            X: Current state [S, v], shape (n_paths, 2).
            t: Current time.

        Returns:
            Drift vector, shape (n_paths, 2).
        """
        S = X[:, 0:1]
        v = X[:, 1:2]

        if self.variance_scheme == "absorption":
            v_drift = np.where(v > 0, self.kappa * (self.theta - v), 0.0)
        else:
            v_drift = self.kappa * (self.theta - v)

        return np.concatenate([self.mu * S, v_drift], axis=1)

    def diffusion(self, X: np.ndarray, t: float) -> np.ndarray:
        """
        Diffusion term with leverage function.

        sigma_S = L(S, t) * sqrt(v_t) * S_t
        sigma_v = sigma_v * sqrt(v_t)

        Args:
            X: Current state [S, v], shape (n_paths, 2).
            t: Current time.

        Returns:
            Diffusion coefficient, shape (n_paths, 2).
        """
        S = X[:, 0:1]
        v = X[:, 1:2]

        if self.variance_scheme == "truncation":
            v_pos = np.maximum(v, 0.0)
        elif self.variance_scheme == "reflection":
            v_pos = np.abs(v)
        elif self.variance_scheme == "absorption":
            v_pos = np.where(v > 0, v, 0.0)
        else:
            v_pos = v

        sqrt_v = np.sqrt(v_pos)

        L = self.leverage_fn(S, t)
        sigma_S = L * sqrt_v * S
        sigma_v = self.sigma_v * sqrt_v

        return np.concatenate([sigma_S, sigma_v], axis=1)

    def _euler_maruyama(self, X0, T, dt, t_grid, config):
        """Override to enforce variance positivity after each step."""
        t_grid, paths = super()._euler_maruyama(X0, T, dt, t_grid, config)

        if self.variance_scheme == "truncation":
            paths[:, :, 1] = np.maximum(paths[:, :, 1], 0.0)
        elif self.variance_scheme == "reflection":
            paths[:, :, 1] = np.abs(paths[:, :, 1])

        return t_grid, paths


# ============================================================================
# Factory: build SLV from a DupireResult + Heston params
# ============================================================================

def from_dupire_result(
    dupire_result,
    heston_params: Dict[str, float],
    mu: float = 0.0,
) -> StochasticLocalVol:
    """
    Build an SLV model from a calibrated Dupire surface and Heston params.

    Uses the simple ratio approximation:
        L(S, t) = sigma_LV(S, t) / sqrt(theta)

    This is a first-order approximation valid when the stochastic variance
    is close to its long-run mean. For a more accurate leverage function,
    use ``calibrate_leverage_mc``.

    Args:
        dupire_result: A DupireResult with ``lv_function(K, T)``.
        heston_params: Dict with keys kappa, theta, sigma_v, rho, v0.
        mu: Drift of the asset.

    Returns:
        Configured StochasticLocalVol instance.
    """
    theta = heston_params['theta']
    sqrt_theta = np.sqrt(theta)

    lv_fn = dupire_result.lv_function

    def leverage_fn(S: np.ndarray, t: float) -> np.ndarray:
        L = np.ones_like(S)
        for i in range(S.shape[0]):
            sigma_lv = lv_fn(float(S[i, 0]), float(t))
            L[i, 0] = sigma_lv / sqrt_theta
        return np.maximum(L, 0.01)

    return StochasticLocalVol(
        mu=mu,
        kappa=heston_params['kappa'],
        theta=theta,
        sigma_v=heston_params['sigma_v'],
        rho=heston_params['rho'],
        leverage_fn=leverage_fn,
        v0=heston_params.get('v0', theta),
    )


# ============================================================================
# MC leverage calibration (particle method)
# ============================================================================

def calibrate_leverage_mc(
    heston_params: Dict[str, float],
    lv_function: Callable[[float, float], float],
    S0: float,
    T_max: float = 1.0,
    n_paths: int = 50_000,
    n_steps: int = 100,
    n_strike_bins: int = 40,
    bandwidth: float = 0.05,
    mu: float = 0.0,
) -> Tuple[StochasticLocalVol, Callable]:
    """
    Calibrate the leverage function via Monte Carlo conditional expectation.

    Procedure (Guyon-Henry-Labordere particle method, simplified):
    1. Simulate Heston paths to get joint (S_t, v_t) samples at each t.
    2. For each time slice, estimate E[v_t | S_t = K] via kernel regression
       on a strike grid.
    3. Compute L(K, t) = sigma_LV(K, t) / sqrt(E[v_t | S_t = K]).
    4. Build a 2-D interpolant for L(S, t).

    Args:
        heston_params: Dict with kappa, theta, sigma_v, rho, v0.
        lv_function: Dupire local vol callable sigma_LV(K, T).
        S0: Spot price.
        T_max: Maximum maturity for the leverage grid.
        n_paths: Number of MC paths for conditional expectation.
        n_steps: Number of time steps.
        n_strike_bins: Number of strike nodes in the leverage grid.
        bandwidth: Gaussian kernel bandwidth (fraction of S0).
        mu: Drift.

    Returns:
        (slv_model, leverage_fn): Configured SLV model and the raw
            leverage interpolant.
    """
    from .heston import Heston
    from .base import SimulationConfig

    dt = T_max / n_steps
    t_grid = np.linspace(0, T_max, n_steps + 1)

    # Simulate Heston paths
    heston = Heston(
        mu=mu,
        kappa=heston_params['kappa'],
        theta=heston_params['theta'],
        sigma_v=heston_params['sigma_v'],
        rho=heston_params['rho'],
        v0=heston_params.get('v0', heston_params['theta']),
    )
    X0 = np.array([[S0, heston_params.get('v0', heston_params['theta'])]])
    cfg = SimulationConfig(n_paths=n_paths, n_steps=n_steps)
    _, paths = heston.simulate(X0, T=T_max, config=cfg)
    # paths shape: (n_steps+1, n_paths, 2)

    # Strike grid for leverage function
    S_all = paths[:, :, 0]
    K_min = max(np.percentile(S_all, 1), S0 * 0.5)
    K_max = min(np.percentile(S_all, 99), S0 * 2.0)
    strike_grid = np.linspace(K_min, K_max, n_strike_bins)

    # Time grid (skip t=0)
    time_indices = range(1, n_steps + 1)
    leverage_grid = np.ones((len(time_indices), n_strike_bins))

    bw = bandwidth * S0

    for ti_idx, ti in enumerate(time_indices):
        t = t_grid[ti]
        S_t = paths[ti, :, 0]
        v_t = paths[ti, :, 1]

        for ki, K in enumerate(strike_grid):
            # Gaussian kernel weights
            weights = np.exp(-0.5 * ((S_t - K) / bw) ** 2)
            w_sum = weights.sum()

            if w_sum < 1e-12:
                leverage_grid[ti_idx, ki] = 1.0
                continue

            # E[v_t | S_t ≈ K]
            cond_var = np.dot(weights, v_t) / w_sum
            cond_var = max(cond_var, 1e-10)

            sigma_lv = lv_function(float(K), float(t))
            leverage_grid[ti_idx, ki] = sigma_lv / np.sqrt(cond_var)

    # Clip to reasonable range
    leverage_grid = np.clip(leverage_grid, 0.01, 10.0)

    # Build 2-D interpolant: L(K, t)
    time_nodes = t_grid[1:]  # skip t=0
    lev_interp = interpolate.RectBivariateSpline(
        time_nodes, strike_grid, leverage_grid, kx=2, ky=2,
    )

    def leverage_fn(S: np.ndarray, t: float) -> np.ndarray:
        t_clip = np.clip(t, time_nodes[0], time_nodes[-1])
        L = np.ones_like(S)
        for i in range(S.shape[0]):
            s_clip = np.clip(float(S[i, 0]), K_min, K_max)
            L[i, 0] = float(lev_interp(t_clip, s_clip)[0, 0])
        return np.maximum(L, 0.01)

    slv = StochasticLocalVol(
        mu=mu,
        kappa=heston_params['kappa'],
        theta=heston_params['theta'],
        sigma_v=heston_params['sigma_v'],
        rho=heston_params['rho'],
        leverage_fn=leverage_fn,
        v0=heston_params.get('v0', heston_params['theta']),
    )

    return slv, leverage_fn
