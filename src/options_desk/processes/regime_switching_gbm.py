"""
Regime-Switching Geometric Brownian Motion

GBM with parameters that switch between regimes

author: Yunian Pan
email: yp1170@nyu.edu
"""
import numpy as np
import warnings
from .regime_switching import RegimeSwitchingProcess
from ._jax_backend import should_fallback_to_numpy


class RegimeSwitchingGBM(RegimeSwitchingProcess):
    """
    Regime-Switching GBM

    dS_t = mu(regime_t) * S_t dt + sigma(regime_t) * S_t dW_t

    where regime_t switches according to continuous-time Markov chain.

    Usage:
        model = RegimeSwitchingGBM(n_regimes=2)
        model.set_regime_params('mu', [0.15, -0.05])     # Bull, Bear
        model.set_regime_params('sigma', [0.20, 0.40])   # Low vol, High vol
        model.set_transition_matrix(Q)
    """

    def __init__(self, n_regimes: int, name: str = "RegimeSwitchingGBM"):
        super().__init__(n_regimes=n_regimes, dim=1, name=name)

    def simulate(self, X0, T, config, scheme='euler'):
        from .base import _JAX_AVAILABLE, validate_simulation_config
        validate_simulation_config(config)
        if (
            _JAX_AVAILABLE
            and scheme.lower() in ("euler", "milstein")
            and not config.use_sobol
            and 'mu' in self.regime_params
            and 'sigma' in self.regime_params
            and self.transition_matrix is not None
        ):
            import jax.numpy as jnp
            from ._process_defs import (
                RegimeSwitchingGBMParams, regime_switching_simulate,
            )
            seed = config.random_seed if config.random_seed is not None else 0
            try:
                t_grid, paths, _ = regime_switching_simulate(
                    RegimeSwitchingGBMParams(
                        mus=jnp.array(self.regime_params['mu']),
                        sigmas=jnp.array(self.regime_params['sigma']),
                        Q=jnp.array(self.transition_matrix),
                    ),
                    X0, T, config.n_paths, config.n_steps, seed=seed, dim=self.dim,
                )
                return t_grid, paths
            except Exception as exc:
                if not should_fallback_to_numpy(exc):
                    raise
                warnings.warn(
                    "JAX backend initialization failed for RegimeSwitchingGBM; falling back to NumPy simulation.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return super().simulate(X0, T, config, scheme=scheme)

    def drift_regime(self, X: np.ndarray, t: float, regime: int) -> np.ndarray:
        """
        Drift for specific regime: mu(regime) * S_t

        Args:
            X: Current state, shape (n_paths, 1)
            t: Current time
            regime: Current regime index

        Returns:
            Drift coefficient, shape (n_paths, 1)
        """
        mu = self.regime_params['mu'][regime]
        return mu * X

    def diffusion_regime(self, X: np.ndarray, t: float, regime: int) -> np.ndarray:
        """
        Diffusion for specific regime: sigma(regime) * S_t

        Args:
            X: Current state, shape (n_paths, 1)
            t: Current time
            regime: Current regime index

        Returns:
            Diffusion coefficient, shape (n_paths, 1)
        """
        sigma = self.regime_params['sigma'][regime]
        return sigma * X
