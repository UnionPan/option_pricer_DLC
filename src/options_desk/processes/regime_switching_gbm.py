"""
Regime-Switching Geometric Brownian Motion

GBM with parameters that switch between regimes

author: Yunian Pan
email: yp1170@nyu.edu
"""
import numpy as np
from .regime_switching import RegimeSwitchingProcess


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
