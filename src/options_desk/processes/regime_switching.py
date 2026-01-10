
"""
Regime switching process class

author: Yunian Pan 
email: yp1170@nyu.edu
"""
import numpy as np
from .base import *

class RegimeSwitchingProcess(StochasticProcess):
    def __init__(self, n_regimes: int, dim: int = 1, name: str = "RegimeSwitching"):
        super().__init__(dim, name)
        self.n_regimes = n_regimes
        self.transition_matrix = None
        self.regime_params = {}

    def set_transition_matrix(self, Q: np.ndarray):
        """Set continuous-time transition rate matrix Q"""
        self.transition_matrix = Q

    def set_regime_params(self, param_name: str, values: list):
        """Set regime-dependent parameters (e.g., mu, sigma for each regime)"""
        self.regime_params[param_name] = np.array(values)
        
    @abstractmethod
    def drift_regime(self, X: np.ndarray, t: float, regime: int) -> np.ndarray:
        """
        Drift for specific regime

        Args:
            X: Current state value (shape: (dim,) for single path)
            t: Current time
            regime: Current regime index

        Returns:
            Drift coefficient for this regime
        """
        pass

    @abstractmethod
    def diffusion_regime(self, X: np.ndarray, t: float, regime: int) -> np.ndarray:
        """
        Diffusion for specific regime

        Args:
            X: Current state value (shape: (dim,) for single path)
            t: Current time
            regime: Current regime index

        Returns:
            Diffusion coefficient for this regime
        """
        pass

    def drift(self, X: np.ndarray, t: float) -> np.ndarray:
        """Not used in regime switching - use drift_regime instead"""
        raise NotImplementedError(
            "Use drift_regime() with regime index for regime-switching models"
        )

    def diffusion(self, X: np.ndarray, t: float) -> np.ndarray:
        """Not used in regime switching - use diffusion_regime instead"""
        raise NotImplementedError(
            "Use diffusion_regime() with regime index for regime-switching models"
        )

    def _simulate_regime_path(self, T: float, dt: float, initial_regime: int, n_paths: int):
        """
        Generate regime switches using continuous-time Markov chain
        Uses exponential holding times and transition rate matrix Q

        Args:
            T: Time horizon
            dt: Time step
            initial_regime: Starting regime index
            n_paths: Number of paths to simulate

        Returns:
            regime_paths: array of shape (n_steps+1, n_paths) with regime indices
        """
        n_steps = int(T / dt) + 1
        t_grid = np.linspace(0, T, n_steps)
        regime_paths = np.zeros((n_steps, n_paths), dtype=int)
        regime_paths[0] = initial_regime

        for path_idx in range(n_paths):
            current_regime = initial_regime
            current_time = 0.0
            step_idx = 0

            while step_idx < n_steps:
                # Exit rate from current regime
                exit_rate = -self.transition_matrix[current_regime, current_regime]

                # Sample holding time (time until next regime switch)
                if exit_rate > 1e-12:
                    holding_time = np.random.exponential(1.0 / exit_rate)
                else:
                    holding_time = np.inf

                next_jump_time = current_time + holding_time

                # Fill regime path until jump or end of simulation
                while step_idx < n_steps and t_grid[step_idx] < next_jump_time:
                    regime_paths[step_idx, path_idx] = current_regime
                    step_idx += 1

                if step_idx >= n_steps:
                    break

                # Sample next regime using transition rates as probabilities
                transition_rates = self.transition_matrix[current_regime].copy()
                transition_rates[current_regime] = 0

                if transition_rates.sum() > 1e-12:
                    transition_probs = transition_rates / transition_rates.sum()
                    current_regime = np.random.choice(self.n_regimes, p=transition_probs)

                current_time = next_jump_time

            # Fill any remaining steps with current regime
            while step_idx < n_steps:
                regime_paths[step_idx, path_idx] = current_regime
                step_idx += 1

        return regime_paths

    def _euler_maruyama(self, X0, T, dt, t_grid, config):
        """Euler-Maruyama discretization with regime switching"""
        n_paths = config.n_paths
        if config.antithetic:
            n_paths = n_paths // 2

        # Simulate regime paths first
        initial_regime = 0
        regime_paths = self._simulate_regime_path(T, dt, initial_regime, n_paths)

        # Simulate X_t using regime-dependent parameters
        paths = np.zeros((len(t_grid), n_paths, self.dim))
        paths[0] = X0
        sqrt_dt = np.sqrt(dt)

        for i, t in enumerate(t_grid[:-1]):
            X_current = paths[i]

            # Generate Brownian increments
            dW = np.random.normal(0, sqrt_dt, size=(n_paths, self.dim))
            if self.cholesky_decomp is not None:
                dW = dW @ self.cholesky_decomp.T

            # Compute regime-specific drift and diffusion
            # Optimize by grouping paths by regime for vectorized computation
            drift_term = np.zeros((n_paths, self.dim))
            diffusion_term = np.zeros((n_paths, self.dim))

            current_regimes = regime_paths[i]
            for regime in range(self.n_regimes):
                mask = (current_regimes == regime)
                if np.any(mask):
                    # Vectorized computation for all paths in this regime
                    drift_term[mask] = self.drift_regime(X_current[mask], t, regime) * dt
                    diffusion_term[mask] = self.diffusion_regime(X_current[mask], t, regime) * dW[mask]

            jump_term = self.jump_component(X_current, t, dt)
            paths[i+1] = X_current + drift_term + diffusion_term + jump_term

        # Antithetic variates
        if config.antithetic:
            paths_anti = np.zeros((len(t_grid), n_paths, self.dim))
            paths_anti[0] = X0

            if config.random_seed is not None:
                np.random.seed(config.random_seed)
            regime_paths_anti = self._simulate_regime_path(T, dt, initial_regime, n_paths)

            for i, t in enumerate(t_grid[:-1]):
                X_current = paths_anti[i]

                dW = -np.random.normal(0, sqrt_dt, size=(n_paths, self.dim))
                if self.cholesky_decomp is not None:
                    dW = dW @ self.cholesky_decomp.T

                # Compute regime-specific drift and diffusion
                # Optimize by grouping paths by regime for vectorized computation
                drift_term = np.zeros((n_paths, self.dim))
                diffusion_term = np.zeros((n_paths, self.dim))

                current_regimes = regime_paths_anti[i]
                for regime in range(self.n_regimes):
                    mask = (current_regimes == regime)
                    if np.any(mask):
                        # Vectorized computation for all paths in this regime
                        drift_term[mask] = self.drift_regime(X_current[mask], t, regime) * dt
                        diffusion_term[mask] = self.diffusion_regime(X_current[mask], t, regime) * dW[mask]

                jump_term = self.jump_component(X_current, t, dt)
                paths_anti[i+1] = X_current + drift_term + diffusion_term + jump_term

            paths = np.concatenate([paths, paths_anti], axis=1)

        return t_grid, paths

    def _milstein(self, X0, T, dt, t_grid, config):
        """Milstein not fully supported for regime-switching, falls back to Euler"""
        return self._euler_maruyama(X0, T, dt, t_grid, config)



