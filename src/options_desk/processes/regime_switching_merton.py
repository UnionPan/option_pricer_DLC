"""
Regime-Switching Merton Jump-Diffusion Model

Combines regime-switching dynamics with jump-diffusion

author: Yunian Pan
email: yp1170@nyu.edu
"""
import numpy as np
from .regime_switching import RegimeSwitchingProcess


class RegimeSwitchingMerton(RegimeSwitchingProcess):
    """
    Regime-Switching Merton Jump-Diffusion Model

    GBM with log-normal jumps where parameters switch between regimes:
        dS_t = mu(R_t) * S_t dt + sigma(R_t) * S_t dW_t + S_{t-} dJ_t

    where:
        - R_t: regime at time t
        - mu(R_t): regime-dependent drift
        - sigma(R_t): regime-dependent volatility
        - J_t: compound Poisson process with regime-dependent parameters
          * lambda(R_t): jump intensity in regime R_t
          * mu_J(R_t), sigma_J(R_t): log-jump distribution parameters in regime R_t

    This captures:
        - Bull/bear market regimes with different drift/vol
        - Crisis regimes with higher jump frequency
        - Different jump severity across regimes
    """

    def __init__(
        self,
        n_regimes: int = 2,
        name: str = "RegimeSwitchingMerton"
    ):
        """
        Initialize Regime-Switching Merton model

        Args:
            n_regimes: Number of regimes
            name: Model name
        """
        super().__init__(n_regimes=n_regimes, dim=1, name=name)

        # Initialize regime-dependent parameters
        # These should be set using set_regime_params()
        self.regime_params['mu'] = None
        self.regime_params['sigma'] = None
        self.regime_params['lambda'] = None
        self.regime_params['mu_J'] = None
        self.regime_params['sigma_J'] = None

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

    def jump_component_regime(self, X: np.ndarray, t: float, dt: float, regime: int) -> np.ndarray:
        """
        Jump component for specific regime

        Generates Merton-style jumps with regime-dependent parameters.

        Args:
            X: Current state, shape (n_paths, 1)
            t: Current time
            dt: Time increment
            regime: Current regime index

        Returns:
            Jump contribution, shape (n_paths, 1)
        """
        n_paths = X.shape[0]
        jumps = np.zeros((n_paths, self.dim))

        # Get regime-specific jump parameters
        lambda_jump = self.regime_params['lambda'][regime]
        mu_J = self.regime_params['mu_J'][regime]
        sigma_J = self.regime_params['sigma_J'][regime]

        # Sample number of jumps for each path
        n_jumps_per_path = np.random.poisson(lambda_jump * dt, n_paths)

        # Optimize by grouping paths with same number of jumps
        unique_jump_counts = np.unique(n_jumps_per_path)

        for n_jumps in unique_jump_counts:
            if n_jumps == 0:
                continue

            mask = (n_jumps_per_path == n_jumps)
            n_paths_with_jumps = np.sum(mask)

            # Generate log-normal jump sizes: Y = exp(Z) - 1, Z ~ N(mu_J, sigma_J^2)
            total_jumps = n_jumps * n_paths_with_jumps
            log_jumps = np.random.normal(mu_J, sigma_J, (total_jumps, self.dim))
            jump_proportions = np.exp(log_jumps) - 1.0

            # Reshape and sum jumps for each path
            jump_proportions = jump_proportions.reshape(n_paths_with_jumps, n_jumps, self.dim)
            total_jump_proportion = jump_proportions.sum(axis=1)

            # Multiplicative jumps: X * sum(Y_i)
            jumps[mask] = X[mask] * total_jump_proportion

        return jumps

    def _euler_maruyama(self, X0, T, dt, t_grid, config):
        """
        Euler-Maruyama with regime-switching and jumps

        Overrides base class to handle regime-dependent jumps.
        """
        n_paths = config.n_paths
        if config.antithetic:
            n_paths = n_paths // 2

        paths = np.zeros((len(t_grid), n_paths, self.dim))
        paths[0] = X0
        sqrt_dt = np.sqrt(dt)

        # Simulate regime paths
        initial_regime = 0  # Start in regime 0
        regime_paths = self._simulate_regime_path(T, dt, initial_regime, n_paths)

        for i, t in enumerate(t_grid[:-1]):
            X_current = paths[i]
            current_regimes = regime_paths[i]

            # Generate Brownian increments
            dW = np.random.normal(0, sqrt_dt, size=(n_paths, self.dim))

            # Initialize step increments
            drift_term = np.zeros((n_paths, self.dim))
            diffusion_term = np.zeros((n_paths, self.dim))
            jump_term = np.zeros((n_paths, self.dim))

            # Loop over regimes and compute for all paths in each regime
            for regime in range(self.n_regimes):
                mask = (current_regimes == regime)
                if not np.any(mask):
                    continue

                # Drift and diffusion for this regime
                drift_term[mask] = self.drift_regime(X_current[mask], t, regime) * dt
                sigma = self.diffusion_regime(X_current[mask], t, regime)
                diffusion_term[mask] = sigma * dW[mask]

                # Jumps for this regime
                jump_term[mask] = self.jump_component_regime(X_current[mask], t, dt, regime)

            paths[i + 1] = X_current + drift_term + diffusion_term + jump_term

        # Antithetic variates
        if config.antithetic:
            paths_anti = np.zeros((len(t_grid), n_paths, self.dim))
            paths_anti[0] = X0

            if config.random_seed is not None:
                np.random.seed(config.random_seed)

            regime_paths_anti = self._simulate_regime_path(T, dt, initial_regime, n_paths)

            for i, t in enumerate(t_grid[:-1]):
                X_current = paths_anti[i]
                current_regimes = regime_paths_anti[i]

                dW = -np.random.normal(0, sqrt_dt, size=(n_paths, self.dim))

                drift_term = np.zeros((n_paths, self.dim))
                diffusion_term = np.zeros((n_paths, self.dim))
                jump_term = np.zeros((n_paths, self.dim))

                for regime in range(self.n_regimes):
                    mask = (current_regimes == regime)
                    if not np.any(mask):
                        continue

                    drift_term[mask] = self.drift_regime(X_current[mask], t, regime) * dt
                    sigma = self.diffusion_regime(X_current[mask], t, regime)
                    diffusion_term[mask] = sigma * dW[mask]
                    jump_term[mask] = self.jump_component_regime(X_current[mask], t, dt, regime)

                paths_anti[i + 1] = X_current + drift_term + diffusion_term + jump_term

            paths = np.concatenate([paths, paths_anti], axis=1)

        return t_grid, paths

    def _milstein(self, X0, T, dt, t_grid, config):
        """
        Milstein scheme not directly applicable with regime-switching and jumps.
        Falls back to Euler-Maruyama.
        """
        return self._euler_maruyama(X0, T, dt, t_grid, config)
