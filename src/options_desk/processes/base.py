"""
Base classes for stochastic processes in financial modeling

This module provides abstract base classes for implmenting various SDEs include mode jump processes

author: Yunian Pan 
email: yp1170@nyu.edu
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Tuple, Callable
from dataclasses import dataclass

@dataclass
class SimulationConfig:
    """configs for Monte-Carlo"""
    n_paths: int = 10000
    n_steps: int = 252
    random_seed: Optional[int] = None
    antithetic: bool = False
    use_sobol: bool = False
    return_full_paths: bool = False
    batch_size: Optional[int] = None 


class StochasticProcess(ABC):
    """
    Abstract base class for stochastic processes.

    All stochastic models will inherit from this class and implemenet
    the required methods for drift, diffusion, simulation.

    Attributes:
        name:
        params:
    """
    def __init__(self, dim: int = 1, name: str = "StochasticProcess"):
        self.dim = dim
        self.name = name
        self.params = {}
        self.correlation_matrix = None
        self.cholesky_decomp = None

    @abstractmethod
    def drift(self, X: np.ndarray, t: float) -> np.ndarray:
        """
        drift terms of:
            dX_t = mu (X, t) dt + sigma(X, t) dW_t
        return drift coefficient
        """
        pass

    @abstractmethod
    def diffusion(self, X: np.ndarray, t: float) -> np.ndarray:
        """
        Diffusion term σ(X,t) of the SDE: dX_t = μ(X_t,t)dt + σ(X_t,t)dW_t

        Args:
            X: Current state value(s), shape (n_paths, dim)
            t: Current time

        Returns:
            Diffusion coefficient - can be either:
                - Shape (n_paths, dim): Element-wise (diagonal diffusion matrix)
                  Example: GBM with diagonal volatility
                - Shape (n_paths, dim, dim): Matrix-valued diffusion
                  Example: GBM with full volatility matrix

        Notes:
            - For element-wise: diffusion_term = σ(X,t) * dW (element-wise mult)
            - For matrix-valued: diffusion_term = σ(X,t) @ dW (matrix mult per path)
        """
        pass

    def jump_component(self, X: np.ndarray, t: float, dt: float) -> np.ndarray:
        """
        jump component for jump-diffusion processes

        return jump contribution after dt

        override this method
        """
        return np.zeros_like(X)

    def set_correlation(self, correlation_matrix: np.ndarray):
        """
        Set correlation matrix for multi-dimensional processes.

        Args:
            correlation_matrix: Correlation matrix (symmetric positive definite)
        """
        self.correlation_matrix = correlation_matrix
        self.cholesky_decomp = np.linalg.cholesky(correlation_matrix)

    def _apply_diffusion(self, sigma: np.ndarray, dW: np.ndarray) -> np.ndarray:
        """
        Apply diffusion term, handling both element-wise and matrix-valued cases

        Args:
            sigma: Diffusion coefficient - shape (n_paths, dim) or (n_paths, dim, dim)
            dW: Brownian increments, shape (n_paths, dim)

        Returns:
            Diffusion term, shape (n_paths, dim)
        """
        if sigma.ndim == 2:
            return sigma * dW
        else:
            # Matrix-valued: vectorized sigma[i] @ dW[i]
            return np.einsum('ijk,ik->ij', sigma, dW)

    def simulate(
        self,
        X0: np.ndarray, 
        T: float,
        config: SimulationConfig,
        scheme: str = "euler"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        X0: initial value
        T: Time horizon
        config: Simulation configuration
        scheme: Numerical scheme ('euler', 'milstein', 'exact') 

        returns:
            Tuple of (time_grid, paths) paths shape: (n_steps, n_paths)
        """
        
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
        
        dt = T/config.n_steps
        t_grid = np.linspace(0, T, config.n_steps+1)

        if scheme.lower() == "euler":
            return self._euler_maruyama(X0, T, dt, t_grid, config)
        elif scheme.lower() == "milstein":
            return self._milstein(X0, T, dt, t_grid, config)
        elif scheme.lower() == "exact":
            return self._exact_simulation(X0, T, dt, t_grid, config)
        else:
            raise ValueError(f"unknown scheme: {scheme}")
        

    
    def _euler_maruyama(
        self, 
        X0: np.ndarray, 
        T: float,
        dt: float,
        t_grid: np.ndarray,
        config: SimulationConfig,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Euler-Maruyama discretization scheme.
            X_{t+dt} = X_t + mu(X_t,t)dt + sigma(X_t,t)√dt * Z_t + J_t
        """ 
        n_paths = config.n_paths
        if config.antithetic:
            n_paths = n_paths // 2
        
        paths = np.zeros((len(t_grid), n_paths, self.dim))
        paths[0] = X0
        sqrt_dt = np.sqrt(dt)
        
        for i, t in enumerate(t_grid[:-1]):
            X_current = paths[i]

            # Generate Brownian increments
            dW = np.random.normal(0, sqrt_dt, size=(n_paths, self.dim))

            # Apply correlation if specified
            if self.cholesky_decomp is not None:
                dW = dW @ self.cholesky_decomp.T

            # the main step
            drift_term = self.drift(X_current, t) * dt
            sigma = self.diffusion(X_current, t)
            diffusion_term = self._apply_diffusion(sigma, dW)
            jump_term = self.jump_component(X_current, t, dt)

            paths[i+1] = X_current + drift_term + diffusion_term + jump_term
            

        if config.antithetic:
            paths_anti = np.zeros((len(t_grid), n_paths, self.dim))
            paths_anti[0] = X0

            # Reset seed for antithetic paths
            if config.random_seed is not None:
                np.random.seed(config.random_seed)

            for i, t in enumerate(t_grid[:-1]):
                X_current = paths_anti[i]

                # Generate antithetic Brownian increments (negated)
                dW = -np.random.normal(0, sqrt_dt, size=(n_paths, self.dim))

                # Apply correlation if specified
                if self.cholesky_decomp is not None:
                    dW = dW @ self.cholesky_decomp.T

                # the main step
                drift_term = self.drift(X_current, t) * dt
                sigma = self.diffusion(X_current, t)
                diffusion_term = self._apply_diffusion(sigma, dW)
                jump_term = self.jump_component(X_current, t, dt)

                paths_anti[i + 1] = X_current + drift_term + diffusion_term + jump_term

            # Concatenate along the path dimension
            paths = np.concatenate([paths, paths_anti], axis=1)

        return t_grid, paths
    

    def _milstein(
        self,
        X0: np.ndarray, 
        T: float,
        dt: float,
        t_grid: np.ndarray,
        config: SimulationConfig
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Milstein discretization scheme (higher order correction).
       
        Requires diffusion_derivative method to be implemented.
        """
        if not hasattr(self, 'diffusion_derivative'):
            # Fall back to Euler if not available
            return self._euler_maruyama(X0, T, dt, t_grid, config)
        
        n_paths = config.n_paths
        paths = np.zeros((len(t_grid), n_paths, self.dim))
        paths[0] = X0
        
        sqrt_dt = np.sqrt(dt)
        
        for i, t in enumerate(t_grid[:-1]):
            X_current = paths[i]

            # Generate Brownian increments
            dW = np.random.normal(0, sqrt_dt, size=(n_paths, self.dim))

            # Apply correlation if specified
            if self.cholesky_decomp is not None:
                dW = dW @ self.cholesky_decomp.T

            drift_term = self.drift(X_current, t) * dt
            sigma = self.diffusion(X_current, t)
            diffusion_term = self._apply_diffusion(sigma, dW)

            # Milstein correction term (only works for element-wise diffusion)
            if sigma.ndim == 2:  # Element-wise case
                sigma_prime = self.diffusion_derivative(X_current, t)
                correction = 0.5 * sigma * sigma_prime * (dW**2 - dt)
            else:
                # Matrix-valued diffusion: Milstein correction complex, skip for now
                correction = 0

            jump_term = self.jump_component(X_current, t, dt)

            paths[i+1] = X_current + drift_term + diffusion_term + jump_term + correction

        if config.antithetic:
            paths_anti = np.zeros((len(t_grid), n_paths, self.dim))
            paths_anti[0] = X0

            # Reset seed for antithetic paths
            if config.random_seed is not None:
                np.random.seed(config.random_seed)

            for i, t in enumerate(t_grid[:-1]):
                X_current = paths_anti[i]

                # Generate antithetic Brownian increments (negated)
                dW = -np.random.normal(0, sqrt_dt, size=(n_paths, self.dim))

                # Apply correlation if specified
                if self.cholesky_decomp is not None:
                    dW = dW @ self.cholesky_decomp.T

                # the main step
                drift_term = self.drift(X_current, t) * dt
                sigma = self.diffusion(X_current, t)
                diffusion_term = self._apply_diffusion(sigma, dW)

                # Milstein correction term (only works for element-wise diffusion)
                if sigma.ndim == 2:  # Element-wise case
                    sigma_prime = self.diffusion_derivative(X_current, t)
                    correction = 0.5 * sigma * sigma_prime * (dW**2 - dt)
                else:
                    # Matrix-valued diffusion: Milstein correction complex, skip for now
                    correction = 0

                jump_term = self.jump_component(X_current, t, dt)

                paths_anti[i + 1] = X_current + drift_term + diffusion_term + jump_term + correction

            # Concatenate along the path dimension
            paths = np.concatenate([paths, paths_anti], axis=1)

        return t_grid, paths

    def _exact_simulation(
        self,
        X0: np.ndarray,
        T: float,
        dt: float,
        t_grid: np.ndarray,
        config: SimulationConfig
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Exact simulation method
        Override in subclasses
        """
        raise NotImplementedError(
            f"Exact simulation method not implemented for {self.name}"
        )

    def expectation(self, X0: np.ndarray, t: float) -> np.ndarray:
        """
        Analytical solution for E[X_t | X_0]

        Override in subclasses where solution exists
        """
        raise NotImplementedError(
            f"analytical solution not available for {self.name}"
        )

    def variance(self, X0: np.ndarray, t: float) -> np.ndarray:
        """
        Analytical variance Var[X_t | X_0] if available.

        Override in subclasses where analytical solution exists.
        """
        raise NotImplementedError(
            f"analytical variance not available for {self.name}"
        )

    def characteristic_function(self, u: complex, X0: np.ndarray, t: float) -> complex:
        """
        Characteristic function phi(u) = E[exp(i*u*X_t) | X_0] if available.

        Useful for Fourier based methods.
        Override in subclasses where analytical form exists.
        """
        raise NotImplementedError(
            f"characteristic function not available for {self.name}"
        )

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k} = {v}" for k, v in self.params.items())
        return f"{self.name}({params_str})"


class SingleFactorProcess(StochasticProcess):
    """
    Base class for 1-dimensional stochastic processes

    Single state variable with autonomous dynamics.
    Examples: GBM, OU, CIR, CEV, Merton jump-diffusion
    """

    def __init__(self, name: str = "SingleFactorProcess"):
        super().__init__(dim=1, name=name)


class MultiFactorProcess(StochasticProcess):
    """
    Base class for multi-dimensional processes with coupled dynamics

    Multiple state variables where dynamics are interdependent.
    Examples: Heston (S_t and v_t coupled), SABR, multi-asset with correlation

    Note: This is different from multiple independent processes.
    Use correlation in base class for independent processes with correlation.
    """

    def __init__(self, dim: int, name: str = "MultiFactorProcess"):
        super().__init__(dim=dim, name=name)


class DriftDiffusionProcess(SingleFactorProcess):
    """
    Pure continuous diffusion processes (no jumps)

    dX_t = mu(X_t, t)dt + sigma(X_t, t)dW_t

    Examples: GBM, CEV, Ornstein-Uhlenbeck, CIR
    """

    def __init__(self, name: str = "DriftDiffusionProcess"):
        super().__init__(name=name)