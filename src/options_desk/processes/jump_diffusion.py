"""
Jump-diffusion process base class

Extends SingleFactorProcess with Poisson jump component

author: Yunian Pan
email: yp1170@nyu.edu
"""
import numpy as np
from .base import SingleFactorProcess
from abc import abstractmethod


class JumpDiffusionProcess(SingleFactorProcess):
    """
    Base class for single-factor jump-diffusion processes

    Models of the form:
        dX_t = mu(X_t, t)dt + sigma(X_t, t)dW_t + dJ_t

    where J_t is a compound Poisson process with intensity lambda
    """

    def __init__(self, name: str = "JumpDiffusion"):
        super().__init__(name)
        self.jump_intensity = None

    def set_jump_intensity(self, lambda_: float):
        """Set the Poisson jump intensity"""
        self.jump_intensity = lambda_

    @abstractmethod
    def jump_size_distribution(self, n_jumps: int) -> np.ndarray:
        """
        Generate jump sizes from the jump distribution

        Args:
            n_jumps: Number of jump sizes to generate

        Returns:
            Array of shape (n_jumps, dim) with jump sizes

        Examples:
            - Merton: log-normal jumps
            - Kou: double exponential (asymmetric)
        """
        pass

    def jump_component(self, X: np.ndarray, t: float, dt: float) -> np.ndarray:
        """
        Compound Poisson jump component

        For each path:
            1. Sample number of jumps N_t ~ Poisson(λ * dt)
            2. Sample N_t jump sizes from jump_size_distribution
            3. Sum all jumps for that path

        Args:
            X: Current state, shape (n_paths, dim)
            t: Current time
            dt: Time increment

        Returns:
            Total jump contribution, shape (n_paths, dim)
        """
        n_paths = X.shape[0]
        jumps = np.zeros((n_paths, self.dim))

        # Sample number of jumps for each path
        n_jumps_per_path = np.random.poisson(self.jump_intensity * dt, n_paths)

        # Optimize by grouping paths with same number of jumps
        unique_jump_counts = np.unique(n_jumps_per_path)

        for n_jumps in unique_jump_counts:
            if n_jumps == 0:
                continue

            # Find all paths with this number of jumps
            mask = (n_jumps_per_path == n_jumps)
            n_paths_with_jumps = np.sum(mask)

            # Generate all jump sizes at once for these paths
            total_jumps = n_jumps * n_paths_with_jumps
            all_jump_sizes = self.jump_size_distribution(total_jumps)

            # Reshape to (n_paths_with_jumps, n_jumps, dim) and sum over jumps
            all_jump_sizes = all_jump_sizes.reshape(n_paths_with_jumps, n_jumps, self.dim)
            jumps[mask] = all_jump_sizes.sum(axis=1)

        return jumps
