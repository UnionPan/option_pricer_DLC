"""
JAX utilities for stochastic process simulation.

Provides shared helpers for PRNG key management, Brownian increment
generation, compound Poisson thinning, and numpy output conversion.

author: Yunian Pan
email: yp1170@nyu.edu
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ._jax_backend import configure_jax_runtime

configure_jax_runtime()


def ensure_jax_key(seed) -> jax.Array:
    """Convert an integer seed or existing key to a JAX PRNG key."""
    if seed is None:
        return jax.random.PRNGKey(0)
    if isinstance(seed, (int, np.integer)):
        return jax.random.PRNGKey(int(seed))
    return seed


def generate_brownian_increments(
    key: jax.Array,
    n_paths: int,
    dim: int,
    sqrt_dt: float,
    cholesky: Optional[jax.Array] = None,
) -> Tuple[jax.Array, jax.Array]:
    """
    Generate correlated Brownian increments.

    Args:
        key: PRNG key
        n_paths: Number of simulation paths
        dim: Dimension of the process
        sqrt_dt: Square root of the time step
        cholesky: Cholesky decomposition of the correlation matrix

    Returns:
        (new_key, dW) where dW has shape (n_paths, dim)
    """
    key, subkey = jax.random.split(key)
    dW = jax.random.normal(subkey, shape=(n_paths, dim)) * sqrt_dt
    if cholesky is not None:
        dW = dW @ cholesky.T
    return key, dW


def generate_compound_poisson_jumps(
    key: jax.Array,
    n_paths: int,
    dim: int,
    lambda_dt: float,
    jump_size_fn,
    max_jumps: int = 10,
) -> Tuple[jax.Array, jax.Array]:
    """
    Fixed-size compound Poisson jump generation for use inside lax.scan.

    Uses thinning: pre-allocates max_jumps candidates per path, masks
    unused ones based on the actual Poisson draw.

    Args:
        key: PRNG key
        n_paths: Number of paths
        dim: Dimension
        lambda_dt: Poisson intensity * dt
        jump_size_fn: Callable(key, shape) -> jump sizes array
        max_jumps: Upper bound on jumps per step per path

    Returns:
        (new_key, total_jumps) where total_jumps has shape (n_paths, dim)
    """
    key, k_poisson, k_sizes = jax.random.split(key, 3)

    # Number of jumps per path: shape (n_paths,)
    n_jumps = jax.random.poisson(k_poisson, lambda_dt, shape=(n_paths,))
    n_jumps = jnp.minimum(n_jumps, max_jumps)

    # Generate max_jumps candidate sizes for every path: (n_paths, max_jumps, dim)
    candidate_sizes = jump_size_fn(k_sizes, (n_paths, max_jumps, dim))

    # Mask: keep only the first n_jumps[i] candidates for path i
    indices = jnp.arange(max_jumps)[None, :]  # (1, max_jumps)
    mask = indices < n_jumps[:, None]  # (n_paths, max_jumps)
    mask = mask[:, :, None]  # (n_paths, max_jumps, 1) for broadcasting

    total_jumps = jnp.sum(candidate_sizes * mask, axis=1)  # (n_paths, dim)
    return key, total_jumps


def to_numpy(x):
    """Convert a JAX array (or numpy array) to a numpy array."""
    if isinstance(x, jnp.ndarray):
        return np.asarray(x)
    return x
