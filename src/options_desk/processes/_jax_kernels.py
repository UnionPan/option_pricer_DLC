"""
Functional JAX simulation kernels for stochastic processes.

All functions are pure: drift/diffusion are passed as callables of the form
    drift_fn(params, X, t) -> X_dot
    diffusion_fn(params, X, t) -> sigma

No class dispatch inside the scan body. Fully JIT-able, vmap-able, grad-able.

author: Yunian Pan
email: yp1170@nyu.edu
"""

from typing import Callable, Optional, Tuple, NamedTuple

import jax
import jax.numpy as jnp

from ._jax_utils import ensure_jax_key, to_numpy


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
DriftFn = Callable  # (params, X, t) -> jnp.ndarray  (n_paths, dim)
DiffusionFn = Callable  # (params, X, t) -> jnp.ndarray  (n_paths, dim) or (n_paths, dim, dim)
JumpFn = Callable  # (params, X, t, dt, key) -> (key, jnp.ndarray)  or None
PostStepFn = Callable  # (params, X) -> X  (e.g. variance floor for Heston)


class SimKernelConfig(NamedTuple):
    """Static configuration for a simulation kernel run."""
    n_paths: int
    n_steps: int
    dt: float
    dim: int
    antithetic: bool = False


# ---------------------------------------------------------------------------
# Core: Euler-Maruyama via lax.scan
# ---------------------------------------------------------------------------
def euler_maruyama(
    drift_fn: DriftFn,
    diffusion_fn: DiffusionFn,
    params,
    X0: jnp.ndarray,
    config: SimKernelConfig,
    key: jax.Array,
    cholesky: Optional[jnp.ndarray] = None,
    jump_fn: Optional[JumpFn] = None,
    post_step_fn: Optional[PostStepFn] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Euler-Maruyama simulation via jax.lax.scan.

    Args:
        drift_fn: Pure function (params, X, t) -> drift, shape (n_paths, dim)
        diffusion_fn: Pure function (params, X, t) -> sigma
            Element-wise: shape (n_paths, dim)
            Matrix-valued: shape (n_paths, dim, dim)
        params: Frozen parameter container (pytree-compatible)
        X0: Initial state, shape (dim,) — broadcast to (n_paths, dim)
        config: SimKernelConfig with n_paths, n_steps, dt, dim, antithetic
        key: JAX PRNG key
        cholesky: Optional Cholesky factor of correlation matrix, shape (dim, dim)
        jump_fn: Optional (params, X, t, dt, key) -> (key, jump_contrib)
        post_step_fn: Optional (params, X) -> X applied after each step
            (e.g. variance truncation for Heston)

    Returns:
        (t_grid, paths)
        t_grid: shape (n_steps + 1,)
        paths: shape (n_steps + 1, n_paths, dim)  — JAX arrays
    """
    n_paths = config.n_paths
    if config.antithetic:
        n_paths = n_paths // 2

    dt = config.dt
    sqrt_dt = jnp.sqrt(dt)
    t_grid = jnp.linspace(0.0, config.n_steps * dt, config.n_steps + 1)

    # Broadcast X0 to (n_paths, dim)
    X0_batch = jnp.broadcast_to(jnp.atleast_1d(X0), (n_paths, config.dim))

    def _run_paths(key_run, negate_dW):
        """Run one set of paths (normal or antithetic)."""

        def scan_body(carry, t):
            X, key = carry

            # --- Brownian increments ---
            key, k_bm = jax.random.split(key)
            dW = jax.random.normal(k_bm, shape=(n_paths, config.dim)) * sqrt_dt
            # Negate for antithetic paths
            dW = jnp.where(negate_dW, -dW, dW)
            # Apply correlation
            if cholesky is not None:
                dW = dW @ cholesky.T

            # --- SDE step ---
            drift_term = drift_fn(params, X, t) * dt
            sigma = diffusion_fn(params, X, t)
            # Element-wise or matrix diffusion
            if sigma.ndim == 2:
                diffusion_term = sigma * dW
            else:
                diffusion_term = jnp.einsum('ijk,ik->ij', sigma, dW)

            X_next = X + drift_term + diffusion_term

            # --- Jumps (optional) ---
            if jump_fn is not None:
                key, jump_contrib = jump_fn(params, X, t, dt, key)
                X_next = X_next + jump_contrib

            # --- Post-step processing (optional) ---
            if post_step_fn is not None:
                X_next = post_step_fn(params, X_next)

            return (X_next, key), X_next

        init_carry = (X0_batch, key_run)
        (_, _), path_body = jax.lax.scan(scan_body, init_carry, t_grid[:-1])

        # Prepend X0: final shape (n_steps+1, n_paths, dim)
        return jnp.concatenate([X0_batch[jnp.newaxis, :, :], path_body], axis=0)

    # Primary paths
    key, key_main = jax.random.split(key)
    paths = _run_paths(key_main, negate_dW=False)

    if config.antithetic:
        key, key_anti = jax.random.split(key)
        paths_anti = _run_paths(key_anti, negate_dW=True)
        paths = jnp.concatenate([paths, paths_anti], axis=1)

    return t_grid, paths


# ---------------------------------------------------------------------------
# Core: Milstein via lax.scan
# ---------------------------------------------------------------------------
def milstein(
    drift_fn: DriftFn,
    diffusion_fn: DiffusionFn,
    diffusion_deriv_fn: Optional[Callable],
    params,
    X0: jnp.ndarray,
    config: SimKernelConfig,
    key: jax.Array,
    cholesky: Optional[jnp.ndarray] = None,
    jump_fn: Optional[JumpFn] = None,
    post_step_fn: Optional[PostStepFn] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Milstein simulation via jax.lax.scan.

    Same interface as euler_maruyama, plus:
        diffusion_deriv_fn: (params, X, t) -> d(sigma)/dX, shape (n_paths, dim)
            If None, falls back to Euler-Maruyama.
    """
    if diffusion_deriv_fn is None:
        return euler_maruyama(
            drift_fn, diffusion_fn, params, X0, config, key,
            cholesky, jump_fn, post_step_fn,
        )

    n_paths = config.n_paths
    if config.antithetic:
        n_paths = n_paths // 2

    dt = config.dt
    sqrt_dt = jnp.sqrt(dt)
    t_grid = jnp.linspace(0.0, config.n_steps * dt, config.n_steps + 1)
    X0_batch = jnp.broadcast_to(jnp.atleast_1d(X0), (n_paths, config.dim))

    def _run_paths(key_run, negate_dW):

        def scan_body(carry, t):
            X, key = carry

            key, k_bm = jax.random.split(key)
            dW = jax.random.normal(k_bm, shape=(n_paths, config.dim)) * sqrt_dt
            dW = jnp.where(negate_dW, -dW, dW)
            if cholesky is not None:
                dW = dW @ cholesky.T

            drift_term = drift_fn(params, X, t) * dt
            sigma = diffusion_fn(params, X, t)

            # Only element-wise Milstein correction
            if sigma.ndim == 2:
                diffusion_term = sigma * dW
                sigma_prime = diffusion_deriv_fn(params, X, t)
                correction = 0.5 * sigma * sigma_prime * (dW ** 2 - dt)
            else:
                diffusion_term = jnp.einsum('ijk,ik->ij', sigma, dW)
                correction = 0.0

            X_next = X + drift_term + diffusion_term + correction

            if jump_fn is not None:
                key, jump_contrib = jump_fn(params, X, t, dt, key)
                X_next = X_next + jump_contrib

            if post_step_fn is not None:
                X_next = post_step_fn(params, X_next)

            return (X_next, key), X_next

        init_carry = (X0_batch, key_run)
        (_, _), path_body = jax.lax.scan(scan_body, init_carry, t_grid[:-1])
        return jnp.concatenate([X0_batch[jnp.newaxis, :, :], path_body], axis=0)

    key, key_main = jax.random.split(key)
    paths = _run_paths(key_main, negate_dW=False)

    if config.antithetic:
        key, key_anti = jax.random.split(key)
        paths_anti = _run_paths(key_anti, negate_dW=True)
        paths = jnp.concatenate([paths, paths_anti], axis=1)

    return t_grid, paths


# ---------------------------------------------------------------------------
# Batched simulation: vmap over parameter sweeps
# ---------------------------------------------------------------------------
def batched_euler_maruyama(
    drift_fn: DriftFn,
    diffusion_fn: DiffusionFn,
    params_batch,
    X0: jnp.ndarray,
    config: SimKernelConfig,
    key: jax.Array,
    cholesky: Optional[jnp.ndarray] = None,
    jump_fn: Optional[JumpFn] = None,
    post_step_fn: Optional[PostStepFn] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Run Euler-Maruyama over a batch of parameter sets via vmap.

    params_batch: pytree where leaf arrays have an extra leading batch dimension.
    Returns: (t_grid, paths) where paths has shape (batch, n_steps+1, n_paths, dim)
    """
    batch_size = jax.tree.leaves(params_batch)[0].shape[0]
    keys = jax.random.split(key, batch_size)

    def _single(params_i, key_i):
        return euler_maruyama(
            drift_fn, diffusion_fn, params_i, X0, config, key_i,
            cholesky, jump_fn, post_step_fn,
        )

    t_grid, paths = jax.vmap(_single)(params_batch, keys)
    # t_grid is duplicated across batch; take first
    return t_grid[0], paths


# ---------------------------------------------------------------------------
# Convenience: simulate and return numpy
# ---------------------------------------------------------------------------
def simulate(
    drift_fn: DriftFn,
    diffusion_fn: DiffusionFn,
    params,
    X0,
    T: float,
    n_paths: int,
    n_steps: int,
    seed: int = 0,
    scheme: str = "euler",
    antithetic: bool = False,
    dim: int = 1,
    cholesky=None,
    jump_fn=None,
    post_step_fn=None,
    diffusion_deriv_fn=None,
) -> Tuple:
    """
    High-level simulation entry point. Returns numpy arrays.

    Args:
        drift_fn, diffusion_fn: Pure functions (params, X, t) -> array
        params: Process parameters (pytree)
        X0: Initial state (scalar or array)
        T: Time horizon
        n_paths: Number of MC paths
        n_steps: Number of time steps
        seed: Random seed (int)
        scheme: 'euler' or 'milstein'
        antithetic: Use antithetic variates
        dim: Process dimension
        cholesky: Correlation Cholesky factor
        jump_fn: Jump component function
        post_step_fn: Per-step post-processing
        diffusion_deriv_fn: For Milstein scheme

    Returns:
        (t_grid, paths) as numpy arrays
        t_grid: shape (n_steps+1,)
        paths: shape (n_steps+1, n_paths, dim)
    """
    key = ensure_jax_key(seed)
    dt = T / n_steps
    config = SimKernelConfig(
        n_paths=n_paths, n_steps=n_steps, dt=dt, dim=dim, antithetic=antithetic,
    )

    X0_jax = jnp.atleast_1d(jnp.array(X0, dtype=jnp.float64))
    cholesky_jax = jnp.array(cholesky) if cholesky is not None else None

    if scheme == "euler":
        t_grid, paths = euler_maruyama(
            drift_fn, diffusion_fn, params, X0_jax, config, key,
            cholesky_jax, jump_fn, post_step_fn,
        )
    elif scheme == "milstein":
        t_grid, paths = milstein(
            drift_fn, diffusion_fn, diffusion_deriv_fn, params, X0_jax,
            config, key, cholesky_jax, jump_fn, post_step_fn,
        )
    else:
        raise ValueError(f"Unknown scheme: {scheme}. Use 'euler' or 'milstein'.")

    return to_numpy(t_grid), to_numpy(paths)
