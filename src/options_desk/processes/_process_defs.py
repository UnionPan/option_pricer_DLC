"""
Functional process definitions for the JAX simulation kernel layer.

Each process is defined as:
  - A NamedTuple param container (automatic JAX pytree)
  - Pure functions: drift_fn(params, X, t), diffusion_fn(params, X, t), etc.

These are the building blocks passed into _jax_kernels.euler_maruyama / milstein.
The OO classes in gbm.py, heston.py, etc. remain as the public API wrapper.

author: Yunian Pan
email: yp1170@nyu.edu
"""

from functools import partial
from typing import NamedTuple
import jax
import jax.numpy as jnp


def _interp1d_clamped(x_grid, values, x):
    """1D linear interpolation with endpoint clamping."""
    x_grid = jnp.asarray(x_grid)
    values = jnp.asarray(values)
    return jnp.interp(x, x_grid, values, left=values[0], right=values[-1])


def _interp2d_rectilinear_clamped(x_grid, y_grid, values, x, y):
    """
    Bilinear interpolation on a rectilinear grid with endpoint clamping.

    Args:
        x_grid: Monotone grid for x/spot dimension, shape (nx,)
        y_grid: Monotone grid for y/time dimension, shape (ny,)
        values: Surface values, shape (ny, nx)
        x: Query x values, arbitrary shape
        y: Scalar query y value
    """
    x_grid = jnp.asarray(x_grid)
    y_grid = jnp.asarray(y_grid)
    values = jnp.asarray(values)
    x_arr = jnp.asarray(x)
    x_flat = x_arr.reshape(-1)
    x_clamped = jnp.clip(x_flat, x_grid[0], x_grid[-1])
    y_clamped = jnp.clip(y, y_grid[0], y_grid[-1])

    ix = jnp.clip(jnp.searchsorted(x_grid, x_clamped, side="right") - 1, 0, x_grid.shape[0] - 2)
    iy = jnp.clip(jnp.searchsorted(y_grid, y_clamped, side="right") - 1, 0, y_grid.shape[0] - 2)

    x0 = x_grid[ix]
    x1 = x_grid[ix + 1]
    y0 = y_grid[iy]
    y1 = y_grid[iy + 1]

    wx = jnp.where(x1 > x0, (x_clamped - x0) / (x1 - x0), 0.0)
    wy = jnp.where(y1 > y0, (y_clamped - y0) / (y1 - y0), 0.0)

    f00 = values[iy, ix]
    f01 = values[iy, ix + 1]
    f10 = values[iy + 1, ix]
    f11 = values[iy + 1, ix + 1]

    interp_x0 = (1.0 - wx) * f00 + wx * f01
    interp_x1 = (1.0 - wx) * f10 + wx * f11
    out = (1.0 - wy) * interp_x0 + wy * interp_x1
    return out.reshape(x_arr.shape)


# ============================================================================
# GBM
# ============================================================================
class GBMParams(NamedTuple):
    mu: float
    sigma: float


def gbm_drift(params: GBMParams, X, t):
    return params.mu * X


def gbm_diffusion(params: GBMParams, X, t):
    return params.sigma * X


def gbm_diffusion_deriv(params: GBMParams, X, t):
    """d(sigma*X)/dX = sigma"""
    return params.sigma * jnp.ones_like(X)


# ============================================================================
# Ornstein-Uhlenbeck
# ============================================================================
class OUParams(NamedTuple):
    theta: float  # mean reversion speed
    mu: float     # long-term mean
    sigma: float  # volatility


def ou_drift(params: OUParams, X, t):
    return params.theta * (params.mu - X)


def ou_diffusion(params: OUParams, X, t):
    return params.sigma * jnp.ones_like(X)


# ============================================================================
# Bachelier (Arithmetic Brownian Motion)
# ============================================================================
class BachelierParams(NamedTuple):
    mu: float     # constant drift
    sigma: float  # normal (absolute) volatility


def bachelier_drift(params: BachelierParams, X, t):
    return params.mu * jnp.ones_like(X)


def bachelier_diffusion(params: BachelierParams, X, t):
    return params.sigma * jnp.ones_like(X)


# ============================================================================
# CEV
# ============================================================================
class CEVParams(NamedTuple):
    mu: float
    sigma: float
    beta: float


def cev_drift(params: CEVParams, X, t):
    return params.mu * X


def cev_diffusion(params: CEVParams, X, t):
    return params.sigma * jnp.sign(X) * jnp.abs(X) ** params.beta


def cev_diffusion_deriv(params: CEVParams, X, t):
    """d(sigma * sign(X) * |X|^beta) / dX = sigma * beta * |X|^(beta-1)"""
    return params.sigma * params.beta * jnp.abs(X) ** (params.beta - 1.0)


# ============================================================================
# Heston
# ============================================================================
class HestonParams(NamedTuple):
    mu: float       # drift
    kappa: float    # mean reversion speed
    theta: float    # long-term variance
    sigma_v: float  # vol of vol
    rho: float      # correlation (stored for reference; correlation applied via cholesky)


def heston_drift(params: HestonParams, X, t):
    S = X[:, 0:1]
    v = X[:, 1:2]
    drift_S = params.mu * S
    drift_v = params.kappa * (params.theta - v)
    return jnp.concatenate([drift_S, drift_v], axis=1)


def heston_diffusion(params: HestonParams, X, t):
    S = X[:, 0:1]
    v = X[:, 1:2]
    v_pos = jnp.maximum(v, 0.0)
    sqrt_v = jnp.sqrt(v_pos)
    sigma_S = sqrt_v * S
    sigma_v = params.sigma_v * sqrt_v
    return jnp.concatenate([sigma_S, sigma_v], axis=1)


def heston_post_step(params: HestonParams, X):
    """Truncation scheme: clamp variance to zero."""
    return X.at[:, 1].set(jnp.maximum(X[:, 1], 0.0))


def heston_cholesky(rho: float) -> jnp.ndarray:
    """Build Cholesky factor for Heston correlation."""
    import numpy as np
    corr = np.array([[1.0, rho], [rho, 1.0]])
    return jnp.array(np.linalg.cholesky(corr))


# ============================================================================
# SABR
# ============================================================================
class SABRParams(NamedTuple):
    beta: float   # CEV exponent
    alpha: float  # vol of vol
    rho: float    # correlation


def sabr_drift(params: SABRParams, X, t):
    """SABR has zero drift under forward measure."""
    return jnp.zeros_like(X)


def sabr_diffusion(params: SABRParams, X, t):
    F = X[:, 0:1]
    sigma = X[:, 1:2]
    F_power = jnp.sign(F) * jnp.abs(F) ** params.beta
    sigma_F = sigma * F_power
    sigma_sigma = params.alpha * sigma
    return jnp.concatenate([sigma_F, sigma_sigma], axis=1)


def sabr_post_step(params: SABRParams, X):
    """Absorb volatility at zero."""
    return X.at[:, 1].set(jnp.maximum(X[:, 1], 0.0))


# ============================================================================
# Merton Jump-Diffusion
# ============================================================================
class MertonJDParams(NamedTuple):
    mu: float         # drift
    sigma: float      # diffusion volatility
    lambda_j: float   # jump intensity
    mu_J: float       # mean of log-jump size
    sigma_J: float    # std of log-jump size


def merton_drift(params: MertonJDParams, X, t):
    return params.mu * X


def merton_diffusion(params: MertonJDParams, X, t):
    return params.sigma * X


def merton_jump_fn(params: MertonJDParams, X, t, dt, key):
    """Multiplicative Merton jumps via thinning."""
    from ._jax_utils import generate_compound_poisson_jumps

    n_paths = X.shape[0]
    dim = X.shape[1]

    def _lognormal_jumps(key, shape):
        # Y = exp(Z) - 1 where Z ~ N(mu_J, sigma_J)
        Z = params.mu_J + params.sigma_J * jax.random.normal(key, shape)
        return jnp.exp(Z) - 1.0

    lambda_dt = params.lambda_j * dt
    max_jumps = int(max(10, 5.0 * lambda_dt + 5))

    key, total_jump_proportion = generate_compound_poisson_jumps(
        key, n_paths, dim, lambda_dt, _lognormal_jumps, max_jumps,
    )
    # Multiplicative: jump contribution = X * sum(Y_i)
    return key, X * total_jump_proportion


# ============================================================================
# Kou Jump-Diffusion
# ============================================================================
class KouJDParams(NamedTuple):
    mu: float          # drift
    sigma: float       # diffusion volatility
    lambda_j: float    # jump intensity
    p: float           # prob of upward jump
    eta_up: float      # rate for upward exponential
    eta_down: float    # rate for downward exponential


def kou_drift(params: KouJDParams, X, t):
    return params.mu * X


def kou_diffusion(params: KouJDParams, X, t):
    return params.sigma * X


def kou_jump_fn(params: KouJDParams, X, t, dt, key):
    """Multiplicative Kou double-exponential jumps via thinning."""
    from ._jax_utils import generate_compound_poisson_jumps

    n_paths = X.shape[0]
    dim = X.shape[1]

    def _double_exp_jumps(key, shape):
        k1, k2 = jax.random.split(key)
        # Direction: up with prob p, down with prob 1-p
        is_up = jax.random.uniform(k1, shape) < params.p
        # Exponential magnitudes
        exp_vals = jax.random.exponential(k2, shape)
        up_jump = exp_vals / params.eta_up
        down_jump = -exp_vals / params.eta_down
        return jnp.where(is_up, up_jump, down_jump)

    lambda_dt = params.lambda_j * dt
    max_jumps = int(max(10, 5.0 * lambda_dt + 5))

    key, total_jump_proportion = generate_compound_poisson_jumps(
        key, n_paths, dim, lambda_dt, _double_exp_jumps, max_jumps,
    )
    return key, X * total_jump_proportion


# ============================================================================
# Variance Gamma (Levy / subordinated BM)
# ============================================================================
class VGParams(NamedTuple):
    theta: float  # drift in subordinated BM
    sigma: float  # vol in subordinated BM
    nu: float     # variance rate (kurtosis control)


def vg_increment_fn(params: VGParams, dt, n_paths, dim, key):
    """
    Simulate VG increments: dX = theta*dT + sigma*sqrt(dT)*Z
    where dT ~ Gamma(dt/nu, nu).
    """
    k1, k2 = jax.random.split(key)
    shape = dt / params.nu
    scale = params.nu
    dT = jax.random.gamma(k1, shape, shape=(n_paths,)) * scale
    Z = jax.random.normal(k2, shape=(n_paths, dim))
    increments = params.theta * dT[:, None] + params.sigma * jnp.sqrt(dT[:, None]) * Z
    return increments


# ============================================================================
# NIG (Levy / subordinated BM with Inverse Gaussian subordinator)
# ============================================================================
class NIGParams(NamedTuple):
    alpha: float  # tail heaviness
    beta: float   # asymmetry
    delta: float  # scale
    mu: float     # location/drift


def nig_increment_fn(params: NIGParams, dt, n_paths, dim, key):
    """
    Simulate NIG increments: dX = mu*dt + beta*dT + sqrt(dT)*Z
    where dT ~ InverseGaussian(delta*dt/gamma, (delta*dt)^2).
    """
    gamma = jnp.sqrt(params.alpha ** 2 - params.beta ** 2)
    mean_ig = params.delta * dt / gamma
    lambda_ig = (params.delta * dt) ** 2

    k1, k2, k3 = jax.random.split(key, 3)

    # Michael-Schucany-Haas algorithm for Inverse Gaussian
    nu = jax.random.normal(k1, shape=(n_paths,))
    y = nu ** 2
    x = mean_ig + (mean_ig ** 2 * y) / (2 * lambda_ig) - \
        (mean_ig / (2 * lambda_ig)) * jnp.sqrt(
            4 * mean_ig * lambda_ig * y + mean_ig ** 2 * y ** 2
        )
    z = jax.random.uniform(k2, shape=(n_paths,))
    mask = z <= mean_ig / (mean_ig + x)
    dT = jnp.where(mask, x, mean_ig ** 2 / x)

    Z = jax.random.normal(k3, shape=(n_paths, dim))
    increments = params.mu * dt + params.beta * dT[:, None] + jnp.sqrt(dT[:, None]) * Z
    return increments


# ============================================================================
# Levy process simulation via lax.scan (VG, NIG)
# ============================================================================
@partial(jax.jit, static_argnames=("increment_fn", "n_paths", "n_steps", "dim", "antithetic"))
def _levy_simulate_core(
    increment_fn,
    params,
    X0,
    T: float,
    n_paths: int,
    n_steps: int,
    key,
    dim: int = 1,
    antithetic: bool = False,
):
    """JIT-compiled Levy process path generator."""
    dt = T / n_steps
    t_grid = jnp.linspace(0.0, T, n_steps + 1)

    actual_n_paths = n_paths // 2 if antithetic else n_paths
    X0_batch = jnp.broadcast_to(
        jnp.atleast_1d(jnp.asarray(X0, dtype=jnp.float64)), (actual_n_paths, dim)
    )

    def _run(key_run, negate):
        def scan_body(carry, _):
            X, key_inner = carry
            key_inner, subkey = jax.random.split(key_inner)
            inc = increment_fn(params, dt, actual_n_paths, dim, subkey)
            inc = jnp.where(negate, -inc, inc)
            X_next = X + inc
            return (X_next, key_inner), X_next

        (_, _), path_body = jax.lax.scan(scan_body, (X0_batch, key_run), jnp.arange(n_steps))
        return jnp.concatenate([X0_batch[None, :, :], path_body], axis=0)

    key, k1 = jax.random.split(key)
    paths = _run(k1, negate=False)

    if antithetic:
        key, k2 = jax.random.split(key)
        paths_anti = _run(k2, negate=True)
        paths = jnp.concatenate([paths, paths_anti], axis=1)

    return t_grid, paths


def levy_simulate(
    increment_fn,
    params,
    X0,
    T: float,
    n_paths: int,
    n_steps: int,
    seed: int = 0,
    dim: int = 1,
    antithetic: bool = False,
):
    """
    Simulate a Levy process by accumulating increments via lax.scan.

    Args:
        increment_fn: (params, dt, n_paths, dim, key) -> increments (n_paths, dim)
        params: Process parameters (NamedTuple)
        X0: Initial state
        T: Time horizon
        n_paths: Number of paths
        n_steps: Number of steps
        seed: Random seed
        dim: Process dimension
        antithetic: Use antithetic variates

    Returns:
        (t_grid, paths) as numpy arrays
    """
    from ._jax_utils import ensure_jax_key, to_numpy

    key = ensure_jax_key(seed)
    t_grid, paths = _levy_simulate_core(
        increment_fn, params, X0, T, n_paths, n_steps, key, dim=dim, antithetic=antithetic,
    )

    return to_numpy(t_grid), to_numpy(paths)


# ============================================================================
# Regime-Switching GBM (discrete-time Markov chain via matrix exponential)
# ============================================================================
class RegimeSwitchingGBMParams(NamedTuple):
    mus: jnp.ndarray       # drift per regime, shape (n_regimes,)
    sigmas: jnp.ndarray    # vol per regime, shape (n_regimes,)
    Q: jnp.ndarray         # transition rate matrix, shape (n_regimes, n_regimes)


@partial(jax.jit, static_argnames=("n_paths", "n_steps", "dim"))
def _regime_switching_simulate_core(
    params: RegimeSwitchingGBMParams,
    X0,
    T: float,
    n_paths: int,
    n_steps: int,
    key,
    dim: int = 1,
):
    """JIT-compiled regime-switching GBM path generator."""
    dt = T / n_steps
    sqrt_dt = jnp.sqrt(dt)
    t_grid = jnp.linspace(0.0, T, n_steps + 1)

    X0_batch = jnp.broadcast_to(
        jnp.atleast_1d(jnp.asarray(X0, dtype=jnp.float64)), (n_paths, dim)
    )

    P = jax.scipy.linalg.expm(params.Q * dt)
    log_P = jnp.log(jnp.maximum(P, 1e-30))

    def scan_body(carry, _):
        X, regimes, key_inner = carry
        key_inner, k_regime, k_bm = jax.random.split(key_inner, 3)
        logits = log_P[regimes]
        next_regimes = jax.random.categorical(k_regime, logits).astype(jnp.int32)

        mu_t = params.mus[next_regimes]
        sigma_t = params.sigmas[next_regimes]

        dW = jax.random.normal(k_bm, shape=(n_paths, dim)) * sqrt_dt
        drift_term = mu_t[:, None] * X * dt
        diffusion_term = sigma_t[:, None] * X * dW
        X_next = X + drift_term + diffusion_term

        return (X_next, next_regimes, key_inner), (X_next, next_regimes)

    init_regimes = jnp.zeros(n_paths, dtype=jnp.int32)
    key, k_sim = jax.random.split(key)
    init_carry = (X0_batch, init_regimes, k_sim)
    _, (path_body, regime_body) = jax.lax.scan(scan_body, init_carry, jnp.arange(n_steps))

    paths = jnp.concatenate([X0_batch[None, :, :], path_body], axis=0)
    regime_paths = jnp.concatenate([init_regimes[None, :], regime_body], axis=0)

    return t_grid, paths, regime_paths


def regime_switching_simulate(
    params: RegimeSwitchingGBMParams,
    X0,
    T: float,
    n_paths: int,
    n_steps: int,
    seed: int = 0,
    dim: int = 1,
):
    """
    Simulate regime-switching GBM using discrete-time Markov chain.

    Regime transitions use expm(Q*dt) for the one-step transition matrix,
    then jax.random.categorical for per-path regime sampling inside lax.scan.

    Returns:
        (t_grid, paths, regime_paths) as numpy arrays
    """
    from ._jax_utils import ensure_jax_key, to_numpy

    key = ensure_jax_key(seed)
    t_grid, paths, regime_paths = _regime_switching_simulate_core(
        params, X0, T, n_paths, n_steps, key, dim=dim,
    )

    return to_numpy(t_grid), to_numpy(paths), to_numpy(regime_paths)


# ============================================================================
# Regime-Switching Merton Jump-Diffusion
# ============================================================================
class RegimeSwitchingMertonParams(NamedTuple):
    mus: jnp.ndarray
    sigmas: jnp.ndarray
    lambdas: jnp.ndarray
    mu_js: jnp.ndarray
    sigma_js: jnp.ndarray
    Q: jnp.ndarray


@partial(jax.jit, static_argnames=("n_paths", "n_steps", "dim", "antithetic", "max_jumps"))
def _regime_switching_merton_simulate_core(
    params: RegimeSwitchingMertonParams,
    X0,
    T: float,
    n_paths: int,
    n_steps: int,
    key,
    dim: int = 1,
    antithetic: bool = False,
    max_jumps: int = 10,
):
    """JIT-compiled regime-switching Merton path generator."""
    dt = T / n_steps
    sqrt_dt = jnp.sqrt(dt)
    t_grid = jnp.linspace(0.0, T, n_steps + 1)

    actual_n_paths = n_paths // 2 if antithetic else n_paths
    X0_batch = jnp.broadcast_to(
        jnp.atleast_1d(jnp.asarray(X0, dtype=jnp.float64)), (actual_n_paths, dim)
    )

    P = jax.scipy.linalg.expm(params.Q * dt)
    log_P = jnp.log(jnp.maximum(P, 1e-30))

    def _run(key_run, negate_dW):
        def scan_body(carry, _):
            X, regimes, key_inner = carry

            key_inner, k_regime, k_bm, k_pois, k_jump = jax.random.split(key_inner, 5)

            logits = log_P[regimes]
            next_regimes = jax.random.categorical(k_regime, logits).astype(jnp.int32)

            mu_t = params.mus[next_regimes]
            sigma_t = params.sigmas[next_regimes]
            lambda_t = params.lambdas[next_regimes]
            mu_j_t = params.mu_js[next_regimes]
            sigma_j_t = params.sigma_js[next_regimes]

            dW = jax.random.normal(k_bm, shape=(actual_n_paths, dim)) * sqrt_dt
            dW = jnp.where(negate_dW, -dW, dW)

            drift_term = mu_t[:, None] * X * dt
            diffusion_term = sigma_t[:, None] * X * dW

            lambda_dt = lambda_t * dt
            n_jumps = jax.random.poisson(k_pois, lambda_dt, shape=(actual_n_paths,))
            n_jumps = jnp.minimum(n_jumps, max_jumps)

            Z = (
                mu_j_t[:, None, None]
                + sigma_j_t[:, None, None]
                * jax.random.normal(k_jump, shape=(actual_n_paths, max_jumps, dim))
            )
            jump_sizes = jnp.exp(Z) - 1.0
            mask = jnp.arange(max_jumps)[None, :] < n_jumps[:, None]
            total_jump = jnp.sum(jump_sizes * mask[:, :, None], axis=1)
            jump_term = X * total_jump

            X_next = X + drift_term + diffusion_term + jump_term
            return (X_next, next_regimes, key_inner), (X_next, next_regimes)

        init_regimes = jnp.zeros(actual_n_paths, dtype=jnp.int32)
        init_carry = (X0_batch, init_regimes, key_run)
        _, (path_body, regime_body) = jax.lax.scan(scan_body, init_carry, jnp.arange(n_steps))

        paths = jnp.concatenate([X0_batch[None, :, :], path_body], axis=0)
        regimes = jnp.concatenate([init_regimes[None, :], regime_body], axis=0)
        return paths, regimes

    key, key_main = jax.random.split(key)
    paths, regime_paths = _run(key_main, negate_dW=False)

    if antithetic:
        key, key_anti = jax.random.split(key)
        paths_anti, regime_paths_anti = _run(key_anti, negate_dW=True)
        paths = jnp.concatenate([paths, paths_anti], axis=1)
        regime_paths = jnp.concatenate([regime_paths, regime_paths_anti], axis=1)

    return t_grid, paths, regime_paths


def regime_switching_merton_simulate(
    params: RegimeSwitchingMertonParams,
    X0,
    T: float,
    n_paths: int,
    n_steps: int,
    seed: int = 0,
    dim: int = 1,
    antithetic: bool = False,
):
    """
    Simulate regime-switching Merton jump-diffusion.

    Each path carries a latent Markov regime and uses regime-specific
    drift, volatility, and log-normal jump parameters.
    """
    from ._jax_utils import ensure_jax_key, to_numpy

    key = ensure_jax_key(seed)
    dt = T / n_steps
    max_lambda_dt = float(jnp.max(params.lambdas)) * dt
    max_jumps = int(max(10, 5.0 * max_lambda_dt + 5))
    t_grid, paths, regime_paths = _regime_switching_merton_simulate_core(
        params, X0, T, n_paths, n_steps, key, dim=dim, antithetic=antithetic, max_jumps=max_jumps,
    )

    return to_numpy(t_grid), to_numpy(paths), to_numpy(regime_paths)


# ============================================================================
# Rough Bergomi (kernel-based, not standard Euler)
# ============================================================================
class RoughBergomiParams(NamedTuple):
    mu: float    # drift
    xi0: float   # initial/forward variance
    eta: float   # vol of vol
    rho: float   # correlation
    H: float     # Hurst parameter


@partial(jax.jit, static_argnames=("n_paths", "n_steps"))
def _rough_bergomi_simulate_core(
    params: RoughBergomiParams,
    X0,
    T: float,
    n_paths: int,
    n_steps: int,
    key,
):
    """JIT-compiled rough Bergomi path generator."""
    dt = T / n_steps
    t_grid = jnp.linspace(0.0, T, n_steps + 1)

    X0_arr = jnp.atleast_1d(jnp.asarray(X0, dtype=jnp.float64))
    S0 = X0_arr[0]

    key, k1, k2 = jax.random.split(key, 3)
    dW1 = jax.random.normal(k1, shape=(n_steps, n_paths)) * jnp.sqrt(dt)
    dW2 = jax.random.normal(k2, shape=(n_steps, n_paths)) * jnp.sqrt(dt)

    coef = jnp.sqrt(2.0 * params.H)
    i_idx = jnp.arange(1, n_steps + 1)[:, None]
    j_idx = jnp.arange(n_steps)[None, :]
    t_i = t_grid[1:][:, None]
    t_j = t_grid[:-1][None, :]
    mask = j_idx < i_idx
    diff = jnp.maximum(t_i - t_j, 1e-30)
    kernel_body = coef * jnp.power(diff, params.H - 0.5) * mask
    kernel = jnp.concatenate([jnp.zeros((1, n_steps)), kernel_body], axis=0)

    W_H = kernel @ dW1
    t_pow = t_grid ** (2 * params.H)
    v_path = params.xi0 * jnp.exp(
        params.eta * W_H - 0.5 * params.eta ** 2 * t_pow[:, None]
    )

    dW_S = params.rho * dW1 + jnp.sqrt(1.0 - params.rho ** 2) * dW2

    def scan_body(log_S, i):
        v_t = v_path[i]
        drift = (params.mu - 0.5 * v_t) * dt
        diff_term = jnp.sqrt(jnp.maximum(v_t, 0.0)) * dW_S[i]
        log_S_next = log_S + drift + diff_term
        return log_S_next, log_S_next

    log_S0 = jnp.full(n_paths, jnp.log(S0))
    _, log_S_body = jax.lax.scan(scan_body, log_S0, jnp.arange(n_steps))
    log_S_all = jnp.concatenate([log_S0[None, :], log_S_body], axis=0)
    S_path = jnp.exp(log_S_all)
    paths = jnp.stack([S_path, v_path], axis=2)

    return t_grid, paths


def rough_bergomi_simulate(
    params: RoughBergomiParams,
    X0,
    T: float,
    n_paths: int,
    n_steps: int,
    seed: int = 0,
):
    """
    Simulate Rough Bergomi using kernel-based fBM approximation.

    This process doesn't fit the standard Euler/Milstein kernel because
    variance is built from a kernel integral over the full Brownian history,
    not from a local SDE step.

    Returns:
        (t_grid, paths) as numpy arrays, paths shape (n_steps+1, n_paths, 2)
    """
    from ._jax_utils import ensure_jax_key, to_numpy

    key = ensure_jax_key(seed)
    t_grid, paths = _rough_bergomi_simulate_core(params, X0, T, n_paths, n_steps, key)

    return to_numpy(t_grid), to_numpy(paths)


# ============================================================================
# Multi-Asset GBM
# ============================================================================
class MultiAssetGBMParams(NamedTuple):
    mus: jnp.ndarray      # drift per asset, shape (n_assets,)
    sigmas: jnp.ndarray   # vol per asset, shape (n_assets,)


def multi_asset_gbm_drift(params: MultiAssetGBMParams, X, t):
    return params.mus[None, :] * X


def multi_asset_gbm_diffusion(params: MultiAssetGBMParams, X, t):
    return params.sigmas[None, :] * X


# ============================================================================
# Bates (Heston + Merton jumps)
# ============================================================================
class BatesParams(NamedTuple):
    mu: float         # drift
    kappa: float      # mean reversion speed
    theta: float      # long-term variance
    sigma_v: float    # vol of vol
    rho: float        # correlation
    lambda_j: float   # jump intensity
    mu_J: float       # mean of log-jump size
    sigma_J: float    # std of log-jump size


def bates_drift(params: BatesParams, X, t):
    S = X[:, 0:1]
    v = X[:, 1:2]
    jump_mean = jnp.exp(params.mu_J + 0.5 * params.sigma_J ** 2) - 1.0
    drift_S = (params.mu - params.lambda_j * jump_mean) * S
    drift_v = params.kappa * (params.theta - v)
    return jnp.concatenate([drift_S, drift_v], axis=1)


def bates_diffusion(params: BatesParams, X, t):
    S = X[:, 0:1]
    v = X[:, 1:2]
    v_pos = jnp.maximum(v, 0.0)
    sqrt_v = jnp.sqrt(v_pos)
    sigma_S = sqrt_v * S
    sigma_v = params.sigma_v * sqrt_v
    return jnp.concatenate([sigma_S, sigma_v], axis=1)


def bates_post_step(params: BatesParams, X):
    """Truncate variance at zero."""
    return X.at[:, 1].set(jnp.maximum(X[:, 1], 0.0))


def bates_jump_fn(params: BatesParams, X, t, dt, key):
    """Multiplicative log-normal jumps on asset price only (variance unaffected)."""
    from ._jax_utils import generate_compound_poisson_jumps

    n_paths = X.shape[0]

    def _lognormal_jumps(key, shape):
        Z = params.mu_J + params.sigma_J * jax.random.normal(key, shape)
        return jnp.exp(Z) - 1.0

    lambda_dt = params.lambda_j * dt
    max_jumps = int(max(10, 5.0 * lambda_dt + 5))

    # Generate jumps for asset dimension only
    key, total_Y = generate_compound_poisson_jumps(
        key, n_paths, 1, lambda_dt, _lognormal_jumps, max_jumps,
    )

    # Multiplicative jump on S, zero jump on v
    jump_contrib = jnp.concatenate([X[:, 0:1] * total_Y, jnp.zeros((n_paths, 1))], axis=1)
    return key, jump_contrib


# ============================================================================
# 3/2 Stochastic Volatility
# ============================================================================
class ThreeHalfParams(NamedTuple):
    mu: float       # drift
    kappa: float    # mean reversion speed
    theta: float    # long-term variance
    sigma_v: float  # vol of vol
    rho: float      # correlation (stored for reference; correlation applied via cholesky)


def three_half_drift(params: ThreeHalfParams, X, t):
    S = X[:, 0:1]
    v = X[:, 1:2]
    drift_S = params.mu * S
    drift_v = params.kappa * v * (params.theta - v)
    return jnp.concatenate([drift_S, drift_v], axis=1)


def three_half_diffusion(params: ThreeHalfParams, X, t):
    S = X[:, 0:1]
    v = X[:, 1:2]
    v_pos = jnp.maximum(v, 0.0)
    sqrt_v = jnp.sqrt(v_pos)
    sigma_S = sqrt_v * S
    sigma_v = params.sigma_v * v_pos * sqrt_v  # v^(3/2)
    return jnp.concatenate([sigma_S, sigma_v], axis=1)


def three_half_post_step(params: ThreeHalfParams, X):
    """Truncation scheme: clamp variance to zero."""
    return X.at[:, 1].set(jnp.maximum(X[:, 1], 0.0))


# ============================================================================
# 4/2 Stochastic Volatility (Grasselli 2017)
# ============================================================================
class FourHalfParams(NamedTuple):
    mu: float       # drift
    kappa: float    # mean reversion speed
    theta: float    # long-term variance
    sigma_v: float  # vol of vol
    rho: float      # correlation (stored for reference; correlation applied via cholesky)
    a: float        # weight on sqrt(v) component
    b: float        # weight on 1/sqrt(v) component


def four_half_drift(params: FourHalfParams, X, t):
    S = X[:, 0:1]
    v = X[:, 1:2]
    drift_S = params.mu * S
    drift_v = params.kappa * (params.theta - v)
    return jnp.concatenate([drift_S, drift_v], axis=1)


def four_half_diffusion(params: FourHalfParams, X, t):
    S = X[:, 0:1]
    v = X[:, 1:2]
    v_pos = jnp.maximum(v, 0.0)
    v_safe = jnp.maximum(v_pos, 1e-12)
    sqrt_v = jnp.sqrt(v_safe)
    # 4/2 spot volatility: a*sqrt(v) + b/sqrt(v)
    spot_vol = params.a * sqrt_v + params.b / sqrt_v
    sigma_S = spot_vol * S
    sigma_v = params.sigma_v * sqrt_v
    return jnp.concatenate([sigma_S, sigma_v], axis=1)


def four_half_post_step(params: FourHalfParams, X):
    """Truncation scheme: clamp variance to zero."""
    return X.at[:, 1].set(jnp.maximum(X[:, 1], 0.0))


# ============================================================================
# CIR (Cox-Ingersoll-Ross) short rate
# ============================================================================
class CIRParams(NamedTuple):
    kappa: float   # mean reversion speed
    theta: float   # long-term mean
    sigma: float   # volatility


def cir_drift(params: CIRParams, X, t):
    return params.kappa * (params.theta - X)


def cir_diffusion(params: CIRParams, X, t):
    return params.sigma * jnp.sqrt(jnp.maximum(X, 0.0))


def cir_diffusion_deriv(params: CIRParams, X, t):
    """d(sigma*sqrt(r))/dr = sigma / (2*sqrt(r))"""
    r_safe = jnp.maximum(X, 1e-12)
    return params.sigma / (2.0 * jnp.sqrt(r_safe))


def cir_post_step(params: CIRParams, X):
    """Floor rate at zero (absorption)."""
    return jnp.maximum(X, 0.0)


# ============================================================================
# Hull-White short rate with structured theta(t)
# ============================================================================
class HullWhiteParams(NamedTuple):
    a: float
    sigma: float
    theta_times: jnp.ndarray
    theta_values: jnp.ndarray


def hull_white_drift(params: HullWhiteParams, X, t):
    theta_t = _interp1d_clamped(params.theta_times, params.theta_values, t)
    return theta_t - params.a * X


def hull_white_diffusion(params: HullWhiteParams, X, t):
    return params.sigma * jnp.ones_like(X)


# ============================================================================
# SLV (Stochastic Local Volatility)
# ============================================================================
class SLVParams(NamedTuple):
    mu: float
    kappa: float
    theta: float
    sigma_v: float
    rho: float
    v0: float
    spot_grid: jnp.ndarray
    time_grid: jnp.ndarray
    leverage_surface: jnp.ndarray


def slv_drift(params: SLVParams, X, t):
    """Truncation/reflection drift is identical to Heston."""
    S = X[..., 0:1]
    v = X[..., 1:2]
    return jnp.concatenate([params.mu * S,
                            params.kappa * (params.theta - v)], axis=-1)


def slv_drift_absorption(params: SLVParams, X, t):
    S = X[..., 0:1]
    v = X[..., 1:2]
    v_drift = jnp.where(v > 0.0, params.kappa * (params.theta - v), 0.0)
    return jnp.concatenate([params.mu * S, v_drift], axis=-1)


def _slv_leverage(params: SLVParams, S, t):
    return _interp2d_rectilinear_clamped(
        params.spot_grid, params.time_grid, params.leverage_surface, S, t,
    )


def slv_diffusion(params: SLVParams, X, t):
    S = X[..., 0:1]
    v = X[..., 1:2]
    v_pos = jnp.maximum(v, 0.0)
    sqrt_v = jnp.sqrt(v_pos)
    L = _slv_leverage(params, S, t)
    sigma_S = L * sqrt_v * S
    sigma_v = params.sigma_v * sqrt_v
    return jnp.concatenate([sigma_S, sigma_v], axis=-1)


def slv_diffusion_reflection(params: SLVParams, X, t):
    S = X[..., 0:1]
    v = X[..., 1:2]
    v_pos = jnp.abs(v)
    sqrt_v = jnp.sqrt(v_pos)
    L = _slv_leverage(params, S, t)
    sigma_S = L * sqrt_v * S
    sigma_v = params.sigma_v * sqrt_v
    return jnp.concatenate([sigma_S, sigma_v], axis=-1)


def slv_diffusion_absorption(params: SLVParams, X, t):
    S = X[..., 0:1]
    v = X[..., 1:2]
    v_pos = jnp.where(v > 0.0, v, 0.0)
    sqrt_v = jnp.sqrt(v_pos)
    L = _slv_leverage(params, S, t)
    sigma_S = L * sqrt_v * S
    sigma_v = params.sigma_v * sqrt_v
    return jnp.concatenate([sigma_S, sigma_v], axis=-1)


def slv_post_step(params: SLVParams, X):
    """Truncation scheme: clamp variance to zero."""
    return X.at[..., 1].set(jnp.maximum(X[..., 1], 0.0))


def slv_post_step_reflection(params: SLVParams, X):
    """Reflection scheme: reflect variance at zero."""
    return X.at[..., 1].set(jnp.abs(X[..., 1]))
