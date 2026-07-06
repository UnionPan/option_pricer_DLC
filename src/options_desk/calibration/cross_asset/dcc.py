"""
DCC(1,1) dynamic conditional correlation model.

Two-step estimation:
1. Fit univariate GARCH(1,1) to each factor via batched GARCH
2. Fit DCC parameters (a, b) on standardized residuals via correlation targeting

References:
    Engle, R. (2002). Dynamic Conditional Correlation: A Simple Class of
    Multivariate Generalized Autoregressive Conditional Heteroskedasticity Models.
    Journal of Business & Economic Statistics, 20(3), 339-350.
"""
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import pandas as pd
import optax
from dataclasses import dataclass

from options_desk.calibration.physical.batched import garch
from options_desk.calibration.physical.batched.common import pad_returns


@dataclass
class DCCResult:
    """Result of DCC(1,1) estimation."""
    a: float  # DCC parameter a
    b: float  # DCC parameter b
    qbar: np.ndarray  # (k, k) unconditional correlation of standardized residuals
    garch_params: pd.DataFrame  # per-factor GARCH params: omega, alpha, beta, mu
    last_corr: np.ndarray  # (k, k) final correlation matrix R_T
    log_likelihood: float  # Total log-likelihood (GARCH + DCC)
    converged: bool  # Convergence flag


def _reconstruct_garch_volatility(returns: np.ndarray, omega: float, alpha: float,
                                    beta: float, mu: float) -> np.ndarray:
    """
    Reconstruct GARCH(1,1) volatility path from parameters.

    Matches the recursion in batched/garch.py:
    - Demean: eps = returns - mu
    - Initialize sigma2[0] = sample variance
    - Recursion: sigma2[t] = omega + alpha * eps[t-1]^2 + beta * sigma2[t-1]

    Args:
        returns: (T,) array of returns
        omega, alpha, beta, mu: GARCH parameters

    Returns:
        sigma: (T,) array of volatilities (sqrt of variance)
    """
    T = len(returns)
    eps = returns - mu

    # Initialize with sample variance
    var_init = np.var(eps, ddof=1)

    sigma2 = np.zeros(T)
    sigma2[0] = var_init

    for t in range(1, T):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        sigma2[t] = max(sigma2[t], 1e-12)  # Ensure positive

    return np.sqrt(sigma2)


def _compute_standardized_residuals(factor_returns: np.ndarray,
                                      garch_params: pd.DataFrame) -> np.ndarray:
    """
    Compute standardized residuals ε_t = r_t / σ_t.

    Args:
        factor_returns: (T, k) array of factor returns
        garch_params: DataFrame with columns [omega, alpha, beta, mu], k rows

    Returns:
        eps: (T, k) array of standardized residuals
    """
    T, k = factor_returns.shape
    eps = np.zeros((T, k))

    for i in range(k):
        omega = garch_params.iloc[i]['omega']
        alpha = garch_params.iloc[i]['alpha']
        beta = garch_params.iloc[i]['beta']
        mu = garch_params.iloc[i]['mu']

        sigma = _reconstruct_garch_volatility(factor_returns[:, i], omega, alpha, beta, mu)
        eps[:, i] = (factor_returns[:, i] - mu) / sigma

    return eps


def _dcc_log_likelihood_jax(params, eps, qbar):
    """
    Compute DCC negative log-likelihood via JAX scan.

    Gaussian copula NLL: 0.5 * sum_t [log|R_t| + eps_t^T R_t^{-1} eps_t - eps_t^T eps_t]

    Args:
        params: (a_raw, b_raw) unconstrained parameters
        eps: (T, k) standardized residuals (JAX array)
        qbar: (k, k) unconditional correlation (JAX array)

    Returns:
        Negative log-likelihood (scalar)
    """
    a_raw, b_raw = params

    # Cast parameters to match eps dtype
    a_raw = jnp.asarray(a_raw, dtype=eps.dtype)
    b_raw = jnp.asarray(b_raw, dtype=eps.dtype)

    # Transform to constrained space: a, b > 0, a + b < 0.999
    # Use sigmoid pair: sum = 0.999 * sigmoid(a_raw), ratio = sigmoid(b_raw)
    sum_ab = eps.dtype.type(0.999) * jax.nn.sigmoid(a_raw)
    ratio = jax.nn.sigmoid(b_raw)
    a = sum_ab * ratio
    b = sum_ab * (jnp.asarray(1.0, dtype=eps.dtype) - ratio)

    T, k = eps.shape

    # Initialize Q_0 = qbar (ensure same dtype)
    Q_init = qbar

    # We need to lag eps by one for the recursion
    # eps[0] is not used in recursion (Q_0 = qbar)
    # For t=1, we use eps[0] to update Q_1, then compute NLL with eps[1]
    eps_lagged = eps[:-1]  # (T-1, k)
    eps_current = eps[1:]  # (T-1, k)

    # Scan over time
    def scan_step(Q_prev, inputs):
        eps_lag, eps_curr = inputs

        # Q_t = (1 - a - b) * qbar + a * eps_{t-1} * eps_{t-1}^T + b * Q_{t-1}
        eps_outer = jnp.outer(eps_lag, eps_lag)
        one = jnp.asarray(1.0, dtype=eps.dtype)
        Q_curr = (one - a - b) * qbar + a * eps_outer + b * Q_prev

        # Normalize to get correlation R_t
        q_diag_sqrt = jnp.sqrt(jnp.diag(Q_curr))
        q_diag_sqrt = jnp.maximum(q_diag_sqrt, eps.dtype.type(1e-8))
        inv_sqrt_diag = one / q_diag_sqrt
        R_curr = Q_curr * jnp.outer(inv_sqrt_diag, inv_sqrt_diag)

        # Ensure R_t is numerically a valid correlation matrix
        R_curr = eps.dtype.type(0.5) * (R_curr + R_curr.T)  # Symmetrize
        R_curr = jnp.clip(R_curr, eps.dtype.type(-0.999), eps.dtype.type(0.999))
        R_curr = R_curr.at[jnp.diag_indices(k)].set(one)

        # Compute NLL term
        sign, logdet = jnp.linalg.slogdet(R_curr)
        # Solve R_curr @ x = eps_curr for x, then compute eps_curr^T @ x
        R_inv_eps = jnp.linalg.solve(R_curr, eps_curr)
        quad_form = jnp.dot(eps_curr, R_inv_eps)
        eps_norm_sq = jnp.dot(eps_curr, eps_curr)

        nll_t = eps.dtype.type(0.5) * (logdet + quad_form - eps_norm_sq)

        return Q_curr, nll_t

    _, nll_terms = lax.scan(scan_step, Q_init, (eps_lagged, eps_current))

    total_nll = jnp.sum(nll_terms)

    return total_nll


@jax.jit
def _fit_dcc_single_start(eps, qbar, init_a, init_b):
    """
    Fit DCC parameters from a single starting point.

    Args:
        eps: (T, k) standardized residuals (JAX array)
        qbar: (k, k) unconditional correlation (JAX array)
        init_a, init_b: initial parameter values

    Returns:
        (a, b, final_nll) tuple
    """
    # Convert to unconstrained space
    # sum_ab = 0.999 * sigmoid(a_raw), so sigmoid(a_raw) = sum_ab / 0.999
    # ratio = sigmoid(b_raw) = a / sum_ab
    sum_ab = init_a + init_b
    sigmoid_a_raw = sum_ab / 0.999
    a_raw_init = jnp.log(sigmoid_a_raw / (1 - sigmoid_a_raw))

    ratio = init_a / jnp.maximum(sum_ab, 1e-10)
    b_raw_init = jnp.log(ratio / (1 - ratio))

    init_params = jnp.array([a_raw_init, b_raw_init])

    # Optimizer
    optimizer = optax.adam(learning_rate=5e-3)
    opt_state = optimizer.init(init_params)
    n_steps = 500

    def step(carry, _):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(_dcc_log_likelihood_jax)(params, eps, qbar)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    # Run optimization
    (final_params, _), losses = lax.scan(step, (init_params, opt_state), None, length=n_steps)

    # Extract constrained parameters
    a_raw, b_raw = final_params
    sum_ab = 0.999 * jax.nn.sigmoid(a_raw)
    ratio = jax.nn.sigmoid(b_raw)
    a = sum_ab * ratio
    b = sum_ab * (1 - ratio)

    # Final NLL
    final_nll = _dcc_log_likelihood_jax(final_params, eps, qbar)

    return a, b, final_nll


def fit_dcc(factor_returns: np.ndarray) -> DCCResult:
    """
    Fit DCC(1,1) model to factor returns.

    Two-step procedure:
    1. Fit univariate GARCH(1,1) to each factor
    2. Fit DCC parameters (a, b) to standardized residuals

    Args:
        factor_returns: (T, k) array of factor returns

    Returns:
        DCCResult with fitted parameters
    """
    T, k = factor_returns.shape

    # Step 1: Fit GARCH(1,1) to each factor
    # Transpose to (k, T) for batched GARCH
    returns_transposed = factor_returns.T  # (k, T)

    # All factors have same length (no padding needed), but batched GARCH expects mask
    mask = np.ones((k, T), dtype=np.float32)

    # dt=1.0: factor returns are already per-period, GARCH operates on return scale
    garch_results = garch.fit_batch(returns_transposed, mask, dt=1.0)

    # Extract GARCH parameters
    garch_params = pd.DataFrame({
        'omega': garch_results['omega'],
        'alpha': garch_results['alpha'],
        'beta': garch_results['beta'],
        'mu': garch_results['mu'],
    })

    # Filter out factors with NaN GARCH parameters (failed fits)
    valid_mask = ~garch_params.isna().any(axis=1)
    if not valid_mask.all():
        n_failed = (~valid_mask).sum()
        print(f"Warning: {n_failed}/{k} factors failed GARCH fit (NaN params). Proceeding with {valid_mask.sum()} valid factors.")

        # Keep only valid factors
        garch_params = garch_params[valid_mask].reset_index(drop=True)
        factor_returns = factor_returns[:, valid_mask.to_numpy()]
        garch_ll_vec = garch_results['log_likelihood'][valid_mask.to_numpy()]
        k = factor_returns.shape[1]  # Update k to reflect valid factors only
    else:
        garch_ll_vec = garch_results['log_likelihood']

    # GARCH log-likelihood
    garch_ll = np.sum(garch_ll_vec[np.isfinite(garch_ll_vec)])

    # Step 2: Compute standardized residuals
    eps = _compute_standardized_residuals(factor_returns, garch_params)

    # Unconditional correlation of standardized residuals
    qbar = np.corrcoef(eps.T)

    # Convert to JAX arrays for optimization
    eps_jax = jnp.asarray(eps, dtype=jnp.float32)
    qbar_jax = jnp.asarray(qbar, dtype=jnp.float32)

    # Multi-start optimization
    starts = [(0.05, 0.90), (0.02, 0.95)]

    best_a = None
    best_b = None
    best_nll = np.inf

    for init_a, init_b in starts:
        a, b, nll = _fit_dcc_single_start(eps_jax, qbar_jax, init_a, init_b)

        # Convert to numpy
        a_np = float(a)
        b_np = float(b)
        nll_np = float(nll)

        if nll_np < best_nll:
            best_a = a_np
            best_b = b_np
            best_nll = nll_np

    # DCC log-likelihood (negative of NLL)
    dcc_ll = -best_nll

    # Total log-likelihood
    total_ll = garch_ll + dcc_ll

    # Convergence check
    converged = bool(np.isfinite(total_ll))

    # Compute final correlation matrix R_T
    R_path = dcc_corr_path_internal(best_a, best_b, qbar, eps)
    last_corr = R_path[-1]

    return DCCResult(
        a=best_a,
        b=best_b,
        qbar=qbar,
        garch_params=garch_params,
        last_corr=last_corr,
        log_likelihood=float(total_ll),
        converged=converged,
    )


def dcc_corr_path_internal(a: float, b: float, qbar: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """
    Internal helper to compute DCC correlation path.

    Args:
        a, b: DCC parameters
        qbar: (k, k) unconditional correlation
        eps: (T, k) standardized residuals

    Returns:
        R_path: (T, k, k) correlation matrices
    """
    T, k = eps.shape
    Q = np.zeros((T, k, k))
    R = np.zeros((T, k, k))

    # Initialize Q_0 = qbar
    Q[0] = qbar.copy()
    R[0] = qbar.copy()

    for t in range(1, T):
        # Q_t = (1 - a - b) * qbar + a * eps_{t-1} * eps_{t-1}^T + b * Q_{t-1}
        eps_outer = np.outer(eps[t - 1], eps[t - 1])
        Q[t] = (1 - a - b) * qbar + a * eps_outer + b * Q[t - 1]

        # Normalize to correlation
        q_diag_sqrt = np.sqrt(np.diag(Q[t]))
        q_diag_sqrt = np.maximum(q_diag_sqrt, 1e-8)
        inv_sqrt_diag = 1.0 / q_diag_sqrt
        R[t] = Q[t] * np.outer(inv_sqrt_diag, inv_sqrt_diag)

        # Ensure symmetric and unit diagonal
        R[t] = 0.5 * (R[t] + R[t].T)
        np.fill_diagonal(R[t], 1.0)

    return R


def dcc_corr_path(result: DCCResult, factor_returns: np.ndarray) -> np.ndarray:
    """
    Reconstruct full DCC correlation path R_t.

    Args:
        result: DCCResult from fit_dcc
        factor_returns: (T, k) array of factor returns

    Returns:
        R_path: (T, k, k) array of correlation matrices
    """
    # Recompute standardized residuals
    eps = _compute_standardized_residuals(factor_returns, result.garch_params)

    # Reconstruct correlation path
    return dcc_corr_path_internal(result.a, result.b, result.qbar, eps)


def _evaluate_dcc_nll(a: float, b: float, qbar: np.ndarray,
                       garch_params: pd.DataFrame, factor_returns: np.ndarray) -> float:
    """
    Evaluate DCC negative log-likelihood at given parameters.

    Helper function for testing objective sanity.

    Args:
        a, b: DCC parameters
        qbar: (k, k) unconditional correlation
        garch_params: DataFrame with GARCH parameters
        factor_returns: (T, k) array of factor returns

    Returns:
        Negative log-likelihood (float)
    """
    # Compute standardized residuals
    eps = _compute_standardized_residuals(factor_returns, garch_params)

    # Convert to JAX
    eps_jax = jnp.asarray(eps, dtype=jnp.float32)
    qbar_jax = jnp.asarray(qbar, dtype=jnp.float32)

    # Transform (a, b) to unconstrained space
    sum_ab = a + b
    sigmoid_a_raw = sum_ab / 0.999
    a_raw = jnp.log(sigmoid_a_raw / (1 - sigmoid_a_raw))

    ratio = a / max(sum_ab, 1e-10)
    b_raw = jnp.log(ratio / (1 - ratio))

    params = jnp.array([a_raw, b_raw])

    # Evaluate NLL
    nll = _dcc_log_likelihood_jax(params, eps_jax, qbar_jax)

    return float(nll)
