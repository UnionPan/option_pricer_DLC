"""
Batched GARCH(1,1) QMLE calibration via JAX.

Matches conventions of physical/garch_calibrator.py (scipy reference).
Uses optax Adam multi-start optimization with 4 initial parameter grids.
"""
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import optax


def _softplus(x):
    """Numerically stable softplus."""
    return jnp.logaddexp(x, 0.0)


def _unconstrain_params(omega, alpha, beta):
    """
    Transform constrained params to unconstrained space.

    omega > 0 via omega = softplus(a)
    alpha + beta < 0.999 via:
        alpha = 0.999 * sigmoid(b) * sigmoid(c)
        beta = 0.999 * sigmoid(b) * (1 - sigmoid(c))

    Returns:
        (a, b, c) unconstrained parameters
    """
    # omega -> a
    a = jnp.log(jnp.exp(omega) - 1.0)  # inverse softplus

    # alpha, beta -> b, c
    # Given alpha, beta, we need to find b, c such that:
    # alpha = 0.999 * sigmoid(b) * sigmoid(c)
    # beta = 0.999 * sigmoid(b) * (1 - sigmoid(c))
    # => alpha + beta = 0.999 * sigmoid(b)
    # => alpha / (alpha + beta) = sigmoid(c)

    sum_ab = alpha + beta
    sigmoid_b = sum_ab / 0.999
    b = jnp.log(sigmoid_b / (1 - sigmoid_b))  # inverse sigmoid (logit)

    ratio_c = alpha / jnp.maximum(sum_ab, 1e-10)
    c = jnp.log(ratio_c / (1 - ratio_c))  # inverse sigmoid (logit)

    return a, b, c


def _constrain_params(a, b, c):
    """
    Transform unconstrained params to constrained space.

    omega = softplus(a)
    alpha = 0.999 * sigmoid(b) * sigmoid(c)
    beta = 0.999 * sigmoid(b) * (1 - sigmoid(c))

    Returns:
        (omega, alpha, beta) constrained parameters
    """
    omega = _softplus(a)

    sigmoid_b = jax.nn.sigmoid(b)
    sigmoid_c = jax.nn.sigmoid(c)

    alpha = 0.999 * sigmoid_b * sigmoid_c
    beta = 0.999 * sigmoid_b * (1 - sigmoid_c)

    return omega, alpha, beta


def _garch_log_likelihood(params, returns, mask, var_init):
    """
    Compute GARCH(1,1) negative log-likelihood (to minimize).

    Matches scipy reference exactly:
    - Demean returns: eps = returns - mu
    - Initialize sigma2[0] = var_init
    - Recursion: sigma2[t] = omega + alpha * eps[t-1]^2 + beta * sigma2[t-1]
    - NLL: -0.5 * sum(log(2π) + log(sigma2) + eps^2 / sigma2)
    - MASKED: padded steps contribute zero NLL and carry sigma2 unchanged

    Args:
        params: (a, b, c, mu) unconstrained parameters
        returns: (T,) array of returns
        mask: (T,) binary mask
        var_init: scalar initial variance (sample variance with ddof=1)

    Returns:
        Negative log-likelihood (scalar)
    """
    a, b, c, mu = params
    omega, alpha, beta = _constrain_params(a, b, c)

    # Cast parameters to match returns dtype
    mu_typed = jnp.asarray(mu, dtype=returns.dtype)
    omega_typed = jnp.asarray(omega, dtype=returns.dtype)
    alpha_typed = jnp.asarray(alpha, dtype=returns.dtype)
    beta_typed = jnp.asarray(beta, dtype=returns.dtype)

    # Demean returns
    eps = returns - mu_typed

    # GARCH recursion with masking
    # Match scipy: sigma2[0] = var_init (given), then for t >= 1:
    # sigma2[t] = omega + alpha * eps[t-1]^2 + beta * sigma2[t-1]
    def step_fn(carry, inputs):
        sigma2_curr, t = carry
        eps_t, mask_t = inputs

        # NLL contribution for current time t using sigma2_curr
        # Use dtype-safe constants
        half = returns.dtype.type(0.5)
        two_pi = returns.dtype.type(2.0 * jnp.pi)
        zero = returns.dtype.type(0.0)

        nll_t = jnp.where(
            mask_t > 0,
            half * (jnp.log(two_pi) + jnp.log(sigma2_curr) + eps_t ** 2 / sigma2_curr),
            zero
        )

        # Update sigma2 for next step: sigma2[t+1] = omega + alpha * eps[t]^2 + beta * sigma2[t]
        # But only if current step is valid (otherwise carry forward)
        sigma2_next = omega_typed + alpha_typed * eps_t ** 2 + beta_typed * sigma2_curr
        sigma2_next = jnp.maximum(sigma2_next, returns.dtype.type(1e-12))

        # If masked (invalid), carry sigma2 forward unchanged
        sigma2_out = jnp.where(mask_t > 0, sigma2_next, sigma2_curr)

        return (sigma2_out, t + 1), nll_t

    # Initial state
    init_carry = (var_init, 0)

    # Run scan
    _, nll_terms = lax.scan(step_fn, init_carry, (eps, mask))

    # Sum NLL
    total_nll = jnp.sum(nll_terms)

    return total_nll


@jax.jit
def _fit_single_asset_single_start(returns, mask, dt, init_omega, init_alpha, init_beta):
    """
    Fit GARCH(1,1) to a single asset from a single starting point.

    Args:
        returns: (T,) array of returns
        mask: (T,) binary mask
        dt: scalar time increment
        init_omega, init_alpha, init_beta: initial parameter values

    Returns:
        (final_params, final_nll) tuple
    """
    # Compute sample statistics
    n = jnp.sum(mask)
    mean_return = jnp.sum(returns * mask) / jnp.maximum(n, 1.0)

    # Sample variance with ddof=1 (Bessel's correction)
    sq_dev = (returns - mean_return) ** 2
    var_init = jnp.sum(sq_dev * mask) / jnp.maximum(n - 1, 1.0)

    # Initial mu
    mu_init = mean_return

    # Convert to unconstrained
    a_init, b_init, c_init = _unconstrain_params(init_omega, init_alpha, init_beta)
    init_params = jnp.array([a_init, b_init, c_init, mu_init])

    # Optimizer
    optimizer = optax.adam(learning_rate=1e-2)
    opt_state = optimizer.init(init_params)
    n_steps = 800

    def step(carry, _):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(_garch_log_likelihood)(
            params, returns, mask, var_init
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    # Run optimization
    (final_params, _), losses = lax.scan(step, (init_params, opt_state), None, length=n_steps)

    # Final loss
    final_nll = _garch_log_likelihood(final_params, returns, mask, var_init)

    return final_params, final_nll, n


def _fit_single_asset(returns, mask, dt, init_params_list):
    """
    Fit GARCH(1,1) to a single asset using multi-start optimization.

    Args:
        returns: (T,) array of returns
        mask: (T,) binary mask
        dt: scalar time increment
        init_params_list: list of (omega, alpha, beta) tuples for initialization

    Returns:
        Dictionary with best fit parameters
    """
    # Run optimization from all starting points
    init_omegas = jnp.array([p[0] for p in init_params_list])
    init_alphas = jnp.array([p[1] for p in init_params_list])
    init_betas = jnp.array([p[2] for p in init_params_list])

    # Vmap over starting points
    final_params_all, final_nlls_all, n_all = jax.vmap(
        lambda o, a, b: _fit_single_asset_single_start(returns, mask, dt, o, a, b)
    )(init_omegas, init_alphas, init_betas)

    # Select best result (minimum NLL)
    best_idx = jnp.argmin(final_nlls_all)
    best_params = final_params_all[best_idx]
    best_nll = final_nlls_all[best_idx]
    n = n_all[best_idx]

    # Extract constrained parameters
    a, b, c, mu = best_params
    omega, alpha, beta = _constrain_params(a, b, c)

    # Log-likelihood (negative of NLL)
    log_likelihood = -best_nll

    # Convergence: finite log-likelihood
    converged = jnp.isfinite(log_likelihood)

    # AIC and BIC
    k = 4  # mu, omega, alpha, beta
    aic = 2 * k - 2 * log_likelihood
    bic = k * jnp.log(n) - 2 * log_likelihood

    return {
        "mu": mu,
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "log_likelihood": log_likelihood,
        "aic": aic,
        "bic": bic,
        "n_observations": n,
        "converged": converged,
    }


def fit_batch(returns, mask, dt):
    """
    Calibrate GARCH(1,1) parameters for a batch of assets.

    Uses optax Adam optimizer with multi-start initialization (4 starting points).

    Args:
        returns: (N, T) array of log-returns
        mask: (N, T) binary mask (1.0 = valid, 0.0 = padding)
        dt: Time increment in years (e.g., 1/252 for daily)

    Returns:
        Dictionary with keys:
            mu: (N,) mean return
            omega: (N,) constant term
            alpha: (N,) ARCH coefficient
            beta: (N,) GARCH coefficient
            log_likelihood: (N,) log-likelihood values
            aic: (N,) Akaike Information Criterion
            bic: (N,) Bayesian Information Criterion
            n_observations: (N,) number of observations per asset
            converged: (N,) boolean array, True if optimization succeeded
    """
    returns_jax = jnp.asarray(returns, dtype=jnp.float32)
    mask_jax = jnp.asarray(mask, dtype=jnp.float32)

    N = returns_jax.shape[0]

    # Define 4 starting points for (alpha, beta)
    # omega will be matched to sample variance via omega = s^2 * (1 - alpha - beta)
    alpha_beta_starts = [
        (0.05, 0.90),
        (0.10, 0.85),
        (0.02, 0.95),
        (0.08, 0.80),
    ]

    # For each asset, compute starting omegas based on sample variance
    def compute_starts_for_asset(returns_row, mask_row):
        n = jnp.sum(mask_row)
        mean_r = jnp.sum(returns_row * mask_row) / jnp.maximum(n, 1.0)
        sq_dev = (returns_row - mean_r) ** 2
        var_sample = jnp.sum(sq_dev * mask_row) / jnp.maximum(n - 1, 1.0)

        starts = []
        for alpha, beta in alpha_beta_starts:
            omega = var_sample * (1 - alpha - beta)
            omega = jnp.maximum(omega, 1e-8)  # Ensure positive
            starts.append((omega, alpha, beta))

        return starts

    # Fit each asset
    def fit_asset(returns_row, mask_row):
        starts = compute_starts_for_asset(returns_row, mask_row)
        return _fit_single_asset(returns_row, mask_row, dt, starts)

    # Vmap over assets
    results = jax.vmap(fit_asset)(returns_jax, mask_jax)

    # Convert to numpy
    return {
        k: np.asarray(v, dtype=np.float64 if v.dtype != bool else bool)
        for k, v in results.items()
    }
