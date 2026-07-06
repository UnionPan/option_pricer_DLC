"""
Batched Merton Jump-Diffusion MLE calibration via JAX.

Matches conventions of physical/merton_calibrator.py (scipy reference).
Uses optax Adam multi-start optimization with 3 initial lambda values.

Model:
    r_t = log(S_t / S_{t-1}) is a Poisson mixture of normals:
    - N_t ~ Poisson(lambda * dt)
    - For k jumps: r_t ~ N((mu - 0.5*sigma^2)*dt + k*mu_j, sigma^2*dt + k*sigma_j^2)

Likelihood is a truncated Poisson mixture (k=0..k_max=5).
"""
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp
import numpy as np
import optax


def _softplus(x):
    """Numerically stable softplus."""
    return jnp.logaddexp(x, 0.0)


def _make_merton_nll_fn(k_max):
    """
    Factory to create a Merton NLL function with fixed k_max.

    This avoids JAX tracing issues with dynamic k_max.
    """
    # Pre-create k_values as a static constant
    k_values_np = np.arange(k_max + 1, dtype=np.float32)

    def _merton_log_likelihood(params, returns, mask, dt):
        """
        Compute Merton jump-diffusion negative log-likelihood (to minimize).

        Matches scipy reference exactly:
        - Truncated Poisson mixture over k=0..k_max
        - For each k: weight = Poisson(k; lambda*dt),
                      mean_k = (mu - 0.5*sigma^2)*dt + k*mu_j,
                      var_k = sigma^2*dt + k*sigma_j^2
        - NLL = -sum_{t} log[ sum_{k=0}^{k_max} weight_k * N(r_t; mean_k, var_k) ]
        - MASKED: padded steps contribute zero NLL

        Args:
            params: (mu, a_sigma, a_lam, mu_j, a_sigma_j) unconstrained parameters
                    where sigma = softplus(a_sigma), lam = softplus(a_lam), sigma_j = softplus(a_sigma_j)
            returns: (T,) array of returns
            mask: (T,) binary mask
            dt: scalar time increment

        Returns:
            Negative log-likelihood (scalar)
        """
        mu, a_sigma, a_lam, mu_j, a_sigma_j = params

        # Transform to constrained space
        sigma = _softplus(a_sigma)
        lam = _softplus(a_lam)
        sigma_j = _softplus(a_sigma_j)

        # Cast to match returns dtype
        mu = jnp.asarray(mu, dtype=returns.dtype)
        sigma = jnp.asarray(sigma, dtype=returns.dtype)
        lam = jnp.asarray(lam, dtype=returns.dtype)
        mu_j = jnp.asarray(mu_j, dtype=returns.dtype)
        sigma_j = jnp.asarray(sigma_j, dtype=returns.dtype)
        dt = jnp.asarray(dt, dtype=returns.dtype)

        # Base mean and variance (for diffusion component)
        # Matches scipy exactly: includes kappa correction term
        # kappa = E[exp(J) - 1] = exp(mu_j + 0.5*sigma_j^2) - 1
        half = returns.dtype.type(0.5)
        one = returns.dtype.type(1.0)
        kappa = jnp.exp(mu_j + half * sigma_j**2) - one
        base_mean = (mu - half * sigma**2 - lam * kappa) * dt
        base_var = sigma**2 * dt

        # Use pre-created k_values
        k_values = jnp.asarray(k_values_np, dtype=returns.dtype)

        def compute_log_prob_k(k_val, r):
            """Compute log P(r | k) * P(k) for a given k and return r."""
            # Poisson weight: P(N=k | lambda*dt)
            lam_dt = lam * dt
            log_weight = k_val * jnp.log(lam_dt) - lam_dt - jax.lax.lgamma(k_val + returns.dtype.type(1.0))

            # Mean and variance for this k
            mean_k = base_mean + k_val * mu_j
            var_k = base_var + k_val * sigma_j**2
            var_k = jnp.maximum(var_k, returns.dtype.type(1e-12))

            # Gaussian log-density: log N(r; mean_k, var_k)
            two_pi = returns.dtype.type(2.0 * jnp.pi)
            log_gauss = -returns.dtype.type(0.5) * (jnp.log(two_pi) + jnp.log(var_k) + (r - mean_k)**2 / var_k)

            return log_weight + log_gauss

        # For each return, compute mixture probability
        def nll_per_return(r, m):
            """Compute -log P(r) for a single return using Poisson mixture."""
            # Compute log probs for all k values
            log_probs = jax.vmap(lambda k: compute_log_prob_k(k, r))(k_values)

            # Log of mixture probability: log( sum_k exp(log_probs[k]) )
            log_mix_prob = logsumexp(log_probs)

            # NLL contribution (masked)
            zero = returns.dtype.type(0.0)
            nll = jnp.where(m > zero, -log_mix_prob, zero)

            return nll

        # Vmap over time
        nll_terms = jax.vmap(nll_per_return)(returns, mask)
        total_nll = jnp.sum(nll_terms)

        return total_nll

    return _merton_log_likelihood


def _make_fit_single_start_fn(k_max):
    """Factory to create a fit function with fixed k_max."""
    nll_fn = _make_merton_nll_fn(k_max)

    @jax.jit
    def _fit_single_asset_single_start(returns, mask, dt, init_mu, init_sigma, init_lam, init_mu_j, init_sigma_j):
        """
        Fit Merton jump-diffusion to a single asset from a single starting point.

        Args:
            returns: (T,) array of returns
            mask: (T,) binary mask
            dt: scalar time increment
            init_mu, init_sigma, init_lam, init_mu_j, init_sigma_j: initial parameter values

        Returns:
            (final_params, final_nll, n) tuple
        """
        # Convert to unconstrained space
        # mu: free
        # sigma = softplus(a_sigma) => a_sigma = log(exp(sigma) - 1)
        # lam = softplus(a_lam) => a_lam = log(exp(lam) - 1)
        # mu_j: free
        # sigma_j = softplus(a_sigma_j) => a_sigma_j = log(exp(sigma_j) - 1)

        a_sigma = jnp.log(jnp.exp(init_sigma) - returns.dtype.type(1.0))
        a_lam = jnp.log(jnp.exp(init_lam) - returns.dtype.type(1.0))
        a_sigma_j = jnp.log(jnp.exp(init_sigma_j) - returns.dtype.type(1.0))

        init_params = jnp.array([init_mu, a_sigma, a_lam, init_mu_j, a_sigma_j], dtype=returns.dtype)

        # Optimizer
        optimizer = optax.adam(learning_rate=5e-3)
        opt_state = optimizer.init(init_params)
        n_steps = 1000

        def step(carry, _):
            params, opt_state = carry
            loss, grads = jax.value_and_grad(nll_fn)(
                params, returns, mask, dt
            )
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss

        # Run optimization
        (final_params, _), losses = lax.scan(step, (init_params, opt_state), None, length=n_steps)

        # Final loss
        final_nll = nll_fn(final_params, returns, mask, dt)

        # Count observations
        n = jnp.sum(mask)

        return final_params, final_nll, n

    return _fit_single_asset_single_start


def _make_fit_single_asset_fn(k_max):
    """Factory to create a fit function with fixed k_max."""
    fit_single_start = _make_fit_single_start_fn(k_max)

    def _fit_single_asset(returns, mask, dt, init_params_list):
        """
        Fit Merton jump-diffusion to a single asset using multi-start optimization.

        Args:
            returns: (T,) array of returns
            mask: (T,) binary mask
            dt: scalar time increment
            init_params_list: list of (mu, sigma, lam, mu_j, sigma_j) tuples for initialization

        Returns:
            Dictionary with best fit parameters
        """
        # Run optimization from all starting points
        init_mus = jnp.array([p[0] for p in init_params_list], dtype=returns.dtype)
        init_sigmas = jnp.array([p[1] for p in init_params_list], dtype=returns.dtype)
        init_lams = jnp.array([p[2] for p in init_params_list], dtype=returns.dtype)
        init_mu_js = jnp.array([p[3] for p in init_params_list], dtype=returns.dtype)
        init_sigma_js = jnp.array([p[4] for p in init_params_list], dtype=returns.dtype)

        # Vmap over starting points
        final_params_all, final_nlls_all, n_all = jax.vmap(
            lambda mu, sig, lam, muj, sigj: fit_single_start(
                returns, mask, dt, mu, sig, lam, muj, sigj
            )
        )(init_mus, init_sigmas, init_lams, init_mu_js, init_sigma_js)

        # Select best result (minimum NLL)
        best_idx = jnp.argmin(final_nlls_all)
        best_params = final_params_all[best_idx]
        best_nll = final_nlls_all[best_idx]
        n = n_all[best_idx]

        # Extract constrained parameters
        mu, a_sigma, a_lam, mu_j, a_sigma_j = best_params
        sigma = _softplus(a_sigma)
        lam = _softplus(a_lam)
        sigma_j = _softplus(a_sigma_j)

        # Log-likelihood (negative of NLL)
        log_likelihood = -best_nll

        # Convergence: finite log-likelihood
        converged = jnp.isfinite(log_likelihood)

        # AIC and BIC (matching scipy: 5 parameters)
        k_params = 5  # mu, sigma, lam, mu_j, sigma_j
        aic = returns.dtype.type(2.0) * k_params - returns.dtype.type(2.0) * log_likelihood
        bic = k_params * jnp.log(n) - returns.dtype.type(2.0) * log_likelihood

        return {
            "mu": mu,
            "sigma": sigma,
            "lam": lam,  # Match scipy's "lambda_" but use "lam" for dict key
            "mu_j": mu_j,
            "sigma_j": sigma_j,
            "log_likelihood": log_likelihood,
            "aic": aic,
            "bic": bic,
            "n_observations": n,
            "converged": converged,
        }

    return _fit_single_asset


def fit_batch(returns, mask, dt, k_max=5):
    """
    Calibrate Merton jump-diffusion parameters for a batch of assets.

    Uses optax Adam optimizer with multi-start initialization (3 starting points for lambda).

    Args:
        returns: (N, T) array of log-returns
        mask: (N, T) binary mask (1.0 = valid, 0.0 = padding)
        dt: Time increment in years (e.g., 1/252 for daily)
        k_max: Truncation level for Poisson mixture (default 5, matching scipy)

    Returns:
        Dictionary with keys:
            mu: (N,) drift parameter
            sigma: (N,) diffusion volatility
            lam: (N,) jump intensity
            mu_j: (N,) jump mean
            sigma_j: (N,) jump volatility
            log_likelihood: (N,) log-likelihood values
            aic: (N,) Akaike Information Criterion
            bic: (N,) Bayesian Information Criterion
            n_observations: (N,) number of observations per asset
            converged: (N,) boolean array, True if optimization succeeded
    """
    returns_jax = jnp.asarray(returns, dtype=jnp.float32)
    mask_jax = jnp.asarray(mask, dtype=jnp.float32)

    N = returns_jax.shape[0]

    # Define 3 starting points for lambda (per year)
    lambda_starts = [5.0, 20.0, 60.0]

    # Create fit function with fixed k_max
    fit_single_asset = _make_fit_single_asset_fn(k_max)

    # For each asset, compute initial parameters based on sample moments
    def compute_starts_for_asset(returns_row, mask_row):
        n = jnp.sum(mask_row)
        mean_r = jnp.sum(returns_row * mask_row) / jnp.maximum(n, returns_row.dtype.type(1.0))
        sq_dev = (returns_row - mean_r) ** 2
        var_sample = jnp.sum(sq_dev * mask_row) / jnp.maximum(n - returns_row.dtype.type(1.0), returns_row.dtype.type(1.0))

        # Initial guesses from sample moments
        mu_init = mean_r / dt
        sigma_init = jnp.sqrt(var_sample / dt)
        sigma_init = jnp.maximum(sigma_init, returns_row.dtype.type(0.01))  # Ensure positive

        # Initial jump parameters
        mu_j_init = returns_row.dtype.type(0.0)
        sigma_j_init = returns_row.dtype.type(0.05)

        starts = []
        for lam in lambda_starts:
            lam_typed = returns_row.dtype.type(lam)
            starts.append((mu_init, sigma_init, lam_typed, mu_j_init, sigma_j_init))

        return starts

    # Fit each asset
    def fit_asset(returns_row, mask_row):
        starts = compute_starts_for_asset(returns_row, mask_row)
        return fit_single_asset(returns_row, mask_row, dt, starts)

    # Vmap over assets
    results = jax.vmap(fit_asset)(returns_jax, mask_jax)

    # Convert to numpy
    return {
        k: np.asarray(v, dtype=np.float64 if v.dtype != bool else bool)
        for k, v in results.items()
    }
