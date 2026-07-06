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

    Uses array broadcasting over ``(T, k_max+1)`` instead of nested
    ``jax.vmap`` to keep the XLA program small and fast to compile.
    """
    # Pre-create k_values as a static constant — shape (K,)
    k_values_np = np.arange(k_max + 1, dtype=np.float32)

    def _merton_nll(params, returns, mask, dt):
        """
        Compute Merton jump-diffusion negative log-likelihood (to minimize).

        Args:
            params: (5,) unconstrained parameters
                    [mu, a_sigma, a_lam, mu_j, a_sigma_j]
            returns: (T,) array of returns
            mask: (T,) binary mask
            dt: scalar time increment

        Returns:
            Negative log-likelihood (scalar)
        """
        mu, a_sigma, a_lam, mu_j, a_sigma_j = params
        sigma = _softplus(a_sigma)
        lam = _softplus(a_lam)
        sigma_j = _softplus(a_sigma_j)

        dt = jnp.asarray(dt, dtype=returns.dtype)

        # Compensated drift
        half = returns.dtype.type(0.5)
        one = returns.dtype.type(1.0)
        kappa = jnp.exp(mu_j + half * sigma_j**2) - one
        base_mean = (mu - half * sigma**2 - lam * kappa) * dt  # scalar
        base_var = sigma**2 * dt                                # scalar

        # k_values: (K,)
        k_values = jnp.asarray(k_values_np, dtype=returns.dtype)

        # --- Poisson log-weights: shape (K,) ---
        lam_dt = lam * dt
        log_weights = (
            k_values * jnp.log(lam_dt)
            - lam_dt
            - jax.lax.lgamma(k_values + one)
        )

        # --- Gaussian mixture components: broadcast (T, K) ---
        r = returns[:, None]                      # (T, 1)
        mean_k = base_mean + k_values * mu_j      # (K,)
        var_k = base_var + k_values * sigma_j**2   # (K,)
        var_k = jnp.maximum(var_k, returns.dtype.type(1e-12))

        two_pi = returns.dtype.type(2.0 * np.pi)
        log_gauss = -half * (
            jnp.log(two_pi)
            + jnp.log(var_k)
            + (r - mean_k)**2 / var_k
        )  # (T, K)

        # --- Log mixture probability per return: (T,) ---
        log_probs = log_weights + log_gauss     # (T, K)
        log_mix = logsumexp(log_probs, axis=1)  # (T,)

        # Masked NLL
        zero = returns.dtype.type(0.0)
        nll_terms = jnp.where(mask > zero, -log_mix, zero)
        return jnp.sum(nll_terms)

    return _merton_nll


def _make_fit_single_start_fn(k_max):
    """
    Factory to create a JIT-compiled single-start fit function.

    Uses ``lax.scan`` over 1500 Adam steps with the vectorized (broadcast)
    NLL.  Because the NLL avoids nested ``jax.vmap``, the XLA program is
    compact and compiles in < 1 s (vs 10+ minutes with the original nested-
    vmap formulation under JAX >= 0.9).
    """
    nll_fn = _make_merton_nll_fn(k_max)
    optimizer = optax.adam(learning_rate=5e-3)

    @jax.jit
    def _fit_single_start(returns, mask, dt, init_mu, init_sigma, init_lam,
                          init_mu_j, init_sigma_j):
        """
        Fit Merton jump-diffusion to one asset from one starting point.

        Returns ``(final_params, final_nll, n_obs)`` tuple.
        """
        a_sigma = jnp.log(jnp.exp(init_sigma) - returns.dtype.type(1.0))
        a_lam = jnp.log(jnp.exp(init_lam) - returns.dtype.type(1.0))
        a_sigma_j = jnp.log(jnp.exp(init_sigma_j) - returns.dtype.type(1.0))

        init_params = jnp.array(
            [init_mu, a_sigma, a_lam, init_mu_j, a_sigma_j],
            dtype=returns.dtype,
        )

        opt_state = optimizer.init(init_params)
        n_steps = 1500

        def step(carry, _):
            params, opt_state = carry
            loss, grads = jax.value_and_grad(nll_fn)(
                params, returns, mask, dt
            )
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss

        (final_params, _), _losses = lax.scan(
            step, (init_params, opt_state), None, length=n_steps
        )

        final_nll = nll_fn(final_params, returns, mask, dt)
        n = jnp.sum(mask)

        return final_params, final_nll, n

    return _fit_single_start


# ---------------------------------------------------------------------------
# Module-level cache so repeated fit_batch calls with the same k_max reuse
# the same JIT-compiled executable (compiled once on first call).
# ---------------------------------------------------------------------------
_FIT_SINGLE_START_CACHE = {}


def fit_batch(returns, mask, dt, k_max=5):
    """
    Calibrate Merton jump-diffusion parameters for a batch of assets.

    Uses optax Adam optimizer with multi-start initialization (3 starting
    points for lambda).

    Both assets and starting points are iterated in Python so the
    JIT-compiled single-start kernel is compiled **once** (for a given
    time-series length) and reused for every ``(asset, start)`` pair.

    The NLL uses array broadcasting over ``(T, k_max+1)`` instead of
    nested ``jax.vmap``, keeping the XLA program compact (< 1 s compile
    on JAX 0.9).

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

    # Reuse the same JIT-compiled function across calls with the same k_max
    if k_max not in _FIT_SINGLE_START_CACHE:
        _FIT_SINGLE_START_CACHE[k_max] = _make_fit_single_start_fn(k_max)
    fit_single_start = _FIT_SINGLE_START_CACHE[k_max]

    # 3 starting points for lambda (per year)
    lambda_starts = [5.0, 20.0, 60.0]

    per_asset_results = []
    for i in range(N):
        r_row = returns_jax[i]
        m_row = mask_jax[i]

        # Compute sample-moment-based initial guesses
        n_obs = float(jnp.sum(m_row))
        mean_r = float(jnp.sum(r_row * m_row) / max(n_obs, 1.0))
        sq_dev = (r_row - mean_r) ** 2
        var_sample = float(jnp.sum(sq_dev * m_row) / max(n_obs - 1.0, 1.0))
        mu_init = mean_r / dt
        sigma_init = max(np.sqrt(var_sample / dt), 0.01)
        mu_j_init = 0.0
        sigma_j_init = 0.05

        # Run each starting point and keep best
        best_params = None
        best_nll = float("inf")
        best_n = None

        for lam_start in lambda_starts:
            params_out, nll_out, n_out = fit_single_start(
                r_row, m_row, dt,
                jnp.float32(mu_init),
                jnp.float32(sigma_init),
                jnp.float32(lam_start),
                jnp.float32(mu_j_init),
                jnp.float32(sigma_j_init),
            )
            nll_val = float(nll_out)
            if nll_val < best_nll:
                best_nll = nll_val
                best_params = params_out
                best_n = n_out

        # Extract constrained parameters from best run
        mu, a_sigma, a_lam, mu_j, a_sigma_j = best_params
        sigma = float(_softplus(a_sigma))
        lam = float(_softplus(a_lam))
        sigma_j = float(_softplus(a_sigma_j))
        log_likelihood = -best_nll
        n_val = float(best_n)
        converged = np.isfinite(log_likelihood)

        k_p = 5
        aic = 2.0 * k_p - 2.0 * log_likelihood
        bic = k_p * np.log(n_val) - 2.0 * log_likelihood

        per_asset_results.append({
            "mu": float(mu),
            "sigma": sigma,
            "lam": lam,
            "mu_j": float(mu_j),
            "sigma_j": sigma_j,
            "log_likelihood": log_likelihood,
            "aic": aic,
            "bic": bic,
            "n_observations": n_val,
            "converged": converged,
        })

    # Stack per-asset dicts into batched arrays (numpy float64 / bool).
    results = {}
    for k in per_asset_results[0]:
        vals = [r[k] for r in per_asset_results]
        arr = np.array(vals)
        if arr.dtype == bool:
            results[k] = arr
        else:
            results[k] = arr.astype(np.float64)

    return results
