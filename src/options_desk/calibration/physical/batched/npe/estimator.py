"""
Amortized Heston NPE estimator for batch calibration.

Implements:
- fit_batch: calibrate N assets' returns via pre-trained NPE checkpoint
- Posterior moments (mean, std) via Monte Carlo sampling with fixed seed
- Per-row log-likelihood diagnostic
- Checkpoint loading with clear error messages
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .model import ConditionalMDN, mdn_nll
from .simulate import summary_features
from .train import load_npe, sample_posterior


# ────────────────────────────────────────────────────────────────────────────
# Repo root and default checkpoint path
# ────────────────────────────────────────────────────────────────────────────

# Derive repo root: this file is at src/options_desk/calibration/physical/batched/npe/estimator.py
# Repo root is 7 levels up
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[6]

DEFAULT_CHECKPOINT = _REPO_ROOT / "data" / "npe" / "heston_mdn.pkl"


# ────────────────────────────────────────────────────────────────────────────
# Batch estimator
# ────────────────────────────────────────────────────────────────────────────

def fit_batch(
    returns: np.ndarray,
    mask: np.ndarray,
    dt: float,
    checkpoint_path: Path | str | None = None,
) -> dict:
    """
    Amortized Heston calibration via pre-trained NPE.

    Process:
    1. Load checkpoint (raises FileNotFoundError if missing)
    2. Compute summary features from returns
    3. Sample posterior via trained MDN (4096 samples, seed 0 for reproducibility)
    4. Compute posterior moments (mean, std) per parameter
    5. Compute log-likelihood diagnostic: MDN log-density at posterior mean

    Args:
        returns: (N, T) log-returns
        mask: (N, T) binary mask for valid observations
        dt: time increment in years (e.g., 1/252 for daily)
        checkpoint_path: path to trained NPE .pkl file (default: DEFAULT_CHECKPOINT)

    Returns:
        dict with keys:
        - kappa, theta, sigma_v, rho, mu, v0: (N,) posterior means
        - kappa_std, theta_std, sigma_v_std, rho_std, mu_std, v0_std: (N,) posterior stds
        - log_likelihood: (N,) MDN log-density at standardized posterior mean (diagnostic)
        - converged: (N,) bool array (True if checkpoint loaded and features finite)
        - n_observations: (N,) number of valid observations per asset

    Raises:
        FileNotFoundError: if checkpoint does not exist (with train-first message)
    """
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT

    checkpoint_path = Path(checkpoint_path)

    # Load checkpoint with clear error message
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"NPE checkpoint not found: {checkpoint_path}\n"
            "Train the model first: scripts/train_npe_heston.py"
        )

    trained_npe = load_npe(str(checkpoint_path))

    # Convert to JAX arrays
    returns_jax = jnp.array(returns, dtype=jnp.float32)
    mask_jax = jnp.array(mask, dtype=jnp.float32)

    # Compute summary features
    features = summary_features(returns_jax, mask_jax)

    # Check for finite features
    features_finite = jnp.all(jnp.isfinite(features), axis=-1)  # (N,) bool

    # Sample posterior with fixed seed for reproducibility
    # (deterministic posterior moments across runs)
    key = jax.random.PRNGKey(0)
    posterior_samples = sample_posterior(
        trained_npe,
        apply_fn_or_none=None,
        s_raw=features,
        key=key,
        n_samples=4096,
    )

    # Compute posterior moments: (N, n_samples, 6) -> (N, 6)
    posterior_mean = jnp.mean(posterior_samples, axis=1)
    posterior_std = jnp.std(posterior_samples, axis=1)

    # Compute log-likelihood diagnostic: MDN log-density at posterior mean
    # (in standardized unconstrained space)
    from .simulate import to_unconstrained

    # Convert posterior mean to unconstrained space
    posterior_mean_unconstrained = to_unconstrained(posterior_mean)

    # Standardize features and parameters
    s_standardized = (features - trained_npe.feature_mean) / trained_npe.feature_std
    z_standardized = (
        (posterior_mean_unconstrained - trained_npe.theta_mean) / trained_npe.theta_std
    )

    # Reconstruct model apply function
    model = ConditionalMDN(
        hidden_dims=trained_npe.config["hidden_dims"],
        n_components=trained_npe.config["n_components"],
        n_outputs=trained_npe.config["n_outputs"],
    )

    # Forward pass to get mixture parameters
    logits, means, log_scales = model.apply(trained_npe.params, s_standardized)
    log_scales = jnp.clip(log_scales, -7.0, 2.0)

    # Compute log mixture weights
    log_mix_weights = jax.nn.log_softmax(logits, axis=-1)  # (N, K)

    # Compute log Gaussian density for each component
    z_expanded = z_standardized[:, None, :]  # (N, 1, 6)
    const = -0.5 * jnp.log(2.0 * jnp.pi)
    log_scales_2 = 2.0 * log_scales
    z_centered = z_expanded - means
    mahalanobis = (z_centered / jnp.exp(log_scales)) ** 2
    log_probs = const * 6 - 0.5 * jnp.sum(log_scales_2 + mahalanobis, axis=-1)  # (N, K)

    # Log-likelihood per observation
    log_likelihood = jax.nn.logsumexp(log_mix_weights + log_probs, axis=-1)  # (N,)

    # Count valid observations per asset
    n_observations = jnp.sum(mask_jax, axis=-1)

    # Converged: checkpoint loaded (always true here) AND features finite
    converged = features_finite

    # Build output dictionary
    param_names = ["kappa", "theta", "sigma_v", "rho", "mu", "v0"]
    result = {}

    for i, name in enumerate(param_names):
        result[name] = np.array(posterior_mean[:, i], dtype=np.float64)
        result[f"{name}_std"] = np.array(posterior_std[:, i], dtype=np.float64)

    result["log_likelihood"] = np.array(log_likelihood, dtype=np.float64)
    result["converged"] = np.array(converged, dtype=bool)
    result["n_observations"] = np.array(n_observations, dtype=np.int64)

    return result
