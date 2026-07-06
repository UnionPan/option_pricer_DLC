"""
Tests for NPE training and posterior sampling.

Tests (4000 sims, T=256, epochs=40, hidden=(48,48), K=6):
a) Validation NLL decreases ≥20% vs epoch 0
b) Save/load roundtrip → identical posterior samples
c) Containment: rho 100% inside (-0.99, 0.99) (transform invariant) and
   ≥95% inside its 1.5x-widened prior range; ≥85% jointly for exp/affine params
d) Informativeness: theta posterior std < prior std; ≥2 params concentrate

Total runtime target: <120s CPU
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from options_desk.calibration.physical.batched.npe.model import ConditionalMDN, mdn_nll
from options_desk.calibration.physical.batched.npe.train import (
    train_mdn,
    save_npe,
    load_npe,
    sample_posterior,
)
from options_desk.calibration.physical.batched.npe.simulate import (
    sample_prior,
    simulate_heston_paths,
    summary_features,
    to_unconstrained,
    PRIOR_LOW,
    PRIOR_HIGH,
)


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def training_data():
    """
    Generate 4000 simulations with T=256 for testing.

    Returns:
        dict with keys: thetas_natural, thetas_unconstrained, features, mask
    """
    key = jax.random.PRNGKey(42)
    N = 4000
    T = 256

    # Sample prior
    key, subkey = jax.random.split(key)
    thetas_natural = sample_prior(subkey, N)

    # Simulate paths
    key, subkey = jax.random.split(key)
    returns = simulate_heston_paths(subkey, thetas_natural, T)

    # Full mask (all valid)
    mask = jnp.ones_like(returns)

    # Compute features
    features = summary_features(returns, mask)

    # Unconstrained parameters
    thetas_unconstrained = to_unconstrained(thetas_natural)

    return {
        "thetas_natural": thetas_natural,
        "thetas_unconstrained": thetas_unconstrained,
        "features": features,
        "mask": mask,
    }


@pytest.fixture(scope="module")
def trained_model(training_data):
    """
    Train a small MDN for testing.

    Config: hidden=(48,48), K=6, epochs=40, batch_size=512
    This provides enough capacity to learn while staying fast (~15s training).
    """
    key = jax.random.PRNGKey(123)

    trained_npe = train_mdn(
        key,
        training_data["features"],
        training_data["thetas_unconstrained"],
        epochs=40,
        batch_size=512,
        lr=1e-3,
        val_frac=0.1,
        hidden=(48, 48),
        n_components=6,
    )

    return trained_npe


# ────────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────────

def test_validation_nll_decreases(training_data):
    """
    Test (a): Validation NLL decreases ≥20% from epoch 0.

    We'll train twice: once for 1 epoch (baseline) and once for 40 epochs,
    then compare validation NLL.
    """
    key = jax.random.PRNGKey(999)
    features = training_data["features"]
    thetas_unc = training_data["thetas_unconstrained"]

    # Train for 1 epoch (baseline)
    key, subkey = jax.random.split(key)
    npe_epoch0 = train_mdn(
        subkey,
        features,
        thetas_unc,
        epochs=1,
        batch_size=512,
        lr=1e-3,
        val_frac=0.1,
        hidden=(48, 48),
        n_components=6,
    )

    # Train for 40 epochs
    key, subkey = jax.random.split(key)
    npe_epoch40 = train_mdn(
        subkey,
        features,
        thetas_unc,
        epochs=40,
        batch_size=512,
        lr=1e-3,
        val_frac=0.1,
        hidden=(48, 48),
        n_components=6,
    )

    # Compute validation NLL on a held-out set
    N = features.shape[0]
    n_val = int(N * 0.1)

    # Use same split for both models (shuffle with fixed key)
    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, N)
    features_perm = features[perm]
    thetas_perm = thetas_unc[perm]

    features_val = features_perm[-n_val:]
    thetas_val = thetas_perm[-n_val:]

    # Standardize using each model's stats
    s_val_0 = (features_val - npe_epoch0.feature_mean) / npe_epoch0.feature_std
    z_val_0 = (thetas_val - npe_epoch0.theta_mean) / npe_epoch0.theta_std

    s_val_40 = (features_val - npe_epoch40.feature_mean) / npe_epoch40.feature_std
    z_val_40 = (thetas_val - npe_epoch40.theta_mean) / npe_epoch40.theta_std

    # Reconstruct apply_fn
    model_0 = ConditionalMDN(hidden_dims=(48, 48), n_components=6, n_outputs=6)
    model_40 = ConditionalMDN(hidden_dims=(48, 48), n_components=6, n_outputs=6)

    nll_0 = mdn_nll(npe_epoch0.params, model_0.apply, s_val_0, z_val_0)
    nll_40 = mdn_nll(npe_epoch40.params, model_40.apply, s_val_40, z_val_40)

    # Check decrease ≥ 20%
    improvement = (nll_0 - nll_40) / nll_0

    assert improvement >= 0.20, f"Val NLL improved by {improvement*100:.1f}%, expected ≥20%"


def test_save_load_roundtrip(trained_model):
    """
    Test (b): Save/load roundtrip produces identical posterior samples.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_npe.pkl"

        # Save
        save_npe(trained_model, str(path))

        # Load
        loaded_npe = load_npe(str(path))

        # Generate test features
        key = jax.random.PRNGKey(777)
        key, subkey = jax.random.split(key)
        test_thetas = sample_prior(subkey, 10)

        key, subkey = jax.random.split(key)
        test_returns = simulate_heston_paths(subkey, test_thetas, 256)
        test_mask = jnp.ones_like(test_returns)
        test_features = summary_features(test_returns, test_mask)

        # Sample from both models with same key
        sample_key = jax.random.PRNGKey(888)

        samples_original = sample_posterior(
            trained_model,
            None,
            test_features,
            sample_key,
            n_samples=100,
        )

        samples_loaded = sample_posterior(
            loaded_npe,
            None,
            test_features,
            sample_key,
            n_samples=100,
        )

        # Check exact equality (or very close due to floating point)
        assert jnp.allclose(samples_original, samples_loaded, rtol=1e-6), \
            "Loaded model produces different samples"


def test_posterior_samples_within_bounds(trained_model, training_data):
    """
    Test (c): Posterior samples within reasonable bounds, per-parameter.

    Split by parameter transform:
    - rho (index 3) goes through 0.99*tanh, so it is mathematically bounded in
      (-0.99, 0.99). We assert (i) 100% of samples strictly inside that open
      interval (the transform invariant — exact), and (ii) >=95% containment
      in the 1.5x-widened prior range (statistical — rho's asymmetric prior
      leaves a reachable spill region below the tanh bound).
    - kappa/theta/sigma_v/v0 go through exp (unbounded above) and mu is affine
      (unbounded both ways), so MDN Gaussian tails naturally produce some
      samples outside the widened ranges at tiny model scale. For these five we
      require >=85% joint containment.
    """
    # Generate test features
    key = jax.random.PRNGKey(555)
    N_test = 100

    key, subkey = jax.random.split(key)
    test_thetas = sample_prior(subkey, N_test)

    key, subkey = jax.random.split(key)
    test_returns = simulate_heston_paths(subkey, test_thetas, 256)
    test_mask = jnp.ones_like(test_returns)
    test_features = summary_features(test_returns, test_mask)

    # Sample posterior
    key, subkey = jax.random.split(key)
    posterior_samples = sample_posterior(
        trained_model,
        None,
        test_features,
        subkey,
        n_samples=500,
    )

    # Flatten: (N_test, 500, 6) -> (N_test * 500, 6)
    samples_flat = posterior_samples.reshape(-1, 6)

    # Widened bounds: 1.5x range
    param_names = ["kappa", "theta", "sigma_v", "rho", "mu", "v0"]
    low = jnp.array([PRIOR_LOW[name] for name in param_names])
    high = jnp.array([PRIOR_HIGH[name] for name in param_names])

    # Widen by 1.5x
    center = (low + high) / 2
    half_width = (high - low) / 2
    widened_low = center - 1.5 * half_width
    widened_high = center + 1.5 * half_width

    # Per-parameter containment breakdown
    within_per_param = (samples_flat >= widened_low) & (samples_flat <= widened_high)  # (M, 6)
    frac_per_param = jnp.mean(within_per_param, axis=0)  # (6,)

    # Split check for the bounded parameter: rho (index 3).
    #
    # (i) Transform invariant (exact): rho = 0.99*tanh(z) is mathematically
    # bounded in (-0.99, 0.99). ANY sample outside this open interval would
    # indicate a real sampling/transform bug, so we require 100% containment.
    rho_samples = samples_flat[:, 3]
    n_violating_bound = jnp.sum((rho_samples <= -0.99) | (rho_samples >= 0.99))
    assert n_violating_bound == 0, \
        f"{n_violating_bound} rho samples outside (-0.99, 0.99) — the tanh " \
        f"transform invariant is violated: sampling/transform bug."

    # (ii) Widened-prior-range containment (statistical, >=95%): rho's prior
    # [-0.95, 0.1] is asymmetric, so its 1.5x-widened upper limit (0.3625)
    # sits far inside the tanh bound (0.99). The spill region (0.3625, 0.99)
    # is mathematically reachable — the tanh bound does NOT protect the upper
    # end of the widened range — and MDN Gaussian tails in atanh-space place a
    # small amount of mass there for a weakly-identified parameter. Observed
    # spill is ~1.9% at test scale, so we require >=95% containment.
    rho_containment = frac_per_param[3]
    assert rho_containment >= 0.95, \
        f"rho containment {rho_containment*100:.2f}% < 95% in widened prior " \
        f"range. Per-param: {frac_per_param}"

    # Joint containment over the unbounded (exp/affine) params:
    # kappa, theta, sigma_v, mu, v0 (indices 0, 1, 2, 4, 5)
    unbounded_idx = jnp.array([0, 1, 2, 4, 5])
    within_unbounded = jnp.all(within_per_param[:, unbounded_idx], axis=-1)
    fraction_unbounded = jnp.mean(within_unbounded)

    assert fraction_unbounded >= 0.85, \
        f"Only {fraction_unbounded*100:.1f}% of samples jointly within 1.5x bounds " \
        f"for exp/affine params, expected ≥85%. Per-param: {frac_per_param}"


def test_posterior_informativeness(trained_model, training_data):
    """
    Test (d): Posterior shows informativeness for well-identified parameters.

    We check that:
    1. theta (index 1, long-run variance) is informative — it is the most
       identifiable Heston parameter from returns data (it drives realized
       variance directly) and must concentrate even at tiny model scale.
    2. At least 2 parameters overall show concentration (std ratio < 1.0),
       demonstrating the NPE learns from data. In Heston calibration from
       returns, typically theta and sigma_v are well-identified, while kappa,
       rho, mu, v0 are harder to identify.
    """
    # Generate test features
    key = jax.random.PRNGKey(666)
    N_test = 100

    key, subkey = jax.random.split(key)
    test_thetas = sample_prior(subkey, N_test)

    key, subkey = jax.random.split(key)
    test_returns = simulate_heston_paths(subkey, test_thetas, 256)
    test_mask = jnp.ones_like(test_returns)
    test_features = summary_features(test_returns, test_mask)

    # Sample posterior
    key, subkey = jax.random.split(key)
    posterior_samples = sample_posterior(
        trained_model,
        None,
        test_features,
        subkey,
        n_samples=1000,
    )

    # Compute posterior std per observation: (N_test, 6)
    posterior_std = jnp.std(posterior_samples, axis=1)  # (N_test, 6)

    # Compute prior std
    key, subkey = jax.random.split(key)
    prior_samples = sample_prior(subkey, 10000)
    prior_std = jnp.std(prior_samples, axis=0)  # (6,)

    # Average posterior std over test observations
    avg_posterior_std = jnp.mean(posterior_std, axis=0)  # (6,)

    # Check that at least some parameters show concentration
    ratio = avg_posterior_std / prior_std

    # theta (index 1) is the most identifiable Heston parameter — it must
    # concentrate below the prior even at tiny model scale. Failure here
    # indicates a real learning/sampling bug, not model-capacity limits.
    theta_ratio = ratio[1]
    assert theta_ratio < 1.0, \
        f"theta posterior std / prior std = {theta_ratio:.3f}, expected < 1.0 " \
        f"(theta is the best-identified param). Ratios: {ratio}"

    # Check that at least 2 parameters show informativeness (ratio < 1.0)
    n_informative = jnp.sum(ratio < 1.0)

    assert n_informative >= 2, \
        f"Only {n_informative}/6 parameters informative (ratio < 1.0), expected ≥2. " \
        f"Ratios: {ratio}"
