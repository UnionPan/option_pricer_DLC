"""
Tests for NPE prior, JAX Heston simulator, and summary features.

Tests:
(a) shapes + determinism per key
(b) prior in bounds + transform round-trip rtol 1e-5
(c) features finite on 500 simulated paths AND on a constant-return path
(d) vol-clustering signal: mean acf(r²,1) feature higher for high-sigma_v (1.2) than low (0.15)
(e) simulator moment check: for θ=(3,0.04,0.4,−0.6,0.05,0.04), mean annualized realized variance over 2000 paths within 15% of 0.04
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.options_desk.calibration.physical.batched.npe.simulate import (
    PRIOR_LOW,
    PRIOR_HIGH,
    FEATURE_NAMES,
    sample_prior,
    simulate_heston_paths,
    summary_features,
    to_unconstrained,
    to_natural,
)


class TestPriorAndTransforms:
    """Test prior sampling and parameter transforms."""

    def test_prior_bounds(self):
        """Prior samples should be within specified bounds."""
        key = jax.random.PRNGKey(42)
        samples = sample_prior(key, n=1000)

        assert samples.shape == (1000, 6)
        assert samples.dtype == jnp.float32

        # Check bounds for each parameter
        param_names = ["kappa", "theta", "sigma_v", "rho", "mu", "v0"]
        for i, name in enumerate(param_names):
            assert jnp.all(samples[:, i] >= PRIOR_LOW[name])
            assert jnp.all(samples[:, i] <= PRIOR_HIGH[name])

    def test_transform_roundtrip(self):
        """to_natural(to_unconstrained(θ)) should equal θ within tolerance."""
        key = jax.random.PRNGKey(123)
        thetas = sample_prior(key, n=100)

        # Round trip
        unconstrained = to_unconstrained(thetas)
        reconstructed = to_natural(unconstrained)

        # Check shape preservation
        assert unconstrained.shape == thetas.shape
        assert reconstructed.shape == thetas.shape

        # Check numerical accuracy
        np.testing.assert_allclose(
            reconstructed, thetas, rtol=1e-5,
            err_msg="Round-trip transform failed"
        )


class TestHestonSimulator:
    """Test JAX Heston path simulator."""

    def test_simulator_shapes_and_determinism(self):
        """Simulator should produce correct shapes and be deterministic."""
        key = jax.random.PRNGKey(999)
        n = 10
        T = 252

        # Sample parameters
        thetas = sample_prior(key, n)

        # Simulate paths
        key1 = jax.random.PRNGKey(1001)
        returns1 = simulate_heston_paths(key1, thetas, T=T, dt=1/252)

        # Check shape
        assert returns1.shape == (n, T)
        assert returns1.dtype == jnp.float32

        # Check determinism: same key -> same output
        returns2 = simulate_heston_paths(key1, thetas, T=T, dt=1/252)
        np.testing.assert_array_equal(returns1, returns2)

        # Different key -> different output
        key2 = jax.random.PRNGKey(1002)
        returns3 = simulate_heston_paths(key2, thetas, T=T, dt=1/252)
        assert not jnp.allclose(returns1, returns3)

    def test_simulator_moment_check(self):
        """Mean annualized realized variance should match theta within 15%."""
        # Fixed parameters: θ=(3,0.04,0.4,−0.6,0.05,0.04)
        theta_fixed = jnp.array([[3.0, 0.04, 0.4, -0.6, 0.05, 0.04]], dtype=jnp.float32)

        # Replicate to 2000 paths
        n_paths = 2000
        thetas = jnp.repeat(theta_fixed, n_paths, axis=0)

        # Simulate 5 years of daily data (T=1260)
        key = jax.random.PRNGKey(7777)
        T = 1260
        dt = 1/252
        returns = simulate_heston_paths(key, thetas, T=T, dt=dt)

        # Compute realized variance for each path
        # RV = sum(r^2) / dt (annualized)
        rv_per_path = jnp.sum(returns**2, axis=1) / dt / T

        # Mean across paths
        mean_rv = jnp.mean(rv_per_path)

        # Should be within 15% of theta=0.04
        expected_var = 0.04
        tolerance = 0.15 * expected_var

        assert jnp.abs(mean_rv - expected_var) < tolerance, \
            f"Mean RV {mean_rv:.6f} not within 15% of {expected_var}"


class TestSummaryFeatures:
    """Test summary feature extraction."""

    def test_feature_names_count(self):
        """FEATURE_NAMES should have exactly 16 entries."""
        assert len(FEATURE_NAMES) == 16
        # Ensure all are strings
        assert all(isinstance(name, str) for name in FEATURE_NAMES)

    def test_features_finite_on_simulated_paths(self):
        """Features should be finite on simulated Heston paths."""
        key = jax.random.PRNGKey(555)
        n = 500
        T = 252

        # Sample parameters and simulate
        thetas = sample_prior(key, n)
        key_sim = jax.random.PRNGKey(556)
        returns = simulate_heston_paths(key_sim, thetas, T=T, dt=1/252)

        # Create mask (all valid)
        mask = jnp.ones_like(returns)

        # Compute features
        features = summary_features(returns, mask)

        # Check shape
        assert features.shape == (n, 16)
        assert features.dtype == jnp.float32

        # All features should be finite
        assert jnp.all(jnp.isfinite(features)), \
            "Some features are not finite on simulated paths"

    def test_features_finite_on_constant_path(self):
        """Features should be finite on a constant-return path (guards)."""
        # Create a constant path (all returns = 0.0)
        returns = jnp.zeros((5, 252), dtype=jnp.float32)
        mask = jnp.ones_like(returns)

        features = summary_features(returns, mask)

        assert features.shape == (5, 16)
        # All features should be finite (no NaN, no Inf)
        assert jnp.all(jnp.isfinite(features)), \
            "Features are not finite on constant path (guards failed)"

    def test_vol_clustering_signal(self):
        """Higher sigma_v should yield higher mean acf(r²,1) feature."""
        # Create two parameter sets differing only in sigma_v
        # Low sigma_v: 0.15
        # High sigma_v: 1.2

        # Base parameters: kappa=3, theta=0.04, rho=-0.6, mu=0.05, v0=0.04
        base = jnp.array([3.0, 0.04, 0.0, -0.6, 0.05, 0.04], dtype=jnp.float32)

        n_paths = 100

        # Low sigma_v group
        thetas_low = jnp.tile(base, (n_paths, 1))
        thetas_low = thetas_low.at[:, 2].set(0.15)

        # High sigma_v group
        thetas_high = jnp.tile(base, (n_paths, 1))
        thetas_high = thetas_high.at[:, 2].set(1.2)

        # Simulate
        T = 252
        dt = 1/252
        key_low = jax.random.PRNGKey(8001)
        key_high = jax.random.PRNGKey(8002)

        returns_low = simulate_heston_paths(key_low, thetas_low, T=T, dt=dt)
        returns_high = simulate_heston_paths(key_high, thetas_high, T=T, dt=dt)

        # Compute features
        mask = jnp.ones_like(returns_low)
        features_low = summary_features(returns_low, mask)
        features_high = summary_features(returns_high, mask)

        # Find acf(r²,1) feature (should be in FEATURE_NAMES)
        # Based on spec: "acf(r²) lags 1,5,10,21"
        # We need to identify the index
        # Assuming it's named something like "acf_r2_lag1"
        # Let's find it
        acf_r2_lag1_idx = None
        for i, name in enumerate(FEATURE_NAMES):
            if "acf" in name.lower() and "r2" in name.lower() and "1" in name:
                acf_r2_lag1_idx = i
                break

        assert acf_r2_lag1_idx is not None, \
            f"Could not find acf(r²,1) in FEATURE_NAMES: {FEATURE_NAMES}"

        # Mean of this feature for each group
        mean_acf_low = jnp.mean(features_low[:, acf_r2_lag1_idx])
        mean_acf_high = jnp.mean(features_high[:, acf_r2_lag1_idx])

        # High sigma_v should have higher autocorrelation (vol clustering)
        assert mean_acf_high > mean_acf_low, \
            f"Vol clustering signal not detected: low={mean_acf_low:.4f}, high={mean_acf_high:.4f}"
