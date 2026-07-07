"""Unit tests for the POMARL tanh-Gaussian policy."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from options_desk.deep_hedging.pomarl.policy import (
    GaussianPolicy,
    mean_action,
    sample_action,
)


def test_mean_action_is_tanh_mu_scaled_and_masked():
    mu = jnp.array([[0.5, -1.2, 0.0, 2.0]])
    mask = jnp.array([[1.0, 1.0, 0.0, 1.0]])
    L = 1.5

    expected = np.tanh(np.asarray(mu)) * L * np.asarray(mask)
    got = np.asarray(mean_action(mu, mask, L))
    np.testing.assert_allclose(got, expected, rtol=1e-6)
    assert got[0, 2] == 0.0  # masked dim is exactly zero


def test_mean_action_deterministic_under_repeated_call():
    mu = jnp.asarray(np.random.RandomState(0).randn(3, 5).astype(np.float32))
    mask = jnp.ones_like(mu)
    a1 = np.asarray(mean_action(mu, mask, 2.0))
    a2 = np.asarray(mean_action(mu, mask, 2.0))
    np.testing.assert_array_equal(a1, a2)


def test_sample_action_log_prob_matches_manual_formula():
    """log_prob should equal Σ_unmasked [log N(z;μ,σ) − log(L(1−tanh²z))]."""
    rng = np.random.RandomState(7)
    B, N = 4, 6
    mu = jnp.asarray(rng.randn(B, N).astype(np.float32))
    log_std = jnp.asarray((rng.randn(B, N) * 0.3 - 1.0).astype(np.float32))
    mask = jnp.asarray(
        np.array([[1, 1, 0, 1, 1, 0]] * B, dtype=np.float32),
    )
    L = 1.5
    key = jax.random.PRNGKey(123)
    action, log_prob = sample_action(mu, log_std, mask, key, L)

    sigma = np.exp(np.asarray(log_std))
    z = (
        np.asarray(mu)
        + sigma * np.asarray(jax.random.normal(key, mu.shape, dtype=mu.dtype))
    )
    tanh_z = np.tanh(z)
    log_prob_z = -0.5 * (
        ((z - np.asarray(mu)) / sigma) ** 2
        + 2.0 * np.asarray(log_std)
        + np.log(2.0 * np.pi)
    )
    jacobian = np.log(L) + np.log(1.0 - tanh_z * tanh_z + 1e-6)
    per_dim = log_prob_z - jacobian
    expected = (per_dim * np.asarray(mask)).sum(axis=-1)
    np.testing.assert_allclose(np.asarray(log_prob), expected, rtol=1e-4)
    # Action is masked
    np.testing.assert_array_equal(
        np.asarray(action)[:, 2], np.zeros(B, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(action)[:, 5], np.zeros(B, dtype=np.float32),
    )


def test_sample_action_masked_dims_zero_log_prob_contribution():
    """All-zero mask ⇒ log_prob == 0 regardless of (μ, σ)."""
    rng = np.random.RandomState(13)
    B, N = 3, 4
    mu = jnp.asarray(rng.randn(B, N).astype(np.float32))
    log_std = jnp.asarray((rng.randn(B, N) * 0.2).astype(np.float32))
    mask = jnp.zeros((B, N), dtype=jnp.float32)
    key = jax.random.PRNGKey(0)
    action, log_prob = sample_action(mu, log_std, mask, key, 1.5)
    np.testing.assert_allclose(np.asarray(log_prob), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(action), 0.0, atol=1e-6)


def test_gaussian_policy_forward_shapes_and_log_std_clip():
    B, H, N = 2, 8, 5
    pol = GaussianPolicy(
        n_instruments=N, hidden_size=16, log_std_min=-3.0, log_std_max=1.0,
    )
    key = jax.random.PRNGKey(0)
    x = jnp.zeros((B, H), dtype=jnp.float32)
    mask = jnp.ones((B, N), dtype=jnp.float32)
    params = pol.init(key, x, mask)
    mu, log_std = pol.apply(params, x, mask)
    assert mu.shape == (B, N)
    assert log_std.shape == (B, N)
    # Force log_std out of range by inputs of arbitrary scale — clip should bound it.
    x_big = jnp.ones((B, H), dtype=jnp.float32) * 100.0
    _, log_std_big = pol.apply(params, x_big, mask)
    arr = np.asarray(log_std_big)
    assert arr.min() >= -3.0 - 1e-6
    assert arr.max() <= 1.0 + 1e-6
