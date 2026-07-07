"""Unit tests for the POMARL AIS GRU encoder."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from options_desk.deep_hedging.pomarl.ais import AISGRUEncoder


def _init_encoder(hidden_size: int, n_layers: int, obs_dim: int, batch: int):
    enc = AISGRUEncoder(hidden_size=hidden_size, n_layers=n_layers)
    key = jax.random.PRNGKey(0)
    dummy_obs = jnp.zeros((batch, obs_dim), dtype=jnp.float32)
    dummy_h = AISGRUEncoder.init_hidden(batch, hidden_size, n_layers)
    params = enc.init(key, dummy_obs, dummy_h)
    return enc, params, dummy_h


def test_encoder_output_shapes_single_layer():
    H, B, D = 8, 4, 10
    enc, params, h0 = _init_encoder(H, 1, D, B)
    obs = jnp.asarray(np.random.RandomState(0).randn(B, D).astype(np.float32))
    x_hat, h_new = enc.apply(params, obs, h0)
    assert x_hat.shape == (B, H)
    assert len(h_new) == 1
    assert h_new[0].shape == (B, H)


def test_encoder_output_shapes_multi_layer():
    H, B, D, L = 6, 3, 7, 2
    enc, params, h0 = _init_encoder(H, L, D, B)
    obs = jnp.asarray(np.random.RandomState(1).randn(B, D).astype(np.float32))
    x_hat, h_new = enc.apply(params, obs, h0)
    assert x_hat.shape == (B, H)
    assert len(h_new) == L
    for h in h_new:
        assert h.shape == (B, H)


def test_encoder_threads_hidden_state_across_history():
    """Identical obs at t but different history ⇒ different x̂_t."""
    H, B, D = 8, 1, 5
    enc, params, h0 = _init_encoder(H, 1, D, B)

    rng = np.random.RandomState(11)
    obs_a = jnp.asarray(rng.randn(B, D).astype(np.float32))
    obs_b = jnp.asarray(rng.randn(B, D).astype(np.float32))
    obs_common = jnp.asarray(rng.randn(B, D).astype(np.float32))

    # Path 1: history = obs_a then obs_common
    _, h1 = enc.apply(params, obs_a, h0)
    x_hat_1, _ = enc.apply(params, obs_common, h1)

    # Path 2: history = obs_b then obs_common
    _, h2 = enc.apply(params, obs_b, h0)
    x_hat_2, _ = enc.apply(params, obs_common, h2)

    arr1 = np.asarray(x_hat_1)
    arr2 = np.asarray(x_hat_2)
    assert not np.allclose(arr1, arr2, atol=1e-5), (
        "encoder must use hidden state — same obs should give different x̂ "
        "under different histories"
    )


def test_encoder_init_hidden_is_zeros():
    h = AISGRUEncoder.init_hidden(4, 8, 2)
    assert len(h) == 2
    for tensor in h:
        np.testing.assert_array_equal(
            np.asarray(tensor), np.zeros((4, 8), dtype=np.float32),
        )
