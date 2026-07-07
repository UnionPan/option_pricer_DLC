"""Unit tests for the AIS reward and transition predictors.

We do not check exact loss values — just that gradient descent on synthetic
toy data decreases the loss substantially, confirming the modules are wired
correctly and JAX gradient flow works end-to-end.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from options_desk.deep_hedging.pomarl.ais import (
    AISRewardModel,
    AISTransitionModel,
)
from options_desk.deep_hedging.pomarl.losses import (
    ais_reward_loss,
    ais_transition_loss,
)


def test_reward_model_descends_on_linear_target():
    """Fit r = w·x̂ + u·a + b on a small batch."""
    rng = np.random.RandomState(0)
    T, B, H, N = 8, 32, 6, 4
    x_hat = jnp.asarray(rng.randn(T, B, H).astype(np.float32))
    actions = jnp.asarray(rng.randn(T, B, N).astype(np.float32))
    true_w = rng.randn(H).astype(np.float32)
    true_u = rng.randn(N).astype(np.float32)
    rewards = jnp.asarray(
        (np.asarray(x_hat) @ true_w + np.asarray(actions) @ true_u + 0.5)
        .astype(np.float32)
    )

    model = AISRewardModel(hidden_size=32)
    params = model.init(jax.random.PRNGKey(0), x_hat[0], actions[0])
    opt = optax.adam(1e-2)
    opt_state = opt.init(params)

    def loss_fn(p):
        return ais_reward_loss(
            x_hat=x_hat, actions=actions, rewards=rewards,
            reward_model_apply=model.apply, reward_params=p,
        )

    initial_loss = float(loss_fn(params))
    for _ in range(200):
        grad = jax.grad(loss_fn)(params)
        updates, opt_state = opt.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
    final_loss = float(loss_fn(params))
    assert final_loss < initial_loss * 0.1, (
        f"reward loss did not descend: initial={initial_loss:.4f} "
        f"final={final_loss:.4f}"
    )


def test_transition_model_descends_on_gaussian_target():
    """Fit P(x̂' | x̂, a) on a linear-Gaussian synthetic transition."""
    rng = np.random.RandomState(1)
    T, B, H, N = 6, 16, 5, 3
    x_hat = rng.randn(T + 1, B, H).astype(np.float32)
    actions = rng.randn(T, B, N).astype(np.float32)
    A = rng.randn(H, H).astype(np.float32) * 0.5
    Bm = rng.randn(N, H).astype(np.float32) * 0.5
    noise = rng.randn(T, B, H).astype(np.float32) * 0.2
    x_hat[1:] = x_hat[:-1] @ A + actions @ Bm + noise
    x_hat_j = jnp.asarray(x_hat)
    actions_j = jnp.asarray(actions)

    model = AISTransitionModel(ais_dim=H, hidden_size=32)
    params = model.init(jax.random.PRNGKey(0), x_hat_j[0], actions_j[0])
    opt = optax.adam(1e-2)
    opt_state = opt.init(params)

    def loss_fn(p):
        return ais_transition_loss(
            x_hat=x_hat_j, actions=actions_j,
            transition_model_apply=model.apply, transition_params=p,
        )

    initial_loss = float(loss_fn(params))
    for _ in range(200):
        grad = jax.grad(loss_fn)(params)
        updates, opt_state = opt.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
    final_loss = float(loss_fn(params))
    assert final_loss < initial_loss - 0.5, (
        f"transition NLL barely moved: initial={initial_loss:.4f} "
        f"final={final_loss:.4f}"
    )


def test_transition_model_has_state_independent_log_sigma_param():
    """The 'log_sigma' parameter is a single (ais_dim,) vector, not per-input."""
    H, N = 4, 3
    model = AISTransitionModel(ais_dim=H)
    params = model.init(
        jax.random.PRNGKey(0),
        jnp.zeros((1, H), dtype=jnp.float32),
        jnp.zeros((1, N), dtype=jnp.float32),
    )
    log_sigma = params["params"]["log_sigma"]
    assert log_sigma.shape == (H,), (
        f"log_sigma should be (ais_dim,), got {log_sigma.shape}"
    )
