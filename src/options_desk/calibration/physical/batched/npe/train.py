"""
NPE training loop and posterior sampling.

Implements:
- Training with standardization (features and unconstrained θ)
- Save/load functionality
- Posterior sampling in natural parameter space
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import pickle
from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from .model import ConditionalMDN, mdn_nll
from .simulate import to_natural


@dataclass
class TrainedNPE:
    """
    Trained NPE model with standardization statistics.

    Attributes:
        params: Flax model parameters
        feature_mean: (16,) mean of summary features
        feature_std: (16,) std of summary features
        theta_mean: (6,) mean of unconstrained θ
        theta_std: (6,) std of unconstrained θ
        config: dict with model config (hidden_dims, n_components)
    """
    params: dict
    feature_mean: jnp.ndarray
    feature_std: jnp.ndarray
    theta_mean: jnp.ndarray
    theta_std: jnp.ndarray
    config: dict


class TrainState(train_state.TrainState):
    """Extended train state for batched training."""
    pass


def train_mdn(
    key,
    features,
    thetas_unconstrained,
    *,
    epochs=100,
    batch_size=512,
    lr=1e-3,
    val_frac=0.1,
    hidden=(128, 128),
    n_components=8,
):
    """
    Train conditional MDN for NPE.

    Standardizes features and unconstrained θ, trains via Adam,
    returns model with standardization statistics.

    Args:
        key: JAX PRNGKey
        features: (N, 16) summary features
        thetas_unconstrained: (N, 6) unconstrained parameters
        epochs: number of training epochs
        batch_size: batch size for SGD
        lr: learning rate
        val_frac: fraction of data for validation
        hidden: tuple of hidden layer sizes
        n_components: number of mixture components

    Returns:
        TrainedNPE with trained parameters and standardization stats
    """
    N = features.shape[0]
    n_val = int(N * val_frac)
    n_train = N - n_val

    # Shuffle and split
    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, N)
    features = features[perm]
    thetas_unconstrained = thetas_unconstrained[perm]

    features_train = features[:n_train]
    features_val = features[n_train:]
    thetas_train = thetas_unconstrained[:n_train]
    thetas_val = thetas_unconstrained[n_train:]

    # Compute standardization statistics (on training set only)
    feature_mean = jnp.mean(features_train, axis=0)
    feature_std = jnp.std(features_train, axis=0) + 1e-8
    theta_mean = jnp.mean(thetas_train, axis=0)
    theta_std = jnp.std(thetas_train, axis=0) + 1e-8

    # Standardize
    s_train = (features_train - feature_mean) / feature_std
    s_val = (features_val - feature_mean) / feature_std
    z_train = (thetas_train - theta_mean) / theta_std
    z_val = (thetas_val - theta_mean) / theta_std

    # Initialize model
    model = ConditionalMDN(hidden_dims=hidden, n_components=n_components, n_outputs=6)
    key, subkey = jax.random.split(key)
    dummy_input = jnp.ones((1, 16))
    params = model.init(subkey, dummy_input)

    # Setup optimizer and train state
    tx = optax.adam(lr)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Training step
    @jax.jit
    def train_step(state, s_batch, z_batch):
        """Single training step."""
        def loss_fn(params):
            return mdn_nll(params, state.apply_fn, s_batch, z_batch)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    # Training loop
    n_batches = (n_train + batch_size - 1) // batch_size
    for epoch in range(epochs):
        # Shuffle training data
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, n_train)
        s_train_shuffled = s_train[perm]
        z_train_shuffled = z_train[perm]

        # Batch training
        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n_train)
            s_batch = s_train_shuffled[start:end]
            z_batch = z_train_shuffled[start:end]

            state, _ = train_step(state, s_batch, z_batch)

    # Return trained model
    config = {
        "hidden_dims": hidden,
        "n_components": n_components,
        "n_outputs": 6,
    }

    return TrainedNPE(
        params=state.params,
        feature_mean=feature_mean,
        feature_std=feature_std,
        theta_mean=theta_mean,
        theta_std=theta_std,
        config=config,
    )


def save_npe(trained_npe: TrainedNPE, path: str):
    """
    Save trained NPE to disk via pickle.

    Args:
        trained_npe: TrainedNPE instance
        path: path to save pickle file
    """
    with open(path, "wb") as f:
        pickle.dump(trained_npe, f)


def load_npe(path: str) -> TrainedNPE:
    """
    Load trained NPE from disk.

    Args:
        path: path to pickle file

    Returns:
        TrainedNPE instance
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def sample_posterior(
    trained_npe: TrainedNPE,
    apply_fn_or_none: Optional[callable],
    s_raw,
    key,
    n_samples=4096,
):
    """
    Sample from posterior p(θ | s) in natural parameter space.

    Process:
    1. Standardize input features s_raw
    2. Forward pass through MDN to get mixture parameters
    3. Sample component assignments via Categorical(π)
    4. Sample from selected Gaussian components
    5. De-standardize to unconstrained space
    6. Transform to natural parameter space

    Args:
        trained_npe: TrainedNPE instance
        apply_fn_or_none: model.apply function (if None, reconstruct from config)
        s_raw: (N, 16) raw summary features (NOT standardized)
        key: JAX PRNGKey
        n_samples: number of posterior samples per observation

    Returns:
        (N, n_samples, 6) posterior samples in natural space
    """
    N = s_raw.shape[0]

    # Standardize features
    s = (s_raw - trained_npe.feature_mean) / trained_npe.feature_std

    # Reconstruct model if needed
    if apply_fn_or_none is None:
        model = ConditionalMDN(
            hidden_dims=trained_npe.config["hidden_dims"],
            n_components=trained_npe.config["n_components"],
            n_outputs=trained_npe.config["n_outputs"],
        )
        apply_fn = model.apply
    else:
        apply_fn = apply_fn_or_none

    # Forward pass to get mixture parameters
    logits, means, log_scales = apply_fn(trained_npe.params, s)
    K = logits.shape[1]

    # Clip log_scales
    log_scales = jnp.clip(log_scales, -7.0, 2.0)

    # Sample component assignments: (N, n_samples)
    # We need to sample n_samples component indices per observation
    # logits: (N, K) -> we want (N, n_samples) component indices
    key, subkey = jax.random.split(key)
    # Generate (N * n_samples,) samples, then reshape
    logits_repeated = jnp.repeat(logits, n_samples, axis=0)  # (N * n_samples, K)
    component_ids_flat = jax.random.categorical(subkey, logits_repeated)  # (N * n_samples,)
    component_ids = component_ids_flat.reshape(N, n_samples)  # (N, n_samples)

    # Sample from selected components
    # Gather means and scales for selected components
    # means: (N, K, 6), component_ids: (N, n_samples) -> gather (N, n_samples, 6)
    batch_idx = jnp.arange(N)[:, None]  # (N, 1)
    selected_means = means[batch_idx, component_ids]  # (N, n_samples, 6)
    selected_log_scales = log_scales[batch_idx, component_ids]  # (N, n_samples, 6)

    # Sample from Gaussian: z ~ N(μ, σ²)
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, shape=(N, n_samples, 6))
    z_standardized = selected_means + jnp.exp(selected_log_scales) * noise

    # De-standardize to unconstrained space
    z_unconstrained = z_standardized * trained_npe.theta_std + trained_npe.theta_mean

    # Transform to natural space
    # Reshape to (N * n_samples, 6) for to_natural, then back to (N, n_samples, 6)
    z_unconstrained_flat = z_unconstrained.reshape(-1, 6)
    theta_natural_flat = to_natural(z_unconstrained_flat)
    theta_natural = theta_natural_flat.reshape(N, n_samples, 6)

    return theta_natural
