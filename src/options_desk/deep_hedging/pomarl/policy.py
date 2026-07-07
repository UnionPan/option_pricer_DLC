"""
Tanh-squashed diagonal Gaussian policy π_θ(a | x̂).

The policy outputs ``(μ, log_σ)`` from the AIS ``x̂`` (already detached by
the rollout, per Algorithm 1). Actions are sampled in pre-tanh space and
squashed to ``[-L, L]`` (``position_limit``); the Jacobian correction
``log(L · (1 − tanh(z)²))`` is subtracted from the Gaussian log-prob to
keep the density well-defined.

The module is purely the network; sampling / log-prob helpers live as plain
JAX functions so they can be vmap'd / scanned freely.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


class GaussianPolicy(nn.Module):
    """Two-layer MLP mapping AIS → ``(μ, log_σ)``.

    The log-std head is clipped to ``[log_std_min, log_std_max]`` so the
    Gaussian doesn't collapse / explode during early REINFORCE updates.
    """

    n_instruments: int
    hidden_size: int = 64
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, x_hat, mask):  # mask unused here; applied at sampling
        h = nn.relu(nn.Dense(self.hidden_size, name="fc1")(x_hat))
        h = nn.relu(nn.Dense(self.hidden_size, name="fc2")(h))
        mu = nn.Dense(self.n_instruments, name="mu")(h)
        log_std = nn.Dense(self.n_instruments, name="log_std")(h)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std


def sample_action(
    mu: jnp.ndarray,
    log_std: jnp.ndarray,
    mask: jnp.ndarray,
    key: jax.Array,
    position_limit: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample a tanh-squashed Gaussian action and return ``(action, log_prob)``.

    ``mask`` is a float vector ``∈ {0,1}^N`` — masked components are zeroed
    in the returned action AND contribute zero to the log-prob (so the
    REINFORCE gradient doesn't try to move masked dimensions).
    """
    sigma = jnp.exp(log_std)
    # Explicit dtype: under jax_enable_x64 the default float dtype is float64,
    # which would promote (mu, log_std, action) and break dtype-strict scan
    # carries downstream.
    z = mu + sigma * jax.random.normal(key, mu.shape, dtype=mu.dtype)
    tanh_z = jnp.tanh(z)
    pre_action = tanh_z * position_limit

    # Gaussian log-prob per dim
    log_prob_z = -0.5 * (((z - mu) / sigma) ** 2 + 2.0 * log_std
                         + jnp.log(2.0 * jnp.pi))
    # Tanh + scaling Jacobian: action = L * tanh(z) ⇒
    #   log|da/dz| = log(L) + log(1 − tanh(z)²)
    jacobian = jnp.log(position_limit) + jnp.log(1.0 - tanh_z * tanh_z + 1e-6)
    per_dim = log_prob_z - jacobian

    # Mask: zero contributions from masked dims.
    per_dim_masked = per_dim * mask
    log_prob = per_dim_masked.sum(axis=-1)
    action = pre_action * mask
    return action, log_prob


def mean_action(
    mu: jnp.ndarray,
    mask: jnp.ndarray,
    position_limit: float,
) -> jnp.ndarray:
    """Deterministic greedy action used at evaluation time."""
    return jnp.tanh(mu) * position_limit * mask
