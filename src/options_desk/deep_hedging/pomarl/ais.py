"""
Approximate Information State (AIS) modules for POMARL.

Three flax modules implement Algorithm 1's parametric pieces:

    σ_φ : H_t → x̂_t           — :class:`AISGRUEncoder`
    r̂_ψ : (x̂_t, a_t) → R       — :class:`AISRewardModel`
    P̂_ψ(x̂_{t+1} | x̂_t, a_t)    — :class:`AISTransitionModel`

The encoder is a stacked GRU on top of an embedding Dense layer; reward and
transition heads are 2-layer MLPs over ``concat([x̂, a])``. The transition
model parameterises a diagonal Gaussian with a global learnable log-std
(state-independent), which matches the auxiliary self-prediction loss in the
notes (a NLL on x̂_{t+1}).
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


class AISGRUEncoder(nn.Module):
    """Stacked-GRU AIS encoder σ_φ.

    Each forward pass consumes the current observation and the previous
    layer-wise hidden state tuple, and emits ``(x̂_t, new_hidden)``.
    """

    hidden_size: int
    n_layers: int = 1

    @nn.compact
    def __call__(self, obs, hidden):
        x = nn.relu(nn.Dense(self.hidden_size, name="embed")(obs))
        new_hidden = []
        for layer in range(self.n_layers):
            cell = nn.GRUCell(features=self.hidden_size, name=f"gru_{layer}")
            h_next, _ = cell(hidden[layer], x)
            x = h_next
            new_hidden.append(h_next)
        return x, tuple(new_hidden)

    @staticmethod
    def init_hidden(batch_size: int, hidden_size: int, n_layers: int) -> tuple:
        zero = jnp.zeros((batch_size, hidden_size), dtype=jnp.float32)
        return tuple(zero for _ in range(n_layers))


class AISRewardModel(nn.Module):
    """Reward predictor r̂_ψ : (x̂, a) → R."""

    hidden_size: int = 64

    @nn.compact
    def __call__(self, x_hat, action):
        z = jnp.concatenate([x_hat, action], axis=-1)
        z = nn.relu(nn.Dense(self.hidden_size, name="fc1")(z))
        return nn.Dense(1, name="head")(z).squeeze(-1)


class AISTransitionModel(nn.Module):
    """Gaussian transition predictor P̂_ψ(x̂' | x̂, a).

    Outputs the mean of next-AIS via an MLP and a state-independent
    learnable log-std (one parameter per AIS dimension). The NLL loss in
    :func:`losses.ais_losses` uses a diagonal Gaussian likelihood.
    """

    ais_dim: int
    hidden_size: int = 64

    @nn.compact
    def __call__(self, x_hat, action):
        z = jnp.concatenate([x_hat, action], axis=-1)
        z = nn.relu(nn.Dense(self.hidden_size, name="fc1")(z))
        mu = nn.Dense(self.ais_dim, name="mu")(z)
        log_sigma = self.param(
            "log_sigma",
            nn.initializers.zeros,
            (self.ais_dim,),
        )
        # Broadcast log_sigma to the leading batch dimensions of mu.
        return mu, jnp.broadcast_to(log_sigma, mu.shape)
