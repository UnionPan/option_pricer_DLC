"""
Helpers for the JAX-native POMARL stack.

Provides:
    * ``build_pomdp_obs``        – variance-hidden observation vector
                                    (mirrors :func:`_build_batch_obs_tensor`
                                    from torch_buehler.py but drops the
                                    variance feature).
    * ``jax_leg_payoff``         – JAX port of :func:`_torch_leg_payoff`
                                    for call/put liabilities (extend as
                                    needed; raises for unsupported kinds).
    * ``discounted_returns``     – reverse cumulative sum
                                    ``G_t = Σ_{k>=t} γ^{k-t} r_k``.
    * ``pomdp_obs_dim``          – obs dimension helper (3N + 2).
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..utils.contracts import LiabilitySpec, _normalize_legs


def pomdp_obs_dim(n_instruments: int) -> int:
    """POMARL observation dimensionality: 3N + 2 (variance hidden)."""
    return 3 * n_instruments + 2


def build_pomdp_obs(
    spot_t: jnp.ndarray,            # (B,)
    option_prices_t: jnp.ndarray,   # (B, N - 1)
    positions: jnp.ndarray,         # (B, N)
    previous_trades: jnp.ndarray,   # (B, N)
    time_index: int,
    horizon: int,
    price_scale: float = 100.0,
    position_scale: float = 1.5,
    price_clip: float = 10.0,
) -> jnp.ndarray:
    """Build the per-step variance-hidden observation matrix.

    Layout (concatenated along the feature axis):
        ``[spot_scaled, option_features, positions/L, prev_trades/L,
           time_feat, bias_feat]``

    Compared to the Buehler tensor builder, the variance entry is removed
    so the agent cannot observe the latent vol — that is the whole point of
    POMARL.
    """
    spot_scaled = jnp.clip(
        spot_t[:, None] / price_scale - 1.0, -price_clip, price_clip,
    )
    option_features = jnp.clip(
        option_prices_t / price_scale, -price_clip, price_clip,
    )
    pos_scaled = jnp.clip(positions / position_scale, -price_clip, price_clip)
    prev_scaled = jnp.clip(
        previous_trades / position_scale, -price_clip, price_clip,
    )
    B = spot_scaled.shape[0]
    time_frac = jnp.full((B, 1), time_index / horizon if horizon > 0 else 0.0)
    bias = jnp.ones((B, 1), dtype=spot_scaled.dtype)
    return jnp.concatenate(
        [spot_scaled, option_features, pos_scaled, prev_scaled, time_frac, bias],
        axis=1,
    )


def _jax_single_leg_payoff(leg: LiabilitySpec, spots: jnp.ndarray) -> jnp.ndarray:
    """JAX port of :func:`_torch_leg_payoff` for vanilla call/put.

    The original Buehler tensor variant supports cliquet/barrier/asian/lookback
    too; here we only need vanilla because the POMARL eval matches the
    LSTM/BS-delta validation script (which trains against a short ATM call).
    """
    kind = leg.kind
    q = jnp.asarray(leg.quantity, dtype=spots.dtype)
    K = jnp.asarray(leg.strike, dtype=spots.dtype)
    S_T = spots[..., -1]
    if kind == "call":
        return q * jnp.maximum(S_T - K, 0.0)
    if kind == "put":
        return q * jnp.maximum(K - S_T, 0.0)
    raise ValueError(
        f"jax_leg_payoff currently only supports kind in {{'call','put'}}, "
        f"got {kind!r}. Extend this helper if a path-dependent liability is "
        f"needed."
    )


def jax_total_payoff(
    legs: Sequence[LiabilitySpec], spots: jnp.ndarray
) -> jnp.ndarray:
    """Sum of leg payoffs from the spot path (JAX equivalent of
    :func:`total_payoff_from_path`)."""
    legs = _normalize_legs(legs) if not isinstance(legs, tuple) else legs
    payoff = _jax_single_leg_payoff(legs[0], spots)
    for leg in legs[1:]:
        payoff = payoff + _jax_single_leg_payoff(leg, spots)
    return payoff


def discounted_returns(rewards: jnp.ndarray, discount: float) -> jnp.ndarray:
    """Compute reverse-cumulative discounted returns.

    For ``rewards`` of shape ``(T, B)`` returns ``G`` of the same shape with
    ``G_t = Σ_{k=t}^{T-1} γ^{k-t} r_k``. Uses ``jax.lax.scan`` so the routine
    JIT-compiles cleanly inside a loss function.
    """
    def step(carry, r_t):
        g = r_t + discount * carry
        return g, g

    _, G_rev = jax.lax.scan(step, jnp.zeros_like(rewards[0]), rewards[::-1])
    return G_rev[::-1]
