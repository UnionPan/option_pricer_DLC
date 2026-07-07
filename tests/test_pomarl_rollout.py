"""Unit tests for the POMARL JAX rollout.

The non-trivial properties we verify here:

* Output shapes match ``(T, B, …)`` and the post-terminal x̂ slot extends
  the encoder sequence to length ``T+1``.
* The per-step reward telescopes — modulo terminal payoff, costs are
  implicit in cash — so ``Σ_t r_t  =  terminal_PnL − initial_cash − payoff``.
* Masked instruments are liquidated (``action = −positions`` on masked slots)
  so the agent can never carry residual exposure forward.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from options_desk.deep_hedging.pomarl.ais import AISGRUEncoder
from options_desk.deep_hedging.pomarl.policy import GaussianPolicy
from options_desk.deep_hedging.pomarl.rollout import (
    greedy_rollout,
    stochastic_rollout,
)
from options_desk.deep_hedging.pomarl.utils import (
    jax_total_payoff,
    pomdp_obs_dim,
)
from options_desk.deep_hedging.utils.contracts import LiabilitySpec


def _build_modules(*, n_instruments, hidden_size, n_layers, policy_hidden,
                   position_limit):
    enc = AISGRUEncoder(hidden_size=hidden_size, n_layers=n_layers)
    pol = GaussianPolicy(
        n_instruments=n_instruments, hidden_size=policy_hidden,
        log_std_min=-5.0, log_std_max=2.0,
    )
    obs_dim = pomdp_obs_dim(n_instruments)
    key_e, key_p = jax.random.split(jax.random.PRNGKey(0))
    enc_params = enc.init(
        key_e,
        jnp.zeros((1, obs_dim), dtype=jnp.float32),
        AISGRUEncoder.init_hidden(1, hidden_size, n_layers),
    )
    pol_params = pol.init(
        key_p,
        jnp.zeros((1, hidden_size), dtype=jnp.float32),
        jnp.ones((1, n_instruments), dtype=jnp.float32),
    )
    return enc, pol, enc_params, pol_params


def _synthetic_market(B: int, T: int, N: int, seed: int = 0):
    """Geometric-Brownian-style spot path with deterministic option prices."""
    rng = np.random.RandomState(seed)
    incr = rng.randn(B, T).astype(np.float32) * 0.01
    log_S = np.cumsum(np.concatenate([np.zeros((B, 1), dtype=np.float32), incr],
                                     axis=1), axis=1)
    spots = (100.0 * np.exp(log_S)).astype(np.float32)
    # Option prices: a deterministic affine transform of the spot so the
    # rollout body has well-defined trade physics. They are NOT realistic
    # options — this is purely a property test of the rollout code.
    extra = np.linspace(0.1, 1.0, N - 1, dtype=np.float32)
    prices = np.zeros((B, T + 1, N), dtype=np.float32)
    prices[..., 0] = spots
    prices[..., 1:] = spots[..., None] * 0.01 + extra[None, None, :]
    masks = np.ones((T + 1, N), dtype=np.float32)
    return jnp.asarray(spots), jnp.asarray(prices), jnp.asarray(masks)


def test_rollout_output_shapes_and_xhat_length_T_plus_one():
    B, T, N, H = 4, 6, 3, 8
    enc, pol, enc_params, pol_params = _build_modules(
        n_instruments=N, hidden_size=H, n_layers=1,
        policy_hidden=16, position_limit=1.5,
    )
    spots, prices, masks = _synthetic_market(B, T, N)
    liab = (LiabilitySpec(kind="call", strike=100.0, maturity=T, quantity=1.0),)
    out = greedy_rollout(
        encoder_apply=enc.apply, policy_apply=pol.apply,
        encoder_params=enc_params, policy_params=pol_params,
        spots=spots, prices=prices, masks=masks,
        transaction_cost_rates=jnp.zeros((N,), dtype=jnp.float32),
        instrument_mask_static=None, liability_legs=liab,
        initial_cash=0.0, position_limit=1.5,
        horizon=T, n_instruments=N,
        encoder_hidden_size=H, encoder_n_layers=1,
        sample=False, key=jax.random.PRNGKey(0),
    )
    assert out.x_hat_seq.shape == (T + 1, B, H)
    assert out.actions.shape == (T, B, N)
    assert out.log_probs.shape == (T, B)
    assert out.rewards.shape == (T, B)
    assert out.costs.shape == (T, B)
    assert out.terminal_pnl.shape == (B,)
    assert out.total_costs.shape == (B,)
    assert out.payoff.shape == (B,)


def test_reward_telescopes_to_terminal_pnl_minus_payoff_zero_costs():
    """With zero costs and zero initial_cash, ``Σ r_t == terminal_PnL − payoff``."""
    B, T, N, H = 8, 5, 3, 6
    enc, pol, enc_params, pol_params = _build_modules(
        n_instruments=N, hidden_size=H, n_layers=1,
        policy_hidden=16, position_limit=1.5,
    )
    spots, prices, masks = _synthetic_market(B, T, N, seed=2)
    liab = (LiabilitySpec(kind="call", strike=100.0, maturity=T, quantity=1.0),)

    out = greedy_rollout(
        encoder_apply=enc.apply, policy_apply=pol.apply,
        encoder_params=enc_params, policy_params=pol_params,
        spots=spots, prices=prices, masks=masks,
        transaction_cost_rates=jnp.zeros((N,), dtype=jnp.float32),
        instrument_mask_static=None, liability_legs=liab,
        initial_cash=0.0, position_limit=1.5,
        horizon=T, n_instruments=N,
        encoder_hidden_size=H, encoder_n_layers=1,
        sample=False, key=jax.random.PRNGKey(1),
    )
    sum_r = np.asarray(out.rewards.sum(axis=0))
    pnl = np.asarray(out.terminal_pnl)
    payoff = np.asarray(out.payoff)
    np.testing.assert_allclose(sum_r, pnl - payoff, atol=1e-3)


def test_reward_telescopes_with_costs_and_initial_cash():
    """With costs > 0, the identity is ``Σ r_t == terminal_PnL − initial_cash − payoff``."""
    B, T, N, H = 6, 5, 3, 6
    enc, pol, enc_params, pol_params = _build_modules(
        n_instruments=N, hidden_size=H, n_layers=1,
        policy_hidden=16, position_limit=1.5,
    )
    spots, prices, masks = _synthetic_market(B, T, N, seed=3)
    liab = (LiabilitySpec(kind="call", strike=100.0, maturity=T, quantity=1.0),)
    tc = jnp.asarray([1e-3] + [1e-2] * (N - 1), dtype=jnp.float32)
    initial_cash = 7.5

    out = stochastic_rollout(
        encoder_apply=enc.apply, policy_apply=pol.apply,
        encoder_params=enc_params, policy_params=pol_params,
        spots=spots, prices=prices, masks=masks,
        transaction_cost_rates=tc, instrument_mask_static=None,
        liability_legs=liab, initial_cash=initial_cash,
        position_limit=1.5, horizon=T, n_instruments=N,
        encoder_hidden_size=H, encoder_n_layers=1,
        sample=True, key=jax.random.PRNGKey(2),
    )
    sum_r = np.asarray(out.rewards.sum(axis=0))
    pnl = np.asarray(out.terminal_pnl)
    payoff = np.asarray(out.payoff)
    np.testing.assert_allclose(sum_r, pnl - initial_cash - payoff, atol=1e-3)


def test_masked_instruments_are_liquidated():
    """Force a mask of 0 on slot 2 throughout — the running position should stay 0."""
    B, T, N, H = 4, 6, 4, 6
    enc, pol, enc_params, pol_params = _build_modules(
        n_instruments=N, hidden_size=H, n_layers=1,
        policy_hidden=16, position_limit=1.5,
    )
    spots, prices, _ = _synthetic_market(B, T, N, seed=4)
    masks = np.ones((T + 1, N), dtype=np.float32)
    masks[:, 2] = 0.0
    masks_j = jnp.asarray(masks)
    liab = (LiabilitySpec(kind="call", strike=100.0, maturity=T, quantity=1.0),)

    out = stochastic_rollout(
        encoder_apply=enc.apply, policy_apply=pol.apply,
        encoder_params=enc_params, policy_params=pol_params,
        spots=spots, prices=prices, masks=masks_j,
        transaction_cost_rates=jnp.zeros((N,), dtype=jnp.float32),
        instrument_mask_static=None, liability_legs=liab,
        initial_cash=0.0, position_limit=1.5,
        horizon=T, n_instruments=N,
        encoder_hidden_size=H, encoder_n_layers=1,
        sample=True, key=jax.random.PRNGKey(7),
    )
    # Cumulative position on slot 2 should remain zero each step.
    cum_pos = np.cumsum(np.asarray(out.actions[:, :, 2]), axis=0)
    np.testing.assert_allclose(cum_pos, 0.0, atol=1e-5)


def test_payoff_matches_jax_total_payoff_helper():
    B, T, N, H = 5, 4, 3, 4
    enc, pol, enc_params, pol_params = _build_modules(
        n_instruments=N, hidden_size=H, n_layers=1,
        policy_hidden=16, position_limit=1.5,
    )
    spots, prices, masks = _synthetic_market(B, T, N, seed=5)
    liab = (LiabilitySpec(kind="call", strike=99.0, maturity=T, quantity=2.0),)
    out = greedy_rollout(
        encoder_apply=enc.apply, policy_apply=pol.apply,
        encoder_params=enc_params, policy_params=pol_params,
        spots=spots, prices=prices, masks=masks,
        transaction_cost_rates=jnp.zeros((N,), dtype=jnp.float32),
        instrument_mask_static=None, liability_legs=liab,
        initial_cash=0.0, position_limit=1.5,
        horizon=T, n_instruments=N,
        encoder_hidden_size=H, encoder_n_layers=1,
        sample=False, key=jax.random.PRNGKey(0),
    )
    expected = np.asarray(jax_total_payoff(liab, spots))
    np.testing.assert_allclose(np.asarray(out.payoff), expected, rtol=1e-6)
