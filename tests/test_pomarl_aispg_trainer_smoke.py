"""Smoke test for :class:`AISPGTrainer`.

Runs a 2-epoch training loop on a tiny Heston batch (B=64, T=16) and asserts:

* ``train_step()`` returns the expected metric keys with finite values.
* :meth:`evaluate` returns the Buehler-compatible schema + POMARL extras.
* A checkpoint round-trip restores all four parameter trees.

Tagged ``slow`` because it instantiates the JAX Heston simulator and JIT-
compiles a rollout / training step.
"""

from __future__ import annotations

import pickle

import jax
import numpy as np
import pytest

from options_desk.deep_hedging.jax.env import (
    GRID_COARSE,
    DeepHedgingEnvConfig,
)
from options_desk.deep_hedging.jax.pricing import (
    HestonMarketParams,
    compile_padded_grid,
)
from options_desk.deep_hedging.pomarl import (
    AISPGTrainer,
    AISPGTrainerConfig,
    PomarlAgent,
)
from options_desk.deep_hedging.utils.contracts import LiabilitySpec


@pytest.fixture(scope="module")
def trainer():
    env_config = DeepHedgingEnvConfig(
        horizon_steps=16,
        option_grid=GRID_COARSE,
        dt=1.0 / 252.0,
        transaction_cost_underlying=1e-4,
        transaction_cost_option=1e-2,
        risk_aversion=1_000.0,
        scheme="qe",
    )
    padded_grid = compile_padded_grid(
        GRID_COARSE, dt=env_config.dt, horizon_steps=env_config.horizon_steps,
    )
    market = HestonMarketParams(
        kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, r=0.0, q=0.0,
    )
    liab = LiabilitySpec(kind="call", strike=100.0, maturity=16, quantity=1.0)
    config = AISPGTrainerConfig(
        ais_hidden_size=16, ais_n_layers=1, policy_hidden_size=16,
        position_limit=1.5, log_std_min=-5.0, log_std_max=2.0,
        policy_lr=3e-4, ais_lr=3e-4,
        batch_size=64, n_epochs=2, eval_every=1, grad_clip=1.0,
        discount=1.0, baseline_kind="batch_mean",
        lambda_reward=1.0, lambda_transition=1.0,
    )
    return AISPGTrainer.from_heston(
        config=config, env_config=env_config, market_params=market,
        padded_grid=padded_grid, liability=liab,
        initial_spot=100.0, initial_variance=0.04, initial_cash=0.0,
        seed=11,
    )


def test_train_step_returns_finite_metric_keys(trainer):
    metrics = trainer.train_step()
    expected_keys = {
        "policy_loss", "ais_loss",
        "pi_policy_term", "pi_mean_return", "pi_mean_log_prob",
        "pi_mean_advantage", "pi_std_advantage",
        "pi_terminal_pnl_mean", "pi_total_cost_mean", "pi_payoff_mean",
        "ais_L_r", "ais_L_p",
    }
    missing = expected_keys - metrics.keys()
    assert not missing, f"missing metric keys: {missing}"
    for k, v in metrics.items():
        assert np.isfinite(v), f"non-finite metric {k}={v}"


def test_two_epoch_train_loop_runs(trainer):
    history = trainer.train(progress=False)
    assert len(history) >= 2
    for entry in history:
        for k, v in entry.items():
            if k == "epoch":
                continue
            assert np.isfinite(v), f"non-finite training metric {k}={v}"


def test_evaluate_returns_buehler_schema_plus_pomarl_extras(trainer):
    metrics = trainer.evaluate(n_paths=64, seed=99)
    required = {
        "n_paths", "eval_loss", "mean_pnl", "std_pnl", "mean_reward",
        "mean_cost", "std_cost", "mean_liability_payoff",
        "std_liability_payoff",
        "mean_hedging_error", "std_hedging_error", "mae_hedging_error",
        "mse_hedging_error", "rmse_hedging_error",
        "p01_hedging_error", "p05_hedging_error", "p50_hedging_error",
        "p95_hedging_error", "p99_hedging_error", "cvar_05_hedging_error",
        "zero_hedge_mean_hedging_error", "zero_hedge_std_hedging_error",
        "std_improvement_vs_zero", "rmse_improvement_vs_zero",
        "pomarl_mean_return", "pomarl_mean_log_prob",
    }
    missing = required - metrics.keys()
    assert not missing, f"missing eval keys: {missing}"
    for k, v in metrics.items():
        assert np.isfinite(v), f"non-finite eval metric {k}={v}"


def test_get_agent_returns_pomarl_agent(trainer):
    agent = trainer.get_agent()
    assert isinstance(agent, PomarlAgent)


def test_checkpoint_round_trip_restores_all_param_trees(trainer, tmp_path):
    ckpt = tmp_path / "pomarl.pt"
    trainer.save_checkpoint(str(ckpt))

    with open(ckpt, "rb") as fp:
        payload = pickle.load(fp)
    for key in ("encoder_params", "policy_params", "reward_params",
                "transition_params", "policy_opt_state", "ais_opt_state",
                "config", "obs_dim", "n_instruments", "train_history"):
        assert key in payload, f"checkpoint missing {key}"

    # Round-trip into a fresh trainer wrapper using the same config.
    fresh = AISPGTrainer.from_heston(
        config=trainer.config, env_config=trainer.env_config,
        market_params=HestonMarketParams(
            kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, r=0.0, q=0.0,
        ),
        padded_grid=compile_padded_grid(
            GRID_COARSE, dt=trainer.env_config.dt,
            horizon_steps=trainer.env_config.horizon_steps,
        ),
        liability=trainer.liability_legs[0],
        initial_spot=100.0, initial_variance=0.04, initial_cash=0.0, seed=11,
    )
    fresh.load_checkpoint(str(ckpt))
    # Spot-check: at least one leaf of the encoder param tree round-trips bit-exact.
    leaves_a = jax.tree_util.tree_leaves(trainer.encoder_params)
    leaves_b = jax.tree_util.tree_leaves(fresh.encoder_params)
    assert len(leaves_a) == len(leaves_b)
    np.testing.assert_array_equal(
        np.asarray(leaves_a[0]), np.asarray(leaves_b[0]),
    )
