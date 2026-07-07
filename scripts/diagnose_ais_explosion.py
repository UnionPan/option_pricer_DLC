"""
Diagnose why the AIS auxiliary loss explodes during training.

The total ais_loss is L_r + L_p where:

    L_r = MSE(reward_predictor(x̂, a), r_t)      — reward prediction MSE
    L_p = NLL(transition(x̂, a), x̂_next)         — Gaussian NLL with
                                                    learned log_sigma

L_p has a learnable log_sigma per AIS dim. If log_sigma collapses
(→ -∞), the (x_next - μ)/σ term explodes — NLL → +∞ even when the
mean prediction is decent. This is a classic Gaussian-NLL failure mode.

This script loads a trained checkpoint, inspects log_sigma, and
computes L_r and L_p separately on held-out data.

Run:
    python scripts/diagnose_ais_explosion.py CHECKPOINT_PATH
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from options_desk.deep_hedging.jax.env import (  # noqa: E402
    DeepHedgingEnvConfig, GRID_COARSE,
)
from options_desk.deep_hedging.jax.pricing import (  # noqa: E402
    HestonMarketParams, compile_padded_grid,
)
from options_desk.deep_hedging.jax.rollout import simulate_heston_market_batch  # noqa: E402
from options_desk.deep_hedging.pomarl.ais import (  # noqa: E402
    AISGRUEncoder, AISRewardModel, AISTransitionModel,
)
from options_desk.deep_hedging.pomarl.utils import build_pomdp_obs  # noqa: E402
from options_desk.deep_hedging.pomarl.policy import GaussianPolicy, sample_action  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=str)
    ap.add_argument("--n-paths", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=252)
    ap.add_argument("--eval-seed", type=int, default=12345)
    args = ap.parse_args()

    print(f"Loading: {args.checkpoint}")
    with open(args.checkpoint, "rb") as fp:
        ckpt = pickle.load(fp)
    cfg = ckpt["config"]

    # ── Inspect transition log_sigma ────────────────────────────────────
    trans_params = ckpt["transition_params"]
    # Tree-walk to find log_sigma
    log_sigma = None
    for path, leaf in jax.tree_util.tree_leaves_with_path(trans_params):
        if any("log_sigma" in str(p) for p in path):
            log_sigma = np.asarray(leaf)
            break
    if log_sigma is None:
        # Fallback: assume it's the only 1-D param matching ais_dim
        for leaf in jax.tree_util.tree_leaves(trans_params):
            arr = np.asarray(leaf)
            if arr.ndim == 1 and arr.shape[0] == cfg.ais_hidden_size:
                log_sigma = arr
                break

    print(f"\n=== Transition model log_sigma ({cfg.ais_hidden_size}-dim) ===")
    if log_sigma is not None:
        sigma = np.exp(log_sigma)
        print(f"  log_sigma min={log_sigma.min():.4f}  max={log_sigma.max():.4f}  mean={log_sigma.mean():.4f}")
        print(f"  sigma     min={sigma.min():.4e}     max={sigma.max():.4e}     mean={sigma.mean():.4e}")
        if sigma.min() < 1e-3:
            print(f"  ⚠️  CRITICAL: sigma → 0 in some dims (min={sigma.min():.2e})")
            print(f"     → NLL explodes: 0.5·((x_next - μ)/σ)² blows up for any non-zero residual")
        elif sigma.max() > 100:
            print(f"  ⚠️  sigma is huge (max={sigma.max():.2f}) → model is essentially saying 'no idea'")
        else:
            print(f"  ✓ sigma values look healthy")
    else:
        print("  could not locate log_sigma param")

    # ── Generate held-out data ──────────────────────────────────────────
    env_config = DeepHedgingEnvConfig(
        horizon_steps=args.horizon, option_grid=GRID_COARSE, dt=1.0/252,
        transaction_cost_underlying=1e-4, transaction_cost_option=1e-2,
        risk_aversion=1000.0, scheme="qe",
    )
    market = HestonMarketParams(
        kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, r=0.0, q=0.0,
    )
    padded_grid = compile_padded_grid(GRID_COARSE, dt=env_config.dt, horizon_steps=args.horizon)
    keys = jax.random.split(jax.random.PRNGKey(args.eval_seed), args.n_paths)
    traj = simulate_heston_market_batch(
        config=env_config, market=market, padded_grid=padded_grid,
        initial_spot=100.0, initial_variance=0.04, keys=keys,
    )
    spots = jnp.asarray(traj.spots, dtype=jnp.float32)
    prices = jnp.asarray(traj.instrument_prices, dtype=jnp.float32)
    B = args.n_paths

    # ── Run encoder + policy + measure L_r, L_p ────────────────────────
    encoder = AISGRUEncoder(hidden_size=cfg.ais_hidden_size, n_layers=cfg.ais_n_layers)
    reward_model = AISRewardModel(hidden_size=cfg.ais_hidden_size)
    transition_model = AISTransitionModel(
        ais_dim=cfg.ais_hidden_size, hidden_size=cfg.ais_hidden_size,
    )
    policy = GaussianPolicy(
        n_instruments=GRID_COARSE.n_instruments,
        hidden_size=cfg.policy_hidden_size,
        log_std_min=cfg.log_std_min, log_std_max=cfg.log_std_max,
    )

    initial_hidden = AISGRUEncoder.init_hidden(B, cfg.ais_hidden_size, cfg.ais_n_layers)
    positions = jnp.zeros((B, GRID_COARSE.n_instruments), dtype=jnp.float32)
    prev_trades = jnp.zeros_like(positions)

    x_hats = []
    actions = []
    rewards = []
    hidden = initial_hidden
    cash = jnp.zeros(B, dtype=jnp.float32)
    prev_V = jnp.zeros(B, dtype=jnp.float32)
    rng = jax.random.PRNGKey(42)
    for t in range(args.horizon):
        obs = build_pomdp_obs(
            spot_t=spots[:, t], option_prices_t=prices[:, t, 1:],
            positions=positions, previous_trades=prev_trades,
            time_index=t, horizon=args.horizon,
        )
        x_hat, hidden = encoder.apply(ckpt["encoder_params"], obs, hidden)
        x_hats.append(x_hat)
        # Greedy actions for diagnostic
        mu, log_std = policy.apply(ckpt["policy_params"], x_hat, jnp.ones((B, GRID_COARSE.n_instruments)))
        rng, key = jax.random.split(rng)
        action, _ = sample_action(mu, log_std, jnp.ones_like(mu), key, cfg.position_limit)
        actions.append(action)
        # Compute reward (same as rollout):
        traded_notional = action * prices[:, t]
        cost = (jnp.full_like(action, 1e-4) * jnp.abs(traded_notional)).sum(axis=-1)  # simple cost
        cash = cash - traded_notional.sum(axis=-1) - cost
        positions = positions + action
        V = cash + (positions * prices[:, t+1]).sum(axis=-1)
        r = V - prev_V
        rewards.append(r)
        prev_V = V
        prev_trades = action

    # Final encoder pass for x_hat_T
    obs_T = build_pomdp_obs(
        spot_t=spots[:, args.horizon], option_prices_t=prices[:, args.horizon, 1:],
        positions=positions, previous_trades=prev_trades,
        time_index=args.horizon, horizon=args.horizon,
    )
    x_hat_T, _ = encoder.apply(ckpt["encoder_params"], obs_T, hidden)
    x_hats.append(x_hat_T)

    x_hat_seq = jnp.stack(x_hats, axis=0)  # (T+1, B, H)
    actions_arr = jnp.stack(actions, axis=0)  # (T, B, N)
    rewards_arr = jnp.stack(rewards, axis=0)  # (T, B)

    # L_r: reward MSE
    r_hat = reward_model.apply(ckpt["reward_params"], x_hat_seq[:-1], actions_arr)
    L_r = jnp.mean((r_hat - rewards_arr) ** 2)

    # L_p: transition NLL
    mu_pred, log_sigma_pred = transition_model.apply(
        ckpt["transition_params"], x_hat_seq[:-1], actions_arr,
    )
    sigma_pred = jnp.exp(log_sigma_pred)
    target = jax.lax.stop_gradient(x_hat_seq[1:])
    per_dim_nll = 0.5 * ((target - mu_pred) / sigma_pred) ** 2 + log_sigma_pred
    L_p = jnp.mean(per_dim_nll)

    # Decompose L_p into squared term and log_sigma term
    L_p_squared = jnp.mean(0.5 * ((target - mu_pred) / sigma_pred) ** 2)
    L_p_log_sigma = jnp.mean(log_sigma_pred)

    print("\n=== Held-out L_r and L_p (on encoder applied with current policy) ===")
    print(f"  L_r (reward MSE)                  = {float(L_r):.4e}")
    print(f"  L_p (transition NLL, total)       = {float(L_p):.4e}")
    print(f"     L_p squared-error component    = {float(L_p_squared):.4e}")
    print(f"     L_p log_sigma component        = {float(L_p_log_sigma):.4e}")

    # Predictive error magnitudes
    resid = target - mu_pred
    resid_rms = float(jnp.sqrt(jnp.mean(resid ** 2)))
    target_rms = float(jnp.sqrt(jnp.mean(target ** 2)))
    print(f"\n  Mean |x_next - μ| (transition prediction RMS): {resid_rms:.4e}")
    print(f"  Mean |x_next| (target RMS):                   {target_rms:.4e}")
    print(f"  Predictive R²(x_next) ≈ {1 - (resid_rms/target_rms)**2:.4f}")

    print(f"\n=== Diagnosis ===")
    if log_sigma is not None and sigma.min() < 0.01:
        print(f"  → log_sigma collapsed (σ_min={sigma.min():.2e}) → L_p dominated by")
        print(f"    (residual/σ)² blow-up. The encoder representation is fine (the probe")
        print(f"    showed R²=0.95 for v_t), but the NLL parameterization is unstable.")
        print(f"    Fix: clamp log_sigma to [-3, 3] or remove log_sigma (use unit variance).")
    elif float(L_p_squared) > 1e6:
        print(f"  → L_p squared term is huge ({float(L_p_squared):.2e}) but log_sigma OK.")
        print(f"    Means transition model can't predict x_next from x_hat + action.")
        print(f"    Could be unstable encoder dynamics or insufficient transition capacity.")
    else:
        print(f"  → L_r={float(L_r):.2e}, L_p={float(L_p):.2e}: both healthy.")
        print(f"    The training-time explosion may have been transient (and the final")
        print(f"    checkpoint stabilized to OK values).")


if __name__ == "__main__":
    main()
