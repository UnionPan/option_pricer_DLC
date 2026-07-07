"""
Linear-probe test: does the AIS encoder recover v_t?

Loads a trained POMARL checkpoint, runs the encoder forward over a held-out
batch of Heston paths, and asks how well the (64-dim) x_hat predicts:
    - The true instantaneous variance v_t
    - The true spot S_t
    - The portfolio value V_t (implicit in obs via positions)

via linear regression with R² and MSE.

If R²(x_hat → v_t) > 0.9, the AIS encoder IS learning the volatility signal
that theory says it must (so POMARL's failure isn't representation, it's
downstream — policy / loss / training). If R²(x_hat → v_t) is low,
the encoder is failing at the basic AIS task of producing a sufficient
statistic — which would explain why POMARL underperforms even a vanilla
LSTM hidden state that's trained end-to-end.

Run:
    python scripts/probe_ais_encoder.py CHECKPOINT_PATH
"""

from __future__ import annotations

import argparse
import json
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
from options_desk.deep_hedging.pomarl.ais import AISGRUEncoder  # noqa: E402
from options_desk.deep_hedging.pomarl.utils import build_pomdp_obs  # noqa: E402


def encoder_forward_pass(
    encoder, encoder_params, spots, prices, horizon, n_inst, hidden_size, n_layers,
):
    """Run the encoder over a batch of paths and return x_hat at each step.

    Returns (T+1, B, hidden_size) — the encoder output at each timestep.
    """
    B = spots.shape[0]
    initial_positions = jnp.zeros((B, n_inst), dtype=jnp.float32)
    initial_prev = jnp.zeros((B, n_inst), dtype=jnp.float32)
    initial_hidden = AISGRUEncoder.init_hidden(B, hidden_size, n_layers)

    def step(carry, t):
        positions, prev_trades, hidden = carry
        obs = build_pomdp_obs(
            spot_t=spots[:, t],
            option_prices_t=prices[:, t, 1:],
            positions=positions,
            previous_trades=prev_trades,
            time_index=t,
            horizon=horizon,
        )
        x_hat, new_hidden = encoder.apply(encoder_params, obs, hidden)
        return (positions, prev_trades, new_hidden), x_hat

    # NOTE: we hold positions and prev_trades fixed at 0 — this is a
    # PURE-ENCODER probe (no policy). What we want to measure is the
    # encoder's ability to recover v_t from the obs stream alone, NOT its
    # interaction with the policy's actions. If the encoder were
    # action-dependent in a non-trivial way (it's not; the policy uses
    # stop_gradient'd x_hat), we'd need to roll out the actual policy here.
    time_indices = jnp.arange(horizon + 1, dtype=jnp.int32)
    _, x_hat_seq = jax.lax.scan(step, (initial_positions, initial_prev, initial_hidden), time_indices)
    return x_hat_seq  # (T+1, B, H)


def linear_probe(X: np.ndarray, y: np.ndarray) -> dict:
    """OLS linear regression diagnostics: y ≈ X β + α.

    X: (N, D)   y: (N,)
    Returns R², MSE, baseline-MSE (predict mean).
    """
    # Add intercept
    X_full = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    # Solve OLS via lstsq
    beta, *_ = np.linalg.lstsq(X_full, y, rcond=None)
    pred = X_full @ beta
    resid = y - pred
    mse = float(np.mean(resid ** 2))
    var = float(np.var(y))
    r2 = 1.0 - mse / max(var, 1e-12)
    return {"r2": r2, "mse": mse, "baseline_mse": var, "n_features": X.shape[1]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=str, help="Path to pomarl_*_seed*.pt")
    ap.add_argument("--n-paths", type=int, default=1024)
    ap.add_argument("--horizon", type=int, default=252)
    ap.add_argument("--eval-seed", type=int, default=12345,
                    help="DIFFERENT from training eval seed to avoid memorization")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")
    with ckpt_path.open("rb") as fp:
        payload = pickle.load(fp)

    cfg = payload["config"]
    print(f"  config: ais_hidden_size={cfg.ais_hidden_size} n_layers={cfg.ais_n_layers} "
          f"reward_kind={cfg.reward_kind} policy_method={cfg.policy_method}")
    print(f"  trained on n_epochs={cfg.n_epochs}, batch={cfg.batch_size}")

    # Build env + market (matches training)
    env_config = DeepHedgingEnvConfig(
        horizon_steps=args.horizon, option_grid=GRID_COARSE, dt=1.0 / 252.0,
        transaction_cost_underlying=1e-4, transaction_cost_option=1e-2,
        risk_aversion=1000.0, scheme="qe",
    )
    market = HestonMarketParams(
        kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, r=0.0, q=0.0,
    )
    padded_grid = compile_padded_grid(GRID_COARSE, dt=env_config.dt, horizon_steps=args.horizon)

    print(f"\nGenerating {args.n_paths} paths with seed={args.eval_seed}")
    key = jax.random.PRNGKey(args.eval_seed)
    keys = jax.random.split(key, args.n_paths)
    traj = simulate_heston_market_batch(
        config=env_config, market=market, padded_grid=padded_grid,
        initial_spot=100.0, initial_variance=0.04, keys=keys,
    )
    spots = np.asarray(traj.spots, dtype=np.float64)           # (B, T+1)
    variances = np.asarray(traj.variances, dtype=np.float64)   # (B, T+1)
    prices = np.asarray(traj.instrument_prices, dtype=np.float64)  # (B, T+1, N)

    # Run the encoder forward
    print("Running trained encoder forward over paths...")
    encoder = AISGRUEncoder(
        hidden_size=cfg.ais_hidden_size, n_layers=cfg.ais_n_layers,
    )
    x_hat_seq = encoder_forward_pass(
        encoder=encoder,
        encoder_params=payload["encoder_params"],
        spots=jnp.asarray(spots, dtype=jnp.float32),
        prices=jnp.asarray(prices, dtype=jnp.float32),
        horizon=args.horizon,
        n_inst=GRID_COARSE.n_instruments,
        hidden_size=cfg.ais_hidden_size,
        n_layers=cfg.ais_n_layers,
    )
    x_hat_seq = np.asarray(x_hat_seq, dtype=np.float64)  # (T+1, B, H)

    print(f"\n  x_hat shape: {x_hat_seq.shape}  (T+1, B, H)")
    print(f"  x_hat range: min={x_hat_seq.min():.4f}  max={x_hat_seq.max():.4f}  std={x_hat_seq.std():.4f}")

    # Flatten time × batch for regression
    # Drop t=0 (encoder just sees initial obs, has no history yet)
    X = x_hat_seq[1:].reshape(-1, cfg.ais_hidden_size)  # (T·B, H)

    targets = {
        "v_t (true variance)": variances[:, 1:].T.ravel(),     # match (T·B,)
        "sigma_t = sqrt(v_t)": np.sqrt(np.maximum(variances[:, 1:].T.ravel(), 1e-12)),
        "log(v_t)": np.log(np.maximum(variances[:, 1:].T.ravel(), 1e-12)),
        "S_t (spot)": spots[:, 1:].T.ravel(),
        "log(S_t)": np.log(np.maximum(spots[:, 1:].T.ravel(), 1e-12)),
        "P_atm_call (option price)": prices[:, 1:, 3].T.ravel(),  # one ATM-ish call
    }

    # Reshape: we want X (N, H) and y (N,) where N = T*B
    # x_hat_seq is (T, B, H) → transpose to (B, T, H) → reshape (B*T, H)
    # Actually our reshape above does (T, B, H) → (T*B, H) which matches the
    # T·B ordering of variances[:, 1:].T.ravel() = (T, B).ravel().
    # Both are row-major reshape, so consistent.

    print("\n=== Linear probe: how well does x_hat predict... ===")
    print(f"  {'target':<30s} {'R²':>8s} {'MSE':>12s} {'baseline_var':>14s}")
    print("  " + "-" * 70)
    results = {}
    for name, y in targets.items():
        res = linear_probe(X, y)
        results[name] = res
        print(f"  {name:<30s} {res['r2']:>8.4f} {res['mse']:>12.6f} {res['baseline_mse']:>14.6f}")

    print("\n=== Interpretation ===")
    v_r2 = results["v_t (true variance)"]["r2"]
    sv_r2 = results["sigma_t = sqrt(v_t)"]["r2"]
    lv_r2 = results["log(v_t)"]["r2"]
    spot_r2 = results["S_t (spot)"]["r2"]
    print(f"  best vol R²: max(v_t={v_r2:.3f}, σ_t={sv_r2:.3f}, log v={lv_r2:.3f}) = {max(v_r2, sv_r2, lv_r2):.3f}")
    print(f"  S_t R²:      {spot_r2:.3f}")
    if max(v_r2, sv_r2, lv_r2) > 0.9:
        print("  → encoder DOES recover v_t (R²>0.9). POMARL's failure is downstream.")
    elif max(v_r2, sv_r2, lv_r2) > 0.5:
        print("  → encoder partially recovers v_t. Some loss of info but not catastrophic.")
    else:
        print("  → encoder DOES NOT recover v_t (R²<0.5). The AIS encoder is the bottleneck.")
        print("    Even though theory says it MUST encode v_t (since it has to predict future")
        print("    rewards + transitions), the trained encoder isn't doing so empirically.")

    out_path = ckpt_path.parent / f"probe_results_{ckpt_path.stem}.json"
    with out_path.open("w") as fp:
        json.dump({
            "checkpoint": str(ckpt_path),
            "n_paths": args.n_paths,
            "horizon": args.horizon,
            "eval_seed": args.eval_seed,
            "x_hat_dim": cfg.ais_hidden_size,
            "probes": results,
        }, fp, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
