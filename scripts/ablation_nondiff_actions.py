"""
Non-differentiable-action ablation: where RL beats supervised pathwise.

Demonstrates the regime in which model-free RL has a genuine structural
advantage over differentiable supervised hedging. We add a NON-
DIFFERENTIABLE operation — rounding trades to discrete lot sizes — to
the rollout. Then we train three configurations on the SAME simulator
with the same liability/grid:

    LSTM            (Buehler / pathwise BPTT)        — expected to FAIL
    POMARL pathwise (SVG / reparameterized gradient) — expected to FAIL
    POMARL REINFORCE (score-function estimator)      — expected to WORK

LSTM and POMARL-pathwise both compute the policy gradient by backprop
through actions → environment. The rounding op has zero gradient
everywhere it's defined; both estimators see zero learning signal and
the policy stays near initialization.

POMARL-REINFORCE estimates the gradient via ∇log π(a) · R, where a is
the CONTINUOUS pre-rounding action. The discretization affects the
return (which the score-function estimator multiplies onto the log-prob)
but does NOT block the gradient path. The policy learns despite the
non-differentiable env op.

Lot size set to 0.1 (10 buckets per unit of position), which is coarse
enough to make the gradient-killing effect dominate but fine enough
that the optimal continuous policy can be approximately represented.

Run:
    python scripts/ablation_nondiff_actions.py                # 200 epochs each
    python scripts/ablation_nondiff_actions.py --smoke        # 10 epochs each (~30 sec)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

import torch  # noqa: E402

from options_desk.deep_hedging.agents import build_option_grid_spec  # noqa: E402
from options_desk.deep_hedging.jax.env import (  # noqa: E402
    DeepHedgingEnvConfig,
    GRID_COARSE,
    build_transaction_cost_vector,
)
from options_desk.deep_hedging.jax.pricing import (  # noqa: E402
    HestonMarketParams,
    compile_padded_grid,
)
from options_desk.deep_hedging.jax.rollout import simulate_heston_market_batch  # noqa: E402
from options_desk.deep_hedging.pomarl import (  # noqa: E402
    AISPGTrainer,
    AISPGTrainerConfig,
)
from options_desk.deep_hedging.training.torch_buehler import (  # noqa: E402
    BuehlerTrainer,
    TrainerConfig,
)
from options_desk.deep_hedging.utils.contracts import (  # noqa: E402
    LiabilitySpec,
    TrajectoryBatch,
)
from options_desk.deep_hedging.utils.eval_baseline import (  # noqa: E402
    evaluate_analytical_hedger_batch,
)


def heston_easy() -> HestonMarketParams:
    return HestonMarketParams(
        kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, r=0.0, q=0.0,
    )


def make_sampler(env_config, market, padded_grid, S0, v0):
    horizon = env_config.horizon_steps
    n_inst = env_config.option_grid.n_instruments

    def sampler(B, key):
        keys = jax.random.split(key, B)
        traj = simulate_heston_market_batch(
            config=env_config, market=market, padded_grid=padded_grid,
            initial_spot=S0, initial_variance=v0, keys=keys,
        )
        return TrajectoryBatch(
            spots=np.asarray(traj.spots, dtype=np.float32),
            variances=np.asarray(traj.variances, dtype=np.float32),
            instrument_prices=np.asarray(traj.instrument_prices, dtype=np.float32),
            action_masks=np.broadcast_to(
                np.asarray(traj.action_masks, dtype=bool),
                (B, horizon + 1, n_inst),
            ).copy(),
        )
    return sampler


def configure_logging(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ablation_nondiff")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout); sh.setLevel(logging.INFO); sh.setFormatter(fmt)
    fh = logging.FileHandler(run_dir / "run.log", mode="w"); fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    for noisy in ("jax", "absl", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logger


def train_lstm(
    env_config, market, padded_grid, liability, S0, v0,
    n_epochs, batch_size, lr, discrete_bucket_size, seed, device, logger,
) -> Dict[str, float]:
    cfg = TrainerConfig(
        position_limit=1.0, risk_aversion=env_config.risk_aversion,
        learning_rate=lr, batch_size=batch_size, n_epochs=n_epochs,
        eval_every=max(1, n_epochs // 10), grad_clip=1.0, policy_kind="lstm",
        lstm_hidden_size=32, lstm_n_blocks=4, lstm_position_limit=1.5,
        lstm_last_layer_scale=1e-3, optimizer_kind="adam",
        discrete_bucket_size=discrete_bucket_size,
    )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    trainer = BuehlerTrainer.from_heston(
        config=cfg, env_config=env_config, market_params=market,
        padded_grid=padded_grid, liability=liability,
        initial_spot=S0, initial_variance=v0, device=device, seed=seed,
    )
    logger.info("[LSTM] training %d epochs, bucket=%.2f", n_epochs, discrete_bucket_size)
    for epoch in range(1, n_epochs + 1):
        m = trainer.train_step()
        if epoch == 1 or epoch % max(1, n_epochs // 5) == 0 or epoch == n_epochs:
            logger.info(
                "  [LSTM] epoch %d/%d  loss=%.2f  err_std=%.3f",
                epoch, n_epochs, m["total_loss"], m["std_hedging_error"],
            )
    ev = trainer.evaluate(n_paths=4096, seed=9999)
    return {
        "agent": "lstm", "std_err": float(ev["std_hedging_error"]),
        "cvar05": float(ev["cvar_05_hedging_error"]),
        "mean_pnl": float(ev["mean_pnl"]),
        "mean_cost": float(ev["mean_cost"]),
        "eval_loss": float(ev["eval_loss"]),
    }


def train_pomarl(
    env_config, market, padded_grid, liability, S0, v0,
    n_epochs, batch_size, lr, discrete_bucket_size, policy_method,
    seed, logger,
) -> Dict[str, float]:
    cfg = AISPGTrainerConfig(
        ais_hidden_size=64, ais_n_layers=1, policy_hidden_size=64,
        position_limit=1.5, log_std_min=-5.0, log_std_max=2.0,
        policy_lr=lr, ais_lr=lr,
        batch_size=batch_size, n_epochs=n_epochs,
        eval_every=max(1, n_epochs // 10), grad_clip=1.0,
        discount=1.0, baseline_kind="batch_mean",
        lambda_reward=1.0, lambda_transition=1.0, instrument_mask=None,
        risk_aversion=env_config.risk_aversion,
        # mean_pnl reward keeps the story focused on the GRADIENT mechanism
        # (REINFORCE-via-log-prob still works under non-diff env;
        #  pathwise / LSTM gradients are zero through round).
        reward_kind="mean_pnl",
        policy_method=policy_method,
        discrete_bucket_size=discrete_bucket_size,
        entropy_coef=0.01 if policy_method == "pathwise" else 0.0,
    )
    trainer = AISPGTrainer.from_heston(
        config=cfg, env_config=env_config, market_params=market,
        padded_grid=padded_grid, liability=liability,
        initial_spot=S0, initial_variance=v0,
        initial_cash=0.0, seed=seed,
    )
    tag = f"POMARL-{policy_method}"
    logger.info("[%s] training %d epochs, bucket=%.2f, policy=%s",
                tag, n_epochs, discrete_bucket_size, policy_method)
    for epoch in range(1, n_epochs + 1):
        m = trainer.train_step()
        if epoch == 1 or epoch % max(1, n_epochs // 5) == 0 or epoch == n_epochs:
            err = m.get("pi_loss", float("nan"))
            logger.info(
                "  [%s] epoch %d/%d  pi_loss=%.2f", tag, epoch, n_epochs, err,
            )
    ev = trainer.evaluate(n_paths=4096, seed=9999, sample=False)
    return {
        "agent": tag, "std_err": float(ev["std_hedging_error"]),
        "cvar05": float(ev["cvar_05_hedging_error"]),
        "mean_pnl": float(ev["mean_pnl"]),
        "mean_cost": float(ev["mean_cost"]),
        "eval_loss": float(ev["eval_loss"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bucket-size", type=float, default=0.1,
                    help="Lot size for trade rounding. 0.1 = 10 buckets/unit.")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--horizon", type=int, default=252)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    if args.smoke:
        args.n_epochs = 10
        args.batch_size = 128
        args.horizon = 32

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.tag:
        stamp = f"{stamp}_{args.tag}"
    run_dir = ROOT / "runs" / "ablation_nondiff" / stamp
    logger = configure_logging(run_dir)

    logger.info("=" * 78)
    logger.info("Non-differentiable-action ablation")
    logger.info(
        "  bucket_size=%.2f  n_epochs=%d  batch=%d  horizon=%d  seed=%d  device=%s",
        args.bucket_size, args.n_epochs, args.batch_size, args.horizon,
        args.seed, args.device,
    )
    logger.info("=" * 78)

    env_config = DeepHedgingEnvConfig(
        horizon_steps=args.horizon, option_grid=GRID_COARSE, dt=1.0 / 252.0,
        transaction_cost_underlying=1e-4, transaction_cost_option=1e-2,
        risk_aversion=1000.0, scheme="qe",
    )
    market = heston_easy()
    padded_grid = compile_padded_grid(
        GRID_COARSE, dt=env_config.dt, horizon_steps=args.horizon,
    )
    liability = LiabilitySpec(
        kind="call", strike=100.0, maturity=args.horizon, quantity=1.0,
    )

    # ── Reference baselines on the continuous-action env ─────────────────
    # (For comparison; the BS-delta hedger doesn't have a "bucket" — it
    #  trades continuously by definition.)
    sampler_continuous = make_sampler(env_config, market, padded_grid, 100.0, 0.04)
    tc = build_transaction_cost_vector(env_config)
    grid_spec = build_option_grid_spec(GRID_COARSE)
    bs_metrics = evaluate_analytical_hedger_batch(
        market_sampler=sampler_continuous, horizon=args.horizon,
        n_instruments=GRID_COARSE.n_instruments, transaction_cost_rates=tc,
        liability=liability, hedge_kind="delta", dt=env_config.dt,
        n_paths=4096, eval_seed=9999, risk_aversion=env_config.risk_aversion,
    )
    logger.info("[BS-delta (continuous, no rounding)] std_err=%.4f  loss=%.2f",
                bs_metrics["std_hedging_error"], bs_metrics["eval_loss"])

    # ── Run the three configurations ─────────────────────────────────────
    results: List[Dict[str, float]] = [
        {"agent": "BS-delta (continuous)", "std_err": float(bs_metrics["std_hedging_error"]),
         "cvar05": float(bs_metrics["cvar_05_hedging_error"]),
         "mean_pnl": float(bs_metrics["mean_pnl"]),
         "mean_cost": float(bs_metrics["mean_cost"]),
         "eval_loss": float(bs_metrics["eval_loss"])},
    ]

    logger.info("\n=== Training LSTM (Buehler pathwise BPTT) with discrete actions ===")
    try:
        lstm_res = train_lstm(
            env_config, market, padded_grid, liability, 100.0, 0.04,
            args.n_epochs, args.batch_size, args.lr, args.bucket_size,
            args.seed, args.device, logger,
        )
        results.append(lstm_res)
    except Exception as e:
        logger.error("LSTM training failed: %s: %s", type(e).__name__, e)
        results.append({"agent": "lstm", "std_err": float("nan"), "error": str(e)})

    logger.info("\n=== Training POMARL-pathwise (SVG) with discrete actions ===")
    try:
        pomarl_pw = train_pomarl(
            env_config, market, padded_grid, liability, 100.0, 0.04,
            args.n_epochs, args.batch_size, args.lr, args.bucket_size,
            policy_method="pathwise", seed=args.seed, logger=logger,
        )
        results.append(pomarl_pw)
    except Exception as e:
        logger.error("POMARL-pathwise failed: %s: %s", type(e).__name__, e)
        results.append({"agent": "POMARL-pathwise", "std_err": float("nan"), "error": str(e)})

    logger.info("\n=== Training POMARL-REINFORCE (score-function) with discrete actions ===")
    try:
        pomarl_rf = train_pomarl(
            env_config, market, padded_grid, liability, 100.0, 0.04,
            args.n_epochs, args.batch_size, args.lr, args.bucket_size,
            policy_method="reinforce", seed=args.seed, logger=logger,
        )
        results.append(pomarl_rf)
    except Exception as e:
        logger.error("POMARL-REINFORCE failed: %s: %s", type(e).__name__, e)
        results.append({"agent": "POMARL-REINFORCE", "std_err": float("nan"), "error": str(e)})

    # ── Comparison table ─────────────────────────────────────────────────
    logger.info("=" * 78)
    logger.info("Non-differentiable-action results (bucket_size=%.2f)", args.bucket_size)
    logger.info("-" * 78)
    logger.info("  %-30s %-10s %-10s %-10s %-10s %-12s",
                "agent", "std_err", "cvar05", "mean_pnl", "mean_cost", "eval_loss")
    logger.info("-" * 78)
    for r in results:
        if "error" in r:
            logger.info("  %-30s  TRAINING FAILED: %s", r["agent"], r["error"])
            continue
        logger.info("  %-30s %-10.3f %-10.3f %-10.3f %-10.4f %-12.2f",
                    r["agent"], r["std_err"], r["cvar05"],
                    r["mean_pnl"], r["mean_cost"], r["eval_loss"])
    logger.info("=" * 78)

    summary_path = run_dir / "ablation_results.json"
    with summary_path.open("w") as fp:
        json.dump({
            "run_id": stamp, "bucket_size": args.bucket_size,
            "n_epochs": args.n_epochs, "batch_size": args.batch_size,
            "horizon": args.horizon, "seed": args.seed,
            "results": results,
        }, fp, indent=2, default=float)
    logger.info("Saved: %s", summary_path)


if __name__ == "__main__":
    main()
