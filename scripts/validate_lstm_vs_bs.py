"""
LSTM deep-hedger vs. Black-Scholes delta-hedger validation.

Two phases on a 252-step Heston environment with a vanilla European
short ATM call liability:

    Phase 1 (sanity):  LSTM constrained to underlying-only via
                       instrument_mask = [1, 0, 0, ...].
                       Expected: LSTM std(hedging error) ≈ BS-delta std.

    Phase 2 (value):   LSTM has full access to GRID_COARSE option grid.
                       Expected: LSTM std(hedging error) < BS-delta std
                                 (cost-aware + gamma exploitation).

5 seeds per phase, 200 epochs, batch_size=2048, Adam, cuda:0.

Logs land in ``runs/validate_lstm_vs_bs/<UTC-timestamp>/``:
    run.log         — DEBUG-level human-readable
    metrics.jsonl   — one JSON line per train epoch and per eval
    summary.json    — final paired comparison (LSTM seeds vs. BS, per phase)

Run:
    python scripts/validate_lstm_vs_bs.py
    python scripts/validate_lstm_vs_bs.py --smoke   # tiny wiring check

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import jax
import numpy as np
import torch

# Ensure src/ is importable when invoked as a script
ROOT = Path(__file__).resolve().parents[1]

from options_desk.deep_hedging.jax.env import (  # noqa: E402
    DeepHedgingEnvConfig,
    FloatingOptionGrid,
    GRID_COARSE,
    build_transaction_cost_vector,
)
from options_desk.deep_hedging.jax.pricing import (  # noqa: E402
    HestonMarketParams,
    compile_padded_grid,
)
from options_desk.deep_hedging.jax.rollout import (  # noqa: E402
    simulate_heston_market,
    simulate_heston_market_batch,
)
from options_desk.deep_hedging.training.torch_buehler import (  # noqa: E402
    BuehlerTrainer,
    TrainerConfig,
)
from options_desk.deep_hedging.utils.contracts import (  # noqa: E402
    LiabilitySpec,
    MarketTrajectory,
    TrajectoryBatch,
)
from options_desk.deep_hedging.utils.eval_baseline import (  # noqa: E402
    evaluate_analytical_delta_hedger_batch,
)


# ─────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────


class JsonLinesHandler(logging.Handler):
    """Emit only records tagged with ``extra={'json': {...}}`` as JSONL."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._fp = path.open("a", buffering=1)  # line-buffered

    def emit(self, record: logging.LogRecord) -> None:
        payload = getattr(record, "json", None)
        if payload is None:
            return
        self._fp.write(json.dumps(payload, default=float) + "\n")

    def close(self) -> None:
        try:
            self._fp.close()
        finally:
            super().close()


def configure_logging(run_dir: Path) -> tuple[logging.Logger, JsonLinesHandler]:
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("validate_lstm_vs_bs")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    # Wipe any previously installed handlers (e.g. pytest pickup).
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(run_dir / "run.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    jsonl_handler = JsonLinesHandler(run_dir / "metrics.jsonl")
    jsonl_handler.setLevel(logging.DEBUG)
    logger.addHandler(jsonl_handler)

    # Quiet down jax / matplotlib chatter
    for noisy in ("jax", "absl", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logger, jsonl_handler


# ─────────────────────────────────────────────────────────────────────────
# Experiment configuration
# ─────────────────────────────────────────────────────────────────────────


def heston_easy() -> HestonMarketParams:
    # Borderline-Feller-safe: 2*kappa*theta = 0.16 > sigma_v^2 = 0.09
    return HestonMarketParams(
        kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, r=0.0, q=0.0,
    )


def make_env_config(grid: FloatingOptionGrid, horizon: int) -> DeepHedgingEnvConfig:
    return DeepHedgingEnvConfig(
        horizon_steps=horizon,
        option_grid=grid,
        dt=1.0 / 252.0,
        transaction_cost_underlying=1.0e-4,
        transaction_cost_option=1.0e-2,
        risk_aversion=1_000.0,
        scheme="qe",  # positivity-preserving — safe even if we change Heston later
    )


def make_market_sampler(env_config, market, padded_grid, S0, v0):
    """Closure over JAX-compiled market sim — re-used for train and eval."""
    horizon = env_config.horizon_steps
    n_instruments = env_config.option_grid.n_instruments

    def sampler(batch_size: int, key) -> TrajectoryBatch:
        keys = jax.random.split(key, batch_size)
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
                (batch_size, horizon + 1, n_instruments),
            ).copy(),
        )

    def single_path_sampler(key) -> MarketTrajectory:
        return simulate_heston_market(
            config=env_config, market=market, padded_grid=padded_grid,
            initial_spot=S0, initial_variance=v0, key=key,
        )

    return sampler, single_path_sampler


# ─────────────────────────────────────────────────────────────────────────
# Per-seed LSTM training run
# ─────────────────────────────────────────────────────────────────────────


def train_one_seed(
    *,
    phase_name: str,
    seed: int,
    env_config,
    market,
    padded_grid,
    liability: LiabilitySpec,
    instrument_mask: tuple[int, ...] | None,
    n_epochs: int,
    batch_size: int,
    eval_n_paths: int,
    eval_seed: int,
    device: str,
    logger: logging.Logger,
    initial_spot: float,
    initial_variance: float,
    cosine_lr: bool = False,
    learning_rate: float = 3e-4,
    eta_min: float = 3e-6,
    checkpoint_dir: Path | None = None,
    observe_variance: bool = True,
    variance_proxy: str = "oracle",
) -> Dict[str, float]:
    config = TrainerConfig(
        position_limit=1.0,
        risk_aversion=env_config.risk_aversion,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_epochs=n_epochs,
        eval_every=max(1, n_epochs // 10),
        grad_clip=1.0,
        policy_kind="lstm",
        lstm_hidden_size=32,
        lstm_n_blocks=4,
        # BS-delta for short ATM call lives in [0, 1]; capping at 1.5 leaves
        # room for over-hedge without letting symexp blow positions to
        # 252*10 shares early in training (which caused err_std=56 oscillation
        # in the previous run).
        lstm_position_limit=1.5,
        lstm_last_layer_scale=1e-3,
        optimizer_kind="adam",
        instrument_mask=instrument_mask,
        observe_variance=observe_variance,
        variance_proxy=variance_proxy,
    )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    trainer = BuehlerTrainer.from_heston(
        config=config,
        env_config=env_config,
        market_params=market,
        padded_grid=padded_grid,
        liability=liability,
        initial_spot=initial_spot,
        initial_variance=initial_variance,
        device=device,
        seed=seed,
    )

    scheduler = None
    if cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer, T_max=n_epochs, eta_min=eta_min,
        )

    logger.info(
        "[phase=%s seed=%d] start train  device=%s n_epochs=%d batch=%d "
        "lr=%.1e schedule=%s mask=%s",
        phase_name, seed, device, n_epochs, batch_size,
        learning_rate, "cosine" if cosine_lr else "constant",
        "underlying-only" if instrument_mask is not None else "full-grid",
    )

    for epoch in range(1, n_epochs + 1):
        metrics = trainer.train_step()
        metrics["epoch"] = epoch
        if scheduler is not None:
            scheduler.step()
            metrics["lr"] = float(scheduler.get_last_lr()[0])
        trainer.train_history.append(metrics)
        logger.debug(
            "[phase=%s seed=%d epoch=%d] loss=%.4f var=%.4f cost=%.6f err_std=%.4f",
            phase_name, seed, epoch,
            metrics["total_loss"], metrics["variance_term"],
            metrics["cost_term"], metrics["std_hedging_error"],
            extra={"json": {
                "kind": "epoch", "phase": phase_name, "seed": seed,
                "epoch": epoch, **{k: float(v) for k, v in metrics.items()},
            }},
        )
        if epoch % config.eval_every == 0 or epoch == n_epochs:
            logger.info(
                "[phase=%s seed=%d epoch=%d/%d] loss=%.4f err_std=%.4f cost=%.6f%s",
                phase_name, seed, epoch, n_epochs,
                metrics["total_loss"], metrics["std_hedging_error"],
                metrics["cost_term"],
                f"  lr={metrics['lr']:.2e}" if "lr" in metrics else "",
            )

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = checkpoint_dir / f"{phase_name}_seed{seed}.pt"
        trainer.save_checkpoint(str(ckpt_path))
        logger.info("[phase=%s seed=%d] checkpoint saved: %s",
                    phase_name, seed, ckpt_path)

    eval_metrics = trainer.evaluate(n_paths=eval_n_paths, seed=eval_seed)
    eval_metrics_out = {
        "kind": "eval", "phase": phase_name, "seed": seed,
        "agent": "lstm", **{k: float(v) for k, v in eval_metrics.items()},
    }
    logger.info(
        "[phase=%s seed=%d] EVAL  std_err=%.4f  cvar05=%.4f  cost=%.6f  loss=%.4f",
        phase_name, seed,
        eval_metrics["std_hedging_error"],
        eval_metrics["cvar_05_hedging_error"],
        eval_metrics["mean_cost"],
        eval_metrics["eval_loss"],
        extra={"json": eval_metrics_out},
    )
    return eval_metrics


# ─────────────────────────────────────────────────────────────────────────
# Phase orchestration
# ─────────────────────────────────────────────────────────────────────────


def run_phase(
    *,
    phase_name: str,
    grid: FloatingOptionGrid,
    instrument_mask: tuple[int, ...] | None,
    seeds: list[int],
    horizon: int,
    n_epochs: int,
    batch_size: int,
    eval_n_paths: int,
    eval_seed: int,
    initial_spot: float,
    initial_variance: float,
    liability: LiabilitySpec,
    device: str,
    logger: logging.Logger,
    cosine_lr: bool = False,
    learning_rate: float = 3e-4,
    checkpoint_dir: Path | None = None,
    observe_variance: bool = True,
    variance_proxy: str = "oracle",
) -> Dict[str, Any]:
    env_config = make_env_config(grid, horizon)
    market = heston_easy()
    padded_grid = compile_padded_grid(grid, dt=env_config.dt, horizon_steps=horizon)
    sampler, single = make_market_sampler(
        env_config, market, padded_grid, initial_spot, initial_variance,
    )

    logger.info(
        "[phase=%s] grid: n_instruments=%d  horizon=%d  liability=%s K=%.2f q=%.2f T=%d",
        phase_name, env_config.option_grid.n_instruments, horizon,
        liability.kind, liability.strike, liability.quantity, liability.maturity,
    )

    lstm_results: List[Dict[str, float]] = []
    for seed in seeds:
        try:
            metrics = train_one_seed(
                phase_name=phase_name, seed=seed,
                env_config=env_config, market=market, padded_grid=padded_grid,
                liability=liability, instrument_mask=instrument_mask,
                n_epochs=n_epochs, batch_size=batch_size,
                eval_n_paths=eval_n_paths, eval_seed=eval_seed,
                device=device, logger=logger,
                initial_spot=initial_spot, initial_variance=initial_variance,
                cosine_lr=cosine_lr, learning_rate=learning_rate,
                checkpoint_dir=checkpoint_dir,
                observe_variance=observe_variance,
                variance_proxy=variance_proxy,
            )
            lstm_results.append({"seed": seed, **metrics})
        except FloatingPointError as exc:
            logger.error(
                "[phase=%s seed=%d] non-finite tensor — skipping seed: %s",
                phase_name, seed, exc,
            )
        except torch.cuda.OutOfMemoryError as exc:
            logger.error(
                "[phase=%s seed=%d] CUDA OOM — skipping seed: %s",
                phase_name, seed, exc,
            )

    bs_metrics = evaluate_analytical_delta_hedger_batch(
        market_sampler=sampler,
        horizon=horizon,
        n_instruments=env_config.option_grid.n_instruments,
        transaction_cost_rates=build_transaction_cost_vector(env_config),
        liability=liability,
        dt=env_config.dt,
        n_paths=eval_n_paths,
        eval_seed=eval_seed,
        risk_free_rate=market.r,
        risk_aversion=env_config.risk_aversion,
    )
    logger.info(
        "[phase=%s] BS-DELTA EVAL  std_err=%.4f  cvar05=%.4f  cost=%.6f  loss=%.4f",
        phase_name,
        bs_metrics["std_hedging_error"],
        bs_metrics["cvar_05_hedging_error"],
        bs_metrics["mean_cost"],
        bs_metrics["eval_loss"],
        extra={"json": {
            "kind": "eval", "phase": phase_name, "agent": "bs_delta",
            **{k: (float(v) if not isinstance(v, str) else v)
               for k, v in bs_metrics.items()},
        }},
    )

    return summarize_phase(phase_name, lstm_results, bs_metrics, logger)


def summarize_phase(
    phase_name: str,
    lstm_results: list[dict],
    bs_metrics: dict,
    logger: logging.Logger,
) -> Dict[str, Any]:
    if not lstm_results:
        logger.warning("[phase=%s] no LSTM seeds completed; skipping summary", phase_name)
        return {"phase": phase_name, "lstm_seeds": [], "verdict": "no_data"}

    keys = ("std_hedging_error", "cvar_05_hedging_error", "eval_loss", "mean_cost")
    summary: Dict[str, Any] = {
        "phase": phase_name,
        "n_seeds": len(lstm_results),
        "lstm_seeds": [r["seed"] for r in lstm_results],
        "bs_delta": {k: float(bs_metrics[k]) for k in keys},
        "lstm": {},
    }

    for k in keys:
        vals = np.asarray([r[k] for r in lstm_results], dtype=np.float64)
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        sem = std / math.sqrt(vals.size) if vals.size > 1 else 0.0
        summary["lstm"][k] = {
            "mean": mean, "std": std, "sem": sem,
            "min": float(vals.min()), "max": float(vals.max()),
            "values": [float(v) for v in vals],
        }

    # Headline metric: std_hedging_error
    delta = summary["lstm"]["std_hedging_error"]["mean"] - summary["bs_delta"]["std_hedging_error"]
    sem = summary["lstm"]["std_hedging_error"]["sem"]
    z = delta / sem if sem > 0 else float("nan")
    summary["std_err_gap"] = {
        "lstm_minus_bs": float(delta),
        "lstm_sem": float(sem),
        "z_score": float(z) if not math.isnan(z) else None,
    }

    # Verdict heuristic: LSTM "wins" if mean LSTM std is lower than BS by > 1 SEM
    if delta < -sem and sem > 0:
        verdict = "lstm_better"
    elif delta > sem and sem > 0:
        verdict = "bs_better"
    else:
        verdict = "indistinguishable"
    summary["verdict"] = verdict

    logger.info(
        "[phase=%s] SUMMARY  LSTM std_err=%.4f±%.4f  BS std_err=%.4f  Δ=%+.4f (z=%.2f)  verdict=%s",
        phase_name,
        summary["lstm"]["std_hedging_error"]["mean"],
        summary["lstm"]["std_hedging_error"]["sem"],
        summary["bs_delta"]["std_hedging_error"],
        summary["std_err_gap"]["lstm_minus_bs"],
        summary["std_err_gap"]["z_score"] or float("nan"),
        verdict,
        extra={"json": {"kind": "phase_summary", **summary}},
    )
    return summary


# ─────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Wiring check: 1 seed, 5 epochs, 32-step horizon")
    parser.add_argument("--device", default=None,
                        help="Override device (cuda:0 / cpu). Default: auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Seeds to run (overrides default 5-seed list)")
    parser.add_argument("--n-epochs", type=int, default=None,
                        help="Override training epochs per seed")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override training batch size")
    parser.add_argument("--phases", choices=["P1", "P2", "both"], default="both",
                        help="Which phase(s) to run (default: both)")
    parser.add_argument("--cosine-lr", action="store_true",
                        help="Use CosineAnnealingLR schedule (lr -> 1/100 of init)")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Initial learning rate (default: 3e-4)")
    parser.add_argument("--tag", default=None,
                        help="Optional run-dir suffix for organizing experiments")
    parser.add_argument("--no-observe-variance", action="store_true",
                        help="POMDP mode: mask variance out of LSTM obs (fair "
                             "comparison vs POMARL which is variance-hidden by "
                             "design — its AIS encoder must infer v_t).")
    parser.add_argument("--variance-proxy",
                        choices=["oracle", "ewma"], default="oracle",
                        help="Source for the variance feature. 'oracle' = "
                             "true Heston v_t (MDP). 'ewma' = RiskMetrics "
                             "EWMA of squared returns (near-sufficient "
                             "proxy — AIS-shortcut ablation).")
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # Defaults (full run): B=1024 keeps wall time ≈3h on the RTX 6000.
    # Profiling showed market-sim cost is linear in batch (B=2048 → 20s/epoch
    # vs B=1024 → 11s/epoch) and per-path gradient quality is already
    # adequate for the variance objective at 1024.
    horizon = 252
    n_epochs = 100
    batch_size = 1024
    eval_n_paths = 4096
    seeds = args.seeds or [11, 23, 42, 97, 113]

    if args.smoke:
        horizon = 32
        n_epochs = 5
        batch_size = 64
        eval_n_paths = 128
        seeds = args.seeds or [42]

    if args.n_epochs is not None:
        n_epochs = args.n_epochs
    if args.batch_size is not None:
        batch_size = args.batch_size

    # Liability: short ATM call expiring at horizon end
    liability = LiabilitySpec(
        kind="call", strike=100.0, maturity=horizon, quantity=1.0,
    )
    initial_spot = 100.0
    initial_variance = 0.04
    eval_seed = 9999

    # Run dir
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.tag:
        stamp = f"{stamp}_{args.tag}"
    run_dir = ROOT / "runs" / "validate_lstm_vs_bs" / stamp
    logger, jsonl_handler = configure_logging(run_dir)

    logger.info("=" * 78)
    logger.info("validate_lstm_vs_bs  device=%s  smoke=%s", device, args.smoke)
    logger.info("horizon=%d  n_epochs=%d  batch=%d  eval_paths=%d  seeds=%s",
                horizon, n_epochs, batch_size, eval_n_paths, seeds)
    logger.info("run dir: %s", run_dir)
    logger.info("=" * 78)

    grid = GRID_COARSE
    n_inst = grid.n_instruments  # 1 underlying + 12 options
    underlying_only_mask = tuple(
        [1] + [0] * (n_inst - 1)
    )

    summaries: List[Dict[str, Any]] = []

    if args.phases in ("P1", "both"):
        summaries.append(run_phase(
            phase_name="P1_underlying_only",
            grid=grid,
            instrument_mask=underlying_only_mask,
            seeds=seeds,
            horizon=horizon, n_epochs=n_epochs, batch_size=batch_size,
            eval_n_paths=eval_n_paths, eval_seed=eval_seed,
            initial_spot=initial_spot, initial_variance=initial_variance,
            liability=liability, device=device, logger=logger,
            cosine_lr=args.cosine_lr, learning_rate=args.learning_rate,
            checkpoint_dir=run_dir / "checkpoints",
            observe_variance=not args.no_observe_variance,
            variance_proxy=args.variance_proxy,
        ))

    if args.phases in ("P2", "both"):
        summaries.append(run_phase(
            phase_name="P2_with_options",
            grid=grid,
            instrument_mask=None,
            seeds=seeds,
            horizon=horizon, n_epochs=n_epochs, batch_size=batch_size,
            eval_n_paths=eval_n_paths, eval_seed=eval_seed,
            initial_spot=initial_spot, initial_variance=initial_variance,
            liability=liability, device=device, logger=logger,
            cosine_lr=args.cosine_lr, learning_rate=args.learning_rate,
            checkpoint_dir=run_dir / "checkpoints",
            observe_variance=not args.no_observe_variance,
            variance_proxy=args.variance_proxy,
        ))

    summary_path = run_dir / "summary.json"
    with summary_path.open("w") as fp:
        json.dump({
            "run_id": stamp,
            "device": device,
            "smoke": bool(args.smoke),
            "horizon": horizon,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "eval_n_paths": eval_n_paths,
            "eval_seed": eval_seed,
            "seeds": seeds,
            "phases_arg": args.phases,
            "cosine_lr": bool(args.cosine_lr),
            "learning_rate": float(args.learning_rate),
            "liability": {
                "kind": liability.kind, "strike": liability.strike,
                "maturity": liability.maturity, "quantity": liability.quantity,
            },
            "phases": summaries,
        }, fp, indent=2, default=float)

    logger.info("=" * 78)
    logger.info("Run complete. Artifacts in: %s", run_dir)
    for s in summaries:
        if "verdict" in s and s["verdict"] != "no_data":
            logger.info(
                "  %s: verdict=%s  LSTM std=%.4f±%.4f  BS std=%.4f",
                s["phase"], s["verdict"],
                s["lstm"]["std_hedging_error"]["mean"],
                s["lstm"]["std_hedging_error"]["sem"],
                s["bs_delta"]["std_hedging_error"],
            )
    logger.info("=" * 78)
    jsonl_handler.close()


if __name__ == "__main__":
    main()
