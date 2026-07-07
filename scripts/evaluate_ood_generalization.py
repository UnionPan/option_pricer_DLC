"""
Held-out generalization protocol for trained hedging policies.

Trains an LSTM on one Heston regime, then evaluates the *same* policy
weights on:
    (a) the in-distribution (training) regime         — sanity check
    (b) different Heston regimes (calm / crisis)      — held-out regimes
    (c) the same regime under Euler vs QE schemes     — held-out simulator
    (d) all three analytical baselines (δ / δ-Γ / δ-Γ-ν) for each regime

Produces a paired table:

    market_regime, simulator_scheme, agent, std_hedging_error, eval_loss, ...

Verifies the resume claim: "evaluated on held-out regimes and simulators."

Run:
    python scripts/evaluate_ood_generalization.py --train-epochs 200 \\
                                                  --seed 23 --tag full
    python scripts/evaluate_ood_generalization.py --smoke   # 3-min wiring check
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import jax
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

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


# ─────────────────────────────────────────────────────────────────────────
# Market regimes (in-distribution + held-out)
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MarketRegime:
    name: str
    market: HestonMarketParams
    initial_variance: float


REGIMES: list[MarketRegime] = [
    # In-distribution (training)
    MarketRegime(
        "ID_easy",
        HestonMarketParams(kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, r=0.0, q=0.0),
        initial_variance=0.04,
    ),
    # Held-out regime: high vol, larger vol-of-vol (crisis)
    MarketRegime(
        "OOD_crisis",
        HestonMarketParams(kappa=2.0, theta=0.09, sigma_v=0.5, rho=-0.8, r=0.0, q=0.0),
        initial_variance=0.09,
    ),
    # Held-out regime: low mean-reversion (vol drifts more)
    MarketRegime(
        "OOD_persistent",
        HestonMarketParams(kappa=0.5, theta=0.04, sigma_v=0.3, rho=-0.7, r=0.0, q=0.0),
        initial_variance=0.04,
    ),
    # Held-out regime: weak leverage effect (rho small)
    MarketRegime(
        "OOD_weak_leverage",
        HestonMarketParams(kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.2, r=0.0, q=0.0),
        initial_variance=0.04,
    ),
]


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def make_sampler(
    env_config: DeepHedgingEnvConfig,
    market: HestonMarketParams,
    padded_grid,
    initial_spot: float,
    initial_variance: float,
) -> Callable:
    horizon = env_config.horizon_steps
    n_inst = env_config.option_grid.n_instruments

    def sampler(B: int, key) -> TrajectoryBatch:
        keys = jax.random.split(key, B)
        traj = simulate_heston_market_batch(
            config=env_config, market=market, padded_grid=padded_grid,
            initial_spot=initial_spot, initial_variance=initial_variance, keys=keys,
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


def configure_logging(run_dir: Path) -> tuple[logging.Logger, object]:
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("evaluate_ood_generalization")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(run_dir / "run.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    for noisy in ("jax", "absl", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logger, fh


def eval_trained_lstm_on_market(
    trainer: BuehlerTrainer,
    env_config: DeepHedgingEnvConfig,
    market: HestonMarketParams,
    padded_grid,
    initial_spot: float,
    initial_variance: float,
    n_paths: int,
    eval_seed: int,
) -> dict:
    """Swap the trainer's sampler for one against ``market`` and evaluate."""
    new_sampler = make_sampler(
        env_config, market, padded_grid, initial_spot, initial_variance,
    )
    saved_sampler = trainer._market_sampler
    trainer._market_sampler = new_sampler
    try:
        metrics = trainer.evaluate(n_paths=n_paths, seed=eval_seed)
    finally:
        trainer._market_sampler = saved_sampler
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=23,
                        help="Training seed for the LSTM")
    parser.add_argument("--train-epochs", type=int, default=200,
                        help="LSTM training budget per seed")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-n-paths", type=int, default=4096)
    parser.add_argument("--eval-seed", type=int, default=9999)
    parser.add_argument("--horizon", type=int, default=252)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="3-min wiring check: 32-step horizon, 10 epochs")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    horizon = args.horizon
    n_epochs = args.train_epochs
    batch_size = args.batch_size
    eval_n_paths = args.eval_n_paths
    if args.smoke:
        horizon = 32
        n_epochs = 10
        batch_size = 64
        eval_n_paths = 256

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.tag:
        stamp = f"{stamp}_{args.tag}"
    run_dir = ROOT / "runs" / "ood_generalization" / stamp
    logger, _ = configure_logging(run_dir)

    logger.info("=" * 78)
    logger.info("OOD generalization protocol")
    logger.info(
        "  device=%s  seed=%d  horizon=%d  n_epochs=%d  batch=%d  eval_paths=%d",
        device, args.seed, horizon, n_epochs, batch_size, eval_n_paths,
    )
    logger.info("  regimes: %s", [r.name for r in REGIMES])
    logger.info("  schemes: ['euler', 'qe']")
    logger.info("=" * 78)

    initial_spot = 100.0
    liability = LiabilitySpec(
        kind="call", strike=100.0, maturity=horizon, quantity=1.0,
    )
    grid = GRID_COARSE
    grid_spec = build_option_grid_spec(grid)

    # ── Train LSTM + Transformer on ID_easy regime, scheme=qe ──────────
    id_regime = REGIMES[0]
    env_train = DeepHedgingEnvConfig(
        horizon_steps=horizon, option_grid=grid, dt=1.0 / 252.0,
        transaction_cost_underlying=1e-4, transaction_cost_option=1e-2,
        risk_aversion=1000.0, scheme="qe",
    )
    padded_grid = compile_padded_grid(grid, dt=env_train.dt, horizon_steps=horizon)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainers: dict[str, BuehlerTrainer] = {}

    def train_one(policy_kind: str) -> BuehlerTrainer:
        cfg = TrainerConfig(
            position_limit=1.0, risk_aversion=1000.0, learning_rate=5e-4,
            batch_size=batch_size, n_epochs=n_epochs,
            eval_every=max(1, n_epochs // 10),
            grad_clip=1.0, policy_kind=policy_kind,
            lstm_hidden_size=32, lstm_n_blocks=4,
            lstm_position_limit=1.5, lstm_last_layer_scale=1e-3,
            transformer_hidden_size=32, transformer_n_layers=2,
            transformer_n_heads=4, transformer_max_history=64,
            transformer_position_limit=1.5,
            optimizer_kind="adam",
        )
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        t = BuehlerTrainer.from_heston(
            config=cfg, env_config=env_train, market_params=id_regime.market,
            padded_grid=padded_grid, liability=liability,
            initial_spot=initial_spot, initial_variance=id_regime.initial_variance,
            device=device, seed=args.seed,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            t.optimizer, T_max=n_epochs, eta_min=5e-6,
        )
        logger.info(
            "[train] %s on regime=%s scheme=%s for %d epochs at lr=5e-4 (cosine→5e-6)",
            policy_kind.upper(), id_regime.name, env_train.scheme, n_epochs,
        )
        for epoch in range(1, n_epochs + 1):
            m = t.train_step()
            sched.step()
            if epoch % cfg.eval_every == 0 or epoch == n_epochs:
                logger.info(
                    "  [%s] epoch %d/%d  loss=%.2f  err_std=%.3f  lr=%.2e",
                    policy_kind, epoch, n_epochs, m["total_loss"],
                    m["std_hedging_error"], sched.get_last_lr()[0],
                )
        cp = ckpt_dir / f"{policy_kind}_seed{args.seed}_id_easy.pt"
        t.save_checkpoint(str(cp))
        logger.info("  [%s] checkpoint: %s", policy_kind, cp)
        return t

    for kind in ("lstm", "transformer"):
        trainers[kind] = train_one(kind)

    # ── Run OOD evaluation grid ─────────────────────────────────────────
    table: list[dict] = []
    tc = build_transaction_cost_vector(env_train)

    for regime in REGIMES:
        for scheme in ("qe", "euler"):
            env_eval = DeepHedgingEnvConfig(
                horizon_steps=horizon, option_grid=grid, dt=1.0 / 252.0,
                transaction_cost_underlying=1e-4, transaction_cost_option=1e-2,
                risk_aversion=1000.0, scheme=scheme,
            )
            # Compile padded_grid for this scheme (cheap)
            pg_eval = compile_padded_grid(grid, dt=env_eval.dt, horizon_steps=horizon)

            # ▸ Neural policies (LSTM, Transformer)
            for kind, tr in trainers.items():
                try:
                    nm = eval_trained_lstm_on_market(
                        trainer=tr, env_config=env_eval, market=regime.market,
                        padded_grid=pg_eval, initial_spot=initial_spot,
                        initial_variance=regime.initial_variance,
                        n_paths=eval_n_paths, eval_seed=args.eval_seed,
                    )
                    table.append({
                        "regime": regime.name, "scheme": scheme, "agent": kind,
                        "std": float(nm["std_hedging_error"]),
                        "cvar05": float(nm["cvar_05_hedging_error"]),
                        "cost": float(nm["mean_cost"]),
                        "loss": float(nm["eval_loss"]),
                    })
                except Exception as e:
                    logger.error("%s eval failed on %s/%s: %s",
                                 kind, regime.name, scheme, e)

            # ▸ Analytical baselines on the same paths
            sampler_eval = make_sampler(
                env_eval, regime.market, pg_eval, initial_spot, regime.initial_variance,
            )
            for hedge_kind in ("delta", "delta_gamma", "delta_gamma_vega"):
                try:
                    am = evaluate_analytical_hedger_batch(
                        market_sampler=sampler_eval, horizon=horizon,
                        n_instruments=grid.n_instruments,
                        transaction_cost_rates=tc, liability=liability,
                        hedge_kind=hedge_kind,
                        option_grid_spec=grid_spec if hedge_kind != "delta" else None,
                        dt=env_eval.dt, n_paths=eval_n_paths,
                        eval_seed=args.eval_seed,
                        risk_free_rate=regime.market.r,
                        risk_aversion=env_eval.risk_aversion,
                    )
                    table.append({
                        "regime": regime.name, "scheme": scheme, "agent": hedge_kind,
                        "std": float(am["std_hedging_error"]),
                        "cvar05": float(am["cvar_05_hedging_error"]),
                        "cost": float(am["mean_cost"]),
                        "loss": float(am["eval_loss"]),
                    })
                except Exception as e:
                    logger.error("%s eval failed on %s/%s: %s",
                                 hedge_kind, regime.name, scheme, e)

    # ── Format + persist the OOD table ──────────────────────────────────
    logger.info("=" * 78)
    logger.info("OOD generalization table")
    logger.info(
        "  %-20s %-7s %-18s %-8s %-8s %-9s %-12s",
        "regime", "scheme", "agent", "std", "cvar05", "cost", "loss",
    )
    for row in table:
        logger.info(
            "  %-20s %-7s %-18s %-8.3f %-8.3f %-9.4f %-12.2f",
            row["regime"], row["scheme"], row["agent"],
            row["std"], row["cvar05"], row["cost"], row["loss"],
        )
    logger.info("=" * 78)

    summary_path = run_dir / "ood_table.json"
    with summary_path.open("w") as fp:
        json.dump({
            "run_id": stamp, "device": device, "smoke": args.smoke,
            "seed": args.seed, "n_epochs": n_epochs, "batch_size": batch_size,
            "horizon": horizon, "eval_n_paths": eval_n_paths,
            "eval_seed": args.eval_seed,
            "liability": {"kind": liability.kind, "strike": liability.strike,
                          "maturity": liability.maturity, "quantity": liability.quantity},
            "regimes": [{"name": r.name, "market": r.market._asdict(),
                         "v0": r.initial_variance} for r in REGIMES],
            "results": table,
        }, fp, indent=2, default=float)
    logger.info("Saved: %s", summary_path)
    logger.info("       checkpoints in: %s", ckpt_dir)


if __name__ == "__main__":
    main()
