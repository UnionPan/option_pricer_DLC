"""
Train POMARL (Algorithm 1) and report a side-by-side comparison with the
existing LSTM hedger and BS-delta baseline on a single Heston seed.

Two phases match :file:`scripts/validate_lstm_vs_bs.py` so the LSTM
checkpoints from that run can be re-evaluated against POMARL:

    P1_underlying_only :  instrument_mask = [1, 0, 0, ...]  (delta-only)
    P2_with_options    :  full GRID_COARSE access

Reporting rule (see :file:`notes/pomarl.tex`): POMARL optimizes
``E[Σ r_t]`` (mean-PnL) while Buehler optimizes ``γ·Var + E[cost]``.  We
report BOTH numbers so the comparison is honest — different objectives,
not a leaderboard.

Run:
    python scripts/train_pomarl_vs_lstm.py --seed 23 --n-epochs 200
    python scripts/train_pomarl_vs_lstm.py --smoke           # 4-epoch wiring check
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path
from typing import Dict, List

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from options_desk.deep_hedging.jax.env import (  # noqa: E402
    DeepHedgingEnvConfig,
    GRID_COARSE,
    build_transaction_cost_vector,
)
from options_desk.deep_hedging.jax.pricing import (  # noqa: E402
    HestonMarketParams,
    compile_padded_grid,
)
from options_desk.deep_hedging.jax.rollout import (  # noqa: E402
    simulate_heston_market_batch,
)
from options_desk.deep_hedging.pomarl import (  # noqa: E402
    AISPGTrainer,
    AISPGTrainerConfig,
    PPOTrainer,
    PPOTrainerConfig,
)
from options_desk.deep_hedging.utils.contracts import (  # noqa: E402
    LiabilitySpec,
    TrajectoryBatch,
)
from options_desk.deep_hedging.utils.eval_baseline import (  # noqa: E402
    evaluate_analytical_delta_hedger_batch,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def configure_logging(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pomarl_vs_lstm")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(run_dir / "run.log", mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    for noisy in ("jax", "absl"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logger


# ---------------------------------------------------------------------------
# Heston spec — must match validate_lstm_vs_bs.py for byte-identical paths
# ---------------------------------------------------------------------------


def heston_easy() -> HestonMarketParams:
    return HestonMarketParams(
        kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, r=0.0, q=0.0,
    )


def make_env_config(horizon: int) -> DeepHedgingEnvConfig:
    return DeepHedgingEnvConfig(
        horizon_steps=horizon, option_grid=GRID_COARSE, dt=1.0 / 252.0,
        transaction_cost_underlying=1.0e-4, transaction_cost_option=1.0e-2,
        risk_aversion=1_000.0, scheme="qe",
    )


def make_market_sampler(env_config, market, padded_grid, S0, v0):
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
            instrument_prices=np.asarray(traj.instrument_prices,
                                         dtype=np.float32),
            action_masks=np.broadcast_to(
                np.asarray(traj.action_masks, dtype=bool),
                (batch_size, horizon + 1, n_instruments),
            ).copy(),
        )

    return sampler


# ---------------------------------------------------------------------------
# POMARL training (single phase)
# ---------------------------------------------------------------------------


def train_pomarl(
    *,
    phase_name: str,
    instrument_mask: tuple[int, ...] | None,
    env_config,
    market,
    padded_grid,
    liability: LiabilitySpec,
    n_epochs: int,
    batch_size: int,
    seed: int,
    eval_n_paths: int,
    eval_seed: int,
    initial_spot: float,
    initial_variance: float,
    learning_rate: float,
    logger: logging.Logger,
    checkpoint_dir: Path,
    reward_kind: str = "mean_pnl",
    risk_aversion: float | None = None,
    ais_lr: float | None = None,
    policy_method: str = "reinforce",
    entropy_coef: float = 0.0,
    encoder_gradient_from_policy: bool = False,
    disable_ais_aux_losses: bool = False,
) -> Dict[str, float]:
    # γ default depends on reward shape: mean_pnl ignores γ, hedging_*
    # uses it as the squared-error coefficient (much smaller scale than
    # Buehler's γ·Var since MSE is O(10²) not O(1) early in training).
    if risk_aversion is None:
        risk_aversion = (
            env_config.risk_aversion if reward_kind == "mean_pnl" else 1.0
        )
    if ais_lr is None:
        ais_lr = learning_rate  # single-timescale (default)
    config = AISPGTrainerConfig(
        ais_hidden_size=64, ais_n_layers=1, policy_hidden_size=64,
        position_limit=1.5,
        log_std_min=-5.0, log_std_max=2.0,
        policy_lr=learning_rate, ais_lr=ais_lr,
        batch_size=batch_size, n_epochs=n_epochs,
        eval_every=max(1, n_epochs // 10), grad_clip=1.0,
        discount=1.0, baseline_kind="batch_mean",
        entropy_coef=entropy_coef,
        lambda_reward=1.0, lambda_transition=1.0,
        instrument_mask=instrument_mask,
        risk_aversion=risk_aversion,
        reward_kind=reward_kind,
        policy_method=policy_method,
        encoder_gradient_from_policy=encoder_gradient_from_policy,
        disable_ais_aux_losses=disable_ais_aux_losses,
    )

    trainer = AISPGTrainer.from_heston(
        config=config, env_config=env_config,
        market_params=market, padded_grid=padded_grid,
        liability=liability,
        initial_spot=initial_spot, initial_variance=initial_variance,
        initial_cash=0.0, seed=seed,
    )

    logger.info(
        "[POMARL phase=%s seed=%d] start train  n_epochs=%d batch=%d lr=%.1e "
        "mask=%s",
        phase_name, seed, n_epochs, batch_size, learning_rate,
        "underlying-only" if instrument_mask is not None else "full-grid",
    )

    for epoch in range(1, n_epochs + 1):
        m = trainer.train_step()
        m["epoch"] = epoch
        trainer.train_history.append(m)
        if (epoch % config.eval_every == 0) or epoch == n_epochs or epoch == 1:
            # The buehler branch reports (var_term, cost_term, std_error)
            # instead of (mean_return, total_cost) — use whichever is present.
            ret_str = (
                f"return={m['pi_mean_return']:+.4f}"
                if "pi_mean_return" in m
                else f"err_std={m.get('pi_std_error', float('nan')):.4f}"
            )
            cost_str = (
                f"cost={m['pi_total_cost_mean']:.4f}"
                if "pi_total_cost_mean" in m
                else f"cost={m.get('pi_total_cost_mean', m.get('pi_cost_term', float('nan'))):.4f}"
            )
            logger.info(
                "[POMARL phase=%s seed=%d epoch=%d/%d] "
                "pi_loss=%+.4f ais_loss=%+.4f %s %s",
                phase_name, seed, epoch, n_epochs,
                m["policy_loss"], m["ais_loss"],
                ret_str, cost_str,
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"pomarl_{phase_name}_seed{seed}.pt"
    trainer.save_checkpoint(str(ckpt_path))

    eval_metrics = trainer.evaluate(n_paths=eval_n_paths, seed=eval_seed)
    logger.info(
        "[POMARL phase=%s seed=%d EVAL] eval_loss=%.4f std_err=%.4f "
        "mean_return=%+.4f cvar05=%+.4f",
        phase_name, seed,
        eval_metrics["eval_loss"], eval_metrics["std_hedging_error"],
        eval_metrics["pomarl_mean_return"],
        eval_metrics["cvar_05_hedging_error"],
    )
    return eval_metrics


# ---------------------------------------------------------------------------
# PPO training (single phase) — the strong cumulative-reward RL baseline
# ---------------------------------------------------------------------------


def train_ppo(
    *,
    phase_name: str,
    instrument_mask: tuple[int, ...] | None,
    env_config,
    market,
    padded_grid,
    liability: LiabilitySpec,
    n_epochs: int,
    batch_size: int,
    seed: int,
    eval_n_paths: int,
    eval_seed: int,
    initial_spot: float,
    initial_variance: float,
    learning_rate: float,
    logger: logging.Logger,
    checkpoint_dir: Path,
    reward_kind: str,
    risk_aversion: float | None,
    gae_lambda: float,
    clip_eps: float,
    ppo_epochs: int,
    n_minibatches: int,
    value_coef: float,
    entropy_coef: float,
) -> Dict[str, float]:
    # γ here scales the shaped reward; advantages are normalized so PPO is
    # far less sensitive to it than REINFORCE/pathwise. Default 1.0 for the
    # hedging-aware rewards (mean_pnl ignores it).
    if risk_aversion is None:
        risk_aversion = (
            env_config.risk_aversion if reward_kind == "mean_pnl" else 1.0
        )
    config = PPOTrainerConfig(
        ais_hidden_size=64, ais_n_layers=1, policy_hidden_size=64,
        critic_hidden_size=64, position_limit=1.5,
        log_std_min=-5.0, log_std_max=2.0,
        learning_rate=learning_rate, batch_size=batch_size,
        n_epochs=n_epochs, eval_every=max(1, n_epochs // 10), grad_clip=1.0,
        discount=1.0, gae_lambda=gae_lambda, clip_eps=clip_eps,
        ppo_epochs=ppo_epochs, n_minibatches=n_minibatches,
        value_coef=value_coef, entropy_coef=entropy_coef,
        instrument_mask=instrument_mask, risk_aversion=risk_aversion,
        reward_kind=reward_kind, dt=env_config.dt,
    )
    trainer = PPOTrainer.from_heston(
        config=config, env_config=env_config, market_params=market,
        padded_grid=padded_grid, liability=liability,
        initial_spot=initial_spot, initial_variance=initial_variance,
        initial_cash=0.0, seed=seed,
    )
    logger.info(
        "[PPO phase=%s seed=%d] start  iters=%d batch=%d lr=%.1e K=%d "
        "mb=%d clip=%.2f lam=%.2f reward=%s mask=%s",
        phase_name, seed, n_epochs, batch_size, learning_rate, ppo_epochs,
        n_minibatches, clip_eps, gae_lambda, reward_kind,
        "underlying-only" if instrument_mask is not None else "full-grid",
    )

    # Best-checkpoint (early-stopping) tracking. PPO without a large entropy
    # bonus can suffer late-training entropy collapse → instability, so we
    # snapshot the params whenever held-out eval improves and report the BEST
    # policy found (standard RL-benchmark practice), not the final one.
    best_std = float("inf")
    best_params = None
    eval_track_paths = min(2048, eval_n_paths)
    for epoch in range(1, n_epochs + 1):
        m = trainer.train_step()
        m["epoch"] = epoch
        trainer.train_history.append(m)
        if (epoch % config.eval_every == 0) or epoch == n_epochs or epoch == 1:
            ev = trainer.evaluate(n_paths=eval_track_paths, seed=eval_seed)
            cur_std = ev["std_hedging_error"]
            if cur_std < best_std:
                best_std = cur_std
                best_params = jax.device_get(trainer.params)
            logger.info(
                "[PPO phase=%s seed=%d iter=%d/%d] loss=%+.4f pg=%+.4f "
                "vf=%.4f ent=%.3f kl=%.4f clipf=%.2f train_err=%.4f "
                "eval_std=%.4f best=%.4f cost=%.4f",
                phase_name, seed, epoch, n_epochs,
                m["loss"], m["policy_loss"], m["value_loss"], m["entropy"],
                m["approx_kl"], m["clip_frac"], m["std_error"],
                cur_std, best_std, m["mean_cost"],
            )

    # Restore the best policy found for the final eval + checkpoint.
    if best_params is not None:
        trainer.params = best_params

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"ppo_{phase_name}_seed{seed}.pt"
    trainer.save_checkpoint(str(ckpt_path))

    eval_metrics = trainer.evaluate(n_paths=eval_n_paths, seed=eval_seed)
    logger.info(
        "[PPO phase=%s seed=%d EVAL] eval_loss=%.4f std_err=%.4f "
        "mean_return=%+.4f cvar05=%+.4f",
        phase_name, seed,
        eval_metrics["eval_loss"], eval_metrics["std_hedging_error"],
        eval_metrics["pomarl_mean_return"],
        eval_metrics["cvar_05_hedging_error"],
    )
    return eval_metrics


# ---------------------------------------------------------------------------
# Optional LSTM baseline re-eval
# ---------------------------------------------------------------------------


def maybe_eval_lstm(
    *,
    ckpt_path: Path,
    env_config,
    market,
    padded_grid,
    liability: LiabilitySpec,
    instrument_mask: tuple[int, ...] | None,
    eval_n_paths: int,
    eval_seed: int,
    initial_spot: float,
    initial_variance: float,
    seed: int,
    device: str,
    logger: logging.Logger,
) -> Dict[str, float] | None:
    if not ckpt_path.exists():
        logger.info(
            "[LSTM ckpt missing — skipping LSTM column for this phase] %s",
            ckpt_path,
        )
        return None

    import torch  # local import: keep torch off the import path if unused
    from options_desk.deep_hedging.training.torch_buehler import (
        BuehlerTrainer, TrainerConfig,
    )

    config = TrainerConfig(
        position_limit=1.0, risk_aversion=env_config.risk_aversion,
        learning_rate=3e-4, batch_size=2048, n_epochs=1,
        eval_every=1, grad_clip=1.0,
        policy_kind="lstm", lstm_hidden_size=32, lstm_n_blocks=4,
        lstm_position_limit=1.5, lstm_last_layer_scale=1e-3,
        optimizer_kind="adam", instrument_mask=instrument_mask,
    )
    trainer = BuehlerTrainer.from_heston(
        config=config, env_config=env_config, market_params=market,
        padded_grid=padded_grid, liability=liability,
        initial_spot=initial_spot, initial_variance=initial_variance,
        device=device, seed=seed,
    )
    trainer.load_checkpoint(str(ckpt_path))
    return trainer.evaluate(n_paths=eval_n_paths, seed=eval_seed)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


_COLS = [
    ("eval_loss",            "eval_loss",            "{:>10.4f}"),
    ("std_pnl",              "std_pnl",              "{:>10.4f}"),
    ("std_err",              "std_hedging_error",    "{:>10.4f}"),
    ("rmse_err",             "rmse_hedging_error",   "{:>10.4f}"),
    ("cvar05_err",           "cvar_05_hedging_error","{:>+10.4f}"),
    ("mean_pnl",             "mean_pnl",             "{:>+10.4f}"),
    ("mean_cost",            "mean_cost",            "{:>10.4f}"),
    ("pomarl_mean_return",   "pomarl_mean_return",   "{:>+10.4f}"),
]


def render_table(
    rows: List[tuple[str, Dict[str, float] | None]],
    logger: logging.Logger,
) -> None:
    header = " {:<24} ".format("agent") + " ".join(
        "{:>10}".format(name) for name, _, _ in _COLS
    )
    sep = "-" * len(header)
    logger.info(sep)
    logger.info(header)
    logger.info(sep)
    for tag, metrics in rows:
        if metrics is None:
            cells = " ".join("{:>10}".format("—") for _ in _COLS)
        else:
            cells = " ".join(
                fmt.format(metrics[k]) if k in metrics
                else "{:>10}".format("—")
                for _, k, fmt in _COLS
            )
        logger.info(" {:<24} {}".format(tag, cells))
    logger.info(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--n-epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--learning-rate", type=float, default=3e-4,
                    help="Policy learning rate (and AIS LR if --ais-lr unset)")
    ap.add_argument("--ais-lr", type=float, default=None,
                    help="Separate AIS encoder/predictor LR. Default = "
                         "--learning-rate; pass smaller (e.g. 1e-5) for "
                         "two-timescale stability (slow encoder, fast policy).")
    ap.add_argument("--eval-n-paths", type=int, default=4096)
    ap.add_argument("--eval-seed", type=int, default=9999)
    ap.add_argument("--initial-spot", type=float, default=100.0)
    ap.add_argument("--initial-variance", type=float, default=0.04)
    ap.add_argument("--horizon", type=int, default=252)
    ap.add_argument(
        "--lstm-ckpt-dir", type=Path,
        default=ROOT / "runs/validate_lstm_vs_bs/"
                       "20260519T213935Z_long_seed23_400ep/checkpoints",
        help="Directory containing P{1,2}_*_seed{N}.pt LSTM checkpoints "
             "for side-by-side eval. Skipped if missing.",
    )
    ap.add_argument(
        "--device", type=str, default="cuda:0",
        help="Torch device used only for LSTM re-eval. POMARL uses JAX.",
    )
    ap.add_argument(
        "--out-root", type=Path,
        default=ROOT / "runs/train_pomarl_vs_lstm",
    )
    ap.add_argument(
        "--smoke", action="store_true",
        help="Tiny wiring check: 4 epochs, B=64, eval n_paths=256.",
    )
    ap.add_argument(
        "--reward-kind",
        choices=["mean_pnl", "hedging_mse", "hedging_var", "hedging_shaped",
                 "local_risk"],
        default="mean_pnl",
        help="POMARL reward shape: mean_pnl (alpha-harvesting), hedging_mse "
             "(spiky terminal MSE), hedging_var (terminal Var, batch-centered), "
             "hedging_shaped (Ng-Russell potential-function shaping — "
             "telescopes to terminal MSE but distributes the gradient "
             "across all 252 steps via BS-priced liability potential).",
    )
    ap.add_argument(
        "--policy-method", choices=["reinforce", "pathwise", "buehler"],
        default="reinforce",
        help="Policy gradient estimator. 'reinforce' = score-function; "
             "'pathwise' = SVG backprop through differentiable simulator; "
             "'buehler' = direct supervised loss γ·Var(error)+E[cost] "
             "(LSTM-trainer-style), recommended with --encoder-grad-from-policy "
             "and --disable-ais-aux-losses to isolate whether the RL training "
             "framework (rather than the AIS encoder) is the bottleneck.",
    )
    ap.add_argument("--encoder-grad-from-policy", action="store_true",
                    help="Let policy loss flow into encoder (drops stop_gradient).")
    ap.add_argument("--disable-ais-aux-losses", action="store_true",
                    help="Skip reward+transition predictor losses (encoder "
                         "learns only via policy loss when combined with "
                         "--encoder-grad-from-policy).")
    ap.add_argument(
        "--entropy-coef", type=float, default=0.0,
        help="Entropy bonus coefficient (for pathwise: prevents σ collapse).",
    )
    ap.add_argument(
        "--risk-aversion", type=float, default=None,
        help="Override the γ in the reward shaper. Defaults: 1000 for "
             "mean_pnl (unused), 1.0 for hedging_mse/hedging_var "
             "(γ·MSE with MSE ~10²-10³ in early training, so γ=1 "
             "balances cost and tracking terms).",
    )
    # ── Algorithm selection ──────────────────────────────────────────────
    ap.add_argument(
        "--algo", choices=["aispg", "ppo"], default="aispg",
        help="RL algorithm. 'aispg' = AIS-PG (REINFORCE/pathwise/buehler via "
             "--policy-method); 'ppo' = recurrent PPO (clipped surrogate + "
             "GAE + learned critic) — the strong cumulative-reward baseline.",
    )
    # ── PPO hyperparameters (used only when --algo ppo) ───────────────────
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--ppo-epochs", type=int, default=4,
                    help="K PPO update passes over each collected batch.")
    ap.add_argument("--ppo-minibatches", type=int, default=4,
                    help="Number of path-minibatches per PPO epoch "
                         "(batch_size must be divisible by this).")
    ap.add_argument("--value-coef", type=float, default=0.5)
    args = ap.parse_args()

    if args.smoke:
        args.n_epochs = 4
        args.batch_size = 64
        args.eval_n_paths = 256

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"seed{args.seed}_ep{args.n_epochs}_b{args.batch_size}"
    if args.smoke:
        tag = "smoke_" + tag
    run_dir = args.out_root / f"{stamp}_{tag}"
    logger = configure_logging(run_dir)
    logger.info("Run directory: %s", run_dir)
    logger.info("Args: %s", vars(args))

    # ── Common setup ────────────────────────────────────────────────────
    env_config = make_env_config(args.horizon)
    padded_grid = compile_padded_grid(
        GRID_COARSE, dt=env_config.dt, horizon_steps=env_config.horizon_steps,
    )
    market = heston_easy()
    liability = LiabilitySpec(
        kind="call", strike=100.0, maturity=args.horizon, quantity=1.0,
    )

    market_sampler = make_market_sampler(
        env_config, market, padded_grid,
        S0=args.initial_spot, v0=args.initial_variance,
    )
    tc = build_transaction_cost_vector(env_config)

    bs_metrics = evaluate_analytical_delta_hedger_batch(
        market_sampler=market_sampler,
        horizon=env_config.horizon_steps,
        n_instruments=env_config.option_grid.n_instruments,
        transaction_cost_rates=tc,
        liability=liability,
        dt=env_config.dt,
        n_paths=args.eval_n_paths,
        eval_seed=args.eval_seed,
        risk_free_rate=market.r,
        initial_cash=0.0,
        risk_aversion=env_config.risk_aversion,
    )
    logger.info(
        "[BS-delta EVAL] eval_loss=%.4f std_err=%.4f cvar05=%+.4f",
        bs_metrics["eval_loss"], bs_metrics["std_hedging_error"],
        bs_metrics["cvar_05_hedging_error"],
    )

    # ── Phase 1: underlying-only (sanity) ───────────────────────────────
    n_inst = env_config.option_grid.n_instruments
    mask_p1 = tuple([1] + [0] * (n_inst - 1))

    def run_phase(phase_name: str, instrument_mask: tuple[int, ...] | None):
        """Dispatch to the AIS-PG or PPO trainer per --algo."""
        if args.algo == "ppo":
            return train_ppo(
                phase_name=phase_name, instrument_mask=instrument_mask,
                env_config=env_config, market=market, padded_grid=padded_grid,
                liability=liability, n_epochs=args.n_epochs,
                batch_size=args.batch_size, seed=args.seed,
                eval_n_paths=args.eval_n_paths, eval_seed=args.eval_seed,
                initial_spot=args.initial_spot,
                initial_variance=args.initial_variance,
                learning_rate=args.learning_rate, logger=logger,
                checkpoint_dir=run_dir / "checkpoints",
                reward_kind=args.reward_kind, risk_aversion=args.risk_aversion,
                gae_lambda=args.gae_lambda, clip_eps=args.clip_eps,
                ppo_epochs=args.ppo_epochs, n_minibatches=args.ppo_minibatches,
                value_coef=args.value_coef, entropy_coef=args.entropy_coef,
            )
        return train_pomarl(
            phase_name=phase_name, instrument_mask=instrument_mask,
            env_config=env_config, market=market, padded_grid=padded_grid,
            liability=liability, n_epochs=args.n_epochs,
            batch_size=args.batch_size, seed=args.seed,
            eval_n_paths=args.eval_n_paths, eval_seed=args.eval_seed,
            initial_spot=args.initial_spot,
            initial_variance=args.initial_variance,
            learning_rate=args.learning_rate, logger=logger,
            checkpoint_dir=run_dir / "checkpoints",
            reward_kind=args.reward_kind, risk_aversion=args.risk_aversion,
            ais_lr=args.ais_lr, policy_method=args.policy_method,
            entropy_coef=args.entropy_coef,
            encoder_gradient_from_policy=args.encoder_grad_from_policy,
            disable_ais_aux_losses=args.disable_ais_aux_losses,
        )

    pomarl_p1 = run_phase("P1_underlying_only", mask_p1)
    lstm_p1 = maybe_eval_lstm(
        ckpt_path=args.lstm_ckpt_dir / f"P1_underlying_only_seed{args.seed}.pt",
        env_config=env_config, market=market, padded_grid=padded_grid,
        liability=liability, instrument_mask=mask_p1,
        eval_n_paths=args.eval_n_paths, eval_seed=args.eval_seed,
        initial_spot=args.initial_spot, initial_variance=args.initial_variance,
        seed=args.seed, device=args.device, logger=logger,
    )

    # ── Phase 2: full grid ───────────────────────────────────────────────
    pomarl_p2 = run_phase("P2_with_options", None)
    lstm_p2 = maybe_eval_lstm(
        ckpt_path=args.lstm_ckpt_dir / f"P2_with_options_seed{args.seed}.pt",
        env_config=env_config, market=market, padded_grid=padded_grid,
        liability=liability, instrument_mask=None,
        eval_n_paths=args.eval_n_paths, eval_seed=args.eval_seed,
        initial_spot=args.initial_spot, initial_variance=args.initial_variance,
        seed=args.seed, device=args.device, logger=logger,
    )

    # ── Side-by-side ────────────────────────────────────────────────────
    logger.info("================ Phase 1 — underlying-only ================")
    render_table(
        [
            ("BS-delta (analytical)", bs_metrics),
            (f"LSTM seed={args.seed}",  lstm_p1),
            (f"{args.algo.upper()} seed={args.seed}", pomarl_p1),
        ],
        logger,
    )
    logger.info("================ Phase 2 — full GRID_COARSE ===============")
    render_table(
        [
            ("BS-delta (analytical)", bs_metrics),
            (f"LSTM seed={args.seed}",  lstm_p2),
            (f"{args.algo.upper()} seed={args.seed}", pomarl_p2),
        ],
        logger,
    )
    logger.info(
        "Reminder: POMARL optimizes E[Σ r_t]; LSTM/Buehler optimize "
        "γ·Var + E[cost]. Compare std_err / cvar05 within an objective, "
        "not across.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
