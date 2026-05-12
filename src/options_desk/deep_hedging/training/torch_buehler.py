"""
Buehler-style deep hedging trainer (PyTorch).

The trainer consumes simulator callables and evaluates on fresh batched
trajectories, so it is decoupled from any specific simulator backend. A
:meth:`BuehlerTrainer.from_heston` classmethod wires the JAX Heston
simulator in for ergonomics.

Differentiable PyTorch rollout through the policy minimises the Buehler
objective::

    L = gamma * Var(terminal_PnL - liability_payoff) + E[transaction_costs]

The policy outputs *trades* (signed delta-positions) at each rebalancing
step.

Reference:
    Buehler, H., Gonon, L., Teichmann, J., & Wood, B. (2019).
    "Deep Hedging." Quantitative Finance, 19(8), 1271-1291.

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np

from ..agents.base import BaseHedgingAgent
from ..agents.torch_policy import HedgingMLPPolicy, TorchPolicyAgent
from ..agents.torch_lstm_policy import HedgingLSTMPolicy
from ..utils.contracts import (
    LiabilityPortfolio,
    LiabilitySpec,
    MarketTrajectory,
    TrajectoryBatch,
    _normalize_legs,
)
from .base import BaseTrainer

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is an optional dep
    _TORCH_AVAILABLE = False

try:
    from tqdm.auto import tqdm

    _TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover - tqdm is an optional dep
    _TQDM_AVAILABLE = False


MarketBatchSampler = Callable[[int, Any], TrajectoryBatch]
SinglePathSampler = Callable[[Any], MarketTrajectory]


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class TrainerConfig:
    """Configuration for the Buehler deep hedging trainer."""

    hidden_sizes: tuple[int, ...] = (64, 64)
    position_limit: float = 1.0
    risk_aversion: float = 1_000.0
    learning_rate: float = 1e-3
    batch_size: int = 1024
    n_epochs: int = 200
    eval_every: int = 20
    grad_clip: float = 1.0
    mlp_last_layer_scale: float = 1e-3

    # Feature scaling used by both Torch training and TorchPolicyAgent
    # inference. Keep price_scale aligned with the simulator's initial spot.
    price_scale: float = 100.0
    position_scale: float | None = None
    price_clip: float = 10.0

    # --- LSTM policy options (used when policy_kind == 'lstm') ---
    policy_kind: str = "mlp"            # 'mlp' (default) or 'lstm'
    lstm_hidden_size: int = 32          # paper: 32
    lstm_n_blocks: int = 4              # paper: 4 residual blocks
    lstm_position_limit: float | None = 10.0   # soft tanh clamp; None disables
    lstm_last_layer_scale: float = 1e-3        # paper: small-init last layer

    # --- Optimizer options ---
    # 'adam' (default) — torch.optim.Adam
    # 'kfac'           — KFACOptimizer (paper-grade second-order); for LSTM
    #                    policies, this auto-enables ManualLSTMCell so KFAC
    #                    can hook into the gate pre-activations.
    optimizer_kind: str = "adam"
    kfac_damping: float = 1e-2
    kfac_ema_decay: float = 0.95
    kfac_n_eigen_decomp: int = 25
    kfac_trust_region: float | None = 1e-2
    kfac_trust_region_decay: float | None = None
    kfac_min_trust_region: float | None = None
    kfac_trust_region_metric: str = "natural"
    kfac_weight_decay: float = 0.0


# ============================================================================
# Differentiable rollout + Buehler loss
# ============================================================================


def _positive_scale(value: float | None, default: float, name: str) -> float:
    scale = default if value is None else float(value)
    if scale <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return scale


def _policy_feature_scale(policy: Any, name: str, default: float) -> float:
    return _positive_scale(getattr(policy, name, default), default, name)


def _assert_finite_tensor(name: str, tensor: "torch.Tensor") -> None:
    """Raise with useful diagnostics when a training tensor becomes non-finite."""
    if not torch.is_tensor(tensor) or not tensor.dtype.is_floating_point:
        return
    finite = torch.isfinite(tensor)
    if bool(finite.all()):
        return

    bad_count = int((~finite).sum().item())
    nan_count = int(torch.isnan(tensor).sum().item())
    inf_count = int(torch.isinf(tensor).sum().item())
    safe = torch.nan_to_num(tensor.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    raise FloatingPointError(
        f"{name} contains non-finite values: "
        f"bad={bad_count}, nan={nan_count}, inf={inf_count}, "
        f"shape={tuple(tensor.shape)}, min={float(safe.min().item()):.6g}, "
        f"max={float(safe.max().item()):.6g}"
    )


def _assert_finite_trajectory(trajectory: Dict[str, "torch.Tensor"]) -> None:
    for name in ("spots", "variances", "instrument_prices"):
        value = trajectory.get(name)
        if value is not None:
            _assert_finite_tensor(f"trajectory[{name!r}]", value)


def _build_batch_obs_tensor(
    trajectory: Dict[str, "torch.Tensor"],
    positions: "torch.Tensor",
    previous_trades: "torch.Tensor",
    time_index: int,
    horizon: int,
    n_instruments: int,
    price_scale: float = 100.0,
    position_scale: float = 100.0,
    price_clip: float = 10.0,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """
    Build batched observation tensors for the differentiable rollout.

    Concatenation order matches :func:`obs_batch_to_tensor` in the
    inference path:
        ``[spot, option_features, positions, previous_trades, context]``
    """
    B = positions.shape[0]
    device = positions.device
    time_ratio = time_index / horizon if horizon > 0 else 0.0
    price_scale = _positive_scale(price_scale, 100.0, "price_scale")
    position_scale = _positive_scale(position_scale, 100.0, "position_scale")
    price_clip = _positive_scale(price_clip, 10.0, "price_clip")

    spot = torch.clamp(
        trajectory["spots"][:, time_index].unsqueeze(1) / price_scale - 1.0,
        min=-price_clip,
        max=price_clip,
    )
    option_features = torch.clamp(
        trajectory["instrument_prices"][:, time_index, 1:] / price_scale,
        min=-price_clip,
        max=price_clip,
    )
    scaled_positions = torch.clamp(
        positions / position_scale,
        min=-price_clip,
        max=price_clip,
    )
    scaled_previous_trades = torch.clamp(
        previous_trades / position_scale,
        min=-price_clip,
        max=price_clip,
    )
    variance = trajectory["variances"][:, time_index].unsqueeze(1)
    time_feat = torch.full((B, 1), time_ratio, dtype=torch.float32, device=device)
    bias_feat = torch.ones(B, 1, dtype=torch.float32, device=device)
    context = torch.cat([variance, time_feat, bias_feat], dim=1)

    obs = torch.cat(
        [spot, option_features, scaled_positions, scaled_previous_trades, context],
        dim=1,
    )
    mask = trajectory["action_masks"][:, time_index].float()
    return obs, mask


def _enforce_trade_mask(
    trades: "torch.Tensor",
    positions: "torch.Tensor",
    mask: "torch.Tensor",
) -> "torch.Tensor":
    """Match agent replay semantics: masked instruments are liquidated."""
    return torch.where(mask.bool(), trades, -positions)


def differentiable_rollout(
    policy: HedgingMLPPolicy,
    trajectory: Dict[str, "torch.Tensor"],
    transaction_cost_rates: "torch.Tensor",
    horizon: int,
    n_instruments: int,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """
    Run the policy through a batch of market trajectories, differentiably.

    Returns:
        terminal_pnl: ``(B,)`` terminal portfolio value (cash + mark-to-market).
        total_costs: ``(B,)`` accumulated transaction costs.
    """
    B = trajectory["spots"].shape[0]
    device = trajectory["spots"].device

    positions = torch.zeros(B, n_instruments, dtype=torch.float32, device=device)
    cash = torch.zeros(B, dtype=torch.float32, device=device)
    previous_trades = torch.zeros(B, n_instruments, dtype=torch.float32, device=device)
    total_costs = torch.zeros(B, dtype=torch.float32, device=device)
    price_scale = _policy_feature_scale(policy, "price_scale", 100.0)
    position_scale = _policy_feature_scale(policy, "position_scale", 100.0)
    price_clip = _policy_feature_scale(policy, "price_clip", 10.0)

    for t in range(horizon):
        obs, mask = _build_batch_obs_tensor(
            trajectory,
            positions,
            previous_trades,
            t,
            horizon,
            n_instruments,
            price_scale=price_scale,
            position_scale=position_scale,
            price_clip=price_clip,
        )
        trades = _enforce_trade_mask(policy(obs, mask), positions, mask)

        prices_t = trajectory["instrument_prices"][:, t]
        notional = (trades * prices_t).sum(dim=1)
        step_cost = (
            transaction_cost_rates.unsqueeze(0) * (trades * prices_t).abs()
        ).sum(dim=1)

        cash = cash - notional - step_cost
        positions = positions + trades
        previous_trades = trades
        total_costs = total_costs + step_cost

    prices_T = trajectory["instrument_prices"][:, horizon]
    terminal_pnl = cash + (positions * prices_T).sum(dim=1)
    return terminal_pnl, total_costs


def differentiable_rollout_recurrent(
    policy: "HedgingLSTMPolicy",
    trajectory: Dict[str, "torch.Tensor"],
    transaction_cost_rates: "torch.Tensor",
    horizon: int,
    n_instruments: int,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """
    Recurrent variant of :func:`differentiable_rollout` for LSTM policies.

    Threads the LSTM hidden state through the time loop so the policy can
    carry market memory across rebalancing dates. Action recurrence is
    implicit via ``previous_trades`` in the obs vector (see
    ``_build_batch_obs_tensor``); hidden recurrence flows through ``state``.

    Returns:
        terminal_pnl: ``(B,)`` terminal portfolio value (cash + mark-to-market).
        total_costs: ``(B,)`` accumulated transaction costs.
    """
    B = trajectory["spots"].shape[0]
    device = trajectory["spots"].device

    positions = torch.zeros(B, n_instruments, dtype=torch.float32, device=device)
    cash = torch.zeros(B, dtype=torch.float32, device=device)
    previous_trades = torch.zeros(B, n_instruments, dtype=torch.float32, device=device)
    total_costs = torch.zeros(B, dtype=torch.float32, device=device)
    price_scale = _policy_feature_scale(policy, "price_scale", 100.0)
    position_scale = _policy_feature_scale(policy, "position_scale", 100.0)
    price_clip = _policy_feature_scale(policy, "price_clip", 10.0)

    state = policy.init_hidden_state(batch_size=B, device=device, dtype=torch.float32)

    for t in range(horizon):
        obs, mask = _build_batch_obs_tensor(
            trajectory,
            positions,
            previous_trades,
            t,
            horizon,
            n_instruments,
            price_scale=price_scale,
            position_scale=position_scale,
            price_clip=price_clip,
        )
        raw_trades, state = policy.step(obs, mask, state)
        trades = _enforce_trade_mask(raw_trades, positions, mask)

        prices_t = trajectory["instrument_prices"][:, t]
        notional = (trades * prices_t).sum(dim=1)
        step_cost = (
            transaction_cost_rates.unsqueeze(0) * (trades * prices_t).abs()
        ).sum(dim=1)

        cash = cash - notional - step_cost
        positions = positions + trades
        previous_trades = trades
        total_costs = total_costs + step_cost

    prices_T = trajectory["instrument_prices"][:, horizon]
    terminal_pnl = cash + (positions * prices_T).sum(dim=1)
    return terminal_pnl, total_costs


def buehler_loss(
    terminal_pnl: "torch.Tensor",
    liability_payoffs: "torch.Tensor",
    total_costs: "torch.Tensor",
    risk_aversion: float,
) -> tuple["torch.Tensor", Dict[str, float]]:
    """
    Compute the Buehler deep hedging objective::

        L = gamma * Var(PnL - payoff) + E[costs]
    """
    hedging_error = terminal_pnl - liability_payoffs
    variance_term = risk_aversion * hedging_error.var()
    cost_term = total_costs.mean()
    loss = variance_term + cost_term

    metrics = {
        "variance_term": float(variance_term.item()),
        "cost_term": float(cost_term.item()),
        "total_loss": float(loss.item()),
        "mean_pnl": float(terminal_pnl.mean().item()),
        "mean_hedging_error": float(hedging_error.mean().item()),
        "std_hedging_error": float(hedging_error.std().item()),
    }
    return loss, metrics


def _safe_population_std(values: "torch.Tensor") -> "torch.Tensor":
    if values.numel() <= 1:
        return torch.zeros((), dtype=values.dtype, device=values.device)
    return values.std(unbiased=False)


def _safe_sample_var(values: "torch.Tensor") -> "torch.Tensor":
    if values.numel() <= 1:
        return torch.zeros((), dtype=values.dtype, device=values.device)
    return values.var(unbiased=True)


def _scalar(value: "torch.Tensor") -> float:
    return float(value.detach().cpu().item())


def _left_tail_mean(values: "torch.Tensor", quantile: float) -> "torch.Tensor":
    threshold = torch.quantile(values, quantile)
    tail = values[values <= threshold]
    if tail.numel() == 0:
        return threshold
    return tail.mean()


def _error_distribution_metrics(
    errors: "torch.Tensor",
    prefix: str = "",
) -> Dict[str, float]:
    abs_errors = errors.abs()
    mse = (errors * errors).mean()
    return {
        f"{prefix}mean_hedging_error": _scalar(errors.mean()),
        f"{prefix}std_hedging_error": _scalar(_safe_population_std(errors)),
        f"{prefix}mae_hedging_error": _scalar(abs_errors.mean()),
        f"{prefix}mse_hedging_error": _scalar(mse),
        f"{prefix}rmse_hedging_error": _scalar(torch.sqrt(mse)),
        f"{prefix}p01_hedging_error": _scalar(torch.quantile(errors, 0.01)),
        f"{prefix}p05_hedging_error": _scalar(torch.quantile(errors, 0.05)),
        f"{prefix}p50_hedging_error": _scalar(torch.quantile(errors, 0.50)),
        f"{prefix}p95_hedging_error": _scalar(torch.quantile(errors, 0.95)),
        f"{prefix}p99_hedging_error": _scalar(torch.quantile(errors, 0.99)),
        f"{prefix}cvar_05_hedging_error": _scalar(_left_tail_mean(errors, 0.05)),
    }


# ============================================================================
# Per-leg torch payoff (used by BuehlerTrainer._compute_liability_payoffs)
# ============================================================================


def _torch_leg_payoff(
    leg: LiabilitySpec, spots: "torch.Tensor"
) -> "torch.Tensor":
    """Compute a single leg's payoff from the spot path in torch.

    Mirrors :meth:`LiabilitySpec.terminal_payoff_from_path` (NumPy) — keep the
    two implementations in sync. Tests in ``test_exotic_liabilities.py``
    enforce numerical parity.

    Args:
        leg: a single ``LiabilitySpec`` (already validated)
        spots: ``(B, T+1)`` torch tensor of spot prices

    Returns:
        ``(B,)`` payoff tensor.
    """
    kind = leg.kind
    q = leg.quantity

    if kind == "call":
        return q * torch.relu(spots[:, -1] - leg.strike)
    if kind == "put":
        return q * torch.relu(leg.strike - spots[:, -1])

    if kind == "cliquet":
        reset_idx = [0, *leg.reset_dates]
        spots_at = spots[:, reset_idx]                              # (B, m+1)
        period_ret = spots_at[:, 1:] / spots_at[:, :-1] - 1.0       # (B, m)
        capped = torch.clamp(period_ret, max=leg.cap)
        summed = capped.sum(dim=1)
        floored = torch.clamp(summed, min=leg.floor)
        return q * floored

    if kind == "barrier":
        S_T = spots[:, -1]
        if leg.option_type == "call":
            european = torch.relu(S_T - leg.strike)
        else:
            european = torch.relu(leg.strike - S_T)
        if leg.barrier_type.startswith("up"):
            triggered = (spots >= leg.barrier).any(dim=1)
        else:
            triggered = (spots <= leg.barrier).any(dim=1)
        triggered_f = triggered.to(european.dtype)
        if leg.barrier_type.endswith("out"):
            alive = 1.0 - triggered_f
        else:  # *_in
            alive = triggered_f
        return q * european * alive

    if kind == "asian":
        if leg.average_type == "arithmetic":
            avg = spots.mean(dim=1)
        else:  # geometric
            avg = torch.exp(torch.log(spots).mean(dim=1))
        if leg.option_type == "call":
            intrinsic = torch.relu(avg - leg.strike)
        else:
            intrinsic = torch.relu(leg.strike - avg)
        return q * intrinsic

    if kind == "lookback":
        S_T = spots[:, -1]
        S_max = spots.max(dim=1).values
        S_min = spots.min(dim=1).values
        if leg.lookback_type == "floating":
            payoff = (S_T - S_min) if leg.option_type == "call" else (S_max - S_T)
        else:  # fixed
            if leg.option_type == "call":
                payoff = torch.relu(S_max - leg.strike)
            else:
                payoff = torch.relu(leg.strike - S_min)
        return q * payoff

    raise ValueError(f"unsupported liability kind: {kind!r}")


# ============================================================================
# Trainer
# ============================================================================


class BuehlerTrainer(BaseTrainer):
    """
    Buehler-style deep hedging trainer driven by injected market samplers.

    The trainer is decoupled from any specific simulator: pass a
    ``market_sampler(batch_size, key) -> TrajectoryBatch`` for batched
    training/evaluation. ``single_path_sampler`` is retained for adapter-level
    debugging and inference rollouts.

    Use :meth:`from_heston` for the common JAX Heston configuration.
    """

    def __init__(
        self,
        config: TrainerConfig,
        env_config: Any,
        liability: LiabilityPortfolio,
        transaction_cost_rates: np.ndarray,
        market_sampler: MarketBatchSampler,
        single_path_sampler: SinglePathSampler,
        initial_cash: float = 0.0,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("BuehlerTrainer requires PyTorch")

        self.config = config
        self.env_config = env_config
        self.liability = liability                       # raw input (single or list)
        self.liability_legs = _normalize_legs(liability) # normalized tuple of legs
        self.initial_cash = initial_cash
        self.device = device

        self.n_instruments = env_config.option_grid.n_instruments
        self.horizon = env_config.horizon_steps

        # obs_dim = spot(1) + option_features(N-1) + positions(N)
        #         + previous_trades(N) + context(3) = 3N + 3
        self.obs_dim = 3 * self.n_instruments + 3

        kind = getattr(config, "policy_kind", "mlp").lower()
        default_position_scale = (
            config.position_limit
            if kind == "mlp"
            else (config.lstm_position_limit or config.position_limit)
        )
        price_scale = _positive_scale(config.price_scale, 100.0, "price_scale")
        position_scale = _positive_scale(
            config.position_scale,
            float(default_position_scale),
            "position_scale",
        )
        price_clip = _positive_scale(config.price_clip, 10.0, "price_clip")
        if kind == "lstm":
            optimizer_kind = getattr(config, "optimizer_kind", "adam").lower()
            # KFAC needs ManualLSTMCell to hook into gate pre-activations
            # (nn.LSTMCell is a fused op and opaque to backward hooks).
            use_manual = (optimizer_kind == "kfac")
            self.policy = HedgingLSTMPolicy(
                obs_dim=self.obs_dim,
                n_instruments=self.n_instruments,
                hidden_size=config.lstm_hidden_size,
                n_blocks=config.lstm_n_blocks,
                position_limit=config.lstm_position_limit,
                last_layer_scale=config.lstm_last_layer_scale,
                price_scale=price_scale,
                position_scale=position_scale,
                price_clip=price_clip,
                use_manual_cells=use_manual,
            ).to(device)
            self._is_recurrent = True
        elif kind == "mlp":
            self.policy = HedgingMLPPolicy(
                obs_dim=self.obs_dim,
                n_instruments=self.n_instruments,
                hidden_sizes=config.hidden_sizes,
                position_limit=config.position_limit,
                last_layer_scale=config.mlp_last_layer_scale,
                price_scale=price_scale,
                position_scale=position_scale,
                price_clip=price_clip,
            ).to(device)
            self._is_recurrent = False
        else:
            raise ValueError(
                f"Unknown policy_kind={kind!r} (expected 'mlp' or 'lstm')"
            )

        optimizer_kind = getattr(config, "optimizer_kind", "adam").lower()
        if optimizer_kind == "kfac":
            from .kfac_optimizer import KFACOptimizer
            self.optimizer = KFACOptimizer(
                self.policy,
                lr=config.learning_rate,
                damping=config.kfac_damping,
                ema_decay=config.kfac_ema_decay,
                n_eigen_decomp=config.kfac_n_eigen_decomp,
                trust_region=config.kfac_trust_region,
                trust_region_decay=config.kfac_trust_region_decay,
                min_trust_region=config.kfac_min_trust_region,
                trust_region_metric=config.kfac_trust_region_metric,
                weight_decay=config.kfac_weight_decay,
            )
        elif optimizer_kind == "adam":
            self.optimizer = torch.optim.Adam(
                self.policy.parameters(), lr=config.learning_rate,
            )
        else:
            raise ValueError(
                f"Unknown optimizer_kind={optimizer_kind!r} "
                f"(expected 'adam' or 'kfac')"
            )

        self.transaction_cost_rates = torch.from_numpy(
            np.asarray(transaction_cost_rates, dtype=np.float32)
        ).to(device)

        self._market_sampler = market_sampler
        self._single_path_sampler = single_path_sampler

        from options_desk.processes._jax_backend import configure_jax_runtime

        configure_jax_runtime()
        import jax
        self._jax_key = jax.random.PRNGKey(seed)

        self.train_history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # Convenience constructor for the JAX Heston backend
    # ------------------------------------------------------------------

    @classmethod
    def from_heston(
        cls,
        config: TrainerConfig,
        env_config: Any,
        market_params: Any,
        padded_grid: Any,
        liability: LiabilityPortfolio,
        initial_spot: float = 100.0,
        initial_variance: float = 0.04,
        initial_cash: float = 0.0,
        device: str = "cpu",
        seed: int = 42,
    ) -> "BuehlerTrainer":
        """Build a trainer wired to the JAX Heston market simulator."""
        import jax

        from ..jax.env import build_transaction_cost_vector
        from ..jax.rollout import simulate_heston_market, simulate_heston_market_batch

        n_instruments = env_config.option_grid.n_instruments
        horizon = env_config.horizon_steps

        def market_sampler(batch_size: int, key: Any) -> TrajectoryBatch:
            keys = jax.random.split(key, batch_size)
            market_traj = simulate_heston_market_batch(
                config=env_config,
                market=market_params,
                padded_grid=padded_grid,
                initial_spot=initial_spot,
                initial_variance=initial_variance,
                keys=keys,
            )
            return TrajectoryBatch(
                spots=np.asarray(market_traj.spots, dtype=np.float32),
                variances=np.asarray(market_traj.variances, dtype=np.float32),
                instrument_prices=np.asarray(market_traj.instrument_prices, dtype=np.float32),
                action_masks=np.broadcast_to(
                    np.asarray(market_traj.action_masks, dtype=bool),
                    (batch_size, horizon + 1, n_instruments),
                ).copy(),
            )

        def single_path_sampler(key: Any) -> MarketTrajectory:
            return simulate_heston_market(
                config=env_config,
                market=market_params,
                padded_grid=padded_grid,
                initial_spot=initial_spot,
                initial_variance=initial_variance,
                key=key,
            )

        tc_np = build_transaction_cost_vector(env_config)
        return cls(
            config=config,
            env_config=env_config,
            liability=liability,
            transaction_cost_rates=tc_np,
            market_sampler=market_sampler,
            single_path_sampler=single_path_sampler,
            initial_cash=initial_cash,
            device=device,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------

    def _generate_batch(self) -> Dict[str, "torch.Tensor"]:
        """Generate a batch of market trajectories and convert to torch tensors."""
        import jax

        from ..utils.adapters import trajectory_batch_to_torch

        self._jax_key, subkey = jax.random.split(self._jax_key)
        batch = self._market_sampler(self.config.batch_size, subkey)
        return trajectory_batch_to_torch(batch, device=self.device)

    def _compute_liability_payoffs(
        self, trajectory: Dict[str, "torch.Tensor"],
    ) -> "torch.Tensor":
        """Sum of leg payoffs along the simulated spot path.

        Supports all liability kinds (vanilla call/put, cliquet, barrier,
        asian, lookback) and multi-leg portfolios. Each leg's payoff is
        computed by :func:`_torch_leg_payoff` and accumulated.
        """
        spots = trajectory["spots"]                                  # (B, T+1)
        total = _torch_leg_payoff(self.liability_legs[0], spots)
        for leg in self.liability_legs[1:]:
            total = total + _torch_leg_payoff(leg, spots)
        return total

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self) -> Dict[str, float]:
        """Run one training step: generate data, rollout, loss, gradient."""
        self.policy.train()

        trajectory = self._generate_batch()
        _assert_finite_trajectory(trajectory)
        liability_payoffs = self._compute_liability_payoffs(trajectory)
        _assert_finite_tensor("liability_payoffs", liability_payoffs)

        rollout_fn = (
            differentiable_rollout_recurrent
            if self._is_recurrent
            else differentiable_rollout
        )
        terminal_pnl, total_costs = rollout_fn(
            policy=self.policy,
            trajectory=trajectory,
            transaction_cost_rates=self.transaction_cost_rates,
            horizon=self.horizon,
            n_instruments=self.n_instruments,
        )

        terminal_pnl = terminal_pnl + self.initial_cash
        _assert_finite_tensor("terminal_pnl", terminal_pnl)
        _assert_finite_tensor("total_costs", total_costs)

        loss, metrics = buehler_loss(
            terminal_pnl=terminal_pnl,
            liability_payoffs=liability_payoffs,
            total_costs=total_costs,
            risk_aversion=self.config.risk_aversion,
        )
        _assert_finite_tensor("loss", loss)

        self.optimizer.zero_grad()
        loss.backward()
        # Adam needs explicit grad clipping; KFAC's trust_region handles
        # update-norm bounding internally (clipping pre-precondition would
        # interfere with the curvature estimate).
        is_kfac = getattr(self.config, "optimizer_kind", "adam").lower() == "kfac"
        if self.config.grad_clip > 0 and not is_kfac:
            nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.config.grad_clip,
            )
        self.optimizer.step()
        if is_kfac:
            trust_region = getattr(self.optimizer, "trust_region", None)
            if trust_region is not None:
                metrics["kfac_trust_region"] = float(trust_region)
            metrics["kfac_damping"] = float(getattr(self.optimizer, "damping"))
            metrics["kfac_lr"] = float(getattr(self.optimizer, "lr"))
            for key, value in getattr(self.optimizer, "last_step_stats", {}).items():
                metrics[f"kfac_{key}"] = float(value)

        return metrics

    def train(self, progress: bool = True) -> List[Dict[str, float]]:
        """Run the full training loop.

        Args:
            progress: If True (default) and ``tqdm`` is installed, show a live
                progress bar with per-epoch metrics in the postfix. Periodic
                summary lines (every ``eval_every`` epochs) are printed via
                ``tqdm.write`` so they do not collide with the bar. When
                ``tqdm`` is unavailable or ``progress=False``, falls back to
                the original ``logging``-only behavior.
        """
        logger.info(
            "Starting training: %d epochs, batch_size=%d, lr=%.1e, "
            "risk_aversion=%.0f, horizon=%d, n_instruments=%d",
            self.config.n_epochs,
            self.config.batch_size,
            self.config.learning_rate,
            self.config.risk_aversion,
            self.horizon,
            self.n_instruments,
        )

        use_tqdm = bool(progress) and _TQDM_AVAILABLE
        epoch_range = range(1, self.config.n_epochs + 1)
        iterator = (
            tqdm(epoch_range, desc="train", dynamic_ncols=True, leave=True)
            if use_tqdm
            else epoch_range
        )

        for epoch in iterator:
            metrics = self.train_step()
            metrics["epoch"] = epoch
            self.train_history.append(metrics)

            if use_tqdm:
                iterator.set_postfix(
                    {
                        "loss": f"{metrics['total_loss']:.3f}",
                        "var": f"{metrics['variance_term']:.3f}",
                        "cost": f"{metrics['cost_term']:.4f}",
                        "err_std": f"{metrics['std_hedging_error']:.3f}",
                        "err_mean": f"{metrics['mean_hedging_error']:+.3f}",
                    },
                    refresh=False,
                )

            if epoch % self.config.eval_every == 0 or epoch == 1:
                summary = (
                    f"Epoch {epoch:>{len(str(self.config.n_epochs))}}/"
                    f"{self.config.n_epochs}  "
                    f"loss={metrics['total_loss']:.4f}  "
                    f"var={metrics['variance_term']:.4f}  "
                    f"cost={metrics['cost_term']:.6f}  "
                    f"hedging_err={metrics['mean_hedging_error']:.4f} "
                    f"+/- {metrics['std_hedging_error']:.4f}"
                )
                if use_tqdm:
                    tqdm.write(summary)
                else:
                    logger.info(summary)

        logger.info("Training complete.")
        return self.train_history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        n_paths: int = 100,
        seed: int = 9999,
    ) -> Dict[str, float]:
        """Evaluate the trained policy on fresh batched market trajectories."""
        if n_paths <= 0:
            raise ValueError(f"n_paths must be positive, got {n_paths}")

        import jax

        from ..utils.adapters import trajectory_batch_to_torch

        was_training = self.policy.training
        self.policy.eval()
        try:
            key = jax.random.PRNGKey(seed)
            batch = self._market_sampler(n_paths, key)
            trajectory = trajectory_batch_to_torch(batch, device=self.device)
            _assert_finite_trajectory(trajectory)

            rollout_fn = (
                differentiable_rollout_recurrent
                if self._is_recurrent
                else differentiable_rollout
            )
            with torch.no_grad():
                liability_payoffs = self._compute_liability_payoffs(trajectory)
                _assert_finite_tensor("eval_liability_payoffs", liability_payoffs)

                terminal_pnl, total_costs = rollout_fn(
                    policy=self.policy,
                    trajectory=trajectory,
                    transaction_cost_rates=self.transaction_cost_rates,
                    horizon=self.horizon,
                    n_instruments=self.n_instruments,
                )
                terminal_pnl = terminal_pnl + self.initial_cash
                _assert_finite_tensor("eval_terminal_pnl", terminal_pnl)
                _assert_finite_tensor("eval_total_costs", total_costs)

                hedging_errors = terminal_pnl - liability_payoffs
                zero_terminal_pnl = torch.full_like(
                    liability_payoffs, float(self.initial_cash)
                )
                zero_errors = zero_terminal_pnl - liability_payoffs
                eval_loss = (
                    self.config.risk_aversion * _safe_sample_var(hedging_errors)
                    + total_costs.mean()
                )

                _assert_finite_tensor("eval_hedging_errors", hedging_errors)
                _assert_finite_tensor("eval_loss", eval_loss)

                metrics = {
                    "n_paths": int(n_paths),
                    "eval_loss": _scalar(eval_loss),
                    "mean_pnl": _scalar(terminal_pnl.mean()),
                    "std_pnl": _scalar(_safe_population_std(terminal_pnl)),
                    "mean_reward": _scalar(
                        (hedging_errors - float(self.initial_cash)).mean()
                    ),
                    "mean_cost": _scalar(total_costs.mean()),
                    "std_cost": _scalar(_safe_population_std(total_costs)),
                    "mean_liability_payoff": _scalar(liability_payoffs.mean()),
                    "std_liability_payoff": _scalar(
                        _safe_population_std(liability_payoffs)
                    ),
                }
                metrics.update(_error_distribution_metrics(hedging_errors))
                metrics.update(_error_distribution_metrics(zero_errors, "zero_hedge_"))
                metrics["std_improvement_vs_zero"] = (
                    metrics["zero_hedge_std_hedging_error"]
                    - metrics["std_hedging_error"]
                )
                metrics["rmse_improvement_vs_zero"] = (
                    metrics["zero_hedge_rmse_hedging_error"]
                    - metrics["rmse_hedging_error"]
                )

            logger.info(
                "Evaluation (%d paths): hedging_error=%.4f +/- %.4f  "
                "zero_std=%.4f  mean_cost=%.6f  eval_loss=%.4f",
                n_paths,
                metrics["mean_hedging_error"],
                metrics["std_hedging_error"],
                metrics["zero_hedge_std_hedging_error"],
                metrics["mean_cost"],
                metrics["eval_loss"],
            )
            return metrics
        finally:
            self.policy.train(was_training)

    # ------------------------------------------------------------------
    # Inference adapter + checkpointing
    # ------------------------------------------------------------------

    def get_agent(self) -> BaseHedgingAgent:
        """Return a :class:`TorchPolicyAgent` wrapping the current policy."""
        return TorchPolicyAgent(self.policy)

    def save_checkpoint(self, path: str) -> None:
        """Save policy weights and optimizer state."""
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "obs_dim": self.obs_dim,
                "n_instruments": self.n_instruments,
                "train_history": self.train_history,
                "policy_kind": "lstm" if self._is_recurrent else "mlp",
            },
            path,
        )
        logger.info("Checkpoint saved to %s", path)

    def load_checkpoint(self, path: str) -> None:
        """Load policy weights and optimizer state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_history = checkpoint.get("train_history", [])
        logger.info("Checkpoint loaded from %s", path)
