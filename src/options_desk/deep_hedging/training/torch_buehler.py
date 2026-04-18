"""
Buehler-style deep hedging trainer (PyTorch).

The trainer consumes two market-data callables -- one for batches of
trajectories (training) and one for single paths (evaluation) -- so it
is decoupled from any specific simulator backend. A
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
from ..utils.contracts import LiabilitySpec, MarketTrajectory, TrajectoryBatch
from .base import BaseTrainer

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is an optional dep
    _TORCH_AVAILABLE = False


MarketBatchSampler = Callable[[int, Any], TrajectoryBatch]
SinglePathSampler = Callable[[Any], MarketTrajectory]


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class TrainerConfig:
    """Configuration for the Buehler deep hedging trainer."""

    hidden_sizes: tuple[int, ...] = (64, 64)
    position_limit: float = 100.0
    risk_aversion: float = 1_000.0
    learning_rate: float = 1e-3
    batch_size: int = 1024
    n_epochs: int = 200
    eval_every: int = 20
    grad_clip: float = 1.0


# ============================================================================
# Differentiable rollout + Buehler loss
# ============================================================================


def _build_batch_obs_tensor(
    trajectory: Dict[str, "torch.Tensor"],
    positions: "torch.Tensor",
    previous_trades: "torch.Tensor",
    time_index: int,
    horizon: int,
    n_instruments: int,
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

    spot = trajectory["spots"][:, time_index].unsqueeze(1)
    option_features = trajectory["instrument_prices"][:, time_index, 1:]
    variance = trajectory["variances"][:, time_index].unsqueeze(1)
    time_feat = torch.full((B, 1), time_ratio, dtype=torch.float32, device=device)
    bias_feat = torch.ones(B, 1, dtype=torch.float32, device=device)
    context = torch.cat([variance, time_feat, bias_feat], dim=1)

    obs = torch.cat([spot, option_features, positions, previous_trades, context], dim=1)
    mask = trajectory["action_masks"][:, time_index].float()
    return obs, mask


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

    for t in range(horizon):
        obs, mask = _build_batch_obs_tensor(
            trajectory, positions, previous_trades, t, horizon, n_instruments,
        )
        trades = policy(obs, mask)

        prices_t = trajectory["instrument_prices"][:, t]
        notional = (trades * prices_t).sum(dim=1)
        step_cost = (transaction_cost_rates.unsqueeze(0) * trades.abs()).sum(dim=1)

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


# ============================================================================
# Trainer
# ============================================================================


class BuehlerTrainer(BaseTrainer):
    """
    Buehler-style deep hedging trainer driven by injected market samplers.

    The trainer is decoupled from any specific simulator: pass a
    ``market_sampler(batch_size, key) -> TrajectoryBatch`` for training
    and a ``single_path_sampler(key) -> MarketTrajectory`` for evaluation.

    Use :meth:`from_heston` for the common JAX Heston configuration.
    """

    def __init__(
        self,
        config: TrainerConfig,
        env_config: Any,
        liability: LiabilitySpec,
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
        self.liability = liability
        self.initial_cash = initial_cash
        self.device = device

        self.n_instruments = env_config.option_grid.n_instruments
        self.horizon = env_config.horizon_steps

        # obs_dim = spot(1) + option_features(N-1) + positions(N)
        #         + previous_trades(N) + context(3) = 3N + 3
        self.obs_dim = 3 * self.n_instruments + 3

        self.policy = HedgingMLPPolicy(
            obs_dim=self.obs_dim,
            n_instruments=self.n_instruments,
            hidden_sizes=config.hidden_sizes,
            position_limit=config.position_limit,
        ).to(device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.learning_rate,
        )

        self.transaction_cost_rates = torch.from_numpy(
            np.asarray(transaction_cost_rates, dtype=np.float32)
        ).to(device)

        self._market_sampler = market_sampler
        self._single_path_sampler = single_path_sampler

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
        liability: LiabilitySpec,
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
        """Compute terminal liability payoffs from spot prices."""
        terminal_spots = trajectory["spots"][:, -1]
        strike = self.liability.strike
        quantity = self.liability.quantity

        if self.liability.kind == "call":
            return quantity * torch.relu(terminal_spots - strike)
        if self.liability.kind == "put":
            return quantity * torch.relu(strike - terminal_spots)
        raise ValueError(f"unsupported liability kind: {self.liability.kind}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self) -> Dict[str, float]:
        """Run one training step: generate data, rollout, loss, gradient."""
        self.policy.train()

        trajectory = self._generate_batch()
        liability_payoffs = self._compute_liability_payoffs(trajectory)

        terminal_pnl, total_costs = differentiable_rollout(
            policy=self.policy,
            trajectory=trajectory,
            transaction_cost_rates=self.transaction_cost_rates,
            horizon=self.horizon,
            n_instruments=self.n_instruments,
        )

        terminal_pnl = terminal_pnl + self.initial_cash

        loss, metrics = buehler_loss(
            terminal_pnl=terminal_pnl,
            liability_payoffs=liability_payoffs,
            total_costs=total_costs,
            risk_aversion=self.config.risk_aversion,
        )

        self.optimizer.zero_grad()
        loss.backward()
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.config.grad_clip,
            )
        self.optimizer.step()

        return metrics

    def train(self) -> List[Dict[str, float]]:
        """Run the full training loop."""
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

        for epoch in range(1, self.config.n_epochs + 1):
            metrics = self.train_step()
            metrics["epoch"] = epoch
            self.train_history.append(metrics)

            if epoch % self.config.eval_every == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d/%d  loss=%.4f  var=%.4f  cost=%.6f  "
                    "hedging_err=%.4f +/- %.4f",
                    epoch,
                    self.config.n_epochs,
                    metrics["total_loss"],
                    metrics["variance_term"],
                    metrics["cost_term"],
                    metrics["mean_hedging_error"],
                    metrics["std_hedging_error"],
                )

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
        """Evaluate the trained policy via ``collect_agent_rollout_from_market``."""
        import jax

        from ..utils.rollouts import collect_agent_rollout_from_market

        self.policy.eval()
        agent = self.get_agent()

        key = jax.random.PRNGKey(seed)
        pnls: list[float] = []
        hedging_errors: list[float] = []
        total_rewards: list[float] = []

        for _ in range(n_paths):
            key, subkey = jax.random.split(key)
            traj = self._single_path_sampler(subkey)
            batch = collect_agent_rollout_from_market(
                agent=agent,
                config=self.env_config,
                trajectory=traj,
                liability=self.liability,
                initial_cash=self.initial_cash,
                action_mode="trades",
                update_agent=False,
            )

            terminal_pnl = float(batch.portfolio_values[0, -1])
            liability_payoff = float(batch.terminal_liability_payoffs[0])
            pnls.append(terminal_pnl)
            hedging_errors.append(terminal_pnl - liability_payoff)
            total_rewards.append(float(batch.rewards.sum()))

        pnls_arr = np.array(pnls)
        errors_arr = np.array(hedging_errors)
        rewards_arr = np.array(total_rewards)

        metrics = {
            "mean_pnl": float(pnls_arr.mean()),
            "std_pnl": float(pnls_arr.std()),
            "mean_hedging_error": float(errors_arr.mean()),
            "std_hedging_error": float(errors_arr.std()),
            "mean_reward": float(rewards_arr.mean()),
            "n_paths": n_paths,
        }

        logger.info(
            "Evaluation (%d paths): hedging_error=%.4f +/- %.4f  "
            "mean_pnl=%.4f  mean_reward=%.4f",
            n_paths,
            metrics["mean_hedging_error"],
            metrics["std_hedging_error"],
            metrics["mean_pnl"],
            metrics["mean_reward"],
        )
        return metrics

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
