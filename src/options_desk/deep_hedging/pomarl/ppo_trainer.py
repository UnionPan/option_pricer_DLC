"""
:class:`PPOTrainer` — recurrent PPO on the JAX POMARL hedging environment.

The strong cumulative-reward RL baseline. Shares the AIS GRU encoder and the
tanh-Gaussian policy with :class:`AISPGTrainer`, adds a critic ``V(x̂)``, and
optimizes the clipped PPO surrogate + GAE instead of REINFORCE / pathwise.

Design (see :mod:`pomarl.ppo` for the rationale):

    1. Collect one on-policy batch with the stochastic rollout — reusing the
       exact reward shaping (mean_pnl / hedging_shaped / local_risk / …) of
       the AIS trainer, so the objective is identical and the comparison is
       apples-to-apples.
    2. Critic the collected ``x_hat_seq`` → V_old; GAE → advantages, returns.
    3. K PPO epochs over minibatches of *paths* (full sequences kept, so the
       GRU encoder is replayed and trained end-to-end through actor+critic).
       No AIS auxiliary losses — the encoder learns from the RL signal alone.

Evaluation reproduces the Buehler eval schema (``std_hedging_error`` etc.) so
PPO slots directly into the LSTM / BS-delta comparison table.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..utils.contracts import (
    LiabilityPortfolio,
    MarketTrajectory,
    TrajectoryBatch,
    _normalize_legs,
)
from ..utils.eval_baseline import _error_distribution_metrics
from .ais import AISGRUEncoder
from .policy import GaussianPolicy
from .ppo import ValueCritic, compute_gae, ppo_loss, replay_forward
from .rollout import pomarl_rollout
from .utils import pomdp_obs_dim

logger = logging.getLogger(__name__)

MarketBatchSampler = Callable[[int, Any], TrajectoryBatch]


@dataclass(frozen=True)
class PPOTrainerConfig:
    """Configuration for :class:`PPOTrainer`."""

    ais_hidden_size: int = 64
    ais_n_layers: int = 1
    policy_hidden_size: int = 64
    critic_hidden_size: int = 64
    position_limit: float = 1.5
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    learning_rate: float = 3e-4
    batch_size: int = 1024
    n_epochs: int = 300            # number of collection iterations
    eval_every: int = 30
    grad_clip: float = 1.0

    # PPO hyperparameters
    discount: float = 1.0          # finite horizon → no discounting by default
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ppo_epochs: int = 4            # K updates per collected batch
    n_minibatches: int = 4
    value_coef: float = 0.5
    entropy_coef: float = 0.0

    instrument_mask: tuple[int, ...] | None = None
    risk_aversion: float = 1_000.0     # eval-loss γ (matches env)
    reward_kind: str = "hedging_shaped"
    dt: float = 1.0 / 252.0


class PPOTrainer:
    """Recurrent PPO trainer for the POMARL hedging environment."""

    def __init__(
        self,
        config: PPOTrainerConfig,
        env_config: Any,
        liability: LiabilityPortfolio,
        transaction_cost_rates: np.ndarray,
        market_sampler: MarketBatchSampler,
        initial_cash: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.config = config
        self.env_config = env_config
        self.liability = liability
        self.liability_legs = tuple(_normalize_legs(liability))
        self.initial_cash = float(initial_cash)

        self.n_instruments = env_config.option_grid.n_instruments
        self.horizon = env_config.horizon_steps
        self.obs_dim = pomdp_obs_dim(self.n_instruments)

        self._market_sampler = market_sampler
        self.transaction_cost_rates = jnp.asarray(
            np.asarray(transaction_cost_rates, dtype=np.float32),
        )

        if config.instrument_mask is not None:
            mask_arr = np.asarray(config.instrument_mask, dtype=np.float32)
            if mask_arr.shape != (self.n_instruments,):
                raise ValueError(
                    f"instrument_mask length {mask_arr.shape[0]} != "
                    f"n_instruments {self.n_instruments}"
                )
            self._instrument_mask_static = jnp.asarray(mask_arr)
        else:
            self._instrument_mask_static = None

        self._encoder = AISGRUEncoder(
            hidden_size=config.ais_hidden_size, n_layers=config.ais_n_layers,
        )
        self._policy = GaussianPolicy(
            n_instruments=self.n_instruments,
            hidden_size=config.policy_hidden_size,
            log_std_min=config.log_std_min,
            log_std_max=config.log_std_max,
        )
        self._critic = ValueCritic(hidden_size=config.critic_hidden_size)

        self._rng_key = jax.random.PRNGKey(seed)
        self._init_params()
        self._init_optimizer()
        self.train_history: List[Dict[str, float]] = []
        self._build_jit_fns()

    # ------------------------------------------------------------------
    def _init_params(self) -> None:
        key_enc, key_pol, key_crit, self._rng_key = jax.random.split(
            self._rng_key, 4,
        )
        dummy_obs = jnp.zeros((1, self.obs_dim), dtype=jnp.float32)
        dummy_hidden = AISGRUEncoder.init_hidden(
            1, self.config.ais_hidden_size, self.config.ais_n_layers,
        )
        dummy_x_hat = jnp.zeros((1, self.config.ais_hidden_size),
                                dtype=jnp.float32)
        dummy_mask = jnp.ones((1, self.n_instruments), dtype=jnp.float32)

        encoder_params = self._encoder.init(key_enc, dummy_obs, dummy_hidden)
        policy_params = self._policy.init(key_pol, dummy_x_hat, dummy_mask)
        critic_params = self._critic.init(key_crit, dummy_x_hat)
        self.params = {
            "encoder": encoder_params,
            "policy": policy_params,
            "critic": critic_params,
        }

    def _init_optimizer(self) -> None:
        self._opt = optax.chain(
            optax.clip_by_global_norm(self.config.grad_clip),
            optax.adam(self.config.learning_rate),
        )
        self._opt_state = self._opt.init(self.params)

    # ------------------------------------------------------------------
    @classmethod
    def from_heston(
        cls,
        config: PPOTrainerConfig,
        env_config: Any,
        market_params: Any,
        padded_grid: Any,
        liability: LiabilityPortfolio,
        initial_spot: float = 100.0,
        initial_variance: float = 0.04,
        initial_cash: float = 0.0,
        seed: int = 42,
    ) -> "PPOTrainer":
        from ..jax.env import build_transaction_cost_vector
        from ..jax.rollout import simulate_heston_market_batch

        n_instruments = env_config.option_grid.n_instruments
        horizon = env_config.horizon_steps

        def market_sampler(batch_size: int, key: Any) -> TrajectoryBatch:
            keys = jax.random.split(key, batch_size)
            market_traj = simulate_heston_market_batch(
                config=env_config, market=market_params,
                padded_grid=padded_grid, initial_spot=initial_spot,
                initial_variance=initial_variance, keys=keys,
            )
            return TrajectoryBatch(
                spots=np.asarray(market_traj.spots, dtype=np.float32),
                variances=np.asarray(market_traj.variances, dtype=np.float32),
                instrument_prices=np.asarray(
                    market_traj.instrument_prices, dtype=np.float32,
                ),
                action_masks=np.broadcast_to(
                    np.asarray(market_traj.action_masks, dtype=bool),
                    (batch_size, horizon + 1, n_instruments),
                ).copy(),
            )

        tc = build_transaction_cost_vector(env_config)
        return cls(
            config=config, env_config=env_config, liability=liability,
            transaction_cost_rates=tc, market_sampler=market_sampler,
            initial_cash=initial_cash, seed=seed,
        )

    # ------------------------------------------------------------------
    def _build_jit_fns(self) -> None:
        encoder_apply = self._encoder.apply
        policy_apply = self._policy.apply
        critic_apply = self._critic.apply

        horizon = self.horizon
        n_inst = self.n_instruments
        hidden_size = self.config.ais_hidden_size
        n_layers = self.config.ais_n_layers
        position_limit = self.config.position_limit
        liability_legs = self.liability_legs
        initial_cash = self.initial_cash
        instrument_mask = self._instrument_mask_static
        tc_rates = self.transaction_cost_rates
        reward_kind = self.config.reward_kind
        risk_aversion = self.config.risk_aversion
        env_dt = getattr(self.env_config, "dt", self.config.dt)
        discount = self.config.discount
        gae_lambda = self.config.gae_lambda
        clip_eps = self.config.clip_eps
        value_coef = self.config.value_coef
        entropy_coef = self.config.entropy_coef

        def _rollout(params, market, key, sample):
            spots, prices, masks, variances = market
            return pomarl_rollout(
                encoder_apply=encoder_apply, policy_apply=policy_apply,
                encoder_params=params["encoder"],
                policy_params=params["policy"],
                spots=spots, prices=prices, masks=masks,
                transaction_cost_rates=tc_rates,
                instrument_mask_static=instrument_mask,
                liability_legs=liability_legs, initial_cash=initial_cash,
                position_limit=position_limit, horizon=horizon,
                n_instruments=n_inst, encoder_hidden_size=hidden_size,
                encoder_n_layers=n_layers, sample=sample, key=key,
                reward_kind=reward_kind, risk_aversion=risk_aversion,
                variances=variances, dt=env_dt,
            )

        # ── Collection: rollout + critic V_old + GAE ──────────────────────
        def collect_pure(params, market, key):
            out = _rollout(params, market, key, True)
            # Reward normalization: scale rewards by the batch std of total
            # per-path return so that returns / values / value-loss are all
            # O(1). Without this, squared-error rewards (e.g. the terminal
            # (V_T−payoff)² in 'hedging_shaped', huge early in training) make
            # the value loss dominate the *shared-encoder* gradient and the
            # global-norm clip starves the policy. Policy-invariant: the
            # advantage is re-normalized in the loss, so scaling only fixes
            # the critic/value-loss magnitude. Eval metrics use terminal_pnl
            # directly and are unaffected.
            returns_per_path = out.rewards.sum(axis=0)            # (B,)
            reward_scale = jnp.std(returns_per_path) + 1e-6
            scaled_rewards = out.rewards / reward_scale
            values_old = critic_apply(params["critic"], out.x_hat_seq)  # (T+1,B)
            advantages, returns = compute_gae(
                scaled_rewards, values_old, discount, gae_lambda,
            )
            return {
                "actions": out.actions,            # (T,B,N)
                "old_log_probs": out.log_probs,    # (T,B)
                "advantages": advantages,          # (T,B)
                "returns": returns,                # (T,B)
                "old_values": values_old,          # (T+1,B)
                "reward_scale": reward_scale,      # scalar
                "terminal_pnl": out.terminal_pnl,  # (B,)
                "payoff": out.payoff,              # (B,)
                "total_costs": out.total_costs,    # (B,)
            }

        self._collect_jit = jax.jit(collect_pure)

        # ── PPO update on one minibatch of paths ──────────────────────────
        def ppo_loss_fn(params, mb):
            new_log_probs, entropies, new_values = replay_forward(
                encoder_apply=encoder_apply, policy_apply=policy_apply,
                critic_apply=critic_apply,
                encoder_params=params["encoder"],
                policy_params=params["policy"],
                critic_params=params["critic"],
                spots=mb["spots"], prices=mb["prices"], masks=mb["masks"],
                actions=mb["actions"],
                instrument_mask_static=instrument_mask,
                position_limit=position_limit, horizon=horizon,
                n_instruments=n_inst, encoder_hidden_size=hidden_size,
                encoder_n_layers=n_layers,
            )
            loss, info = ppo_loss(
                new_log_probs=new_log_probs,
                old_log_probs=mb["old_log_probs"],
                advantages=mb["advantages"], returns=mb["returns"],
                new_values=new_values, old_values=mb["old_values"],
                entropies=entropies, clip_eps=clip_eps,
                value_coef=value_coef, entropy_coef=entropy_coef,
            )
            return loss, info

        loss_grad_fn = jax.value_and_grad(ppo_loss_fn, has_aux=True)

        def update_step_pure(params, opt_state, mb):
            (loss, info), grad = loss_grad_fn(params, mb)
            updates, opt_state = self._opt.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            metrics = {
                "loss": loss,
                "policy_loss": info.policy_loss,
                "value_loss": info.value_loss,
                "entropy": info.entropy,
                "approx_kl": info.approx_kl,
                "clip_frac": info.clip_frac,
            }
            return params, opt_state, metrics

        self._update_jit = jax.jit(update_step_pure)
        self._eval_jit = jax.jit(
            lambda params, market, key: _rollout(params, market, key, False),
        )

    # ------------------------------------------------------------------
    def _sample_market_jax(self, batch_size: int, key) -> tuple:
        batch = self._market_sampler(batch_size, key)
        spots = jnp.asarray(batch.spots, dtype=jnp.float32)
        prices = jnp.asarray(batch.instrument_prices, dtype=jnp.float32)
        variances = jnp.asarray(batch.variances, dtype=jnp.float32)
        masks_np = np.asarray(batch.action_masks)
        if masks_np.ndim == 3:
            masks_np = masks_np[0]
        masks = jnp.asarray(masks_np, dtype=jnp.float32)
        return spots, prices, masks, variances

    # ------------------------------------------------------------------
    def train_step(self) -> Dict[str, float]:
        self._rng_key, key_market, key_roll, key_perm = jax.random.split(
            self._rng_key, 4,
        )
        market = self._sample_market_jax(self.config.batch_size, key_market)
        spots, prices, masks, variances = market

        batch = self._collect_jit(self.params, market, key_roll)

        B = self.config.batch_size
        n_mb = self.config.n_minibatches
        mb_size = B // n_mb
        if mb_size * n_mb != B:
            raise ValueError(
                f"batch_size {B} not divisible by n_minibatches {n_mb}"
            )

        last: Dict[str, float] = {}
        for _ in range(self.config.ppo_epochs):
            self._rng_key, key_perm = jax.random.split(self._rng_key)
            perm = jax.random.permutation(key_perm, B)
            for mb in range(n_mb):
                idx = perm[mb * mb_size:(mb + 1) * mb_size]
                mb_data = {
                    # path axis 0
                    "spots": spots[idx],
                    "prices": prices[idx],
                    "masks": masks,
                    # path axis 1
                    "actions": batch["actions"][:, idx],
                    "old_log_probs": batch["old_log_probs"][:, idx],
                    "advantages": batch["advantages"][:, idx],
                    "returns": batch["returns"][:, idx],
                    "old_values": batch["old_values"][:, idx],
                }
                self.params, self._opt_state, metrics = self._update_jit(
                    self.params, self._opt_state, mb_data,
                )
                last = metrics

        # Collection-level diagnostics (on the freshly collected batch).
        error = batch["terminal_pnl"] - batch["payoff"]
        out = {k: float(jax.device_get(v)) for k, v in last.items()}
        out["mean_return"] = float(jax.device_get(
            batch["returns"][0].mean()
        ))
        out["std_error"] = float(np.std(np.asarray(error), ddof=1))
        out["mean_cost"] = float(jax.device_get(batch["total_costs"].mean()))
        return out

    # ------------------------------------------------------------------
    def evaluate(
        self, n_paths: int = 4096, seed: int = 9999, sample: bool = False,
    ) -> Dict[str, float]:
        if n_paths <= 0:
            raise ValueError(f"n_paths must be positive, got {n_paths}")
        key = jax.random.PRNGKey(seed)
        key_market, key_roll = jax.random.split(key)
        market = self._sample_market_jax(n_paths, key_market)
        out = self._eval_jit(self.params, market, key_roll)

        terminal_pnl = np.asarray(out.terminal_pnl, dtype=np.float64)
        total_costs = np.asarray(out.total_costs, dtype=np.float64)
        payoff = np.asarray(out.payoff, dtype=np.float64)
        rewards = np.asarray(out.rewards, dtype=np.float64)

        hedging_errors = terminal_pnl - payoff
        zero_errors = float(self.initial_cash) - payoff
        eval_loss = (
            self.config.risk_aversion * float(np.var(hedging_errors, ddof=1))
            + float(total_costs.mean())
        )
        metrics: Dict[str, float] = {
            "n_paths": int(n_paths),
            "eval_loss": eval_loss,
            "mean_pnl": float(terminal_pnl.mean()),
            "std_pnl": float(terminal_pnl.std(ddof=0)),
            "mean_cost": float(total_costs.mean()),
            "std_cost": float(total_costs.std(ddof=0)),
            "mean_liability_payoff": float(payoff.mean()),
        }
        metrics.update(_error_distribution_metrics(hedging_errors))
        metrics.update(_error_distribution_metrics(zero_errors, "zero_hedge_"))
        metrics["pomarl_mean_return"] = float(rewards.sum(axis=0).mean())
        return metrics

    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str) -> None:
        payload = {
            "params": jax.device_get(self.params),
            "opt_state": jax.device_get(self._opt_state),
            "config": self.config,
            "obs_dim": self.obs_dim,
            "n_instruments": self.n_instruments,
            "train_history": self.train_history,
        }
        with open(path, "wb") as fp:
            pickle.dump(payload, fp)
        logger.info("PPO checkpoint saved to %s", path)

    def load_checkpoint(self, path: str) -> None:
        with open(path, "rb") as fp:
            payload = pickle.load(fp)
        self.params = payload["params"]
        self._opt_state = payload["opt_state"]
        self.train_history = payload.get("train_history", [])
        logger.info("PPO checkpoint loaded from %s", path)
