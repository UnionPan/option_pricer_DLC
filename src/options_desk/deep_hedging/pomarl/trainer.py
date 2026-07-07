"""
:class:`AISPGTrainer` — JAX-native realisation of Algorithm 1 from pomarl.tex.

Wires the AIS encoder, the tanh-Gaussian policy, the reward / transition
predictors, and two optax optimizers into a :class:`BaseTrainer` that

    * uses the existing JAX Heston simulator (via :meth:`from_heston`),
    * trains the policy by REINFORCE on the per-step PnL reward, and
    * trains the AIS encoder + reward/transition models on the same
      on-policy batch (Algorithm 1 ``single-loop`` updates).

The Buehler eval schema (``eval_loss``, ``std_hedging_error`` etc.) is
reproduced in :meth:`evaluate` so POMARL results are directly comparable to
the LSTM / BS-delta baselines on the same ``eval_seed``.
"""

from __future__ import annotations

import logging
import math
import pickle
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..agents.base import BaseHedgingAgent
from ..utils.contracts import (
    LiabilityPortfolio,
    MarketTrajectory,
    TrajectoryBatch,
    _normalize_legs,
)
from ..utils.eval_baseline import _error_distribution_metrics
from .agent import PomarlAgent
from .ais import AISGRUEncoder, AISRewardModel, AISTransitionModel
from .losses import ais_reward_loss, ais_transition_loss, reinforce_loss
from .policy import GaussianPolicy
from .rollout import pomarl_rollout
from .utils import pomdp_obs_dim

logger = logging.getLogger(__name__)

MarketBatchSampler = Callable[[int, Any], TrajectoryBatch]
SinglePathSampler = Callable[[Any], MarketTrajectory]


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class AISPGTrainerConfig:
    """Configuration for :class:`AISPGTrainer`."""

    ais_hidden_size: int = 64
    ais_n_layers: int = 1
    policy_hidden_size: int = 64
    position_limit: float = 1.5
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    policy_lr: float = 3e-4
    ais_lr: float = 3e-4
    batch_size: int = 1024
    n_epochs: int = 200
    eval_every: int = 20
    grad_clip: float = 1.0
    discount: float = 1.0
    baseline_kind: str = "batch_mean"
    entropy_coef: float = 0.0
    lambda_reward: float = 1.0
    lambda_transition: float = 1.0
    instrument_mask: tuple[int, ...] | None = None
    risk_aversion: float = 1_000.0
    # ── Reward shaping ───────────────────────────────────────────────────
    # 'mean_pnl'        — original: per-step r = ΔV, terminal r -= payoff
    #                     → maximize E[hedging_error] (alpha-harvesting)
    # 'hedging_mse'     — Föllmer-Schweizer minimum-variance hedging:
    #                     per-step r = -cost, terminal r = -γ·(V_T − payoff)²
    #                     → minimize E[cost] + γ·MSE(hedging_error).
    #                     SPIKY terminal reward → high-variance encoder grad.
    # 'hedging_var'     — variance only at the terminal (no mean penalty),
    #                     centered on batch mean.
    # 'hedging_shaped'  — Ng-Russell potential-function shaping. Per-step
    #                     reward = -cost + (Φ_t − Φ_{t-1}) where
    #                     Φ_t = -γ·(V_t − L_t)², L_t = BS-priced liability.
    #                     Telescopes to (V_T−payoff)² = MSE (mean-inclusive).
    # 'local_risk'      — Föllmer-Schweizer / local risk minimization via the
    #                     Doob decomposition. r_t = -γ·(Δe_t)² - cost_t with
    #                     Δe_t = ΔV_hedge_t − ΔL_t. SUM of squared increments
    #                     (not square of the sum) = Var(error) exactly under
    #                     the Q-martingale property. This is the rigorous
    #                     cumulative-reward form of γ·Var + E[cost] — needs
    #                     trajectory (on-policy) structure to be valid.
    reward_kind: str = "mean_pnl"
    dt: float = 1.0 / 252.0
    discrete_bucket_size: float = 0.0  # round trades to lot size (non-diff)

    # ── Policy gradient estimator ────────────────────────────────────────
    # 'reinforce'  — score-function: ∇J = E[Σ ∇log π(a) · R]. High variance.
    # 'pathwise'   — SVG-style: backprop through reparameterized policy
    #                into differentiable sim. Loss = -E[Σ r].
    # 'buehler'    — direct supervised-style loss γ·Var(V_T-payoff) + E[cost],
    #                exactly as in BuehlerTrainer. Differs from 'pathwise':
    #                (a) uses Buehler variance form (NOT reward-based)
    #                (b) gradient through encoder is OPTIONAL via
    #                    encoder_gradient_from_policy. When True and
    #                    disable_ais_aux_losses=True, the trainer becomes
    #                    a "POMARL-architecture-with-Buehler-training"
    #                    hybrid that lets us test if the RL framework
    #                    itself (rather than the encoder) is the bottleneck.
    policy_method: str = "reinforce"

    # When True, drops jax.lax.stop_gradient between encoder output and
    # policy, letting policy loss flow back into the encoder. Standard AIS
    # design says no — encoder is task-agnostic. But probe shows AIS
    # transition loss is essentially a no-op (R²=0.06), so the "task-
    # agnostic encoder" theory doesn't operate as advertised. Try True
    # with policy_method='buehler' to mimic the LSTM trainer's end-to-end
    # gradient flow on top of the AIS encoder architecture.
    encoder_gradient_from_policy: bool = False

    # When True, skips the AIS reward + transition auxiliary losses
    # entirely (only the policy loss runs). Combined with
    # encoder_gradient_from_policy=True, the AIS auxiliary structure is
    # removed — the encoder becomes a vanilla recurrent feature extractor.
    disable_ais_aux_losses: bool = False


# ============================================================================
# Trainer
# ============================================================================


class AISPGTrainer:
    """Single-loop AIS-PG trainer (POMARL Algorithm 1)."""

    def __init__(
        self,
        config: AISPGTrainerConfig,
        env_config: Any,
        liability: LiabilityPortfolio,
        transaction_cost_rates: np.ndarray,
        market_sampler: MarketBatchSampler,
        single_path_sampler: SinglePathSampler | None = None,
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
        self._single_path_sampler = single_path_sampler

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

        # ── Build flax modules ────────────────────────────────────────────
        self._encoder = AISGRUEncoder(
            hidden_size=config.ais_hidden_size, n_layers=config.ais_n_layers,
        )
        self._policy = GaussianPolicy(
            n_instruments=self.n_instruments,
            hidden_size=config.policy_hidden_size,
            log_std_min=config.log_std_min,
            log_std_max=config.log_std_max,
        )
        self._reward_model = AISRewardModel()
        self._transition_model = AISTransitionModel(
            ais_dim=config.ais_hidden_size,
        )

        self._rng_key = jax.random.PRNGKey(seed)
        self._init_params()
        self._init_optimizers()

        self.train_history: List[Dict[str, float]] = []
        self._build_jit_fns()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_params(self) -> None:
        key_enc, key_pol, key_r, key_p, self._rng_key = jax.random.split(
            self._rng_key, 5,
        )
        dummy_obs = jnp.zeros((1, self.obs_dim), dtype=jnp.float32)
        dummy_hidden = AISGRUEncoder.init_hidden(
            1, self.config.ais_hidden_size, self.config.ais_n_layers,
        )
        dummy_x_hat = jnp.zeros((1, self.config.ais_hidden_size),
                                dtype=jnp.float32)
        dummy_mask = jnp.ones((1, self.n_instruments), dtype=jnp.float32)
        dummy_action = jnp.zeros((1, self.n_instruments), dtype=jnp.float32)

        self.encoder_params = self._encoder.init(key_enc, dummy_obs, dummy_hidden)
        self.policy_params = self._policy.init(key_pol, dummy_x_hat, dummy_mask)
        self.reward_params = self._reward_model.init(
            key_r, dummy_x_hat, dummy_action,
        )
        self.transition_params = self._transition_model.init(
            key_p, dummy_x_hat, dummy_action,
        )

    def _init_optimizers(self) -> None:
        clip = optax.clip_by_global_norm(self.config.grad_clip)
        self._policy_opt = optax.chain(clip, optax.adam(self.config.policy_lr))
        self._policy_opt_state = self._policy_opt.init(self.policy_params)

        # Single optimiser updates encoder + reward + transition params jointly.
        self._ais_opt = optax.chain(clip, optax.adam(self.config.ais_lr))
        self._ais_opt_state = self._ais_opt.init(self._ais_params_tree())

    def _ais_params_tree(self) -> dict:
        return {
            "encoder": self.encoder_params,
            "reward": self.reward_params,
            "transition": self.transition_params,
        }

    # ------------------------------------------------------------------
    # Convenience constructor: JAX Heston backend
    # ------------------------------------------------------------------

    @classmethod
    def from_heston(
        cls,
        config: AISPGTrainerConfig,
        env_config: Any,
        market_params: Any,
        padded_grid: Any,
        liability: LiabilityPortfolio,
        initial_spot: float = 100.0,
        initial_variance: float = 0.04,
        initial_cash: float = 0.0,
        seed: int = 42,
    ) -> "AISPGTrainer":
        """Wire a trainer to the JAX Heston simulator (mirrors
        :meth:`BuehlerTrainer.from_heston`)."""
        from ..jax.env import build_transaction_cost_vector
        from ..jax.rollout import (
            simulate_heston_market,
            simulate_heston_market_batch,
        )

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
                instrument_prices=np.asarray(
                    market_traj.instrument_prices, dtype=np.float32,
                ),
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

        tc = build_transaction_cost_vector(env_config)
        return cls(
            config=config,
            env_config=env_config,
            liability=liability,
            transaction_cost_rates=tc,
            market_sampler=market_sampler,
            single_path_sampler=single_path_sampler,
            initial_cash=initial_cash,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # JIT entry points
    # ------------------------------------------------------------------

    def _build_jit_fns(self) -> None:
        encoder_apply = self._encoder.apply
        policy_apply = self._policy.apply
        reward_apply = self._reward_model.apply
        transition_apply = self._transition_model.apply

        horizon = self.horizon
        n_inst = self.n_instruments
        hidden_size = self.config.ais_hidden_size
        n_layers = self.config.ais_n_layers
        position_limit = self.config.position_limit
        discount = self.config.discount
        baseline_kind = self.config.baseline_kind
        lambda_r = self.config.lambda_reward
        lambda_p = self.config.lambda_transition
        liability_legs = self.liability_legs
        initial_cash = self.initial_cash
        instrument_mask = self._instrument_mask_static
        tc_rates = self.transaction_cost_rates

        reward_kind = self.config.reward_kind
        risk_aversion_reward = self.config.risk_aversion
        # dt sourced from env_config (authoritative) rather than the trainer
        # config; falls back to the trainer config default if env_config
        # doesn't carry one.
        env_dt = getattr(self.env_config, "dt", self.config.dt)

        def _rollout(encoder_params, policy_params, market, key, sample):
            spots, prices, masks, variances = market
            return pomarl_rollout(
                encoder_apply=encoder_apply,
                policy_apply=policy_apply,
                encoder_params=encoder_params,
                policy_params=policy_params,
                spots=spots,
                prices=prices,
                masks=masks,
                transaction_cost_rates=tc_rates,
                instrument_mask_static=instrument_mask,
                liability_legs=liability_legs,
                initial_cash=initial_cash,
                position_limit=position_limit,
                horizon=horizon,
                n_instruments=n_inst,
                encoder_hidden_size=hidden_size,
                encoder_n_layers=n_layers,
                sample=sample,
                key=key,
                reward_kind=reward_kind,
                risk_aversion=risk_aversion_reward,
                variances=variances,
                dt=env_dt,
                discrete_bucket_size=self.config.discrete_bucket_size,
                encoder_gradient_from_policy=self.config.encoder_gradient_from_policy,
            )

        policy_method = self.config.policy_method
        entropy_coef = self.config.entropy_coef

        def policy_loss_fn(policy_params, encoder_params, market, key):
            out = _rollout(encoder_params, policy_params, market, key, True)
            if policy_method == "reinforce":
                loss, info = reinforce_loss(
                    log_probs=out.log_probs,
                    rewards=out.rewards,
                    discount=discount,
                    baseline_kind=baseline_kind,
                )
            elif policy_method == "pathwise":
                # SVG: backprop directly through the differentiable rollout.
                returns_per_path = out.rewards.sum(axis=0)
                mean_return = returns_per_path.mean()
                loss = -mean_return
                if entropy_coef > 0.0:
                    entropy_proxy = -jnp.mean(out.log_probs)
                    loss = loss - entropy_coef * entropy_proxy
                info = {
                    "policy_method": 1.0, "mean_return": mean_return,
                    "return_std": returns_per_path.std(),
                }
            elif policy_method == "buehler":
                # LSTM-trainer-style supervised loss: γ·Var(error) + E[cost],
                # backprop end-to-end through (encoder?) + policy + rollout.
                # Decouples representation learning from reward semantics —
                # mirrors what works in the BuehlerTrainer. With
                # encoder_gradient_from_policy=True, the encoder is trained
                # by the hedging objective directly (like the LSTM trainer
                # trains the LSTM hidden state).
                terminal_pnl = out.terminal_pnl
                payoff = out.payoff
                error = terminal_pnl - payoff
                # Use sample variance (ddof=1) to match BuehlerTrainer.
                B_paths = error.shape[0]
                var_term = (
                    risk_aversion_reward
                    * jnp.sum((error - error.mean()) ** 2) / (B_paths - 1)
                )
                cost_term = out.total_costs.mean()
                loss = var_term + cost_term
                if entropy_coef > 0.0:
                    entropy_proxy = -jnp.mean(out.log_probs)
                    loss = loss - entropy_coef * entropy_proxy
                info = {
                    "policy_method": 2.0,
                    "var_term": var_term,
                    "cost_term": cost_term,
                    "mean_error": error.mean(),
                    "std_error": jnp.sqrt(jnp.var(error, ddof=1)),
                }
            else:
                raise ValueError(
                    f"unknown policy_method={policy_method!r}; "
                    f"expected 'reinforce', 'pathwise', or 'buehler'"
                )
            info = {
                **info,
                "terminal_pnl_mean": out.terminal_pnl.mean(),
                "total_cost_mean": out.total_costs.mean(),
                "payoff_mean": out.payoff.mean(),
            }
            return loss, info

        def ais_loss_fn(ais_params, policy_params, market, key):
            out = _rollout(
                ais_params["encoder"], policy_params, market, key, True,
            )
            x_hat_t = out.x_hat_seq[:-1]
            actions = jax.lax.stop_gradient(out.actions)
            rewards = jax.lax.stop_gradient(out.rewards)
            L_r = ais_reward_loss(
                x_hat=x_hat_t,
                actions=actions,
                rewards=rewards,
                reward_model_apply=reward_apply,
                reward_params=ais_params["reward"],
            )
            L_p = ais_transition_loss(
                x_hat=out.x_hat_seq,
                actions=actions,
                transition_model_apply=transition_apply,
                transition_params=ais_params["transition"],
            )
            loss = lambda_r * L_r + lambda_p * L_p
            return loss, {"L_r": L_r, "L_p": L_p}

        policy_grad_fn = jax.value_and_grad(policy_loss_fn, has_aux=True)
        ais_grad_fn = jax.value_and_grad(ais_loss_fn, has_aux=True)

        disable_aux = bool(self.config.disable_ais_aux_losses)
        enc_grad_from_policy = bool(self.config.encoder_gradient_from_policy)

        # When the policy loss is meant to train the encoder end-to-end
        # (encoder_gradient_from_policy=True), we need a policy_grad_fn
        # that exposes encoder_params as a differentiated argument too.
        if enc_grad_from_policy:
            def joint_loss_fn(joint_params, market, key):
                # Re-pack and call existing policy_loss_fn so the encoder
                # appears as a differentiated argument.
                return policy_loss_fn(
                    joint_params["policy"], joint_params["encoder"], market, key,
                )
            joint_grad_fn = jax.value_and_grad(joint_loss_fn, has_aux=True)
        else:
            joint_grad_fn = None

        def train_step_pure(
            encoder_params, policy_params, reward_params, transition_params,
            policy_opt_state, ais_opt_state, market, key,
        ):
            key_pi, key_ais = jax.random.split(key)

            if enc_grad_from_policy:
                joint_params = {"policy": policy_params, "encoder": encoder_params}
                (pi_loss, pi_info), joint_grad = joint_grad_fn(
                    joint_params, market, key_pi,
                )
                # Apply gradient to policy via policy_opt; encoder also
                # updates through the AIS optimizer's encoder slot when
                # not disabled. To keep this clean, channel encoder grad
                # through ais_opt below by injecting into ais_grad.
                pi_grad = joint_grad["policy"]
                encoder_grad_from_policy = joint_grad["encoder"]
                updates, policy_opt_state = self._policy_opt.update(
                    pi_grad, policy_opt_state, policy_params,
                )
                policy_params = optax.apply_updates(policy_params, updates)
            else:
                (pi_loss, pi_info), pi_grad = policy_grad_fn(
                    policy_params, encoder_params, market, key_pi,
                )
                encoder_grad_from_policy = None
                updates, policy_opt_state = self._policy_opt.update(
                    pi_grad, policy_opt_state, policy_params,
                )
                policy_params = optax.apply_updates(policy_params, updates)

            ais_params = {
                "encoder": encoder_params,
                "reward": reward_params,
                "transition": transition_params,
            }
            if disable_aux:
                # No aux losses; encoder only updates via policy-loss
                # gradient (if encoder_gradient_from_policy=True), routed
                # through the policy optimizer for simplicity.
                ais_loss = jnp.float32(0.0)
                ais_info: dict = {}
                if encoder_grad_from_policy is not None:
                    # Apply encoder gradient via AIS optimizer (its slot
                    # for encoder params); reward/transition heads get
                    # zero gradient (no aux loss).
                    fake_ais_grad = {
                        "encoder": encoder_grad_from_policy,
                        "reward": jax.tree_util.tree_map(jnp.zeros_like, reward_params),
                        "transition": jax.tree_util.tree_map(jnp.zeros_like, transition_params),
                    }
                    ais_updates, ais_opt_state = self._ais_opt.update(
                        fake_ais_grad, ais_opt_state, ais_params,
                    )
                    ais_params = optax.apply_updates(ais_params, ais_updates)
            else:
                (ais_loss, ais_info), ais_grad = ais_grad_fn(
                    ais_params, policy_params, market, key_ais,
                )
                # When policy also contributes encoder grad, combine.
                if encoder_grad_from_policy is not None:
                    ais_grad = {
                        **ais_grad,
                        "encoder": jax.tree_util.tree_map(
                            lambda a, b: a + b,
                            ais_grad["encoder"], encoder_grad_from_policy,
                        ),
                    }
                ais_updates, ais_opt_state = self._ais_opt.update(
                    ais_grad, ais_opt_state, ais_params,
                )
                ais_params = optax.apply_updates(ais_params, ais_updates)

            metrics = {
                "policy_loss": pi_loss,
                "ais_loss": ais_loss,
                **{f"pi_{k}": v for k, v in pi_info.items()},
                **{f"ais_{k}": v for k, v in ais_info.items()},
            }
            return (
                ais_params["encoder"], policy_params,
                ais_params["reward"], ais_params["transition"],
                policy_opt_state, ais_opt_state,
                metrics,
            )

        self._train_step_jit = jax.jit(train_step_pure)

        def eval_pure(
            encoder_params, policy_params, market, key, sample,
        ):
            out = _rollout(encoder_params, policy_params, market, key, sample)
            return out

        self._eval_jit = jax.jit(eval_pure, static_argnums=(4,))

    # ------------------------------------------------------------------
    # Data generation helpers
    # ------------------------------------------------------------------

    def _sample_market_jax(self, batch_size: int, key) -> tuple:
        """Generate a market batch as (spots, prices, masks, variances)."""
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
    # Training
    # ------------------------------------------------------------------

    def train_step(self) -> Dict[str, float]:
        self._rng_key, key_market, key_step = jax.random.split(self._rng_key, 3)
        market = self._sample_market_jax(self.config.batch_size, key_market)

        (
            self.encoder_params, self.policy_params,
            self.reward_params, self.transition_params,
            self._policy_opt_state, self._ais_opt_state,
            metrics_jnp,
        ) = self._train_step_jit(
            self.encoder_params, self.policy_params,
            self.reward_params, self.transition_params,
            self._policy_opt_state, self._ais_opt_state,
            market, key_step,
        )
        return {k: float(jax.device_get(v)) for k, v in metrics_jnp.items()}

    def train(self, progress: bool = True) -> List[Dict[str, float]]:
        try:
            from tqdm.auto import tqdm
            use_tqdm = bool(progress)
        except ImportError:  # pragma: no cover
            use_tqdm = False
            tqdm = None  # type: ignore

        rng = range(1, self.config.n_epochs + 1)
        it = tqdm(rng, desc="pomarl") if use_tqdm else rng
        for epoch in it:
            metrics = self.train_step()
            metrics["epoch"] = epoch
            self.train_history.append(metrics)
            if use_tqdm:
                it.set_postfix({
                    "pi": f"{metrics['policy_loss']:+.3f}",
                    "ais": f"{metrics['ais_loss']:+.3f}",
                    "ret": f"{metrics['pi_mean_return']:+.3f}",
                })
            if epoch % self.config.eval_every == 0 or epoch == 1:
                logger.info(
                    "[POMARL epoch %d] pi=%+.4f ais=%+.4f return=%+.4f",
                    epoch, metrics["policy_loss"], metrics["ais_loss"],
                    metrics["pi_mean_return"],
                )
        return self.train_history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        n_paths: int = 4096,
        seed: int = 9999,
        sample: bool = False,
    ) -> Dict[str, float]:
        if n_paths <= 0:
            raise ValueError(f"n_paths must be positive, got {n_paths}")
        key = jax.random.PRNGKey(seed)
        key_market, key_roll = jax.random.split(key)
        market = self._sample_market_jax(n_paths, key_market)
        out = self._eval_jit(
            self.encoder_params, self.policy_params, market, key_roll, sample,
        )

        terminal_pnl = np.asarray(out.terminal_pnl, dtype=np.float64)
        total_costs = np.asarray(out.total_costs, dtype=np.float64)
        payoff = np.asarray(out.payoff, dtype=np.float64)
        rewards = np.asarray(out.rewards, dtype=np.float64)
        log_probs = np.asarray(out.log_probs, dtype=np.float64)

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
            "mean_reward": float((hedging_errors - float(self.initial_cash)).mean()),
            "mean_cost": float(total_costs.mean()),
            "std_cost": float(total_costs.std(ddof=0)),
            "mean_liability_payoff": float(payoff.mean()),
            "std_liability_payoff": float(payoff.std(ddof=0)),
        }
        metrics.update(_error_distribution_metrics(hedging_errors))
        metrics.update(_error_distribution_metrics(zero_errors, "zero_hedge_"))
        metrics["std_improvement_vs_zero"] = (
            metrics["zero_hedge_std_hedging_error"] - metrics["std_hedging_error"]
        )
        metrics["rmse_improvement_vs_zero"] = (
            metrics["zero_hedge_rmse_hedging_error"] - metrics["rmse_hedging_error"]
        )
        # POMARL-specific extras
        metrics["pomarl_mean_return"] = float(rewards.sum(axis=0).mean())
        metrics["pomarl_mean_log_prob"] = float(log_probs.mean())
        return metrics

    # ------------------------------------------------------------------
    # Inference adapter / checkpoints
    # ------------------------------------------------------------------

    def get_agent(self) -> BaseHedgingAgent:
        return PomarlAgent(
            encoder_params=self.encoder_params,
            policy_params=self.policy_params,
            encoder_hidden_size=self.config.ais_hidden_size,
            encoder_n_layers=self.config.ais_n_layers,
            n_instruments=self.n_instruments,
            position_limit=self.config.position_limit,
            horizon=self.horizon,
            policy_hidden_size=self.config.policy_hidden_size,
            log_std_min=self.config.log_std_min,
            log_std_max=self.config.log_std_max,
        )

    def save_checkpoint(self, path: str) -> None:
        payload = {
            "encoder_params": jax.device_get(self.encoder_params),
            "policy_params": jax.device_get(self.policy_params),
            "reward_params": jax.device_get(self.reward_params),
            "transition_params": jax.device_get(self.transition_params),
            "policy_opt_state": jax.device_get(self._policy_opt_state),
            "ais_opt_state": jax.device_get(self._ais_opt_state),
            "config": self.config,
            "obs_dim": self.obs_dim,
            "n_instruments": self.n_instruments,
            "train_history": self.train_history,
        }
        with open(path, "wb") as fp:
            pickle.dump(payload, fp)
        logger.info("POMARL checkpoint saved to %s", path)

    def load_checkpoint(self, path: str) -> None:
        with open(path, "rb") as fp:
            payload = pickle.load(fp)
        self.encoder_params = payload["encoder_params"]
        self.policy_params = payload["policy_params"]
        self.reward_params = payload["reward_params"]
        self.transition_params = payload["transition_params"]
        self._policy_opt_state = payload["policy_opt_state"]
        self._ais_opt_state = payload["ais_opt_state"]
        self.train_history = payload.get("train_history", [])
        logger.info("POMARL checkpoint loaded from %s", path)
