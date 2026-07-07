"""
Inference-only :class:`BaseHedgingAgent` adapter for the POMARL stack.

Reconstructs the policy's flax modules at construction time, threads the GRU
hidden state across ``act`` calls, and returns deterministic (mean-action)
trades — mirroring how :class:`TorchPolicyAgent` wraps the Buehler MLP/LSTM.
"""

from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np

from ..agents.base import BaseHedgingAgent
from .ais import AISGRUEncoder
from .policy import GaussianPolicy, mean_action
from .utils import build_pomdp_obs


class PomarlAgent(BaseHedgingAgent):
    """Deterministic POMARL inference agent.

    Each ``act`` call:
      1. Builds the variance-hidden obs from a ``MarketTrajectory`` / Gym-style
         dict (whichever the env emits).
      2. Steps the GRU encoder, updating the hidden state.
      3. Returns ``tanh(μ) · L · mask`` as the trade vector.
    """

    def __init__(
        self,
        *,
        encoder_params,
        policy_params,
        encoder_hidden_size: int,
        encoder_n_layers: int,
        n_instruments: int,
        position_limit: float,
        horizon: int,
        policy_hidden_size: int = 64,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        name: str = "PomarlAgent",
    ) -> None:
        super().__init__(
            n_instruments=n_instruments,
            position_limits=position_limit,
            name=name,
        )
        self._encoder = AISGRUEncoder(
            hidden_size=encoder_hidden_size, n_layers=encoder_n_layers,
        )
        self._policy = GaussianPolicy(
            n_instruments=n_instruments,
            hidden_size=policy_hidden_size,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )
        self._encoder_params = encoder_params
        self._policy_params = policy_params
        self._encoder_hidden_size = encoder_hidden_size
        self._encoder_n_layers = encoder_n_layers
        self._horizon = horizon
        self._position_limit = float(position_limit)

        self._hidden = AISGRUEncoder.init_hidden(
            1, encoder_hidden_size, encoder_n_layers,
        )
        self._prev_trades = np.zeros(n_instruments, dtype=np.float32)
        self._time_index = 0

    def reset(self, observation: Any, info: Dict[str, Any]) -> None:
        super().reset(observation, info)
        self._hidden = AISGRUEncoder.init_hidden(
            1, self._encoder_hidden_size, self._encoder_n_layers,
        )
        self._prev_trades = np.zeros(self.n_instruments, dtype=np.float32)
        self._time_index = 0

    def act(self, observation: Any, info: Dict[str, Any]) -> np.ndarray:
        spot = self._extract(observation, "spot")
        instrument_prices = self._extract(observation, "instrument_prices")
        positions = self._extract(observation, "positions",
                                  default=self.current_positions)
        mask = self._extract(observation, "action_mask",
                             default=np.ones(self.n_instruments, dtype=bool))

        spot_b = jnp.asarray(spot, dtype=jnp.float32).reshape(1)
        opt_prices_b = jnp.asarray(
            instrument_prices[1:], dtype=jnp.float32,
        ).reshape(1, -1)
        positions_b = jnp.asarray(positions, dtype=jnp.float32).reshape(
            1, self.n_instruments,
        )
        prev_b = jnp.asarray(self._prev_trades, dtype=jnp.float32).reshape(
            1, self.n_instruments,
        )
        mask_b = jnp.asarray(mask, dtype=jnp.float32).reshape(
            1, self.n_instruments,
        )

        obs = build_pomdp_obs(
            spot_t=spot_b,
            option_prices_t=opt_prices_b,
            positions=positions_b,
            previous_trades=prev_b,
            time_index=self._time_index,
            horizon=self._horizon,
        )
        x_hat, self._hidden = self._encoder.apply(
            self._encoder_params, obs, self._hidden,
        )
        mu, _ = self._policy.apply(self._policy_params, x_hat, mask_b)
        action = mean_action(mu, mask_b, self._position_limit)
        action_np = np.asarray(action, dtype=np.float32).reshape(-1)
        self._prev_trades = action_np
        self.current_positions = (
            np.zeros(self.n_instruments, dtype=np.float32)
            if self.current_positions is None
            else self.current_positions
        ) + action_np
        self._time_index += 1
        self.step_count += 1
        return action_np

    # ------------------------------------------------------------------
    # tolerant observation extractor
    # ------------------------------------------------------------------

    @staticmethod
    def _extract(observation: Any, name: str, default=None) -> np.ndarray:
        if hasattr(observation, name):
            return np.asarray(getattr(observation, name))
        if isinstance(observation, dict) and name in observation:
            return np.asarray(observation[name])
        if default is not None:
            return np.asarray(default)
        raise KeyError(f"observation has no attribute / key {name!r}")
