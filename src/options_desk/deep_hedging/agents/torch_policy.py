"""
Torch-based hedging policy and its inference-only agent wrapper.

The policy network maps a flat observation to bounded *trades* (signed
delta positions). The agent wraps a trained policy behind the
:class:`BaseHedgingAgent` interface so it can be evaluated through
``collect_agent_rollout_from_market`` without any training-side
dependencies.

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..utils.contracts import ObservationBatch
from .base import BaseHedgingAgent

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is an optional dep
    _TORCH_AVAILABLE = False


def _validate_positive(value: float, name: str) -> float:
    value_f = float(value)
    if value_f <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return value_f


def normalize_observation_parts(
    spot: np.ndarray,
    option_features: np.ndarray,
    portfolio_features: np.ndarray,
    previous_action: np.ndarray,
    context_features: np.ndarray,
    *,
    price_scale: float = 100.0,
    position_scale: float = 100.0,
    price_clip: float = 10.0,
) -> np.ndarray:
    """Flatten observation parts onto the same feature scale used in training."""
    price_scale = _validate_positive(price_scale, "price_scale")
    position_scale = _validate_positive(position_scale, "position_scale")
    price_clip = _validate_positive(price_clip, "price_clip")

    spot_feature = np.clip(
        np.asarray(spot, dtype=np.float32).ravel() / price_scale - 1.0,
        -price_clip,
        price_clip,
    )
    option_scaled = np.clip(
        np.asarray(option_features, dtype=np.float32).ravel() / price_scale,
        -price_clip,
        price_clip,
    )
    portfolio_scaled = np.clip(
        np.asarray(portfolio_features, dtype=np.float32).ravel() / position_scale,
        -price_clip,
        price_clip,
    )
    previous_scaled = np.clip(
        np.asarray(previous_action, dtype=np.float32).ravel() / position_scale,
        -price_clip,
        price_clip,
    )
    context = np.asarray(context_features, dtype=np.float32).ravel()
    return np.concatenate(
        [spot_feature, option_scaled, portfolio_scaled, previous_scaled, context]
    ).astype(np.float32)


class HedgingMLPPolicy(nn.Module if _TORCH_AVAILABLE else object):
    """
    MLP policy for deep hedging.

    Maps a flat observation vector to bounded trade actions::

        obs -> Linear -> ReLU -> ... -> Linear -> tanh -> scale -> mask
    """

    def __init__(
        self,
        obs_dim: int,
        n_instruments: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
        position_limit: float = 1.0,
        last_layer_scale: float = 1e-3,
        price_scale: float = 100.0,
        position_scale: float = 100.0,
        price_clip: float = 10.0,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("HedgingMLPPolicy requires PyTorch")
        super().__init__()
        self.obs_dim = obs_dim
        self.n_instruments = n_instruments
        self.position_limit = position_limit
        self.last_layer_scale = last_layer_scale
        self.price_scale = _validate_positive(price_scale, "price_scale")
        self.position_scale = _validate_positive(position_scale, "position_scale")
        self.price_clip = _validate_positive(price_clip, "price_clip")

        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.output = nn.Linear(prev, n_instruments)
        layers.append(self.output)
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Small output init prevents saturated tanh trades at epoch 0."""
        for module in self.net:
            if not isinstance(module, nn.Linear):
                continue
            nonlinearity = "linear" if module is self.output else "relu"
            nn.init.kaiming_normal_(module.weight, nonlinearity=nonlinearity)
            nn.init.zeros_(module.bias)
        with torch.no_grad():
            self.output.weight.mul_(self.last_layer_scale)

    def forward(
        self,
        obs: "torch.Tensor",
        action_mask: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Args:
            obs: ``(B, obs_dim)`` flat observation tensor.
            action_mask: ``(B, N)`` 1/0 mask -- 1 for tradable instruments.

        Returns:
            trades: ``(B, N)`` bounded trade actions.
        """
        raw = self.net(obs)
        return torch.tanh(raw) * self.position_limit * action_mask


def obs_batch_to_tensor(
    obs: ObservationBatch,
    *,
    price_scale: float = 100.0,
    position_scale: float = 100.0,
    price_clip: float = 10.0,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """
    Flatten an :class:`ObservationBatch` into ``(obs_tensor, mask_tensor)``.

    Concatenation order:
        ``[spot, option_features, portfolio_features, previous_action, context_features]``

    Returns:
        obs_tensor: ``(1, obs_dim)``
        mask_tensor: ``(1, N)``
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("obs_batch_to_tensor requires PyTorch")

    flat = normalize_observation_parts(
        spot=obs.spot,
        option_features=obs.option_features,
        portfolio_features=obs.portfolio_features,
        previous_action=obs.previous_action,
        context_features=obs.context_features,
        price_scale=price_scale,
        position_scale=position_scale,
        price_clip=price_clip,
    )
    obs_tensor = torch.from_numpy(flat).unsqueeze(0)

    if obs.action_mask is not None:
        mask_tensor = torch.from_numpy(
            np.asarray(obs.action_mask, dtype=np.float32).reshape(1, -1)
        )
    else:
        mask_tensor = torch.ones(1, obs.spot.shape[-1], dtype=torch.float32)
    return obs_tensor, mask_tensor


class TorchPolicyAgent(BaseHedgingAgent):
    """
    Inference-only adapter around a trained :class:`HedgingMLPPolicy`.

    Always runs under ``torch.no_grad`` and returns NumPy actions so it
    plugs into ``collect_agent_rollout_from_market`` without any training
    dependencies.
    """

    def __init__(
        self,
        policy: "HedgingMLPPolicy",
        name: str = "TorchPolicyAgent",
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("TorchPolicyAgent requires PyTorch")
        # position_limit may be None for recurrent policies that disable the
        # tanh clamp; fall back to +inf for the BaseHedgingAgent bound.
        position_limit = getattr(policy, "position_limit", None)
        if position_limit is None:
            position_limit = float("inf")
        super().__init__(
            n_instruments=policy.n_instruments,
            position_limits=position_limit,
            name=name,
        )
        self.policy = policy
        self.policy.eval()
        self.price_scale = getattr(policy, "price_scale", 100.0)
        self.position_scale = getattr(policy, "position_scale", 100.0)
        self.price_clip = getattr(policy, "price_clip", 10.0)

        # Duck-type detect a recurrent policy that exposes init_hidden_state
        # and step. Hidden state is carried across .act() calls and reset on
        # .reset(). Non-recurrent policies leave _hidden_state at None.
        self._is_recurrent = hasattr(policy, "init_hidden_state") and hasattr(
            policy, "step"
        )
        self._hidden_state = None

    def _policy_device(self) -> "torch.device":
        try:
            return next(self.policy.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def reset(self, observation: Any, info: Dict[str, Any]) -> None:
        super().reset(observation, info)
        if self._is_recurrent:
            self._hidden_state = self.policy.init_hidden_state(
                batch_size=1, device=self._policy_device(),
            )

    def act(
        self,
        observation: ObservationBatch,
        info: Dict[str, Any],
    ) -> np.ndarray:
        obs_tensor, mask_tensor = obs_batch_to_tensor(
            observation,
            price_scale=self.price_scale,
            position_scale=self.position_scale,
            price_clip=self.price_clip,
        )
        with torch.no_grad():
            if self._is_recurrent:
                if self._hidden_state is None:
                    self._hidden_state = self.policy.init_hidden_state(
                        batch_size=1, device=obs_tensor.device,
                    )
                trades, self._hidden_state = self.policy.step(
                    obs_tensor, mask_tensor, self._hidden_state,
                )
            else:
                trades = self.policy(obs_tensor, mask_tensor)
        trade_np = trades.squeeze(0).cpu().numpy().astype(np.float32)
        if self.current_positions is None:
            self.current_positions = np.zeros(self.n_instruments, dtype=np.float32)
        self.current_positions = self.current_positions + trade_np
        return trade_np
