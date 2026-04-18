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
        position_limit: float = 100.0,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("HedgingMLPPolicy requires PyTorch")
        super().__init__()
        self.obs_dim = obs_dim
        self.n_instruments = n_instruments
        self.position_limit = position_limit

        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, n_instruments))
        self.net = nn.Sequential(*layers)

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

    parts = [
        np.asarray(obs.spot, dtype=np.float32).ravel(),
        np.asarray(obs.option_features, dtype=np.float32).ravel(),
        np.asarray(obs.portfolio_features, dtype=np.float32).ravel(),
        np.asarray(obs.previous_action, dtype=np.float32).ravel(),
        np.asarray(obs.context_features, dtype=np.float32).ravel(),
    ]
    flat = np.concatenate(parts)
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
        policy: HedgingMLPPolicy,
        name: str = "TorchPolicyAgent",
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("TorchPolicyAgent requires PyTorch")
        super().__init__(
            n_instruments=policy.n_instruments,
            position_limits=policy.position_limit,
            name=name,
        )
        self.policy = policy
        self.policy.eval()

    def reset(self, observation: Any, info: Dict[str, Any]) -> None:
        super().reset(observation, info)

    def act(
        self,
        observation: ObservationBatch,
        info: Dict[str, Any],
    ) -> np.ndarray:
        obs_tensor, mask_tensor = obs_batch_to_tensor(observation)
        with torch.no_grad():
            trades = self.policy(obs_tensor, mask_tensor)
        trade_np = trades.squeeze(0).cpu().numpy().astype(np.float32)
        if self.current_positions is None:
            self.current_positions = np.zeros(self.n_instruments, dtype=np.float32)
        self.current_positions = self.current_positions + trade_np
        return trade_np
