"""
Inference-only base class for hedging agents.

The base ABC describes how a *policy* responds to environment observations.
It deliberately does not own any learning loop -- that responsibility lives
in ``deep_hedging.training``. A hedging agent only needs to:

- Reset internal state at the start of an episode.
- Produce an action given an observation.
- Optionally expose an opaque snapshot of its parameters for checkpointing.

Concrete agents (delta hedging, neural-network policies, scripted baselines)
inherit from :class:`BaseHedgingAgent` and implement :meth:`act`. Trainers may
extend a concrete agent with extra learning hooks, but those hooks live on the
trainer side, not on this contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class BaseHedgingAgent(ABC):
    """
    Abstract base class for inference-only hedging policies.

    Interface:
        - reset(observation, info): initialise per-episode state.
        - act(observation, info) -> np.ndarray: return an action vector of
          shape ``(n_instruments,)``.
        - get_state() / set_state(): optional snapshotting hooks.
    """

    def __init__(
        self,
        n_instruments: int,
        position_limits: float = 100.0,
        name: str = "BaseAgent",
    ):
        self.n_instruments = n_instruments
        self.position_limits = position_limits
        self.name = name

        self.current_positions: Optional[np.ndarray] = None
        self.episode_count = 0
        self.step_count = 0

    @abstractmethod
    def reset(
        self,
        observation: Any,
        info: Dict[str, Any],
    ) -> None:
        """Initialise per-episode state. Subclasses should call ``super().reset(...)``."""
        self.current_positions = np.zeros(self.n_instruments, dtype=np.float32)
        self.episode_count += 1
        self.step_count = 0

    @abstractmethod
    def act(
        self,
        observation: Any,
        info: Dict[str, Any],
    ) -> np.ndarray:
        """
        Produce an action for the given observation.

        Returns:
            np.ndarray of shape ``(n_instruments,)`` containing target
            positions (or trades, depending on the rollout's action mode).
        """
        raise NotImplementedError

    def get_state(self) -> Dict[str, Any]:
        """Return an opaque snapshot of agent state for checkpointing."""
        return {
            "name": self.name,
            "n_instruments": self.n_instruments,
            "position_limits": self.position_limits,
            "current_positions": (
                self.current_positions.copy()
                if self.current_positions is not None
                else None
            ),
            "episode_count": self.episode_count,
            "step_count": self.step_count,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore agent state from a snapshot."""
        self.current_positions = state.get("current_positions")
        self.episode_count = state.get("episode_count", 0)
        self.step_count = state.get("step_count", 0)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"n_instruments={self.n_instruments}, "
            f"position_limits={self.position_limits})"
        )


class RandomAgent(BaseHedgingAgent):
    """Baseline that samples uniform actions inside the position box."""

    def __init__(
        self,
        n_instruments: int,
        position_limits: float = 100.0,
        scale: float = 0.1,
        seed: Optional[int] = None,
    ):
        super().__init__(n_instruments, position_limits, name="RandomAgent")
        self.scale = scale
        self.rng = np.random.RandomState(seed)

    def reset(self, observation: Any, info: Dict[str, Any]) -> None:
        super().reset(observation, info)

    def act(self, observation: Any, info: Dict[str, Any]) -> np.ndarray:
        action = self.rng.uniform(
            low=-self.position_limits * self.scale,
            high=self.position_limits * self.scale,
            size=self.n_instruments,
        ).astype(np.float32)
        self.current_positions = action
        return action


class DoNothingAgent(BaseHedgingAgent):
    """Baseline that always returns a zero action vector."""

    def __init__(
        self,
        n_instruments: int,
        position_limits: float = 100.0,
    ):
        super().__init__(n_instruments, position_limits, name="DoNothingAgent")

    def reset(self, observation: Any, info: Dict[str, Any]) -> None:
        super().reset(observation, info)

    def act(self, observation: Any, info: Dict[str, Any]) -> np.ndarray:
        action = np.zeros(self.n_instruments, dtype=np.float32)
        self.current_positions = action
        return action
