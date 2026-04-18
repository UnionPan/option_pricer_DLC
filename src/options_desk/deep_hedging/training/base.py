"""
Abstract base class for deep hedging trainers.

A trainer owns the learning loop and produces a trained agent. The exact
data backend (JAX Heston, recorded market trajectories, GAN simulator,
etc.) is injected via callables, so this contract has no dependency on
any specific simulator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

from ..agents.base import BaseHedgingAgent


class BaseTrainer(ABC):
    """
    Minimal contract every deep hedging trainer must implement.

    Subclasses are free to add more knobs, but anything calling a trainer
    polymorphically can rely on these four methods:

    - :meth:`train_step` -- one optimisation step, returns a metrics dict.
    - :meth:`train` -- run the full training loop, returns the metric history.
    - :meth:`evaluate` -- measure the current policy on fresh trajectories.
    - :meth:`get_agent` -- return an inference-only :class:`BaseHedgingAgent`
      that wraps the trained policy.
    """

    @abstractmethod
    def train_step(self) -> Dict[str, float]:
        """Run one training step and return scalar metrics."""

    @abstractmethod
    def train(self) -> List[Dict[str, float]]:
        """Run the full training loop and return per-epoch metric history."""

    @abstractmethod
    def evaluate(self, n_paths: int, seed: int) -> Dict[str, float]:
        """Evaluate the current policy on ``n_paths`` fresh trajectories."""

    @abstractmethod
    def get_agent(self) -> BaseHedgingAgent:
        """Return an inference-only agent wrapping the trained policy."""

    def save_checkpoint(self, path: str) -> None:
        """Persist trainer state. Subclasses should override if applicable."""
        raise NotImplementedError

    def load_checkpoint(self, path: str) -> None:
        """Restore trainer state. Subclasses should override if applicable."""
        raise NotImplementedError
