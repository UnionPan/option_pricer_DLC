"""
Deep hedging training algorithms.

Trainers consume market-data callables (decoupled from any specific
simulator backend) and produce trained inference-only agents that live
in :mod:`options_desk.deep_hedging.agents`.

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

from __future__ import annotations

from .base import BaseTrainer

try:
    from .torch_buehler import (
        BuehlerTrainer,
        TrainerConfig,
        differentiable_rollout,
        buehler_loss,
    )

    _TORCH_TRAINER_AVAILABLE = True
except ImportError:
    _TORCH_TRAINER_AVAILABLE = False

__all__ = [
    "BaseTrainer",
]

if _TORCH_TRAINER_AVAILABLE:
    __all__ += [
        "BuehlerTrainer",
        "TrainerConfig",
        "differentiable_rollout",
        "buehler_loss",
    ]
