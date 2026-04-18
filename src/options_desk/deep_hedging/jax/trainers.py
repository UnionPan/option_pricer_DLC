"""
Trainer configuration for the JAX deep hedging line.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class TrainerConfig:
    """
    Training defaults aligned with the notes' first-pass Adam baseline.
    """

    batch_size: int = 2048
    learning_rate: float = 1.0e-3
    optimizer: str = "adam"


@dataclass(frozen=True)
class HedgingLossBreakdown:
    """
    Separated objective terms for diagnostics and training logs.
    """

    variance_term: float
    cost_term: float
    total_loss: float


def compute_hedging_loss(
    terminal_portfolio_values: np.ndarray,
    terminal_liability_payoffs: np.ndarray,
    accumulated_transaction_costs: np.ndarray,
    risk_aversion: float,
) -> HedgingLossBreakdown:
    """
    Compute the note-aligned hedging objective:

    gamma * Var(PnL - payoff) + E[cost]
    """

    portfolio = jnp.asarray(terminal_portfolio_values, dtype=jnp.float32)
    liability = jnp.asarray(terminal_liability_payoffs, dtype=jnp.float32)
    costs = jnp.asarray(accumulated_transaction_costs, dtype=jnp.float32)

    hedging_error = portfolio - liability
    variance_term = risk_aversion * jnp.var(hedging_error)
    cost_term = jnp.mean(costs)
    total_loss = variance_term + cost_term

    return HedgingLossBreakdown(
        variance_term=float(variance_term),
        cost_term=float(cost_term),
        total_loss=float(total_loss),
    )
