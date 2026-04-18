"""
Policy configuration for the JAX deep hedging line.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class PolicyConfig:
    """
    Defaults aligned with the fast hedging notes.
    """

    hidden_size: int = 32
    n_recurrent_blocks: int = 4
    use_previous_action: bool = True
    use_action_mask: bool = True


@dataclass(frozen=True)
class PolicyState:
    """
    Minimal recurrent policy state.
    """

    hidden: jnp.ndarray
    previous_action: jnp.ndarray


@dataclass(frozen=True)
class PolicyParams:
    """
    Minimal deterministic linear policy parameters.
    """

    weights: jnp.ndarray
    bias: jnp.ndarray


def _flatten_observation(observation: dict[str, np.ndarray]) -> np.ndarray:
    """
    Flatten the observation bundle into a single feature vector.
    """

    ordered_keys = (
        "spot",
        "variance",
        "time_features",
        "positions",
        "previous_action",
        "option_features",
    )
    parts = [
        np.asarray(observation[key], dtype=np.float32).ravel()
        for key in ordered_keys
    ]
    return np.concatenate(parts, axis=0)


def init_policy_state(
    config: PolicyConfig,
    n_instruments: int,
) -> PolicyState:
    """
    Initialize recurrent hidden state and previous-action memory.
    """

    return PolicyState(
        hidden=jnp.zeros(
            (config.n_recurrent_blocks, config.hidden_size),
            dtype=jnp.float32,
        ),
        previous_action=jnp.zeros((n_instruments,), dtype=jnp.float32),
    )


def init_linear_policy_params(
    obs_dim: int,
    n_instruments: int,
    bias: np.ndarray | None = None,
) -> PolicyParams:
    """
    Initialize a deterministic linear policy with zero weights.
    """

    bias_vec = np.zeros((n_instruments,), dtype=np.float32)
    if bias is not None:
        bias_vec = np.asarray(bias, dtype=np.float32)
        if bias_vec.shape != (n_instruments,):
            raise ValueError(
                f"bias must have shape ({n_instruments},), got {bias_vec.shape}"
            )

    return PolicyParams(
        weights=jnp.zeros((n_instruments, obs_dim), dtype=jnp.float32),
        bias=jnp.asarray(bias_vec, dtype=jnp.float32),
    )


def apply_action_mask_to_action(
    action: np.ndarray,
    action_mask: np.ndarray,
) -> np.ndarray:
    """
    Zero out unavailable instruments at the policy output boundary.
    """

    action = np.asarray(action, dtype=np.float32)
    action_mask = np.asarray(action_mask, dtype=bool)
    if action.shape != action_mask.shape:
        raise ValueError(
            f"action and action_mask must have matching shapes, got {action.shape} and {action_mask.shape}"
        )
    return np.where(action_mask, action, 0.0).astype(np.float32)


def apply_linear_policy(
    params: PolicyParams,
    policy_state: PolicyState,
    observation: dict[str, np.ndarray],
    action_mask: np.ndarray,
    position_limit: float,
) -> tuple[np.ndarray, PolicyState]:
    """
    Apply a deterministic masked linear policy.
    """

    features = _flatten_observation(observation)
    logits = np.asarray(params.weights, dtype=np.float32) @ features + np.asarray(
        params.bias,
        dtype=np.float32,
    )
    raw_action = np.tanh(logits) * np.float32(position_limit)
    action = apply_action_mask_to_action(raw_action, action_mask)

    next_policy_state = PolicyState(
        hidden=policy_state.hidden,
        previous_action=jnp.asarray(action, dtype=jnp.float32),
    )
    return action, next_policy_state
