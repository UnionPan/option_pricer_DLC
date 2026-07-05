"""
Common utilities for batched calibration: padding and masked statistics.
"""
import numpy as np
import jax.numpy as jnp


def pad_returns(returns_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Pad list of variable-length return arrays to uniform shape.

    Args:
        returns_list: List of 1-D float arrays (log-returns)

    Returns:
        returns: Shape (N, T) float32, right-padded with zeros
        mask: Shape (N, T) float32, 1.0 for valid entries, 0.0 for padding

    where N = len(returns_list), T = max length.
    """
    if not returns_list:
        return np.array([]).reshape(0, 0), np.array([]).reshape(0, 0)

    n = len(returns_list)
    t = max(len(r) for r in returns_list)

    returns = np.zeros((n, t), dtype=np.float32)
    mask = np.zeros((n, t), dtype=np.float32)

    for i, r in enumerate(returns_list):
        L = len(r)
        returns[i, :L] = r
        mask[i, :L] = 1.0

    return returns, mask


def masked_mean(x, mask, axis=-1):
    """
    Compute mean along axis, ignoring masked (zero) entries.

    Args:
        x: Input array
        mask: Binary mask (1.0 = valid, 0.0 = ignore)
        axis: Reduction axis

    Returns:
        Mean of x where mask is nonzero
    """
    return jnp.sum(x * mask, axis=axis) / jnp.sum(mask, axis=axis)


def masked_var(x, mask, axis=-1):
    """
    Compute MLE variance (ddof=0) along axis, ignoring masked entries.

    Args:
        x: Input array
        mask: Binary mask (1.0 = valid, 0.0 = ignore)
        axis: Reduction axis

    Returns:
        Variance of x where mask is nonzero (ddof=0)
    """
    mean = jnp.expand_dims(masked_mean(x, mask, axis=axis), axis=axis)
    sq_dev = (x - mean) ** 2
    return jnp.sum(sq_dev * mask, axis=axis) / jnp.sum(mask, axis=axis)
