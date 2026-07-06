"""
Conditional Mixture Density Network for NPE posterior approximation.

Implements a conditional MDN with diagonal Gaussian components:
- Input: summary features s ∈ ℝ¹⁶
- Output: mixture of K diagonal Gaussians over θ ∈ ℝ⁶
- Architecture: MLP with ReLU activations
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple


class ConditionalMDN(nn.Module):
    """
    Conditional Mixture Density Network.

    Architecture:
    - Hidden layers with ReLU activations
    - Output heads for mixture logits, component means, and component log-scales
    - K diagonal Gaussian components

    Attributes:
        hidden_dims: tuple of hidden layer sizes (default: (128, 128))
        n_components: number of mixture components (default: 8)
        n_outputs: output dimension (default: 6 for Heston parameters)
    """
    hidden_dims: Tuple[int, ...] = (128, 128)
    n_components: int = 8
    n_outputs: int = 6

    @nn.compact
    def __call__(self, s):
        """
        Forward pass.

        Args:
            s: (B, 16) summary features

        Returns:
            logits: (B, K) unnormalized mixture weights
            means: (B, K, 6) component means
            log_scales: (B, K, 6) component log standard deviations
        """
        # Shared trunk
        x = s
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)

        # Mixture logits (unnormalized log-weights)
        logits = nn.Dense(self.n_components)(x)

        # Component means
        means = nn.Dense(self.n_components * self.n_outputs)(x)
        means = means.reshape(-1, self.n_components, self.n_outputs)

        # Component log-scales (will be clipped during training)
        log_scales = nn.Dense(self.n_components * self.n_outputs)(x)
        log_scales = log_scales.reshape(-1, self.n_components, self.n_outputs)

        return logits, means, log_scales


def mdn_nll(params, apply_fn, s, z):
    """
    Compute negative log-likelihood for conditional MDN.

    Uses stable logsumexp for numerically robust mixture density evaluation:
        log p(z|s) = logsumexp_k [log π_k + log 𝒩(z | μ_k, Σ_k)]

    Args:
        params: Flax model parameters
        apply_fn: model.apply function
        s: (B, 16) summary features
        z: (B, 6) target parameters (unconstrained, standardized)

    Returns:
        scalar: mean negative log-likelihood over batch
    """
    # Forward pass
    logits, means, log_scales = apply_fn(params, s)

    # Clip log_scales to [-7, 2] for numerical stability
    log_scales = jnp.clip(log_scales, -7.0, 2.0)

    # Compute log π_k (normalized log mixture weights)
    log_mix_weights = jax.nn.log_softmax(logits, axis=-1)  # (B, K)

    # Compute log 𝒩(z | μ_k, diag(σ_k²)) for each component
    # z: (B, 6) -> (B, 1, 6) for broadcasting
    # means: (B, K, 6)
    # log_scales: (B, K, 6)

    z_expanded = z[:, None, :]  # (B, 1, 6)

    # Log of diagonal Gaussian density:
    # log 𝒩(z | μ, σ²) = -0.5 * [log(2π) + 2*log(σ) + ((z-μ)/σ)²]
    #                  = -0.5 * [D*log(2π) + sum_d(2*log(σ_d) + ((z_d-μ_d)/σ_d)²)]

    const = -0.5 * jnp.log(2.0 * jnp.pi)  # per dimension

    # Per-dimension log density contributions
    log_scales_2 = 2.0 * log_scales  # (B, K, 6)
    z_centered = z_expanded - means  # (B, K, 6)
    mahalanobis = (z_centered / jnp.exp(log_scales)) ** 2  # (B, K, 6)

    # Sum over dimensions to get log density per component
    log_probs = const * z.shape[-1] - 0.5 * jnp.sum(log_scales_2 + mahalanobis, axis=-1)  # (B, K)

    # Combine with mixture weights: log p(z|s) = logsumexp_k [log π_k + log p_k(z)]
    log_likelihood = jax.nn.logsumexp(log_mix_weights + log_probs, axis=-1)  # (B,)

    # Return mean negative log-likelihood
    return -jnp.mean(log_likelihood)
