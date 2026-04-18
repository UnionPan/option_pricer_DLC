"""
Deep Hedging Agent

Neural network-based hedging agent using policy gradient (REINFORCE).
Maps observations (spot price, option features, time step, portfolio weights)
to target positions via a configurable MLP, trained end-to-end to minimize
hedging cost.

Reference:
    Buehler, H., Gonon, L., Teichmann, J., & Wood, B. (2019).
    "Deep Hedging." Quantitative Finance, 19(8), 1271-1291.

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from options_desk.deep_hedging.utils.adapters import adapt_observation_batch
from options_desk.deep_hedging.utils.contracts import ObservationBatch

from .base import BaseHedgingAgent

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning(
        "PyTorch not installed. DeepHedgingAgent requires torch. "
        "Install with: pip install torch"
    )


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: Tuple[int, ...],
) -> "nn.Sequential":
    """
    Build an MLP with ReLU activations between hidden layers.

    The final layer has no activation -- callers apply tanh scaling externally.

    Args:
        input_dim: Dimensionality of the input feature vector.
        output_dim: Number of output units (one per instrument).
        hidden_sizes: Widths of hidden layers.

    Returns:
        A torch.nn.Sequential module.
    """
    layers: list = []
    prev = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class DeepHedgingAgent(BaseHedgingAgent):
    """
    Deep hedging agent trained via REINFORCE (policy gradient).

    Architecture
    ------------
    A small MLP maps the flattened observation vector to a mean action.
    A learned log-std parameter controls exploration noise.  During
    ``act`` the agent samples from ``N(mu, sigma)`` and squashes
    the result through ``tanh * position_limits`` so that actions always
    respect the position box.

    Training
    --------
    Episode trajectories (log-probs, rewards) are stored in memory.
    At episode termination the discounted returns are computed, a
    standardised REINFORCE loss is formed, and a single gradient step
    is taken.

    Usage::

        agent = DeepHedgingAgent(n_instruments=31, position_limits=100.0)
        agent.train_episodes(env, n_episodes=500)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        n_instruments: int,
        position_limits: float = 100.0,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        entropy_coeff: float = 0.01,
        log_std_init: float = -0.5,
        seed: Optional[int] = None,
        name: str = "DeepHedgingAgent",
    ):
        """
        Initialise deep hedging agent.

        Args:
            n_instruments: Number of tradable instruments
                (1 underlying + n_options).
            position_limits: Maximum absolute position per instrument.
            hidden_sizes: Widths of MLP hidden layers.
            learning_rate: Adam learning rate.
            gamma: Discount factor for computing returns.
            entropy_coeff: Coefficient for the entropy bonus that
                encourages exploration.
            log_std_init: Initial value for the learnable log-std
                parameter (per instrument).
            seed: Random seed for reproducibility.
            name: Agent name for logging.
        """


        super().__init__(n_instruments, position_limits, name)

        # Hyper-parameters (stored immutably for checkpointing)
        self.hidden_sizes = tuple(hidden_sizes)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff

        # Seed
        if seed is not None:
            torch.manual_seed(seed)

        # Input dimension is determined lazily on the first observation
        # because the exact option-feature length varies with the grid.
        self._input_dim: Optional[int] = None
        self._policy_net: Optional["nn.Sequential"] = None
        self._log_std: Optional["nn.Parameter"] = None
        self._optimizer: Optional["optim.Adam"] = None
        self._log_std_init = log_std_init

        # Episode trajectory buffer
        self._log_probs: list = []
        self._rewards: list = []
        self._entropies: list = []

        # Training metrics (running averages)
        self._episode_returns: list = []

    # ------------------------------------------------------------------
    # Lazy initialisation helpers
    # ------------------------------------------------------------------
    def _coerce_observation_batch(
        self,
        observation: ObservationBatch | Dict[str, np.ndarray],
        info: Dict[str, Any],
    ) -> ObservationBatch:
        """Normalise legacy env observations and shared batches."""
        if isinstance(observation, ObservationBatch):
            return observation
        return adapt_observation_batch(observation, info)

    def _obs_to_tensor(
        self,
        observation: ObservationBatch | Dict[str, np.ndarray],
        info: Dict[str, Any],
    ) -> tuple["torch.Tensor", Optional["torch.Tensor"]]:
        """
        Flatten observations into a single 1-D tensor and optional action mask.

        Shared ``ObservationBatch`` inputs already carry richer recurrent features.
        Legacy Gym observations are adapted into the same contract first.
        """
        batch = self._coerce_observation_batch(observation, info)
        parts = [
            np.asarray(batch.spot, dtype=np.float32).ravel(),
            np.asarray(batch.option_features, dtype=np.float32).ravel(),
        ]

        if batch.context_features.size == 0:
            parts.append(np.asarray(batch.time_index, dtype=np.float32).ravel())

        parts.append(np.asarray(batch.portfolio_features, dtype=np.float32).ravel())

        if batch.previous_action.size > 0:
            parts.append(np.asarray(batch.previous_action, dtype=np.float32).ravel())
        if batch.context_features.size > 0:
            parts.append(np.asarray(batch.context_features, dtype=np.float32).ravel())

        flat = np.concatenate(parts)
        mask = None
        if batch.action_mask is not None:
            mask = torch.from_numpy(
                np.asarray(batch.action_mask, dtype=np.float32).reshape(1, -1)
            )
        return torch.from_numpy(flat).unsqueeze(0), mask  # (1, D)

    def _lazy_init(self, input_dim: int) -> None:
        """Build network and optimiser on first observation."""
        self._input_dim = input_dim
        self._policy_net = _build_mlp(
            input_dim, self.n_instruments, self.hidden_sizes
        )
        self._log_std = nn.Parameter(
            torch.full((self.n_instruments,), self._log_std_init)
        )
        self._optimizer = optim.Adam(
            list(self._policy_net.parameters()) + [self._log_std],
            lr=self.learning_rate,
        )
        logger.info(
            "DeepHedgingAgent network initialised: input_dim=%d, "
            "hidden=%s, output_dim=%d",
            input_dim,
            self.hidden_sizes,
            self.n_instruments,
        )

    # ------------------------------------------------------------------
    # BaseHedgingAgent interface
    # ------------------------------------------------------------------
    def reset(
        self,
        observation: ObservationBatch | Dict[str, np.ndarray],
        info: Dict[str, Any],
    ) -> None:
        """Reset agent state and trajectory buffer at episode start."""
        super().reset(observation, info)
        self._log_probs = []
        self._rewards = []
        self._entropies = []

    def act(
        self,
        observation: ObservationBatch | Dict[str, np.ndarray],
        info: Dict[str, Any],
    ) -> np.ndarray:
        """
        Sample an action from the policy and record the log-probability.

        Args:
            observation: Dict observation from HestonEnv.
            info: Auxiliary information from the environment.

        Returns:
            action: Target positions of shape ``(n_instruments,)``.
        """
        x, action_mask = self._obs_to_tensor(observation, info)

        # Lazy network construction on first call
        if self._policy_net is None:
            self._lazy_init(x.shape[1])

        # Forward pass
        mean = self._policy_net(x)  # (1, n_instruments)
        std = self._log_std.exp().unsqueeze(0)  # (1, n_instruments)
        dist = Normal(mean, std)

        raw_action = dist.rsample()  # reparameterised sample

        # Squash through tanh and scale to position limits
        squashed = torch.tanh(raw_action)
        if action_mask is not None:
            squashed = squashed * action_mask
        action_tensor = squashed * self.position_limits

        # Log-probability under the *squashed* distribution
        # Correction for tanh: log p(a) = log p(u) - sum log(1 - tanh^2(u))
        log_prob = dist.log_prob(raw_action) - torch.log(
            1.0 - squashed.pow(2) + 1e-6
        )
        if action_mask is not None:
            log_prob = log_prob * action_mask
        log_prob = log_prob.sum(dim=-1)  # sum over instruments

        entropy = dist.entropy().sum(dim=-1)
        if action_mask is not None:
            entropy = (dist.entropy() * action_mask).sum(dim=-1)

        self._log_probs.append(log_prob.squeeze(0))
        self._entropies.append(entropy.squeeze(0))

        action = action_tensor.detach().squeeze(0).numpy().astype(np.float32)
        self.current_positions = action.copy()
        return action

    def update(
        self,
        observation: Dict[str, np.ndarray],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Store reward and, at episode end, run a REINFORCE gradient step.

        Args:
            observation: Current observation after the step.
            reward: Scalar reward from the environment.
            terminated: Whether the episode ended normally.
            truncated: Whether the episode was cut short.
            info: Auxiliary information.

        Returns:
            metrics: Empty dict mid-episode; contains ``policy_loss``,
                ``mean_return``, and ``entropy`` at episode end.
        """
        self.step_count += 1
        self._rewards.append(float(reward))

        if not (terminated or truncated):
            return {}

        # --- End of episode: compute REINFORCE update ---
        returns = self._compute_returns()
        episode_return = float(sum(self._rewards))
        self._episode_returns.append(episode_return)

        # Standardise returns for variance reduction
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        if returns_tensor.numel() > 1:
            std = returns_tensor.std() + 1e-8
            returns_tensor = (returns_tensor - returns_tensor.mean()) / std

        # Policy gradient loss: -E[log pi(a|s) * G]
        policy_loss = torch.tensor(0.0)
        entropy_bonus = torch.tensor(0.0)
        for log_prob, G, ent in zip(
            self._log_probs, returns_tensor, self._entropies
        ):
            policy_loss = policy_loss - log_prob * G.detach()
            entropy_bonus = entropy_bonus + ent

        # Average over time steps
        n_steps = max(len(self._log_probs), 1)
        loss = policy_loss / n_steps - self.entropy_coeff * entropy_bonus / n_steps

        # Gradient step
        self._optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(
            list(self._policy_net.parameters()) + [self._log_std],
            max_norm=1.0,
        )
        self._optimizer.step()

        metrics = {
            "policy_loss": float(policy_loss.item()) / n_steps,
            "entropy": float(entropy_bonus.item()) / n_steps,
            "episode_return": episode_return,
            "mean_return_50": float(
                np.mean(self._episode_returns[-50:])
            ),
        }

        return metrics

    # ------------------------------------------------------------------
    # Convenience training loop
    # ------------------------------------------------------------------
    def train_episodes(
        self,
        env: Any,
        n_episodes: int = 500,
        log_interval: int = 50,
        seed: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        """
        Run a training loop over *n_episodes*.

        Args:
            env: A Gymnasium-compatible environment (e.g. HestonEnv).
            n_episodes: Number of episodes to train.
            log_interval: Print progress every *log_interval* episodes.
            seed: Seed passed to ``env.reset``.

        Returns:
            all_metrics: Per-episode metric dicts (only non-empty ones,
                i.e. one per episode).
        """
        all_metrics: list = []

        for ep in range(1, n_episodes + 1):
            reset_kwargs: Dict[str, Any] = {}
            if seed is not None:
                reset_kwargs["seed"] = seed + ep

            obs, info = env.reset(**reset_kwargs)
            self.reset(obs, info)

            done = False
            metrics: Dict[str, float] = {}
            while not done:
                action = self.act(obs, info)
                obs, reward, terminated, truncated, info = env.step(action)
                metrics = self.update(obs, reward, terminated, truncated, info)
                done = terminated or truncated

            if metrics:
                all_metrics.append(metrics)

            if ep % log_interval == 0:
                recent = all_metrics[-log_interval:] if all_metrics else []
                mean_ret = (
                    np.mean([m["episode_return"] for m in recent])
                    if recent
                    else float("nan")
                )
                mean_loss = (
                    np.mean([m["policy_loss"] for m in recent])
                    if recent
                    else float("nan")
                )
                logger.info(
                    "[%s] Episode %d/%d  mean_return=%.4f  mean_loss=%.4f",
                    self.name,
                    ep,
                    n_episodes,
                    mean_ret,
                    mean_loss,
                )

        return all_metrics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_returns(self) -> List[float]:
        """Compute discounted cumulative returns (reward-to-go)."""
        returns: list = []
        g = 0.0
        for r in reversed(self._rewards):
            g = r + self.gamma * g
            returns.insert(0, g)
        return returns

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        """Get agent state including network weights for checkpointing."""
        state = super().get_state()
        state["hidden_sizes"] = self.hidden_sizes
        state["learning_rate"] = self.learning_rate
        state["gamma"] = self.gamma
        state["entropy_coeff"] = self.entropy_coeff
        state["episode_returns"] = list(self._episode_returns)

        if self._policy_net is not None:
            state["policy_net_state_dict"] = self._policy_net.state_dict()
            state["log_std"] = self._log_std.data.clone()
            state["optimizer_state_dict"] = self._optimizer.state_dict()
            state["input_dim"] = self._input_dim

        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore agent state from checkpoint."""
        super().set_state(state)
        self._episode_returns = state.get("episode_returns", [])

        if "policy_net_state_dict" in state:
            input_dim = state["input_dim"]
            self._lazy_init(input_dim)
            self._policy_net.load_state_dict(state["policy_net_state_dict"])
            self._log_std.data = state["log_std"]
            self._optimizer.load_state_dict(state["optimizer_state_dict"])

    def __repr__(self) -> str:
        net_status = (
            f"input_dim={self._input_dim}" if self._input_dim else "uninitialised"
        )
        return (
            f"DeepHedgingAgent("
            f"name='{self.name}', "
            f"n_instruments={self.n_instruments}, "
            f"hidden_sizes={self.hidden_sizes}, "
            f"lr={self.learning_rate}, "
            f"gamma={self.gamma}, "
            f"net={net_status})"
        )
