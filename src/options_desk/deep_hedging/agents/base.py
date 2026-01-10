"""
Base Agent Classes for Hedging Strategies

Provides abstract base classes for:
- Hedging agents (delta hedging, deep hedging, RL agents)
- Consistent interface for action selection
- State management and observation processing

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np


class BaseHedgingAgent(ABC):
    """
    Abstract base class for hedging agents.

    All hedging agents (delta hedging, deep hedging, RL agents) should inherit
    from this class and implement the abstract methods.

    Interface:
        - reset(observation, info): Initialize agent state at episode start
        - select_action(observation, info): Choose action given observation
        - update(observation, reward, terminated, truncated, info): Learn from experience (optional)
    """

    def __init__(
        self,
        n_instruments: int,
        position_limits: float = 100.0,
        name: str = "BaseAgent",
    ):
        """
        Initialize base hedging agent.

        Args:
            n_instruments: Number of tradable instruments (1 underlying + n_options)
            position_limits: Max absolute position per instrument
            name: Agent name for logging
        """
        self.n_instruments = n_instruments
        self.position_limits = position_limits
        self.name = name

        # Agent state
        self.current_positions = None
        self.episode_count = 0
        self.step_count = 0

    @abstractmethod
    def reset(
        self,
        observation: Dict[str, np.ndarray],
        info: Dict[str, Any],
    ) -> None:
        """
        Reset agent state at the beginning of an episode.

        Args:
            observation: Initial observation from environment
            info: Initial info dict from environment
        """
        self.current_positions = np.zeros(self.n_instruments, dtype=np.float32)
        self.episode_count += 1
        self.step_count = 0

    @abstractmethod
    def select_action(
        self,
        observation: Dict[str, np.ndarray],
        info: Dict[str, Any],
    ) -> np.ndarray:
        """
        Select action given current observation.

        Args:
            observation: Current observation dict with keys:
                - 'spot_price': Current spot price
                - 'option_features': Option prices and IVs
                - 'time_step': Current time step
                - 'portfolio_weights': Current portfolio weights
            info: Info dict with additional state information:
                - 'S': Spot price
                - 'v': Variance (hidden from observation)
                - 't': Time step
                - 'cash': Cash balance
                - 'positions': Current positions
                - 'option_chain': Current option chain object
                - etc.

        Returns:
            action: Array of shape (n_instruments,) with target positions
                - action[0]: Target underlying position
                - action[1:]: Target positions for each option on grid
        """
        raise NotImplementedError

    def update(
        self,
        observation: Dict[str, np.ndarray],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Update agent from experience (for learning agents).

        Args:
            observation: Current observation
            reward: Reward received
            terminated: Episode terminated early
            truncated: Episode truncated (max steps)
            info: Additional info

        Returns:
            metrics: Dict of training metrics (loss, etc.)
        """
        # Base implementation does nothing (for model-free agents like delta hedging)
        self.step_count += 1
        return {}

    def get_state(self) -> Dict[str, Any]:
        """Get agent state for checkpointing."""
        return {
            'name': self.name,
            'n_instruments': self.n_instruments,
            'position_limits': self.position_limits,
            'current_positions': self.current_positions.copy() if self.current_positions is not None else None,
            'episode_count': self.episode_count,
            'step_count': self.step_count,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore agent state from checkpoint."""
        self.current_positions = state.get('current_positions')
        self.episode_count = state.get('episode_count', 0)
        self.step_count = state.get('step_count', 0)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"n_instruments={self.n_instruments}, "
            f"position_limits={self.position_limits})"
        )


class RandomAgent(BaseHedgingAgent):
    """
    Random agent for baseline comparison.

    Samples random actions from the action space.
    """

    def __init__(
        self,
        n_instruments: int,
        position_limits: float = 100.0,
        scale: float = 0.1,
        seed: Optional[int] = None,
    ):
        """
        Initialize random agent.

        Args:
            n_instruments: Number of tradable instruments
            position_limits: Max absolute position per instrument
            scale: Scale factor for random actions (0.1 = 10% of limits)
            seed: Random seed for reproducibility
        """
        super().__init__(n_instruments, position_limits, name="RandomAgent")
        self.scale = scale
        self.rng = np.random.RandomState(seed)

    def reset(
        self,
        observation: Dict[str, np.ndarray],
        info: Dict[str, Any],
    ) -> None:
        """Reset agent state."""
        super().reset(observation, info)

    def select_action(
        self,
        observation: Dict[str, np.ndarray],
        info: Dict[str, Any],
    ) -> np.ndarray:
        """Select random action."""
        action = self.rng.uniform(
            low=-self.position_limits * self.scale,
            high=self.position_limits * self.scale,
            size=self.n_instruments,
        ).astype(np.float32)

        self.current_positions = action
        return action


class DoNothingAgent(BaseHedgingAgent):
    """
    Do-nothing agent for baseline comparison.

    Always returns zero action (no trading).
    """

    def __init__(
        self,
        n_instruments: int,
        position_limits: float = 100.0,
    ):
        """Initialize do-nothing agent."""
        super().__init__(n_instruments, position_limits, name="DoNothingAgent")

    def reset(
        self,
        observation: Dict[str, np.ndarray],
        info: Dict[str, Any],
    ) -> None:
        """Reset agent state."""
        super().reset(observation, info)

    def select_action(
        self,
        observation: Dict[str, np.ndarray],
        info: Dict[str, Any],
    ) -> np.ndarray:
        """Select zero action."""
        action = np.zeros(self.n_instruments, dtype=np.float32)
        self.current_positions = action
        return action
