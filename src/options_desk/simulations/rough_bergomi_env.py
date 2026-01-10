"""
Rough Bergomi RL Environment

Trading environment with rough volatility spot dynamics.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    print("Warning: gymnasium not installed. Install with: pip install gymnasium")
    gym = None
    spaces = None

from options_desk.processes import RoughBergomi
from options_desk.processes.base import SimulationConfig


@dataclass
class RoughBergomiParams:
    """Rough Bergomi parameters."""
    S_0: float = 100.0
    mu: float = 0.05
    xi0: float = 0.04
    eta: float = 1.0
    rho: float = -0.7
    H: float = 0.1


class RoughBergomiEnv(gym.Env if gym else object):
    """
    Rough Bergomi trading environment (underlying only).

    Action:
        target position in underlying (continuous)

    Observation:
        dict with spot_price, time_step, portfolio_value, position
    """

    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(
        self,
        params: Optional[RoughBergomiParams] = None,
        max_steps: int = 252,
        dt: float = 1 / 252,
        initial_cash: float = 10_000.0,
        transaction_cost_pct: float = 0.001,
        position_limits: float = 100.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.params = params if params is not None else RoughBergomiParams()
        self.max_steps = max_steps
        self.dt = dt
        self.T = max_steps * dt
        self.initial_cash = initial_cash
        self.transaction_cost_pct = transaction_cost_pct
        self.position_limits = position_limits
        self.render_mode = render_mode

        self.process = RoughBergomi(
            mu=self.params.mu,
            xi0=self.params.xi0,
            eta=self.params.eta,
            rho=self.params.rho,
            H=self.params.H,
        )

        self.action_space = spaces.Box(
            low=-self.position_limits,
            high=self.position_limits,
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict({
            'spot_price': spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            'time_step': spaces.Box(low=0, high=max_steps, shape=(1,), dtype=np.float32),
            'portfolio_value': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'position': spaces.Box(
                low=-self.position_limits,
                high=self.position_limits,
                shape=(1,),
                dtype=np.float32,
            ),
        })

        self.path = None
        self.var_path = None
        self.S = None
        self.v = None
        self.t = None
        self.cash = None
        self.position = None
        self.portfolio_value = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        X0 = np.array([[self.params.S_0, self.params.xi0]])
        config = SimulationConfig(
            n_paths=1,
            n_steps=self.max_steps,
            random_seed=seed,
        )
        _, paths = self.process.simulate(
            X0=X0,
            T=self.T,
            config=config,
        )

        self.path = paths[:, 0, 0]
        self.var_path = paths[:, 0, 1]
        self.S = float(self.path[0])
        self.v = float(self.var_path[0])
        self.t = 0
        self.cash = self.initial_cash
        self.position = 0.0
        self.portfolio_value = self.cash

        return self._get_observation(), self._get_info()

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        action = np.array(action, dtype=np.float32).flatten()
        if action.shape[0] != 1:
            raise ValueError(f"Action must have shape (1,), got {action.shape}")

        target_position = float(np.clip(action[0], -self.position_limits, self.position_limits))
        position_change = target_position - self.position
        transaction_cost = abs(position_change) * self.S * self.transaction_cost_pct

        self.cash -= position_change * self.S + transaction_cost
        self.position = target_position

        self.t += 1
        terminated = self.t >= self.max_steps
        truncated = False

        if not terminated:
            self.S = float(self.path[self.t])
            self.v = float(self.var_path[self.t])

        old_portfolio_value = self.portfolio_value
        self.portfolio_value = self.cash + self.position * self.S
        reward = self.portfolio_value - old_portfolio_value

        return self._get_observation(), float(reward), terminated, truncated, self._get_info()

    def _get_observation(self) -> Dict[str, np.ndarray]:
        return {
            'spot_price': np.array([self.S], dtype=np.float32),
            'time_step': np.array([self.t], dtype=np.float32),
            'portfolio_value': np.array([self.portfolio_value], dtype=np.float32),
            'position': np.array([self.position], dtype=np.float32),
        }

    def _get_info(self) -> Dict[str, Any]:
        return {
            'S': self.S,
            'v': self.v,
            't': self.t,
            'cash': self.cash,
            'position': self.position,
            'portfolio_value': self.portfolio_value,
        }

    def render(self):
        if self.render_mode == 'human':
            print(f"Step {self.t}/{self.max_steps} | S={self.S:.4f} | v={self.v:.6f} | Pos={self.position:.2f} | PV={self.portfolio_value:.2f}")

    def close(self):
        pass
