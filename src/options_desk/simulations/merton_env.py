"""
Merton Jump-Diffusion RL Environment

Trading environment with jump-diffusion spot dynamics.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Any, Optional, Tuple, List

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    print("Warning: gymnasium not installed. Install with: pip install gymnasium")
    gym = None
    spaces = None

from options_desk.processes.merton import MertonJD
from options_desk.processes.base import SimulationConfig
from options_desk.calibration.data.synthetic_merton_equity import (
    SyntheticMertonOptionChainGenerator,
    MertonVolatilityProfile,
)


@dataclass
class MertonParams:
    """Merton jump-diffusion parameters."""
    S_0: float = 100.0
    mu: float = 0.05
    sigma: float = 0.2
    lambda_jump: float = 0.2
    mu_J: float = -0.05
    sigma_J: float = 0.2


@dataclass
class Liability:
    """Liability to be hedged."""
    option_type: str
    strike: float
    maturity_days: int
    quantity: float = -1.0
    initial_price: float = None
    current_price: float = None
    payoff: float = None


class MertonEnv(gym.Env if gym else object):
    """
    Merton jump-diffusion trading environment (underlying only).

    Action:
        target position in underlying (continuous)

    Observation:
        dict with spot_price, time_step, portfolio_value, position
    """

    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(
        self,
        params: Optional[MertonParams] = None,
        max_steps: int = 252,
        dt: float = 1 / 252,
        discretization: str = "euler",
        include_options: bool = True,
        option_maturities: Optional[List[int]] = None,
        option_moneyness: Optional[List[float]] = None,
        task: str = "trading",
        liability: Optional[Liability] = None,
        initial_cash: float = 10_000.0,
        transaction_cost_pct: float = 0.001,
        position_limits: float = 100.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.params = params if params is not None else MertonParams()
        self.max_steps = max_steps
        self.dt = dt
        self.T = max_steps * dt
        self.discretization = discretization
        self.include_options = include_options
        self.task = task
        self.initial_cash = initial_cash
        self.transaction_cost_pct = transaction_cost_pct
        self.position_limits = position_limits
        self.render_mode = render_mode

        self.process = MertonJD(
            mu=self.params.mu,
            sigma=self.params.sigma,
            lambda_jump=self.params.lambda_jump,
            mu_J=self.params.mu_J,
            sigma_J=self.params.sigma_J,
        )

        if task == "hedging" and liability is None:
            liability = Liability(
                option_type="call",
                strike=self.params.S_0,
                maturity_days=90,
                quantity=-1.0,
            )
        self.liability = liability

        if option_maturities is None:
            option_maturities = [30, 60, 90]
        if option_moneyness is None:
            option_moneyness = [0.95, 0.97, 0.99, 1.0, 1.01, 1.03, 1.05]

        self.option_maturities = sorted(option_maturities)
        self.option_moneyness = sorted(option_moneyness)
        self.option_grid = {ttm: self.option_moneyness for ttm in self.option_maturities}
        self.n_options = len(self.option_maturities) * len(self.option_moneyness) * 2 if include_options else 0
        self.n_instruments = 1 + self.n_options

        self.option_generator = None
        if self.include_options:
            self.option_generator = SyntheticMertonOptionChainGenerator(
                maturities_days=self.option_maturities,
                moneyness_by_maturity=self.option_grid,
                risk_free_rate=0.03,
                add_noise=False,
                random_seed=42,
            )

        self.action_space = spaces.Box(
            low=-self.position_limits,
            high=self.position_limits,
            shape=(self.n_instruments,),
            dtype=np.float32,
        )

        if self.include_options:
            self.option_feature_dim = self.n_options * 2
            self.observation_space = spaces.Dict({
                'spot_price': spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
                'option_features': spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.option_feature_dim,),
                    dtype=np.float32
                ),
                'time_step': spaces.Box(low=0, high=max_steps, shape=(1,), dtype=np.float32),
                'portfolio_weights': spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.n_instruments,),
                    dtype=np.float32
                ),
            })
        else:
            self.option_feature_dim = 0
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
        self.S = None
        self.t = None
        self.cash = None
        self.positions = None
        self.portfolio_value = None
        self.current_option_chain = None
        self.option_grid_prices = None
        self.liability_mtm = None
        self.hedge_portfolio_value = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        X0 = np.array([[self.params.S_0]])
        config = SimulationConfig(
            n_paths=1,
            n_steps=self.max_steps,
            random_seed=seed,
        )
        _, paths = self.process.simulate(
            X0=X0,
            T=self.T,
            config=config,
            scheme=self.discretization,
        )

        self.path = paths[:, 0, 0]
        self.S = float(self.path[0])
        self.t = 0
        self.cash = self.initial_cash
        self.positions = np.zeros(self.n_instruments, dtype=np.float32)
        self.portfolio_value = self.cash

        if self.include_options:
            self._generate_option_chain()

        if self.liability is not None:
            self._price_liability()
            if self.liability.initial_price is None:
                self.liability.initial_price = self.liability_mtm
            self.hedge_portfolio_value = self.portfolio_value + self.liability_mtm
        else:
            self.liability_mtm = 0.0
            self.hedge_portfolio_value = self.portfolio_value

        return self._get_observation(), self._get_info()

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        action = np.array(action, dtype=np.float32).flatten()
        if action.shape[0] != self.n_instruments:
            raise ValueError(f"Action must have shape ({self.n_instruments},), got {action.shape}")

        action = np.clip(action, -self.position_limits, self.position_limits)
        old_positions = self.positions.copy()
        position_changes = action - old_positions

        transaction_cost = self._compute_transaction_cost(position_changes)
        self.positions = action
        self.cash -= transaction_cost

        self.cash -= position_changes[0] * self.S

        # Move to next step
        self.t += 1
        terminated = self.t >= self.max_steps
        truncated = False

        if not terminated:
            self.S = float(self.path[self.t])
            if self.include_options:
                self._generate_option_chain()

        old_portfolio_value = self.portfolio_value
        self.portfolio_value = self.cash + self.positions[0] * self.S
        if self.include_options and self.option_grid_prices is not None:
            self.portfolio_value += float(np.sum(self.positions[1:] * self.option_grid_prices))

        if self.liability is not None:
            self._price_liability()
            self.hedge_portfolio_value = self.portfolio_value + self.liability_mtm

        reward = self.portfolio_value - old_portfolio_value

        return self._get_observation(), float(reward), terminated, truncated, self._get_info()

    def _get_observation(self) -> Dict[str, np.ndarray]:
        if self.include_options:
            option_features = self._vectorize_option_chain()
            portfolio_weights = self.positions / (self.position_limits + 1e-8)
            return {
                'spot_price': np.array([self.S], dtype=np.float32),
                'option_features': option_features,
                'time_step': np.array([self.t], dtype=np.float32),
                'portfolio_weights': portfolio_weights.astype(np.float32),
            }

        return {
            'spot_price': np.array([self.S], dtype=np.float32),
            'time_step': np.array([self.t], dtype=np.float32),
            'portfolio_value': np.array([self.portfolio_value], dtype=np.float32),
            'position': np.array([self.positions[0]], dtype=np.float32),
        }

    def _get_info(self) -> Dict[str, Any]:
        return {
            'S': self.S,
            't': self.t,
            'cash': self.cash,
            'positions': self.positions.copy(),
            'portfolio_value': self.portfolio_value,
            'n_instruments': self.n_instruments,
            'option_chain': self.current_option_chain,
            'option_grid_prices': self.option_grid_prices,
            'liability_mtm': self.liability_mtm,
        }

    def render(self):
        if self.render_mode == 'human':
            print(f"Step {self.t}/{self.max_steps} | S={self.S:.4f} | PV={self.portfolio_value:.2f}")

    def close(self):
        pass

    def _generate_option_chain(self):
        if not self.include_options:
            return
        vol_profile = MertonVolatilityProfile(
            sigma=self.params.sigma,
            lambda_jump=self.params.lambda_jump,
            mu_J=self.params.mu_J,
            sigma_J=self.params.sigma_J,
            atm_iv=max(np.sqrt(self.params.sigma**2 + self.params.lambda_jump * (self.params.mu_J**2 + self.params.sigma_J**2)), 0.01),
        )
        reference_date = date(2024, 1, 1) + timedelta(days=int(self.t))
        self.current_option_chain = self.option_generator.generate_single_chain(
            reference_date=reference_date,
            spot_price=self.S,
            vol_profile=vol_profile,
        )
        self.option_grid_prices = self._extract_grid_prices()

    def _extract_grid_prices(self) -> np.ndarray:
        if self.current_option_chain is None:
            return np.zeros(self.n_options, dtype=np.float32)
        prices = []
        all_options = self.current_option_chain.options
        for ttm in sorted(self.option_grid.keys()):
            moneyness_list = self.option_grid[ttm]
            ttm_options = [
                opt for opt in all_options
                if abs((opt.expiry - self.current_option_chain.reference_date).days - ttm) < 1
            ]
            for moneyness_target in sorted(moneyness_list):
                tolerance = 0.01
                matching_calls = [
                    opt for opt in ttm_options
                    if opt.option_type == 'call' and abs(opt.strike / self.S - moneyness_target) < tolerance
                ]
                matching_puts = [
                    opt for opt in ttm_options
                    if opt.option_type == 'put' and abs(opt.strike / self.S - moneyness_target) < tolerance
                ]
                prices.append(matching_calls[0].mid if matching_calls else 0.0)
                prices.append(matching_puts[0].mid if matching_puts else 0.0)
        return np.array(prices, dtype=np.float32)

    def _vectorize_option_chain(self) -> np.ndarray:
        if self.current_option_chain is None:
            return np.zeros(self.option_feature_dim, dtype=np.float32)
        features = []
        all_options = self.current_option_chain.options
        for ttm in sorted(self.option_grid.keys()):
            moneyness_list = self.option_grid[ttm]
            ttm_options = [
                opt for opt in all_options
                if abs((opt.expiry - self.current_option_chain.reference_date).days - ttm) < 1
            ]
            for moneyness_target in sorted(moneyness_list):
                tolerance = 0.01
                matching_calls = [
                    opt for opt in ttm_options
                    if opt.option_type == 'call' and abs(opt.strike / self.S - moneyness_target) < tolerance
                ]
                matching_puts = [
                    opt for opt in ttm_options
                    if opt.option_type == 'put' and abs(opt.strike / self.S - moneyness_target) < tolerance
                ]
                if matching_calls:
                    opt = matching_calls[0]
                    features.extend([opt.mid / (self.S + 1e-8), opt.implied_volatility or 0.0])
                else:
                    features.extend([0.0, 0.0])
                if matching_puts:
                    opt = matching_puts[0]
                    features.extend([opt.mid / (self.S + 1e-8), opt.implied_volatility or 0.0])
                else:
                    features.extend([0.0, 0.0])
        return np.array(features, dtype=np.float32)

    def _compute_transaction_cost(self, position_changes: np.ndarray) -> float:
        total_cost = 0.0
        underlying_change = abs(position_changes[0])
        total_cost += underlying_change * self.S * self.transaction_cost_pct

        if self.include_options and self.option_grid_prices is not None:
            option_changes = np.abs(position_changes[1:])
            option_cost = np.sum(option_changes * self.option_grid_prices) * self.transaction_cost_pct
            total_cost += option_cost
        return float(total_cost)

    def _price_liability(self):
        if self.liability is None or self.current_option_chain is None:
            self.liability_mtm = 0.0
            return
        # Find closest option in chain by moneyness/maturity
        target_maturity = self.liability.maturity_days
        best_opt = None
        best_distance = float('inf')
        for opt in self.current_option_chain.options:
            if opt.option_type != self.liability.option_type:
                continue
            days_to_expiry = (opt.expiry - self.current_option_chain.reference_date).days
            distance = abs(days_to_expiry - target_maturity) + abs(opt.strike - self.liability.strike)
            if distance < best_distance:
                best_distance = distance
                best_opt = opt
        if best_opt is None:
            self.liability_mtm = 0.0
        else:
            self.liability_mtm = self.liability.quantity * best_opt.mid
