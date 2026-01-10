"""
Heston cache builder and cached RL environment.

Uses Zarr to store precomputed spot/variance paths and option chains.
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

try:
    import zarr
except ImportError:  # pragma: no cover
    zarr = None

from options_desk.simulations.heston_env import HestonParams, Liability
from options_desk.processes import Heston
from options_desk.processes.base import SimulationConfig
from options_desk.calibration.data.synthetic_equity import (
    SyntheticEquityOptionChainGenerator,
    HestonVolatilityProfile,
)


def build_heston_cache(
    store_path: str,
    n_paths: int,
    max_steps: int,
    dt: float,
    params: Optional[HestonParams] = None,
    option_maturities: Optional[List[int]] = None,
    option_moneyness: Optional[List[float]] = None,
    seed: Optional[int] = None,
    chunk_paths: int = 32,
    chunk_steps: int = 4,
) -> None:
    """
    Build a Zarr cache with precomputed Heston paths + option chains.

    Stored arrays:
      - spot: (n_paths, n_steps+1)
      - variance: (n_paths, n_steps+1)
      - option_prices: (n_paths, n_steps+1, n_options)
      - option_features: (n_paths, n_steps+1, n_feature_dim)
    """
    if zarr is None:
        raise ImportError("zarr is required. Install with: pip install zarr")

    params = params if params is not None else HestonParams()
    if option_maturities is None:
        option_maturities = [30, 60, 90]
    if option_moneyness is None:
        option_moneyness = [0.95, 0.97, 0.99, 1.0, 1.01, 1.03, 1.05]

    option_maturities = sorted(option_maturities)
    option_moneyness = sorted(option_moneyness)
    option_grid = {ttm: option_moneyness for ttm in option_maturities}
    n_options = len(option_maturities) * len(option_moneyness) * 2
    feature_dim = n_options * 2

    if seed is not None:
        np.random.seed(seed)

    process = Heston(
        mu=params.mu,
        kappa=params.kappa,
        theta=params.theta,
        sigma_v=params.xi,
        rho=params.rho,
        v0=params.v_0,
        variance_scheme="truncation",
    )
    config = SimulationConfig(n_paths=n_paths, n_steps=max_steps, random_seed=seed)
    _, paths = process.simulate(
        X0=np.array([[params.S_0, params.v_0]]),
        T=max_steps * dt,
        config=config,
        scheme="milstein",
    )

    generator = SyntheticEquityOptionChainGenerator(
        maturities_days=option_maturities,
        moneyness_by_maturity=option_grid,
        add_noise=False,
        random_seed=42,
    )

    root = zarr.open_group(store_path, mode="w")
    root.attrs.update({
        "model": "heston",
        "n_paths": n_paths,
        "n_steps": max_steps,
        "dt": dt,
        "option_maturities": option_maturities,
        "option_moneyness": option_moneyness,
    })

    spot_arr = root.create_dataset(
        "spot",
        shape=(n_paths, max_steps + 1),
        chunks=(chunk_paths, chunk_steps),
        dtype="f4",
    )
    var_arr = root.create_dataset(
        "variance",
        shape=(n_paths, max_steps + 1),
        chunks=(chunk_paths, chunk_steps),
        dtype="f4",
    )
    option_prices_arr = root.create_dataset(
        "option_prices",
        shape=(n_paths, max_steps + 1, n_options),
        chunks=(chunk_paths, chunk_steps, n_options),
        dtype="f4",
    )
    option_features_arr = root.create_dataset(
        "option_features",
        shape=(n_paths, max_steps + 1, feature_dim),
        chunks=(chunk_paths, chunk_steps, feature_dim),
        dtype="f4",
    )

    for p in range(n_paths):
        S_path = paths[:, p, 0]
        v_path = paths[:, p, 1]
        spot_arr[p, :] = S_path.astype(np.float32)
        var_arr[p, :] = v_path.astype(np.float32)

        for t in range(max_steps + 1):
            S_t = float(S_path[t])
            v_t = float(v_path[t])

            vol_profile = HestonVolatilityProfile(
                kappa=params.kappa,
                theta=params.theta,
                xi=params.xi,
                rho=params.rho,
                v0=v_t,
                atm_iv=max(np.sqrt(max(v_t, 0.0)), 0.01),
            )
            ref_date = date(2024, 1, 1) + timedelta(days=int(t * dt * 365))
            chain = generator.generate_single_chain(
                reference_date=ref_date,
                spot_price=S_t,
                vol_profile=vol_profile,
            )

            option_prices = _extract_grid_prices(chain, S_t, option_grid)
            option_features = _vectorize_option_chain(chain, S_t, option_grid)

            option_prices_arr[p, t, :] = option_prices.astype(np.float32)
            option_features_arr[p, t, :] = option_features.astype(np.float32)


def _extract_grid_prices(option_chain: Any, S: float, option_grid: Dict[int, List[float]]) -> np.ndarray:
    if option_chain is None:
        return np.zeros(sum(len(v) for v in option_grid.values()) * 2, dtype=np.float32)

    prices = []
    all_options = option_chain.options
    for ttm in sorted(option_grid.keys()):
        moneyness_list = option_grid[ttm]
        ttm_options = [
            opt for opt in all_options
            if abs((opt.expiry - option_chain.reference_date).days - ttm) < 1
        ]
        for moneyness_target in sorted(moneyness_list):
            tolerance = 0.01
            matching_calls = [
                opt for opt in ttm_options
                if opt.option_type == 'call' and abs(opt.strike / S - moneyness_target) < tolerance
            ]
            matching_puts = [
                opt for opt in ttm_options
                if opt.option_type == 'put' and abs(opt.strike / S - moneyness_target) < tolerance
            ]
            prices.append(matching_calls[0].mid if matching_calls else 0.0)
            prices.append(matching_puts[0].mid if matching_puts else 0.0)

    return np.array(prices, dtype=np.float32)


def _vectorize_option_chain(option_chain: Any, S: float, option_grid: Dict[int, List[float]]) -> np.ndarray:
    if option_chain is None:
        feature_dim = sum(len(v) for v in option_grid.values()) * 4
        return np.zeros(feature_dim, dtype=np.float32)

    features = []
    all_options = option_chain.options
    for ttm in sorted(option_grid.keys()):
        moneyness_list = option_grid[ttm]
        ttm_options = [
            opt for opt in all_options
            if abs((opt.expiry - option_chain.reference_date).days - ttm) < 1
        ]
        for moneyness_target in sorted(moneyness_list):
            tolerance = 0.01
            matching_calls = [
                opt for opt in ttm_options
                if opt.option_type == 'call' and abs(opt.strike / S - moneyness_target) < tolerance
            ]
            matching_puts = [
                opt for opt in ttm_options
                if opt.option_type == 'put' and abs(opt.strike / S - moneyness_target) < tolerance
            ]
            if matching_calls:
                opt = matching_calls[0]
                features.extend([opt.mid / (S + 1e-8), opt.implied_volatility or 0.0])
            else:
                features.extend([0.0, 0.0])
            if matching_puts:
                opt = matching_puts[0]
                features.extend([opt.mid / (S + 1e-8), opt.implied_volatility or 0.0])
            else:
                features.extend([0.0, 0.0])

    return np.array(features, dtype=np.float32)


class CachedHestonEnv(gym.Env if gym else object):
    """
    Heston env that reads precomputed paths + option chains from Zarr.
    """

    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(
        self,
        store_path: str,
        task: str = "hedging",
        liability: Optional[Liability] = None,
        initial_cash: float = 10.0,
        transaction_cost_pct: float = 0.001,
        position_limits: float = 100.0,
        hedge_error_penalty: float = 1.0,
        render_mode: Optional[str] = None,
        random_seed: Optional[int] = None,
    ):
        if zarr is None:
            raise ImportError("zarr is required. Install with: pip install zarr")

        self.root = zarr.open_group(store_path, mode="r")
        self.spot_arr = self.root["spot"]
        self.var_arr = self.root["variance"]
        self.option_prices_arr = self.root["option_prices"]
        self.option_features_arr = self.root["option_features"]

        self.n_paths = self.spot_arr.shape[0]
        self.max_steps = self.spot_arr.shape[1] - 1
        self.dt = float(self.root.attrs.get("dt", 1 / 252))
        self.option_maturities = list(self.root.attrs.get("option_maturities", []))
        self.option_moneyness = list(self.root.attrs.get("option_moneyness", []))
        self.option_grid = {ttm: self.option_moneyness for ttm in self.option_maturities}
        self.n_options = len(self.option_maturities) * len(self.option_moneyness) * 2
        self.option_feature_dim = self.n_options * 2
        self.n_instruments = 1 + self.n_options

        self.task = task
        self.initial_cash = initial_cash
        self.transaction_cost_pct = transaction_cost_pct
        self.position_limits = position_limits
        self.hedge_error_penalty = hedge_error_penalty
        self.render_mode = render_mode

        if task == "hedging" and liability is None:
            liability = Liability(
                option_type="call",
                strike=1.0,
                maturity_days=90,
                quantity=-1.0,
            )
        self.liability = liability

        self._rng = np.random.RandomState(random_seed)

        self.action_space = spaces.Box(
            low=-self.position_limits,
            high=self.position_limits,
            shape=(self.n_instruments,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict({
            'spot_price': spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            'option_features': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.option_feature_dim,),
                dtype=np.float32,
            ),
            'time_step': spaces.Box(low=0, high=self.max_steps, shape=(1,), dtype=np.float32),
            'portfolio_weights': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.n_instruments,),
                dtype=np.float32,
            ),
        })

        self.path_index = None
        self.S = None
        self.v = None
        self.t = None
        self.cash = None
        self.positions = None
        self.portfolio_value = None
        self.hedge_portfolio_value = None
        self.liability_mtm = None
        self.prev_hedge_error = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)

        self.path_index = self._rng.randint(0, self.n_paths)
        self.t = 0
        self.S = float(self.spot_arr[self.path_index, self.t])
        self.v = float(self.var_arr[self.path_index, self.t])
        self.cash = self.initial_cash
        self.positions = np.zeros(self.n_instruments, dtype=np.float32)
        self.portfolio_value = self.cash

        if self.liability is not None:
            self._price_liability()
            if self.liability.initial_price is None:
                self.liability.initial_price = self.liability_mtm
            self.hedge_portfolio_value = self.portfolio_value + self.liability_mtm
            self.prev_hedge_error = abs(self.hedge_portfolio_value)
        else:
            self.liability_mtm = 0.0
            self.hedge_portfolio_value = self.portfolio_value
            self.prev_hedge_error = 0.0

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

        self.t += 1
        terminated = self.t >= self.max_steps
        truncated = False

        if not terminated:
            self.S = float(self.spot_arr[self.path_index, self.t])
            self.v = float(self.var_arr[self.path_index, self.t])

        old_portfolio_value = self.portfolio_value
        self.portfolio_value = self._compute_portfolio_value()

        if self.liability is not None:
            self._price_liability()
            self.hedge_portfolio_value = self.portfolio_value + self.liability_mtm

        reward = self._compute_reward(transaction_cost, old_portfolio_value)

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self) -> Dict[str, np.ndarray]:
        option_features = self.option_features_arr[self.path_index, self.t]
        portfolio_weights = self.positions / (self.position_limits + 1e-8)
        return {
            'spot_price': np.array([self.S], dtype=np.float32),
            'option_features': option_features.astype(np.float32),
            'time_step': np.array([self.t], dtype=np.float32),
            'portfolio_weights': portfolio_weights.astype(np.float32),
        }

    def _get_info(self) -> Dict[str, Any]:
        return {
            'S': self.S,
            'v': self.v,
            't': self.t,
            'cash': self.cash,
            'positions': self.positions.copy(),
            'portfolio_value': self.portfolio_value,
            'liability_mtm': self.liability_mtm,
            'n_instruments': self.n_instruments,
        }

    def _compute_portfolio_value(self) -> float:
        value = self.cash
        value += self.positions[0] * self.S
        option_prices = self.option_prices_arr[self.path_index, self.t]
        value += float(np.sum(self.positions[1:] * option_prices))
        return float(value)

    def _compute_reward(self, transaction_cost: float, prev_value: float) -> float:
        if self.task == 'hedging':
            current_hedge_error = abs(self.hedge_portfolio_value)
            hedge_error_reduction = self.prev_hedge_error - current_hedge_error
            inventory_penalty = 0.001 * np.sum(np.abs(self.positions))
            reward = hedge_error_reduction - transaction_cost - inventory_penalty
            self.prev_hedge_error = current_hedge_error
            return float(reward)

        pnl = self.portfolio_value - prev_value
        return float(pnl - transaction_cost)

    def _compute_transaction_cost(self, position_changes: np.ndarray) -> float:
        total_cost = abs(position_changes[0]) * self.S * self.transaction_cost_pct
        option_prices = self.option_prices_arr[self.path_index, self.t]
        total_cost += float(np.sum(np.abs(position_changes[1:]) * option_prices) * self.transaction_cost_pct)
        return float(total_cost)

    def _price_liability(self) -> None:
        if self.liability is None:
            self.liability_mtm = 0.0
            return
        days_elapsed = self.t * self.dt * 365
        days_to_maturity = max(self.liability.maturity_days - days_elapsed, 0)
        if not self.option_maturities or not self.option_moneyness:
            self.liability_mtm = 0.0
            return
        target_ttm = min(self.option_maturities, key=lambda x: abs(x - days_to_maturity))
        target_moneyness = min(self.option_moneyness, key=lambda x: abs(x - self.liability.strike / self.S))
        idx = _option_grid_index(target_ttm, target_moneyness, self.option_maturities, self.option_moneyness, self.liability.option_type)
        option_prices = self.option_prices_arr[self.path_index, self.t]
        self.liability_mtm = float(self.liability.quantity * option_prices[idx])

    def render(self):
        if self.render_mode == 'human':
            print(f"Step {self.t}/{self.max_steps} | S={self.S:.4f} | PV={self.portfolio_value:.2f}")

    def close(self):
        pass


def _option_grid_index(
    maturity: int,
    moneyness: float,
    maturities: List[int],
    moneyness_list: List[float],
    option_type: str,
) -> int:
    idx = 0
    for ttm in sorted(maturities):
        for m in sorted(moneyness_list):
            if ttm == maturity and abs(m - moneyness) < 1e-8:
                if option_type == 'call':
                    return idx
                return idx + 1
            idx += 2
    return 0
