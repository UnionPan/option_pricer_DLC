"""
Heston RL Environment with Multi-Asset Trading

Professional implementation with:
- processes.Heston simulation (Milstein/Euler schemes)
- Multi-dimensional action space (trade underlying + options)
- Fixed grid representation (moneyness x TTM)
- SyntheticEquityOptionChainGenerator for realistic pricing
- T=246 steps, dt=1/252 (approximately 1 trading year)

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    print("Warning: gymnasium not installed. Install with: pip install gymnasium")
    gym = None
    spaces = None

from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from datetime import date, timedelta

# Import professional Heston process
from options_desk.processes import Heston
from options_desk.processes.base import SimulationConfig

# Import equity option chain generator
from options_desk.calibration.data.synthetic_equity import (
    SyntheticEquityOptionChainGenerator,
    HestonVolatilityProfile,
)

# Import renderer
from options_desk.simulations.renderer import HestonEnvRenderer


@dataclass
class HestonParams:
    """
    Heston model parameters.

    Default parameters calibrated for realistic volatility smile visible even
    in short-dated options (7-30 days):
    - Fast mean reversion (kappa=6.0)
    - High vol-of-vol (xi=1.0) - creates visible smile in short maturities
    - Strong negative correlation (rho=-0.8) - creates equity-style left skew

    These parameters satisfy Feller condition: 2κθ = 1.08 > ξ² = 1.0
    and create typical equity/crypto volatility patterns:
    - OTM puts: Higher IV (crash protection premium)
    - ATM: Peak IV
    - OTM calls: Lower IV (smile/smirk pattern)

    Note: 30% vol represents volatile assets (crypto, meme stocks).
    For standard equities, use v_0=theta=0.04 (20% vol), xi=0.5, kappa=4.0.
    """
    S_0: float = 1.0        # Initial stock price
    v_0: float = 0.09       # Initial variance (σ₀ = 30%)
    mu: float = 0.0         # Drift (risk-neutral: mu=0)
    kappa: float = 6.0      # Mean reversion speed (fast)
    theta: float = 0.09     # Long-run variance (σ_LR = 30%)
    xi: float = 1.0         # Vol-of-vol (high for visible short-dated smile)
    rho: float = -0.8       # Correlation (strong negative = equity-style skew)


@dataclass
class Liability:
    """
    Liability to be hedged.

    For hedging task, agent is short this liability and must hedge it
    using underlying + options from the grid.
    """
    option_type: str        # 'call' or 'put'
    strike: float          # Strike price (can be relative to S_0)
    maturity_days: int     # Days to maturity
    quantity: float = -1.0  # Negative = short position

    # MTM tracking
    initial_price: float = None
    current_price: float = None
    payoff: float = None


class HestonEnv(gym.Env if gym else object):
    """
    Heston RL Environment with multi-dimensional trading.

    Key Features:
    - Professional Heston simulation via processes.Heston
    - Multi-dimensional action space: [underlying_position, option_positions...]
    - Fixed grid for options: (moneyness, TTM) x enables consistent observation
    - Agent can trade underlying + multiple options simultaneously
    - POMDP: variance v_t is hidden, must infer from option prices

    Action Space:
        Continuous: Box(shape=(1 + n_options,))
            - action[0]: target underlying position
            - action[1:]: target positions for each option on fixed grid

    Observation Space:
        Dict with:
            - 'spot_price': Current spot price S_t
            - 'option_features': [normalized_price, IV] for each grid point
            - 'time_step': Current time step
            - 'portfolio_weights': Current portfolio allocation (optional)

    Fixed Grid:
        - Moneyness levels: [0.90, 0.95, 1.0, 1.05, 1.10]
        - TTM levels: [30, 60, 90] days
        - Each (moneyness, TTM): call + put = 2 options
        - Total: 5 x 3 x 2 = 30 options
    """

    metadata = {'render_modes': ['human', 'matplotlib'], 'render_fps': 4}

    def __init__(
        self,
        params: Optional[HestonParams] = None,
        max_steps: int = 246,                    # ~ 1 trading year (NOT capped by liability)
        dt: float = 1/252,                       # 1 day
        discretization: str = 'milstein',        # Milstein for accuracy
        variance_scheme: str = 'truncation',     # Variance positivity
        # Option chain settings
        include_options: bool = True,
        option_maturities: List[int] = None,     # Fixed TTM grid (days) OR None for option_grid
        option_moneyness: List[float] = None,    # Fixed moneyness grid OR None for option_grid
        option_grid: Optional[Dict[int, List[float]]] = None,  # Buehler-style: {τ: [K/S]}
        # Task settings
        task: str = 'hedging',                   # 'trading' or 'hedging'
        liability: Optional[Liability] = None,   # Liability to hedge (for hedging task)
        initial_cash: float = 10.0,              # Starting cash
        transaction_cost_pct: float = 0.001,     # 10 bps transaction cost
        position_limits: float = 100.0,          # Max absolute position per instrument
        hedge_error_penalty: float = 1.0,        # Penalty weight for hedge error
        # Rendering
        render_mode: Optional[str] = None,
    ):
        """
        Initialize Heston environment with hedging or trading task.

        Args:
            params: Heston model parameters
            max_steps: Episode length (NOT capped by liability maturity)
            dt: Time step (1/252 = 1 day)
            discretization: 'euler' or 'milstein'
            variance_scheme: 'truncation', 'reflection', 'absorption'
            include_options: Generate option chains
            option_maturities: Fixed TTM grid [30, 60, 90] (ignored if option_grid provided)
            option_moneyness: Fixed moneyness grid [0.95, 0.97, ...] (ignored if option_grid provided)
            option_grid: Buehler-style grid {τ_days: [K/S ratios]}
                         Example: {10: [0.99, 1.0, 1.01], 20: [0.97, 0.99, 1.0, 1.01, 1.03]}
            task: 'hedging' (hedge liability) or 'trading' (maximize PnL)
            liability: Liability to hedge (required for hedging task)
            initial_cash: Starting cash
            transaction_cost_pct: Transaction cost as % of notional
            position_limits: Max position per instrument
            hedge_error_penalty: Weight for hedge error penalty in reward
            render_mode: 'human' for console output
        """
        super().__init__()

        # Model parameters
        self.params = params if params is not None else HestonParams()
        self.max_steps = max_steps
        self.dt = dt
        self.T = max_steps * dt
        self.discretization = discretization
        self.variance_scheme = variance_scheme
        self.include_options = include_options
        self.task = task
        self.initial_cash = initial_cash
        self.transaction_cost_pct = transaction_cost_pct
        self.position_limits = position_limits
        self.hedge_error_penalty = hedge_error_penalty
        self.render_mode = render_mode

        # Initialize renderer if matplotlib mode (will set n_instruments later)
        self.renderer = None
        self.render_mode_requested = render_mode  # Store for later initialization

        # Liability for hedging task
        if task == 'hedging':
            if liability is None:
                # Default: short 1 ATM call, 90 days maturity
                liability = Liability(
                    option_type='call',
                    strike=self.params.S_0,  # ATM
                    maturity_days=90,
                    quantity=-1.0,
                )
            self.liability = liability
        else:
            self.liability = None

        # Fixed grid for options
        # Support two formats:
        # 1. option_grid (Buehler-style): {τ: [K/S]} with different strikes per maturity
        # 2. option_maturities + option_moneyness: Cartesian product

        if option_grid is not None:
            # Buehler-style grid: {10: [0.99, 1.0, 1.01], 20: [0.97, 0.99, 1.0, 1.01, 1.03], ...}
            self.option_grid = option_grid
            self.use_buehler_grid = True

            # Extract unique maturities and count total options
            self.option_maturities = sorted(option_grid.keys())
            self.n_maturities = len(self.option_maturities)

            # Count total options: sum over all (maturity, moneyness, type) combinations
            total_options = 0
            for ttm, moneyness_list in option_grid.items():
                total_options += len(moneyness_list) * 2  # calls + puts

            self.n_options = total_options
            self.n_option_types = 2

            # Store for compatibility (these are not uniform grids anymore)
            self.option_moneyness = None  # Not uniform across maturities
            self.n_moneyness = None

        else:
            # Original Cartesian product grid
            if option_maturities is None:
                # Default: wider range of maturities
                option_maturities = [7, 14, 30, 60, 90]
            if option_moneyness is None:
                # Use adaptive grid by default for better coverage
                # Import here to avoid circular dependency
                from options_desk.calibration.data.synthetic_equity import get_default_moneyness_by_maturity

                # Get default adaptive grid
                default_adaptive = get_default_moneyness_by_maturity()

                # Create finer adaptive grid based on maturity
                option_grid = {}
                for mat in option_maturities:
                    # Find closest maturity in default grid
                    if mat in default_adaptive:
                        option_grid[mat] = default_adaptive[mat]
                    elif mat <= 10:
                        # Very short: tight grid
                        option_grid[mat] = [0.95, 0.975, 1.0, 1.025, 1.05]
                    elif mat <= 30:
                        # Short: medium-tight grid
                        option_grid[mat] = [0.90, 0.95, 0.975, 1.0, 1.025, 1.05, 1.10]
                    elif mat <= 60:
                        # Medium: wider grid
                        option_grid[mat] = [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15]
                    else:
                        # Long: widest grid
                        option_grid[mat] = [0.80, 0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15, 1.20]

                # Use as Buehler grid
                self.option_grid = option_grid
                self.use_buehler_grid = True
                self.option_maturities = sorted(option_grid.keys())
                self.n_maturities = len(self.option_maturities)

                # Count total options
                total_options = 0
                for ttm, moneyness_list in option_grid.items():
                    total_options += len(moneyness_list) * 2  # calls + puts

                self.n_options = total_options
                self.n_option_types = 2
                self.option_moneyness = None  # Not uniform
                self.n_moneyness = None
            else:
                # User specified uniform moneyness grid
                self.use_buehler_grid = False
                self.option_maturities = sorted(option_maturities)
                self.option_moneyness = sorted(option_moneyness)
                self.n_maturities = len(self.option_maturities)
                self.n_moneyness = len(self.option_moneyness)
                self.n_option_types = 2

                # Total options: moneyness x TTM x type
                self.n_options = self.n_moneyness * self.n_maturities * self.n_option_types

                # Convert to Buehler format internally for unified handling
                self.option_grid = {ttm: self.option_moneyness for ttm in self.option_maturities}

        # Create Heston process
        self.heston_process = Heston(
            mu=self.params.mu,
            kappa=self.params.kappa,
            theta=self.params.theta,
            sigma_v=self.params.xi,
            rho=self.params.rho,
            v0=self.params.v_0,
            variance_scheme=self.variance_scheme,
        )

        # Option chain generator
        if self.include_options:
            # For Buehler grid: use adaptive moneyness by maturity
            # For uniform grid: use single moneyness list for all maturities
            if self.use_buehler_grid:
                # Pass maturity-specific moneyness grid (adaptive)
                self.option_generator = SyntheticEquityOptionChainGenerator(
                    maturities_days=self.option_maturities,
                    moneyness_by_maturity=self.option_grid,  # Adaptive grid
                    add_noise=False,
                    random_seed=42,
                )
            else:
                # Pass uniform moneyness for all maturities
                self.option_generator = SyntheticEquityOptionChainGenerator(
                    maturities_days=self.option_maturities,
                    moneyness_range=self.option_moneyness,  # Uniform grid
                    add_noise=False,
                    random_seed=42,
                )

            # Features per option: [normalized_price, IV]
            self.option_feature_dim = self.n_options * 2
        else:
            self.option_generator = None
            self.option_feature_dim = 0
            self.n_options = 0

        # Number of tradable instruments: 1 underlying + n_options
        self.n_instruments = 1 + self.n_options

        # Initialize renderer now that we know n_instruments
        if self.render_mode_requested == 'matplotlib':
            self.renderer = HestonEnvRenderer(
                max_steps=max_steps,
                n_instruments=self.n_instruments
            )

        # Action space: continuous positions for [underlying, opt1, opt2, ..., optN]
        self.action_space = spaces.Box(
            low=-self.position_limits,
            high=self.position_limits,
            shape=(self.n_instruments,),
            dtype=np.float32
        )

        # Observation space
        if self.include_options:
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
            self.observation_space = spaces.Box(
                low=np.array([0.0, 0.0], dtype=np.float32),
                high=np.array([np.inf, float(max_steps)], dtype=np.float32),
                dtype=np.float32
            )

        # State variables
        self.S = None
        self.v = None
        self.t = None
        self.cash = None
        self.positions = None  # Array: [underlying_pos, opt1_pos, ..., optN_pos]
        self.portfolio_value = None
        self.hedge_portfolio_value = None  # Portfolio + liability MTM
        self.current_option_chain = None
        self.option_grid_prices = None  # Mid prices on fixed grid
        self.path = None  # Pre-simulated full path
        self.liability_mtm = None  # Current liability mark-to-market
        self.prev_hedge_error = None  # Previous hedge error for incremental rewards

        # History
        self.history = {
            'S': [],
            'v': [],
            't': [],
            'cash': [],
            'positions': [],
            'portfolio_value': [],
            'hedge_portfolio_value': [],
            'liability_mtm': [],
            'hedge_error': [],
            'action': [],
            'reward': [],
            'transaction_costs': [],
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment and pre-simulate full episode."""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # Simulate entire path using processes.Heston
        X0 = np.array([[self.params.S_0, self.params.v_0]])
        config = SimulationConfig(
            n_paths=1,
            n_steps=self.max_steps,
            random_seed=seed,
        )

        # Pre-simulate full trajectory
        t_grid, paths = self.heston_process.simulate(
            X0=X0,
            T=self.T,
            config=config,
            scheme=self.discretization,
        )

        # Store full path: shape (n_steps+1, 1, 2)
        self.path = paths
        self.t_grid = t_grid

        # Initialize state
        self.S = self.path[0, 0, 0]
        self.v = self.path[0, 0, 1]
        self.t = 0
        self.cash = self.initial_cash
        self.positions = np.zeros(self.n_instruments, dtype=np.float32)
        self.portfolio_value = self.cash

        # Generate initial option chain
        if self.include_options:
            self._generate_option_chain()

        # Price liability (for hedging task)
        if self.liability is not None:
            self._price_liability()
            # Set initial liability price
            if self.liability.initial_price is None:
                self.liability.initial_price = self.liability_mtm
            # Initialize hedge portfolio value
            self.hedge_portfolio_value = self.portfolio_value + self.liability_mtm
            self.prev_hedge_error = abs(self.hedge_portfolio_value)
        else:
            self.liability_mtm = 0.0
            self.hedge_portfolio_value = self.portfolio_value
            self.prev_hedge_error = 0.0

        # Clear history
        self.history = {k: [] for k in self.history.keys()}
        self._record_state(action=np.zeros(self.n_instruments), reward=0.0, cost=0.0)

        # Reset renderer if using matplotlib
        if self.renderer is not None:
            self.renderer.reset()
            self.renderer.initialize(interactive=True)
            # Initial render
            self.renderer.update(
                t=self.t,
                S=self.S,
                v=self.v,
                positions=self.positions,
                portfolio_value=self.portfolio_value,
                option_chain=self.current_option_chain if self.include_options else None
            )

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step with multi-dimensional trading.

        Args:
            action: Array of shape (n_instruments,)
                - action[0]: target position in underlying
                - action[1:]: target positions in options (on fixed grid)

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Ensure action is correct shape
        action = np.array(action, dtype=np.float32).flatten()
        if action.shape[0] != self.n_instruments:
            raise ValueError(f"Action must have shape ({self.n_instruments},), got {action.shape}")

        # Clip action to position limits
        action = np.clip(action, -self.position_limits, self.position_limits)

        # Store old state
        old_positions = self.positions.copy()
        old_portfolio_value = self.portfolio_value
        S_old = self.S

        # Compute position changes
        position_changes = action - old_positions

        # Compute transaction costs
        transaction_cost = self._compute_transaction_cost(position_changes)

        # Update positions and cash
        self.positions = action
        self.cash -= transaction_cost

        # Move to next time step (from pre-simulated path)
        self.t += 1
        self.S = self.path[self.t, 0, 0]
        self.v = self.path[self.t, 0, 1]

        # Generate option chain at new state
        if self.include_options:
            self._generate_option_chain()

            # Settle any expired options on the grid
            settlement_pnl = self._settle_expired_options()
            self.cash += settlement_pnl

        # Update liability MTM (for hedging task)
        if self.liability is not None:
            self._price_liability()

        # Mark-to-market portfolio value
        self.portfolio_value = self._compute_portfolio_value()

        # Compute hedge portfolio value (portfolio + liability)
        if self.liability is not None:
            self.hedge_portfolio_value = self.portfolio_value + self.liability_mtm
        else:
            self.hedge_portfolio_value = self.portfolio_value

        # Compute reward with proper shaping
        reward = self._compute_reward(transaction_cost)

        # Check termination
        terminated = self._is_terminated()
        truncated = self.t >= self.max_steps

        # Handle terminal settlement
        if terminated or truncated:
            # Settle all hedging positions (options + underlying)
            settlement_pnl = self._settle_all_positions()
            self.cash += settlement_pnl

            # Compute final liability payoff (if hedging)
            if self.liability is not None:
                final_payoff = self._compute_liability_payoff()
                self.liability.payoff = final_payoff
                # Liability P&L: quantity * payoff (quantity is negative for short)
                liability_pnl = self.liability.quantity * final_payoff
                self.cash += liability_pnl

            # Final portfolio value after settlement
            final_portfolio_value = self.cash  # All positions liquidated
            final_pnl = final_portfolio_value - self.initial_cash

            # Add terminal P&L to reward
            reward += final_pnl

        # Record
        self._record_state(action, reward, transaction_cost)

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _generate_option_chain(self):
        """Generate option chain on fixed grid at current state."""
        if not self.include_options:
            return

        # Update volatility profile with current variance
        vol_profile = HestonVolatilityProfile(
            kappa=self.params.kappa,
            theta=self.params.theta,
            xi=self.params.xi,
            rho=self.params.rho,
            v0=self.v,
            atm_iv=np.sqrt(self.v),
        )

        # Generate chain
        reference_date = date(2024, 1, 1) + timedelta(days=int(self.t))
        self.current_option_chain = self.option_generator.generate_single_chain(
            reference_date=reference_date,
            spot_price=self.S,
            vol_profile=vol_profile,
        )

        # Extract prices on fixed grid (consistent ordering)
        self.option_grid_prices = self._extract_grid_prices()

    def _extract_grid_prices(self) -> np.ndarray:
        """
        Extract option prices on fixed grid in consistent order.

        For Buehler grid: Filter options based on {τ: [K/S]} specification
        For uniform grid: Use all (TTM, moneyness, type) combinations

        Returns:
            Array of shape (n_options,) with mid prices
            Order: sorted by (maturity, moneyness, call/put)
        """
        if self.current_option_chain is None:
            return np.zeros(self.n_options, dtype=np.float32)

        prices = []

        # Get all options from chain
        all_options = self.current_option_chain.options

        # Filter and sort based on grid specification
        for ttm in sorted(self.option_grid.keys()):
            moneyness_list = self.option_grid[ttm]

            # Find options matching this maturity
            ttm_options = [
                opt for opt in all_options
                if abs((opt.expiry - self.current_option_chain.reference_date).days - ttm) < 1
            ]

            # For each specified moneyness at this maturity
            for moneyness_target in sorted(moneyness_list):
                # Find matching calls and puts (with tolerance for floating point)
                tolerance = 0.01  # 1% moneyness tolerance

                matching_calls = [
                    opt for opt in ttm_options
                    if opt.option_type == 'call' and
                    abs(opt.strike / self.S - moneyness_target) < tolerance
                ]

                matching_puts = [
                    opt for opt in ttm_options
                    if opt.option_type == 'put' and
                    abs(opt.strike / self.S - moneyness_target) < tolerance
                ]

                # Add call price (or 0 if not found)
                if matching_calls:
                    prices.append(matching_calls[0].mid)
                else:
                    prices.append(0.0)

                # Add put price (or 0 if not found)
                if matching_puts:
                    prices.append(matching_puts[0].mid)
                else:
                    prices.append(0.0)

        return np.array(prices, dtype=np.float32)

    def _compute_transaction_cost(self, position_changes: np.ndarray) -> float:
        """
        Compute transaction costs for position changes.

        Args:
            position_changes: Array of position changes

        Returns:
            Total transaction cost in dollars
        """
        total_cost = 0.0

        # Underlying transaction cost
        underlying_change = abs(position_changes[0])
        underlying_cost = underlying_change * self.S * self.transaction_cost_pct
        total_cost += underlying_cost

        # Option transaction costs (use bid-ask spread)
        if self.include_options and self.current_option_chain is not None:
            for i, opt in enumerate(sorted(
                self.current_option_chain.options,
                key=lambda opt: (
                    (opt.expiry - self.current_option_chain.reference_date).days,
                    opt.strike / self.S,
                    0 if opt.option_type == 'call' else 1
                )
            )):
                option_change = abs(position_changes[1 + i])
                if option_change > 0:
                    # Pay bid-ask spread on trades
                    spread = opt.ask - opt.bid
                    option_cost = option_change * spread / 2.0
                    total_cost += option_cost

        return total_cost

    def _compute_portfolio_value(self) -> float:
        """
        Mark-to-market portfolio value.

        Returns:
            Total portfolio value = cash + underlying_value + options_value
        """
        value = self.cash

        # Underlying value
        value += self.positions[0] * self.S

        # Option values (at mid prices)
        if self.include_options and self.option_grid_prices is not None:
            option_positions = self.positions[1:]
            option_values = option_positions * self.option_grid_prices
            value += np.sum(option_values)

        return float(value)

    def _compute_reward(self, transaction_cost: float) -> float:
        """
        Compute reward with proper shaping.

        For hedging task:
            reward = -hedge_error_change - transaction_cost - inventory_penalty
            where hedge_error = |hedge_portfolio_value|

        For trading task:
            reward = portfolio_pnl - transaction_cost

        Args:
            transaction_cost: Transaction cost paid this step

        Returns:
            Reward scalar
        """
        if self.task == 'hedging':
            # Compute current hedge error (distance from zero)
            current_hedge_error = abs(self.hedge_portfolio_value)

            # Incremental reward: reduction in hedge error
            hedge_error_reduction = self.prev_hedge_error - current_hedge_error

            # Penalty for transaction costs
            cost_penalty = transaction_cost

            # Optional: inventory penalty (discourage large positions)
            inventory_penalty = 0.001 * np.sum(np.abs(self.positions))

            # Reward = improvement - costs
            reward = (
                hedge_error_reduction
                - cost_penalty
                - inventory_penalty
            )

            # Update previous hedge error for next step
            self.prev_hedge_error = current_hedge_error

            return float(reward)

        elif self.task == 'trading':
            # Simple PnL minus costs
            # Get PnL from history
            if len(self.history['portfolio_value']) > 0:
                prev_value = self.history['portfolio_value'][-1]
                pnl = self.portfolio_value - prev_value
            else:
                pnl = self.portfolio_value - self.initial_cash

            reward = pnl - transaction_cost
            return float(reward)

        else:
            # Default: portfolio returns
            returns = (self.portfolio_value - self.initial_cash) / self.initial_cash
            return float(returns)

    def _is_terminated(self) -> bool:
        """Check early termination conditions."""
        # Bankruptcy
        if self.portfolio_value <= 0:
            return True

        # Invalid price
        if self.S <= 0 or np.isnan(self.S) or np.isinf(self.S):
            return True

        return False

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get observation (agent's view)."""
        if not self.include_options:
            return np.array([self.S, self.t], dtype=np.float32)

        # Vectorize option chain features
        option_features = self._vectorize_option_chain()

        # Portfolio weights (for observation)
        portfolio_weights = self.positions / (self.position_limits + 1e-8)

        obs = {
            'spot_price': np.array([self.S], dtype=np.float32),
            'option_features': option_features,
            'time_step': np.array([self.t], dtype=np.float32),
            'portfolio_weights': portfolio_weights.astype(np.float32),
        }

        return obs

    def _vectorize_option_chain(self) -> np.ndarray:
        """
        Vectorize option chain to feature vector on fixed grid.

        For Buehler grid: Filter options based on {τ: [K/S]} specification
        For uniform grid: Use all (TTM, moneyness, type) combinations

        Returns:
            Array of shape (n_options * 2,) with [normalized_price, IV] for each option
        """
        if self.current_option_chain is None:
            return np.zeros(self.option_feature_dim, dtype=np.float32)

        features = []
        all_options = self.current_option_chain.options

        # Iterate through grid in same order as _extract_grid_prices()
        for ttm in sorted(self.option_grid.keys()):
            moneyness_list = self.option_grid[ttm]

            # Find options matching this maturity
            ttm_options = [
                opt for opt in all_options
                if abs((opt.expiry - self.current_option_chain.reference_date).days - ttm) < 1
            ]

            # For each specified moneyness at this maturity
            for moneyness_target in sorted(moneyness_list):
                tolerance = 0.01

                matching_calls = [
                    opt for opt in ttm_options
                    if opt.option_type == 'call' and
                    abs(opt.strike / self.S - moneyness_target) < tolerance
                ]

                matching_puts = [
                    opt for opt in ttm_options
                    if opt.option_type == 'put' and
                    abs(opt.strike / self.S - moneyness_target) < tolerance
                ]

                # Add call features
                if matching_calls:
                    opt = matching_calls[0]
                    norm_price = opt.mid / (self.S + 1e-8)
                    iv = opt.implied_volatility
                    features.extend([norm_price, iv])
                else:
                    features.extend([0.0, 0.0])

                # Add put features
                if matching_puts:
                    opt = matching_puts[0]
                    norm_price = opt.mid / (self.S + 1e-8)
                    iv = opt.implied_volatility
                    features.extend([norm_price, iv])
                else:
                    features.extend([0.0, 0.0])

        return np.array(features, dtype=np.float32)

    def _record_state(self, action: np.ndarray, reward: float, cost: float):
        """Record current state to history."""
        self.history['S'].append(self.S)
        self.history['v'].append(self.v)
        self.history['t'].append(self.t)
        self.history['cash'].append(self.cash)
        self.history['positions'].append(self.positions.copy())
        self.history['portfolio_value'].append(self.portfolio_value)
        self.history['hedge_portfolio_value'].append(self.hedge_portfolio_value)
        self.history['liability_mtm'].append(self.liability_mtm if self.liability_mtm is not None else 0.0)
        self.history['hedge_error'].append(abs(self.hedge_portfolio_value))
        self.history['action'].append(action.copy())
        self.history['reward'].append(reward)
        self.history['transaction_costs'].append(cost)

    def _get_info(self) -> Dict[str, Any]:
        """Get info dict (includes hidden state for analysis)."""
        info = {
            'S': self.S,
            'v': self.v,  # HIDDEN from agent
            't': self.t,
            'cash': self.cash,
            'positions': self.positions.copy(),
            'portfolio_value': self.portfolio_value,
            'volatility': np.sqrt(self.v),
            'n_instruments': self.n_instruments,
        }

        if self.include_options and self.current_option_chain is not None:
            info['option_chain'] = self.current_option_chain
            info['n_options'] = len(self.current_option_chain.options)
            info['option_grid_prices'] = self.option_grid_prices

            # Find ATM call
            atm_calls = [
                opt for opt in self.current_option_chain.options
                if opt.option_type == 'call' and abs(opt.strike / self.S - 1.0) < 0.02
            ]
            if atm_calls:
                info['atm_call_price'] = atm_calls[0].mid
                info['atm_iv'] = atm_calls[0].implied_volatility

        return info

    def _settle_all_positions(self) -> float:
        """
        Settle all remaining positions at episode termination.

        Liquidates:
        - All option positions at current market value (mid price)
        - Underlying position at current spot price

        This simulates closing out the entire portfolio at episode end.

        Returns:
            Total settlement proceeds (added to cash)
        """
        total_proceeds = 0.0

        # 1. Settle underlying position
        underlying_position = self.positions[0]
        if abs(underlying_position) > 1e-8:
            underlying_value = underlying_position * self.S
            total_proceeds += underlying_value
            self.positions[0] = 0.0

        # 2. Settle all option positions at market value
        if self.include_options and self.option_grid_prices is not None:
            option_positions = self.positions[1:]
            option_values = option_positions * self.option_grid_prices
            total_proceeds += np.sum(option_values)
            # Zero out all option positions
            self.positions[1:] = 0.0

        return float(total_proceeds)

    def _settle_expired_options(self) -> float:
        """
        Settle options that have expired at current timestep.

        Checks each option on the grid:
        - If days_to_maturity <= 0: option has expired
        - Compute payoff, settle position, zero it out

        Returns:
            Total settlement P&L added to cash
        """
        if not self.include_options or self.current_option_chain is None:
            return 0.0

        total_pnl = 0.0
        days_elapsed = self.t * self.dt * 365

        # Get sorted options (same order as positions array)
        sorted_options = sorted(
            self.current_option_chain.options,
            key=lambda opt: (
                (opt.expiry - self.current_option_chain.reference_date).days,
                opt.strike / self.S,
                0 if opt.option_type == 'call' else 1
            )
        )

        # Check each option for expiry
        for idx, opt in enumerate(sorted_options):
            # Original maturity of this option
            original_maturity_days = (opt.expiry - date(2024, 1, 1)).days
            days_to_expiry = original_maturity_days - days_elapsed

            # If option has expired or is expiring now
            if days_to_expiry <= 0:
                position = self.positions[1 + idx]  # positions[0] is underlying

                if abs(position) > 1e-8:  # Has non-zero position
                    # Compute payoff
                    if opt.option_type == 'call':
                        payoff = max(self.S - opt.strike, 0.0)
                    else:  # put
                        payoff = max(opt.strike - self.S, 0.0)

                    # Settlement P&L
                    settlement = position * payoff
                    total_pnl += settlement

                    # Zero out position
                    self.positions[1 + idx] = 0.0

        return total_pnl

    def _price_liability(self):
        """
        Price the liability using current state.

        Uses the same Heston pricer (via option chain generator) to ensure
        consistent pricing between hedging instruments and liability.
        """
        if self.liability is None or not self.include_options:
            self.liability_mtm = 0.0
            return

        # Compute days to maturity
        days_elapsed = self.t * self.dt * 365
        days_to_maturity = max(self.liability.maturity_days - days_elapsed, 0)

        if days_to_maturity <= 0:
            # At or past maturity, use payoff
            self.liability_mtm = self.liability.quantity * self._compute_liability_payoff()
            self.liability.current_price = self.liability_mtm / self.liability.quantity
            return

        # Create volatility profile with current variance
        vol_profile = HestonVolatilityProfile(
            kappa=self.params.kappa,
            theta=self.params.theta,
            xi=self.params.xi,
            rho=self.params.rho,
            v0=self.v,
            atm_iv=np.sqrt(self.v),
        )

        # Create temporary generator with single maturity and strike
        temp_generator = SyntheticEquityOptionChainGenerator(
            maturities_days=[int(days_to_maturity)],
            moneyness_range=[self.liability.strike / self.S],
            add_noise=False,
        )

        # Generate chain
        reference_date = date(2024, 1, 1) + timedelta(days=int(self.t))
        temp_chain = temp_generator.generate_single_chain(
            reference_date=reference_date,
            spot_price=self.S,
            vol_profile=vol_profile,
        )

        # Find matching option
        matching_opt = next(
            (opt for opt in temp_chain.options
             if opt.option_type == self.liability.option_type),
            None
        )

        if matching_opt is not None:
            # MTM = quantity * price (quantity is negative for short)
            self.liability_mtm = self.liability.quantity * matching_opt.mid
            self.liability.current_price = matching_opt.mid
        else:
            # Fallback to intrinsic value
            intrinsic = self._compute_liability_payoff()
            self.liability_mtm = self.liability.quantity * intrinsic
            self.liability.current_price = intrinsic

    def _compute_liability_payoff(self) -> float:
        """
        Compute terminal payoff of liability.

        Returns:
            Payoff value (positive = liability owes you, negative = you owe)
        """
        if self.liability is None:
            return 0.0

        K = self.liability.strike
        S = self.S

        if self.liability.option_type == 'call':
            payoff = max(S - K, 0.0)
        elif self.liability.option_type == 'put':
            payoff = max(K - S, 0.0)
        else:
            payoff = 0.0

        return payoff

    def render(self):
        """Render environment state."""
        if self.render_mode == 'human':
            print(f"\nStep {self.t}/{self.max_steps} (T={self.t*self.dt:.4f})")
            print(f"  S_t = {self.S:.4f}")
            print(f"  v_t = {self.v:.4f} (sigma = {np.sqrt(self.v):.2%})")
            print(f"  Cash = ${self.cash:.2f}")
            print(f"  Portfolio Value = ${self.portfolio_value:.2f}")
            print(f"  Positions:")
            print(f"    Underlying: {self.positions[0]:.2f}")
            if self.include_options:
                option_pos = self.positions[1:]
                active_options = np.sum(np.abs(option_pos) > 0.01)
                print(f"    Active options: {active_options}/{self.n_options}")
                print(f"    Total option notional: {np.sum(np.abs(option_pos)):.2f}")

            if self.include_options and self.current_option_chain:
                atm_call = next(
                    (opt for opt in self.current_option_chain.options
                     if opt.option_type == 'call' and abs(opt.strike / self.S - 1.0) < 0.02),
                    None
                )
                if atm_call:
                    print(f"  ATM Call: ${atm_call.mid:.4f}, IV={atm_call.implied_volatility:.2%}")

            print("-" * 60)

        elif self.render_mode == 'matplotlib':
            if self.renderer is not None:
                # Update matplotlib renderer with current state
                self.renderer.update(
                    t=self.t,
                    S=self.S,
                    v=self.v,
                    positions=self.positions,
                    portfolio_value=self.portfolio_value,
                    option_chain=self.current_option_chain if self.include_options else None
                )

    def get_history(self) -> Dict[str, list]:
        """Get episode history."""
        return self.history

    def close(self):
        """Clean up."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def make_heston_env(
    include_options: bool = True,
    max_steps: int = 246,
    dt: float = 1/252,
    discretization: str = 'milstein',
    option_maturities: List[int] = None,
    option_moneyness: List[float] = None,
    option_grid: Optional[Dict[int, List[float]]] = None,
    task: str = 'hedging',
    liability: Optional[Liability] = None,
    # Heston parameters (can be passed individually or as params object)
    S_0: Optional[float] = None,
    v_0: Optional[float] = None,
    mu: Optional[float] = None,
    kappa: Optional[float] = None,
    theta: Optional[float] = None,
    xi: Optional[float] = None,
    volvol: Optional[float] = None,  # Alias for xi
    rho: Optional[float] = None,
    **kwargs
) -> HestonEnv:
    """
    Create Heston environment with proper hedging or trading task.

    Args:
        include_options: Generate option chains
        max_steps: Episode length (NOT capped by liability maturity)
        dt: Time step (1/252 = 1 day)
        discretization: 'euler' or 'milstein'
        option_maturities: Fixed TTM grid [30, 60, 90] (ignored if option_grid provided)
        option_moneyness: Fixed moneyness grid [0.95, 0.97, ...] (ignored if option_grid provided)
        option_grid: Buehler-style grid {τ: [K/S]}
                     Example: {10: [0.99, 1.0, 1.01], 20: [0.97, 0.99, 1.0, 1.01, 1.03]}
        task: 'hedging' (hedge liability) or 'trading' (maximize PnL)
        liability: Liability to hedge (auto-created if task='hedging' and None)
        S_0: Initial spot price (default: 1.0)
        v_0: Initial variance (default: 0.09 = 30% vol)
        mu: Drift (default: 0.0 for risk-neutral)
        kappa: Mean reversion speed (default: 6.0)
        theta: Long-run variance (default: 0.09 = 30% vol)
        xi: Vol-of-vol (default: 1.0)
        volvol: Alias for xi (use either xi or volvol)
        rho: Correlation spot-vol (default: -0.8)
        **kwargs: Additional HestonEnv arguments (hedge_error_penalty, etc.)

    Returns:
        Configured Heston environment

    Example (Hedging):
        >>> env = make_heston_env(
        ...     task='hedging',
        ...     liability=Liability(
        ...         option_type='call',
        ...         strike=1.0,
        ...         maturity_days=90,
        ...         quantity=-1.0  # Short 1 call
        ...     ),
        ...     kappa=8.0,
        ...     theta=0.0625,
        ...     volvol=1.0,
        ...     rho=-0.7,
        ... )
        >>> obs, info = env.reset()
        >>> # Agent must hedge the short call using underlying + grid options

    Example (Trading):
        >>> env = make_heston_env(task='trading')
        >>> obs, info = env.reset()
        >>> # Agent trades for profit, no liability
    """
    # Set defaults only if not using Buehler grid
    if option_grid is None:
        if option_maturities is None:
            option_maturities = [30, 60, 90]

        # If no moneyness specified, use adaptive grid by default
        if option_moneyness is None:
            # Import here to avoid circular dependency
            from options_desk.calibration.data.synthetic_equity import get_default_moneyness_by_maturity

            # Get default adaptive grid
            default_adaptive = get_default_moneyness_by_maturity()

            # Filter to only requested maturities
            option_grid = {
                mat: default_adaptive.get(mat, [0.95, 1.0, 1.05])  # Fallback if maturity not in default
                for mat in option_maturities
            }
            # Will be handled below as Buehler grid

    # Handle Heston parameters
    # If 'params' is in kwargs, use it; otherwise construct from individual parameters
    if 'params' not in kwargs:
        # Handle volvol alias
        if volvol is not None and xi is None:
            xi = volvol

        # Build params dict with only provided values
        params_dict = {}
        if S_0 is not None:
            params_dict['S_0'] = S_0
        if v_0 is not None:
            params_dict['v_0'] = v_0
        if mu is not None:
            params_dict['mu'] = mu
        if kappa is not None:
            params_dict['kappa'] = kappa
        if theta is not None:
            params_dict['theta'] = theta
        if xi is not None:
            params_dict['xi'] = xi
        if rho is not None:
            params_dict['rho'] = rho

        # Create HestonParams with provided values (uses defaults for missing)
        if params_dict:
            kwargs['params'] = HestonParams(**params_dict)

    return HestonEnv(
        include_options=include_options,
        max_steps=max_steps,
        dt=dt,
        discretization=discretization,
        option_maturities=option_maturities,
        option_moneyness=option_moneyness,
        option_grid=option_grid,
        task=task,
        liability=liability,
        **kwargs
    )
