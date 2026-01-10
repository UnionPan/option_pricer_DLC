"""
Delta Hedging Agent

Implements analytical delta hedging using Black-Scholes formula.
Observes underlying + option prices (bid/ask spreads) in real-time
and calculates analytical delta to hedge the liability.

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Any, Optional
from .base import BaseHedgingAgent


class DeltaHedgingAgent(BaseHedgingAgent):
    """
    Delta hedging agent using Black-Scholes analytical delta.

    Strategy:
    1. Observe current spot price S_t and option implied volatilities
    2. Compute Black-Scholes delta of the liability
    3. Hedge delta using underlying position
    4. Optionally rebalance based on threshold

    Key Features:
    - Model-free: Uses implied volatility from market (observations)
    - Analytical: Closed-form Black-Scholes delta
    - Rebalancing: Only trades when delta drift exceeds threshold
    """

    def __init__(
        self,
        n_instruments: int,
        liability_option_type: str,
        liability_strike: float,
        liability_maturity_days: int,
        liability_quantity: float = -1.0,
        position_limits: float = 100.0,
        rebalance_threshold: float = 0.05,
        risk_free_rate: float = 0.0,
        hedge_underlying_only: bool = True,
        gamma_hedge: bool = False,
        name: str = "DeltaHedgingAgent",
    ):
        """
        Initialize delta hedging agent.

        Args:
            n_instruments: Number of tradable instruments
            liability_option_type: 'call' or 'put'
            liability_strike: Strike price of liability
            liability_maturity_days: Maturity in days
            liability_quantity: Quantity (negative = short)
            position_limits: Max position per instrument
            rebalance_threshold: Only rebalance if delta drift > threshold
            risk_free_rate: Risk-free rate (annualized)
            hedge_underlying_only: If True, only hedge with underlying (no options)
            gamma_hedge: If True, also hedge gamma using options
            name: Agent name
        """
        super().__init__(n_instruments, position_limits, name)

        # Liability specification
        self.liability_option_type = liability_option_type
        self.liability_strike = liability_strike
        self.liability_maturity_days = liability_maturity_days
        self.liability_quantity = liability_quantity
        self.risk_free_rate = risk_free_rate

        # Hedging parameters
        self.rebalance_threshold = rebalance_threshold
        self.hedge_underlying_only = hedge_underlying_only
        self.gamma_hedge = gamma_hedge

        # State tracking
        self.current_delta = 0.0
        self.current_gamma = 0.0
        self.target_underlying_position = 0.0
        self.last_hedge_delta = 0.0

    def reset(
        self,
        observation: Dict[str, np.ndarray],
        info: Dict[str, Any],
    ) -> None:
        """Reset agent state at episode start."""
        super().reset(observation, info)
        self.current_delta = 0.0
        self.current_gamma = 0.0
        self.target_underlying_position = 0.0
        self.last_hedge_delta = 0.0

    def select_action(
        self,
        observation: Dict[str, np.ndarray],
        info: Dict[str, Any],
    ) -> np.ndarray:
        """
        Select hedging action using Black-Scholes delta.

        Steps:
        1. Extract spot price and time to maturity
        2. Estimate implied volatility from observations
        3. Calculate liability delta using Black-Scholes
        4. Determine target underlying position = -liability_delta
        5. Optionally hedge gamma using options

        Args:
            observation: Dict with 'spot_price', 'option_features', 'time_step'
            info: Dict with 'S', 't', 'option_chain', etc.

        Returns:
            action: Target positions [underlying, opt1, opt2, ...]
        """
        # Extract state
        S = float(observation['spot_price'][0])
        t = int(observation['time_step'][0])
        dt = 1/252  # Daily steps

        # Time to maturity (in years)
        days_elapsed = t * dt * 365
        tau = max((self.liability_maturity_days - days_elapsed) / 365.0, 1e-6)

        # Estimate implied volatility from option chain
        sigma = self._estimate_implied_volatility(observation, info, S)

        # Calculate Black-Scholes delta and gamma of liability
        delta, gamma = self._black_scholes_greeks(
            S=S,
            K=self.liability_strike,
            tau=tau,
            sigma=sigma,
            r=self.risk_free_rate,
            option_type=self.liability_option_type,
        )

        # Store current greeks
        self.current_delta = delta
        self.current_gamma = gamma

        # Liability delta (quantity is negative for short)
        liability_delta = self.liability_quantity * delta

        # Target underlying position to hedge delta
        # If short call with delta +0.6, liability_delta = -1 * 0.6 = -0.6
        # Need to buy +0.6 shares to offset
        target_underlying = -liability_delta

        # Rebalancing threshold: only trade if delta drift is significant
        delta_drift = abs(target_underlying - self.last_hedge_delta)
        if delta_drift < self.rebalance_threshold and t > 0:
            # No rebalancing needed, keep current positions
            action = info['positions'].copy()
        else:
            # Rebalance
            action = np.zeros(self.n_instruments, dtype=np.float32)
            action[0] = target_underlying

            # Optional: Gamma hedging using options on the grid
            if self.gamma_hedge and not self.hedge_underlying_only:
                action = self._add_gamma_hedge(action, observation, info, S, tau, sigma)

            # Update last hedge delta
            self.last_hedge_delta = target_underlying

        # Clip to position limits
        action = np.clip(action, -self.position_limits, self.position_limits)

        # Update state
        self.current_positions = action
        self.target_underlying_position = target_underlying

        return action

    def _estimate_implied_volatility(
        self,
        observation: Dict[str, np.ndarray],
        info: Dict[str, Any],
        S: float,
    ) -> float:
        """
        Estimate implied volatility from option chain observations.

        Strategy:
        1. Look for ATM options with similar maturity as liability
        2. Extract implied volatility from option_features
        3. Fallback to ATM IV if available in info

        Args:
            observation: Observation dict
            info: Info dict
            S: Current spot price

        Returns:
            sigma: Estimated implied volatility (annualized)
        """
        # Try to get ATM IV from info (if available)
        if 'atm_iv' in info:
            return float(info['atm_iv'])

        # Parse option_features: [normalized_price, IV] for each option
        option_features = observation['option_features']
        n_options = len(option_features) // 2

        # Extract IVs
        ivs = option_features[1::2]  # Every second element is IV

        if len(ivs) > 0:
            # Use median IV as estimate (robust to outliers)
            sigma = float(np.median(ivs))
            return max(sigma, 0.01)  # Floor at 1%

        # Fallback: use 25% volatility
        return 0.25

    def _black_scholes_greeks(
        self,
        S: float,
        K: float,
        tau: float,
        sigma: float,
        r: float,
        option_type: str,
    ) -> tuple:
        """
        Calculate Black-Scholes delta and gamma.

        Delta:
            Call: N(d1)
            Put: N(d1) - 1

        Gamma:
            Both: phi(d1) / (S * sigma * sqrt(tau))

        where:
            d1 = [ln(S/K) + (r + 0.5*sigma^2)*tau] / (sigma*sqrt(tau))
            d2 = d1 - sigma*sqrt(tau)
            N(x) = standard normal CDF
            phi(x) = standard normal PDF

        Args:
            S: Spot price
            K: Strike price
            tau: Time to maturity (years)
            sigma: Implied volatility (annualized)
            r: Risk-free rate (annualized)
            option_type: 'call' or 'put'

        Returns:
            (delta, gamma): Greeks
        """
        if tau <= 0:
            # At maturity, delta is discontinuous
            if option_type == 'call':
                delta = 1.0 if S > K else 0.0
            else:
                delta = -1.0 if S < K else 0.0
            gamma = 0.0
            return delta, gamma

        # Black-Scholes d1, d2
        sqrt_tau = np.sqrt(tau)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
        d2 = d1 - sigma * sqrt_tau

        # Delta
        if option_type == 'call':
            delta = norm.cdf(d1)
        elif option_type == 'put':
            delta = norm.cdf(d1) - 1.0
        else:
            raise ValueError(f"Invalid option_type: {option_type}")

        # Gamma (same for call and put)
        gamma = norm.pdf(d1) / (S * sigma * sqrt_tau)

        return float(delta), float(gamma)

    def _add_gamma_hedge(
        self,
        action: np.ndarray,
        observation: Dict[str, np.ndarray],
        info: Dict[str, Any],
        S: float,
        tau: float,
        sigma: float,
    ) -> np.ndarray:
        """
        Add gamma hedge using options on the grid.

        Strategy:
        1. Calculate liability gamma
        2. Find ATM option on grid with similar maturity
        3. Determine option position to neutralize gamma
        4. Add to action vector

        Args:
            action: Current action (underlying position set)
            observation: Observation dict
            info: Info dict
            S: Spot price
            tau: Time to maturity of liability
            sigma: Implied volatility

        Returns:
            action: Updated action with gamma hedge
        """
        # Get option chain from info
        if 'option_chain' not in info or info['option_chain'] is None:
            return action

        option_chain = info['option_chain']
        reference_date = option_chain.reference_date

        # Find ATM option with similar maturity to liability
        best_option_idx = None
        min_distance = float('inf')

        for idx, opt in enumerate(sorted(
            option_chain.options,
            key=lambda opt: (
                (opt.expiry - reference_date).days,
                opt.strike / S,
                0 if opt.option_type == 'call' else 1
            )
        )):
            # Maturity
            days_to_expiry = (opt.expiry - reference_date).days
            tau_opt = days_to_expiry / 365.0

            # Moneyness
            moneyness = opt.strike / S

            # Distance metric: |tau - tau_opt| + |moneyness - 1.0|
            distance = abs(tau - tau_opt) + abs(moneyness - 1.0)

            if distance < min_distance and opt.option_type == 'call':
                min_distance = distance
                best_option_idx = idx

        if best_option_idx is None:
            return action

        # Get option
        best_opt = sorted(
            option_chain.options,
            key=lambda opt: (
                (opt.expiry - reference_date).days,
                opt.strike / S,
                0 if opt.option_type == 'call' else 1
            )
        )[best_option_idx]

        # Calculate gamma of the hedging option
        K_hedge = best_opt.strike
        tau_hedge = (best_opt.expiry - reference_date).days / 365.0
        _, gamma_hedge = self._black_scholes_greeks(
            S=S,
            K=K_hedge,
            tau=tau_hedge,
            sigma=sigma,
            r=self.risk_free_rate,
            option_type='call',
        )

        # Position to neutralize gamma
        liability_gamma = self.liability_quantity * self.current_gamma

        if gamma_hedge > 1e-8:
            option_position = -liability_gamma / gamma_hedge
        else:
            option_position = 0.0

        # Add to action (index is 1 + best_option_idx)
        action[1 + best_option_idx] = option_position

        return action

    def get_state(self) -> Dict[str, Any]:
        """Get agent state for checkpointing."""
        state = super().get_state()
        state.update({
            'current_delta': self.current_delta,
            'current_gamma': self.current_gamma,
            'target_underlying_position': self.target_underlying_position,
            'last_hedge_delta': self.last_hedge_delta,
        })
        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore agent state from checkpoint."""
        super().set_state(state)
        self.current_delta = state.get('current_delta', 0.0)
        self.current_gamma = state.get('current_gamma', 0.0)
        self.target_underlying_position = state.get('target_underlying_position', 0.0)
        self.last_hedge_delta = state.get('last_hedge_delta', 0.0)
