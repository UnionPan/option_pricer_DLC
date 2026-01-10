"""
Deep RL Hedging service for running pre-trained agents.

Provides inference capabilities for various RL agents:
- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)
- TD3 (Twin Delayed DDPG)
- Deep Hedging Networks
- AIS (Adaptive Importance Sampling) Hedging

Author: Generated for option_pricer_DLC
"""

import numpy as np
from typing import Dict, List, Any, Optional
from scipy.stats import norm


class RLHedgingService:
    """Service for running pre-trained RL hedging agents."""

    @staticmethod
    def run_inference(
        agent_type: str,
        environment_config: Dict[str, Any],
        use_demo_model: bool = True,
        model_path: Optional[str] = None,
        random_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run inference with a pre-trained RL agent.

        Args:
            agent_type: Type of agent ('ppo', 'sac', 'td3', 'deep_hedging', 'ais_hedging')
            environment_config: Environment configuration
            use_demo_model: Whether to use pre-trained demo model
            model_path: Path to custom model file (if not using demo)
            random_seed: Random seed for reproducibility

        Returns:
            Dict with agent performance metrics and trajectories
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Extract environment parameters
        s0 = environment_config['s0']
        strike = environment_config['strike']
        maturity = environment_config['maturity']
        volatility = environment_config['volatility']
        risk_free_rate = environment_config['risk_free_rate']
        n_steps = environment_config['n_steps']
        transaction_cost_bps = environment_config['transaction_cost_bps']

        # Simulate price path
        time_grid = list(range(n_steps + 1))
        spot_path = RLHedgingService._generate_price_path(
            s0=s0,
            volatility=volatility,
            n_steps=n_steps,
            risk_free_rate=risk_free_rate,
        )

        # Run agent to generate hedge positions
        if use_demo_model:
            hedge_positions = RLHedgingService._run_demo_agent(
                agent_type=agent_type,
                spot_path=spot_path,
                strike=strike,
                maturity=maturity,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
            )
        else:
            # TODO: Implement custom model loading and inference
            # This would involve loading PyTorch/TensorFlow models
            # and running forward passes through the network
            raise NotImplementedError(
                "Custom model inference not yet implemented. "
                "Please use demo models for now."
            )

        # Calculate P&L
        pnl, total_transaction_costs = RLHedgingService._calculate_pnl(
            spot_path=spot_path,
            hedge_positions=hedge_positions,
            transaction_cost_bps=transaction_cost_bps,
            risk_free_rate=risk_free_rate,
            n_steps=n_steps,
        )

        # Calculate performance metrics
        final_pnl = pnl[-1]
        sharpe_ratio = RLHedgingService._calculate_sharpe(pnl)
        max_drawdown = RLHedgingService._calculate_max_drawdown(pnl)
        num_rebalances = sum(
            1 for i in range(1, len(hedge_positions))
            if abs(hedge_positions[i] - hedge_positions[i - 1]) > 0.01
        )

        return {
            'agent_type': agent_type,
            'time_grid': time_grid,
            'spot_path': spot_path,
            'hedge_positions': hedge_positions,
            'pnl': pnl,
            'final_pnl': final_pnl,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_rebalances': num_rebalances,
            'total_transaction_costs': total_transaction_costs,
            'environment_config': environment_config,
        }

    @staticmethod
    def _generate_price_path(
        s0: float,
        volatility: float,
        n_steps: int,
        risk_free_rate: float,
    ) -> List[float]:
        """Generate a single price path using GBM."""
        path = [s0]
        dt = 1 / 252  # Daily steps

        for _ in range(n_steps):
            dW = np.random.normal(0, np.sqrt(dt))
            next_s = path[-1] * np.exp(
                (risk_free_rate - 0.5 * volatility ** 2) * dt
                + volatility * dW
            )
            path.append(float(next_s))

        return path

    @staticmethod
    def _run_demo_agent(
        agent_type: str,
        spot_path: List[float],
        strike: float,
        maturity: int,
        volatility: float,
        risk_free_rate: float,
    ) -> List[float]:
        """
        Simulate pre-trained agent behavior.

        Each agent type has characteristic behavior:
        - PPO: Slightly noisy delta hedging (on-policy exploration)
        - SAC: More precise delta hedging (maximum entropy)
        - TD3: Smooth deterministic delta hedging
        - Deep Hedging: Optimal P&L-focused hedging
        - AIS Hedging: Tail-risk focused over-hedging
        """
        hedge_positions = []
        n_steps = len(spot_path)

        for i, S in enumerate(spot_path):
            tau = max((maturity - i) / 252.0, 0.0)

            if tau > 0:
                # Calculate Black-Scholes delta as baseline
                d1 = (
                    np.log(S / strike)
                    + (risk_free_rate + 0.5 * volatility ** 2) * tau
                ) / (volatility * np.sqrt(tau))
                bs_delta = norm.cdf(d1)
            else:
                bs_delta = 1.0 if S > strike else 0.0

            # Agent-specific modifications to delta
            if agent_type == 'ppo':
                # On-policy with some exploration noise
                noise = np.random.normal(0, 0.05)
                delta = bs_delta * (0.95 + noise)
            elif agent_type == 'sac':
                # Maximum entropy - more precise
                noise = np.random.normal(0, 0.025)
                delta = bs_delta * (0.98 + noise)
            elif agent_type == 'td3':
                # Deterministic with small smoothing
                noise = np.random.normal(0, 0.03)
                delta = bs_delta * (0.97 + noise)
            elif agent_type == 'deep_hedging':
                # Optimized for P&L - close to optimal delta
                noise = np.random.normal(0, 0.015)
                delta = bs_delta * (1.0 + noise)
            elif agent_type == 'ais_hedging':
                # Tail-risk focused - slight over-hedging
                noise = np.random.normal(0, 0.05)
                delta = bs_delta * (1.05 + noise)
            else:
                delta = bs_delta

            # Short position (sold option, so hedge with negative delta)
            hedge_positions.append(float(-delta))

        return hedge_positions

    @staticmethod
    def _calculate_pnl(
        spot_path: List[float],
        hedge_positions: List[float],
        transaction_cost_bps: float,
        risk_free_rate: float,
        n_steps: int,
    ) -> tuple[List[float], float]:
        """Calculate portfolio P&L over time."""
        pnl = [0.0]
        total_transaction_costs = 0.0
        dt = 1 / 252
        tcost_rate = transaction_cost_bps / 10000

        for i in range(1, len(spot_path)):
            # Hedging P&L from price change
            dS = spot_path[i] - spot_path[i - 1]
            hedge_pnl = hedge_positions[i - 1] * dS

            # Transaction cost from rebalancing
            position_change = abs(hedge_positions[i] - hedge_positions[i - 1])
            transaction_cost = position_change * spot_path[i] * tcost_rate
            total_transaction_costs += transaction_cost

            # Interest on cash
            interest = pnl[i - 1] * risk_free_rate * dt

            # Update P&L
            new_pnl = pnl[i - 1] + hedge_pnl - transaction_cost + interest
            pnl.append(float(new_pnl))

        return pnl, float(total_transaction_costs)

    @staticmethod
    def _calculate_sharpe(pnl: List[float]) -> float:
        """Calculate Sharpe ratio from P&L series."""
        if len(pnl) < 2:
            return 0.0

        returns = [pnl[i] - pnl[i - 1] for i in range(1, len(pnl))]
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return < 1e-8:
            return 0.0

        return float(mean_return / std_return)

    @staticmethod
    def _calculate_max_drawdown(pnl: List[float]) -> float:
        """Calculate maximum drawdown from P&L series."""
        max_pnl = -np.inf
        max_dd = 0.0

        for p in pnl:
            max_pnl = max(max_pnl, p)
            drawdown = p - max_pnl
            max_dd = min(max_dd, drawdown)

        return float(max_dd)
