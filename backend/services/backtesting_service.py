"""
Backtesting service for hedging strategies.

Simulates price dynamics using calibrated P-measure models and runs
hedging agents (Delta hedging, Gamma hedging, etc.) to evaluate performance.

Author: Generated for option_pricer_DLC
"""

import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Tuple
from scipy.stats import norm

from options_desk.processes import GBM, Heston
from options_desk.derivatives import EuropeanCall, EuropeanPut
from options_desk.pricer import COSPricer, HestonAnalyticalPricer

# Import Heston option chain generator (same as HestonEnv)
from options_desk.calibration.data.synthetic_equity import (
    SyntheticEquityOptionChainGenerator,
    HestonVolatilityProfile,
)


class BacktestingService:
    """Service for backtesting hedging strategies with P-measure models."""

    @staticmethod
    def _price_option_model(
        model: str,
        S: float,
        K: float,
        tau: float,
        params: Dict[str, float],
        r: float,
        option_type: str,
        variance_t: Optional[float] = None,
        pricer: Optional[HestonAnalyticalPricer] = None,
        heston_pricer: str = "mgf",
    ) -> float:
        if tau <= 0:
            return max(S - K, 0.0) if option_type == 'call' else max(K - S, 0.0)

        if model == 'heston':
            sigma_v = params.get('sigma_v', params.get('xi', 0.3))
            rho = params.get('rho', 0.0)
            kappa = params.get('kappa', 2.0)
            theta = params.get('theta', 0.04)
            v0 = float(variance_t if variance_t is not None else params.get('v0', theta))
            if heston_pricer == "mgf":
                from options_desk.pricer.heston_mgf_pricer import heston_price_vanilla
                return float(
                    heston_price_vanilla(
                        S=S, K=K, T=tau, r=r, q=0.0,
                        v0=v0, theta=theta, kappa=kappa, volvol=sigma_v, rho=rho,
                        is_call=(option_type == 'call'),
                    )
                )

            heston = Heston(
                mu=r,
                kappa=kappa,
                theta=theta,
                sigma_v=sigma_v,
                rho=rho,
                v0=v0,
            )
            derivative = EuropeanCall(K, tau) if option_type == 'call' else EuropeanPut(K, tau)
            active_pricer = pricer or HestonAnalyticalPricer(risk_free_rate=r)
            return float(active_pricer.price(derivative, heston, np.array([S, v0])).price)

        if model == 'ou':
            return BacktestingService._ou_option_price(
                S=S,
                K=K,
                tau=tau,
                kappa=params.get('kappa', 1.0),
                mu=params.get('mu', S),
                sigma=params.get('sigma', 0.2),
                r=r,
                option_type=option_type,
            )

        sigma = params.get('sigma', 0.2)
        return BacktestingService._black_scholes_price(S, K, tau, sigma, r, option_type)

    @staticmethod
    def _compute_greeks_model(
        model: str,
        S: float,
        K: float,
        tau: float,
        params: Dict[str, float],
        r: float,
        option_type: str,
        variance_t: Optional[float] = None,
        pricer: Optional[HestonAnalyticalPricer] = None,
        heston_pricer: str = "mgf",
    ) -> Tuple[float, float, float, float]:
        if tau <= 0:
            if option_type == 'call':
                delta = 1.0 if S > K else 0.0
            else:
                delta = -1.0 if S < K else 0.0
            return delta, 0.0, 0.0, 0.0

        spot_bump = max(1e-4, 0.01 * S)

        def price_for_spot(spot: float, var_override: Optional[float] = None) -> float:
            return BacktestingService._price_option_model(
                model=model,
                S=spot,
                K=K,
                tau=tau,
                params=params,
                r=r,
                option_type=option_type,
                variance_t=var_override if var_override is not None else variance_t,
                pricer=pricer,
                heston_pricer=heston_pricer,
            )

        price_center = price_for_spot(S)
        price_up = price_for_spot(S + spot_bump)
        price_down = price_for_spot(S - spot_bump)

        delta = (price_up - price_down) / (2.0 * spot_bump)
        gamma = (price_up - 2.0 * price_center + price_down) / (spot_bump ** 2)

        if model == 'heston':
            base_vol = float(np.sqrt(max(variance_t if variance_t is not None else params.get('v0', 0.04), 1e-8)))
            vol_bump = max(1e-4, 0.01 * base_vol)
            v0_up = max((base_vol + vol_bump) ** 2, 1e-8)
            v0_down = max((base_vol - vol_bump) ** 2, 1e-8)
            price_vol_up = price_for_spot(S, var_override=v0_up)
            price_vol_down = price_for_spot(S, var_override=v0_down)
            vega = (price_vol_up - price_vol_down) / (2.0 * vol_bump)
        else:
            base_sigma = float(params.get('sigma', 0.2))
            sigma_bump = max(1e-4, 0.01 * base_sigma)
            params_up = dict(params, sigma=base_sigma + sigma_bump)
            params_down = dict(params, sigma=max(base_sigma - sigma_bump, 1e-8))
            price_vol_up = BacktestingService._price_option_model(
                model=model,
                S=S,
                K=K,
                tau=tau,
                params=params_up,
                r=r,
                option_type=option_type,
                variance_t=variance_t,
                pricer=pricer,
                heston_pricer=heston_pricer,
            )
            price_vol_down = BacktestingService._price_option_model(
                model=model,
                S=S,
                K=K,
                tau=tau,
                params=params_down,
                r=r,
                option_type=option_type,
                variance_t=variance_t,
                pricer=pricer,
                heston_pricer=heston_pricer,
            )
            vega = (price_vol_up - price_vol_down) / (2.0 * sigma_bump)

        theta_bump = min(1.0 / 365.0, tau * 0.2)
        price_t_forward = BacktestingService._price_option_model(
            model=model,
            S=S,
            K=K,
            tau=max(tau - theta_bump, 1e-8),
            params=params,
            r=r,
            option_type=option_type,
            variance_t=variance_t,
            pricer=pricer,
            heston_pricer=heston_pricer,
        )
        theta = (price_t_forward - price_center) / theta_bump

        return delta, gamma, vega, theta

    @staticmethod
    def _ou_option_price(
        S: float,
        K: float,
        tau: float,
        kappa: float,
        mu: float,
        sigma: float,
        r: float,
        option_type: str,
    ) -> float:
        if tau <= 0:
            return max(S - K, 0.0) if option_type == 'call' else max(K - S, 0.0)

        if kappa <= 1e-8:
            variance = sigma ** 2 * tau
            mean = S
        else:
            exp_k = np.exp(-kappa * tau)
            mean = S * exp_k + mu * (1.0 - exp_k)
            variance = (sigma ** 2) * (1.0 - np.exp(-2.0 * kappa * tau)) / (2.0 * kappa)

        std = np.sqrt(max(variance, 1e-12))
        d = (mean - K) / std

        if option_type == 'call':
            price = (mean - K) * norm.cdf(d) + std * norm.pdf(d)
        else:
            price = (K - mean) * norm.cdf(-d) + std * norm.pdf(d)

        return float(np.exp(-r * tau) * price)

    @staticmethod
    def run_backtest(
        model: str,
        parameters: Dict[str, float],
        liability_spec: Dict[str, Any],
        hedging_strategy: str,
        s0: float,
        n_steps: int,
        n_paths: int,
        dt: float,
        risk_free_rate: float = 0.05,
        transaction_cost_bps: float = 0.0,
        rebalance_threshold: float = 0.05,
        random_seed: Optional[int] = None,
        full_visualization: bool = True,
        hedge_options: Optional[List[Dict[str, Any]]] = None,
        heston_pricer: Optional[str] = "mgf",
    ) -> Dict[str, Any]:
        """
        Run backtest of hedging strategy on simulated paths.

        Args:
            model: Model name ('gbm', 'heston', 'ou')
            parameters: Calibrated model parameters
            liability_spec: Dict with 'option_type', 'strike', 'maturity_days', 'quantity'
            hedging_strategy: Strategy name ('delta_hedge', 'delta_gamma_hedge', 'delta_vega_hedge', 'delta_gamma_vega_hedge', 'no_hedge')
            hedge_options: Optional hedge option specs for gamma/vega hedging
            heston_pricer: 'mgf' (fast) or 'analytical' (slow)
            s0: Initial spot price
            n_steps: Number of time steps
            n_paths: Number of Monte Carlo paths to simulate
            dt: Time step size (in years)
            risk_free_rate: Risk-free rate (annualized)
            transaction_cost_bps: Transaction cost in basis points
            rebalance_threshold: Minimum delta change to trigger rebalancing
            random_seed: Random seed for reproducibility

        Returns:
            Dict with:
                - time_grid: Array of times
                - paths: Array of shape (n_paths, n_steps+1) with price paths
                - hedge_positions: Array of hedge positions over time
                - pnl: Array of P&L over time
                - greeks: Dict of Greeks over time
                - transactions: List of transaction events
                - summary_stats: Dict of summary statistics
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Parse liability specification
        option_type = liability_spec['option_type']
        strike = liability_spec['strike']
        maturity_days = liability_spec['maturity_days']
        quantity = liability_spec.get('quantity', -1.0)  # Negative = short position

        # Time grid
        time_grid = np.linspace(0, n_steps * dt, n_steps + 1)
        maturity_time = maturity_days / 365.0

        heston_pricer_choice = (heston_pricer or "mgf").lower()
        heston_pricer_obj = (
            HestonAnalyticalPricer(risk_free_rate=risk_free_rate)
            if model == 'heston' and heston_pricer_choice != "mgf"
            else None
        )

        required_hedge_legs = 0
        if hedging_strategy in ['delta_gamma_hedge', 'delta_vega_hedge']:
            required_hedge_legs = 1
        elif hedging_strategy == 'delta_gamma_vega_hedge':
            required_hedge_legs = 2

        hedge_option_specs: List[Dict[str, Any]] = []
        if required_hedge_legs > 0:
            if hedge_options:
                hedge_option_specs = list(hedge_options)
            while len(hedge_option_specs) < required_hedge_legs:
                hedge_option_specs.append({
                    'option_type': 'call',
                    'strike': float(strike * 1.1),
                    'maturity_days': int(maturity_days),
                })
            if len(hedge_option_specs) > required_hedge_legs:
                hedge_option_specs = hedge_option_specs[:required_hedge_legs]

        # Simulate price paths using calibrated model
        paths, variance_paths = BacktestingService._simulate_paths(
            model=model,
            parameters=parameters,
            s0=s0,
            n_steps=n_steps,
            n_paths=n_paths,
            dt=dt,
        )

        # Initialize storage for one representative path (for detailed tracking)
        representative_path_idx = 0
        representative_path = paths[representative_path_idx]
        representative_variance_path = variance_paths[representative_path_idx] if variance_paths is not None else None

        # Storage for hedging results
        hedge_positions = np.zeros(n_steps + 1)
        n_hedge_options = len(hedge_option_specs)
        hedge_option_positions = np.zeros((n_hedge_options, n_steps + 1)) if n_hedge_options > 0 else None
        cash = np.zeros(n_steps + 1)
        portfolio_value = np.zeros(n_steps + 1)
        option_value = np.zeros(n_steps + 1)
        hedge_option_value = np.zeros((n_hedge_options, n_steps + 1)) if n_hedge_options > 0 else None
        pnl = np.zeros(n_steps + 1)

        # Greeks storage
        deltas = np.zeros(n_steps + 1)
        gammas = np.zeros(n_steps + 1)
        vegas = np.zeros(n_steps + 1)
        thetas = np.zeros(n_steps + 1)

        # Transaction log
        transactions = []

        # Option chain storage (for visualization)
        option_chains_data = []

        # Volatility path (sqrt of variance for Heston, constant for GBM/OU)
        volatility_path = np.zeros(n_steps + 1)
        base_sigma = parameters.get('sigma', 0.2)

        # Initialize hedging at t=0
        S0 = representative_path[0]
        tau0 = maturity_time

        # Get current volatility (for Heston, use variance path; for others, use constant)
        if model == 'heston' and representative_variance_path is not None:
            current_vol = np.sqrt(representative_variance_path[0])
        else:
            current_vol = base_sigma
        volatility_path[0] = current_vol

        option_value[0] = BacktestingService._price_option_model(
            model=model,
            S=S0,
            K=strike,
            tau=tau0,
            params=parameters,
            r=risk_free_rate,
            option_type=option_type,
            variance_t=representative_variance_path[0] if representative_variance_path is not None else None,
            pricer=heston_pricer_obj,
            heston_pricer=heston_pricer_choice,
        )

        delta0, gamma0, vega0, theta0 = BacktestingService._compute_greeks_model(
            model=model,
            S=S0,
            K=strike,
            tau=tau0,
            params=parameters,
            r=risk_free_rate,
            option_type=option_type,
            variance_t=representative_variance_path[0] if representative_variance_path is not None else None,
            pricer=heston_pricer_obj,
            heston_pricer=heston_pricer_choice,
        )

        deltas[0] = delta0
        gammas[0] = gamma0
        vegas[0] = vega0
        thetas[0] = theta0

        # Generate initial option chain (only if full visualization enabled)
        if full_visualization:
            option_chain_t0 = BacktestingService._generate_option_chain(
                S=S0,
                time_step=0,
                time_grid=time_grid,
                risk_free_rate=risk_free_rate,
                model=model,
                parameters=parameters,
                variance_t=representative_variance_path[0] if representative_variance_path is not None else None,
                pricer=heston_pricer_obj,
                heston_pricer=heston_pricer_choice,
            )
            option_chains_data.append(option_chain_t0)

        # Initial hedge option position(s)
        hedge_delta_exposure = 0.0
        if hedge_option_positions is not None and hedge_option_value is not None:
            hedge_deltas0 = np.zeros(n_hedge_options)
            hedge_gammas0 = np.zeros(n_hedge_options)
            hedge_vegas0 = np.zeros(n_hedge_options)
            for i, spec in enumerate(hedge_option_specs):
                hedge_option_value[i, 0] = BacktestingService._price_option_model(
                    model=model,
                    S=S0,
                    K=spec['strike'],
                    tau=spec['maturity_days'] / 365.0,
                    params=parameters,
                    r=risk_free_rate,
                    option_type=spec['option_type'],
                    variance_t=representative_variance_path[0] if representative_variance_path is not None else None,
                    pricer=heston_pricer_obj,
                    heston_pricer=heston_pricer_choice,
                )
                hedge_deltas0[i], hedge_gammas0[i], hedge_vegas0[i], _ = BacktestingService._compute_greeks_model(
                    model=model,
                    S=S0,
                    K=spec['strike'],
                    tau=spec['maturity_days'] / 365.0,
                    params=parameters,
                    r=risk_free_rate,
                    option_type=spec['option_type'],
                    variance_t=representative_variance_path[0] if representative_variance_path is not None else None,
                    pricer=heston_pricer_obj,
                    heston_pricer=heston_pricer_choice,
                )

            target_hedge_option = np.zeros(n_hedge_options)
            if hedging_strategy == 'delta_gamma_hedge' and abs(hedge_gammas0[0]) > 1e-8:
                target_hedge_option[0] = -quantity * gamma0 / hedge_gammas0[0]
            elif hedging_strategy == 'delta_vega_hedge' and abs(hedge_vegas0[0]) > 1e-8:
                target_hedge_option[0] = -quantity * vega0 / hedge_vegas0[0]
            elif hedging_strategy == 'delta_gamma_vega_hedge':
                if abs(hedge_gammas0[0]) > 1e-8:
                    target_hedge_option[0] = -quantity * gamma0 / hedge_gammas0[0]
                if n_hedge_options > 1 and abs(hedge_vegas0[1]) > 1e-8:
                    target_hedge_option[1] = -quantity * vega0 / hedge_vegas0[1]

            hedge_option_positions[:, 0] = target_hedge_option
            hedge_delta_exposure = float(np.dot(target_hedge_option, hedge_deltas0))

        # Initial hedge position (underlying)
        if hedging_strategy in ['delta_hedge', 'delta_gamma_hedge', 'delta_vega_hedge', 'delta_gamma_vega_hedge']:
            hedge_positions[0] = -quantity * delta0 - hedge_delta_exposure
        else:
            hedge_positions[0] = 0.0

        # Initial cash (from selling option)
        hedge_option_cash = 0.0
        if hedge_option_positions is not None and hedge_option_value is not None:
            hedge_option_cash = float(np.dot(hedge_option_positions[:, 0], hedge_option_value[:, 0]))
        cash[0] = -quantity * option_value[0] - hedge_positions[0] * S0 - hedge_option_cash
        portfolio_value[0] = cash[0] + hedge_positions[0] * S0 + hedge_option_cash + quantity * option_value[0]
        pnl[0] = portfolio_value[0]

        # Record initial trade
        if abs(hedge_positions[0]) > 1e-6:
            transactions.append({
                'time': 0.0,
                'spot': float(S0),
                'action': 'initial_hedge',
                'shares_traded': float(hedge_positions[0]),
                'cost': float(hedge_positions[0] * S0),
                'delta': float(delta0),
            })
        if hedge_option_positions is not None and hedge_option_value is not None:
            for i, spec in enumerate(hedge_option_specs):
                if abs(hedge_option_positions[i, 0]) > 1e-6:
                    transactions.append({
                        'time': 0.0,
                        'spot': float(S0),
                        'action': 'initial_hedge_option',
                        'option_strike': float(spec['strike']),
                        'option_type': spec['option_type'],
                        'contracts_traded': float(hedge_option_positions[i, 0]),
                        'cost': float(hedge_option_positions[i, 0] * hedge_option_value[i, 0]),
                        'hedge_leg': i,
                    })

        # Simulate forward
        last_hedge_delta = hedge_positions[0]

        for t in range(1, n_steps + 1):
            S_t = representative_path[t]
            tau_t = max(maturity_time - time_grid[t], 0.0)

            # Get current volatility
            if model == 'heston' and representative_variance_path is not None:
                current_vol = np.sqrt(representative_variance_path[t])
            else:
                current_vol = base_sigma
            volatility_path[t] = current_vol

            # Generate option chain at current state (only if full visualization enabled)
            if full_visualization:
                option_chain_t = BacktestingService._generate_option_chain(
                    S=S_t,
                    time_step=t,
                    time_grid=time_grid,
                    risk_free_rate=risk_free_rate,
                    model=model,
                    parameters=parameters,
                    variance_t=representative_variance_path[t] if representative_variance_path is not None else None,
                    pricer=heston_pricer_obj,
                    heston_pricer=heston_pricer_choice,
                )
                option_chains_data.append(option_chain_t)

            # Calculate option value and Greeks
            if tau_t > 0:
                option_value[t] = BacktestingService._price_option_model(
                    model=model,
                    S=S_t,
                    K=strike,
                    tau=tau_t,
                    params=parameters,
                    r=risk_free_rate,
                    option_type=option_type,
                    variance_t=representative_variance_path[t] if representative_variance_path is not None else None,
                    pricer=heston_pricer_obj,
                    heston_pricer=heston_pricer_choice,
                )

                delta_t, gamma_t, vega_t, theta_t = BacktestingService._compute_greeks_model(
                    model=model,
                    S=S_t,
                    K=strike,
                    tau=tau_t,
                    params=parameters,
                    r=risk_free_rate,
                    option_type=option_type,
                    variance_t=representative_variance_path[t] if representative_variance_path is not None else None,
                    pricer=heston_pricer_obj,
                    heston_pricer=heston_pricer_choice,
                )
            else:
                # At maturity
                if option_type == 'call':
                    option_value[t] = max(S_t - strike, 0.0)
                else:
                    option_value[t] = max(strike - S_t, 0.0)

                delta_t = 1.0 if (option_type == 'call' and S_t > strike) else 0.0
                if option_type == 'put':
                    delta_t = -1.0 if S_t < strike else 0.0
                gamma_t = 0.0
                vega_t = 0.0
                theta_t = 0.0

            deltas[t] = delta_t
            gammas[t] = gamma_t
            vegas[t] = vega_t
            thetas[t] = theta_t

            # Determine if rebalancing is needed
            target_hedge = 0.0
            target_hedge_options = np.zeros(n_hedge_options) if n_hedge_options > 0 else None
            hedge_deltas_t = None
            hedge_gammas_t = None
            hedge_vegas_t = None

            if hedging_strategy == 'no_hedge':
                target_hedge = 0.0
            elif hedging_strategy == 'delta_hedge':
                target_hedge = -quantity * delta_t
            else:
                if hedge_option_positions is not None and hedge_option_value is not None:
                    hedge_deltas_t = np.zeros(n_hedge_options)
                    hedge_gammas_t = np.zeros(n_hedge_options)
                    hedge_vegas_t = np.zeros(n_hedge_options)
                    for i, spec in enumerate(hedge_option_specs):
                        tau_remaining = max(spec['maturity_days'] / 365.0 - time_grid[t], 0.0)
                        hedge_option_value[i, t] = BacktestingService._price_option_model(
                            model=model,
                            S=S_t,
                            K=spec['strike'],
                            tau=tau_remaining,
                            params=parameters,
                            r=risk_free_rate,
                            option_type=spec['option_type'],
                            variance_t=representative_variance_path[t] if representative_variance_path is not None else None,
                            pricer=heston_pricer_obj,
                            heston_pricer=heston_pricer_choice,
                        )
                        hedge_deltas_t[i], hedge_gammas_t[i], hedge_vegas_t[i], _ = BacktestingService._compute_greeks_model(
                            model=model,
                            S=S_t,
                            K=spec['strike'],
                            tau=tau_remaining,
                            params=parameters,
                            r=risk_free_rate,
                            option_type=spec['option_type'],
                            variance_t=representative_variance_path[t] if representative_variance_path is not None else None,
                            pricer=heston_pricer_obj,
                            heston_pricer=heston_pricer_choice,
                        )

                    if hedging_strategy == 'delta_gamma_hedge' and abs(hedge_gammas_t[0]) > 1e-8:
                        target_hedge_options[0] = -quantity * gamma_t / hedge_gammas_t[0]
                    elif hedging_strategy == 'delta_vega_hedge' and abs(hedge_vegas_t[0]) > 1e-8:
                        target_hedge_options[0] = -quantity * vega_t / hedge_vegas_t[0]
                    elif hedging_strategy == 'delta_gamma_vega_hedge':
                        if abs(hedge_gammas_t[0]) > 1e-8:
                            target_hedge_options[0] = -quantity * gamma_t / hedge_gammas_t[0]
                        if n_hedge_options > 1 and abs(hedge_vegas_t[1]) > 1e-8:
                            target_hedge_options[1] = -quantity * vega_t / hedge_vegas_t[1]

                hedge_delta_exposure_t = float(np.dot(target_hedge_options, hedge_deltas_t)) if hedge_deltas_t is not None else 0.0
                target_hedge = -quantity * delta_t - hedge_delta_exposure_t

            delta_drift = abs(target_hedge - last_hedge_delta)

            if hedging_strategy in ['delta_gamma_hedge', 'delta_vega_hedge', 'delta_gamma_vega_hedge']:
                should_rebalance = True
            else:
                should_rebalance = delta_drift >= rebalance_threshold or tau_t <= 0.0

            if should_rebalance and hedging_strategy != 'no_hedge':
                # Rebalance
                shares_to_trade = target_hedge - hedge_positions[t-1]
                transaction_cost = abs(shares_to_trade) * S_t * (transaction_cost_bps / 10000.0)

                hedge_positions[t] = target_hedge
                hedge_option_trade_total = 0.0
                hedge_option_cost_total = 0.0
                if hedge_option_positions is not None and hedge_option_value is not None and target_hedge_options is not None:
                    for i in range(n_hedge_options):
                        hedge_option_trade = target_hedge_options[i] - hedge_option_positions[i, t-1]
                        hedge_option_positions[i, t] = target_hedge_options[i]
                        hedge_option_cost = abs(hedge_option_trade) * hedge_option_value[i, t] * (transaction_cost_bps / 10000.0)
                        hedge_option_trade_total += hedge_option_trade * hedge_option_value[i, t]
                        hedge_option_cost_total += hedge_option_cost

                cash[t] = (
                    cash[t-1] * np.exp(risk_free_rate * dt)
                    - shares_to_trade * S_t
                    - transaction_cost
                    - hedge_option_trade_total
                    - hedge_option_cost_total
                )
                last_hedge_delta = target_hedge

                # Record transaction
                if abs(shares_to_trade) > 1e-6:
                    transactions.append({
                        'time': float(time_grid[t]),
                        'spot': float(S_t),
                        'action': 'rebalance',
                        'shares_traded': float(shares_to_trade),
                        'cost': float(shares_to_trade * S_t + transaction_cost),
                        'delta': float(delta_t),
                        'transaction_cost': float(transaction_cost),
                    })
                if hedge_option_positions is not None and hedge_option_value is not None and target_hedge_options is not None:
                    for i, spec in enumerate(hedge_option_specs):
                        hedge_option_trade = target_hedge_options[i] - hedge_option_positions[i, t-1]
                        hedge_option_cost = abs(hedge_option_trade) * hedge_option_value[i, t] * (transaction_cost_bps / 10000.0)
                        if abs(hedge_option_trade) > 1e-6:
                            transactions.append({
                                'time': float(time_grid[t]),
                                'spot': float(S_t),
                                'action': 'rebalance_hedge_option',
                                'option_strike': float(spec['strike']),
                                'option_type': spec['option_type'],
                                'contracts_traded': float(hedge_option_trade),
                                'cost': float(hedge_option_trade * hedge_option_value[i, t] + hedge_option_cost),
                                'transaction_cost': float(hedge_option_cost),
                                'hedge_leg': i,
                            })
            else:
                # No rebalancing
                hedge_positions[t] = hedge_positions[t-1]
                if hedge_option_positions is not None:
                    hedge_option_positions[:, t] = hedge_option_positions[:, t-1]
                cash[t] = cash[t-1] * np.exp(risk_free_rate * dt)

            # Mark-to-market P&L
            hedge_option_value_t = 0.0
            if hedge_option_positions is not None and hedge_option_value is not None:
                hedge_option_value_t = float(np.dot(hedge_option_positions[:, t], hedge_option_value[:, t]))
            portfolio_value[t] = cash[t] + hedge_positions[t] * S_t + hedge_option_value_t + quantity * option_value[t]
            pnl[t] = portfolio_value[t]

        # Calculate summary statistics across all paths
        final_pnls = BacktestingService._calculate_final_pnl_distribution(
            paths=paths,
            variance_paths=variance_paths,
            option_type=option_type,
            strike=strike,
            quantity=quantity,
            maturity_time=maturity_time,
            time_grid=time_grid,
            hedging_strategy=hedging_strategy,
            parameters=parameters,
            model=model,
            dt=dt,
            risk_free_rate=risk_free_rate,
            rebalance_threshold=rebalance_threshold,
            transaction_cost_bps=transaction_cost_bps,
            hedge_option_specs=hedge_option_specs,
            heston_pricer=heston_pricer_choice,
        )

        summary_stats = {
            'mean_pnl': float(np.mean(final_pnls)),
            'std_pnl': float(np.std(final_pnls)),
            'median_pnl': float(np.median(final_pnls)),
            'min_pnl': float(np.min(final_pnls)),
            'max_pnl': float(np.max(final_pnls)),
            'sharpe_ratio': float(np.mean(final_pnls) / (np.std(final_pnls) + 1e-8)),
            'var_95': float(np.percentile(final_pnls, 5)),
            'cvar_95': float(np.mean(final_pnls[final_pnls <= np.percentile(final_pnls, 5)])),
            'num_rebalances': len(transactions),
            'total_transaction_costs': float(sum(t.get('transaction_cost', 0.0) for t in transactions)),
        }

        # Compute IV surface from option chains (only if full visualization enabled)
        if full_visualization and len(option_chains_data) > 0:
            iv_surface = BacktestingService._compute_iv_surface(option_chains_data)
        else:
            iv_surface = None

        return {
            'time_grid': time_grid.tolist(),
            'representative_path': representative_path.tolist(),
            'all_paths': paths[:min(20, n_paths)].tolist(),  # Return first 20 paths for visualization
            'variance_path': representative_variance_path.tolist() if representative_variance_path is not None else None,
            'volatility_path': volatility_path.tolist(),
            'hedge_positions': hedge_positions.tolist(),
            'hedge_option_positions': hedge_option_positions.tolist() if hedge_option_positions is not None else None,
            'hedge_option_value': hedge_option_value.tolist() if hedge_option_value is not None else None,
            'cash': cash.tolist(),
            'portfolio_value': portfolio_value.tolist(),
            'option_value': option_value.tolist(),
            'pnl': pnl.tolist(),
            'greeks': {
                'delta': deltas.tolist(),
                'gamma': gammas.tolist(),
                'vega': vegas.tolist(),
                'theta': thetas.tolist(),
            },
            'transactions': transactions,
            'summary_stats': summary_stats,
            'final_pnl_distribution': final_pnls.tolist(),
            'option_chains': option_chains_data if full_visualization else None,  # Only if full viz
            'iv_surface': iv_surface,  # Only if full viz
            'liability_spec': liability_spec,
            'hedge_option_specs': hedge_option_specs if hedge_option_specs else None,
            'hedging_strategy': hedging_strategy,
            'model': model,
            'parameters': parameters,
        }

    @staticmethod
    def _simulate_paths(
        model: str,
        parameters: Dict[str, float],
        s0: float,
        n_steps: int,
        n_paths: int,
        dt: float,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Simulate price paths using calibrated model.

        Returns:
            Tuple of (paths, variance_paths)
            - paths: shape (n_paths, n_steps+1) price paths
            - variance_paths: shape (n_paths, n_steps+1) variance paths (None for GBM/OU)
        """

        if model == 'gbm':
            mu = parameters.get('mu', 0.0)
            sigma = parameters.get('sigma', 0.2)

            paths = np.zeros((n_paths, n_steps + 1))
            paths[:, 0] = s0

            for t in range(n_steps):
                dW = np.random.normal(0, np.sqrt(dt), n_paths)
                paths[:, t+1] = paths[:, t] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)

            return paths, None

        elif model == 'heston':
            # Use Heston process simulation
            mu = parameters.get('mu', 0.0)
            kappa = parameters.get('kappa', 2.0)
            theta = parameters.get('theta', 0.04)
            sigma_v = parameters.get('sigma_v', parameters.get('xi', 0.3))  # Support both sigma_v and xi
            rho = parameters.get('rho', 0.0)  # Default to 0 if not provided
            v0 = parameters.get('v0', 0.04)

            paths = np.zeros((n_paths, n_steps + 1))
            variance_paths = np.zeros((n_paths, n_steps + 1))

            paths[:, 0] = s0
            variance_paths[:, 0] = v0

            for t in range(n_steps):
                dW1 = np.random.normal(0, np.sqrt(dt), n_paths)
                dW2_indep = np.random.normal(0, np.sqrt(dt), n_paths)
                dW2 = rho * dW1 + np.sqrt(1 - rho**2) * dW2_indep

                # Milstein scheme for variance
                v_t = np.maximum(variance_paths[:, t], 0.0)
                variance_paths[:, t+1] = v_t + kappa * (theta - v_t) * dt + sigma_v * np.sqrt(v_t) * dW2 + \
                                        0.25 * sigma_v**2 * (dW2**2 - dt)
                variance_paths[:, t+1] = np.maximum(variance_paths[:, t+1], 0.0)

                # Euler scheme for log-price
                paths[:, t+1] = paths[:, t] * np.exp((mu - 0.5 * v_t) * dt + np.sqrt(v_t) * dW1)

            return paths, variance_paths

        elif model == 'ou':
            mu = parameters.get('mu', 100.0)
            kappa = parameters.get('kappa', 1.0)
            sigma = parameters.get('sigma', 10.0)

            paths = np.zeros((n_paths, n_steps + 1))
            paths[:, 0] = s0

            for t in range(n_steps):
                dW = np.random.normal(0, np.sqrt(dt), n_paths)
                paths[:, t+1] = paths[:, t] + kappa * (mu - paths[:, t]) * dt + sigma * dW
                paths[:, t+1] = np.maximum(paths[:, t+1], 0.01)  # Floor at 0.01

            return paths, None

        else:
            raise ValueError(f"Unknown model: {model}")

    @staticmethod
    def _black_scholes_price(
        S: float,
        K: float,
        tau: float,
        sigma: float,
        r: float,
        option_type: str,
    ) -> float:
        """Calculate Black-Scholes option price."""
        if tau <= 0:
            if option_type == 'call':
                return max(S - K, 0.0)
            else:
                return max(K - S, 0.0)

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)

        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return price

    @staticmethod
    def _calculate_greeks(
        S: float,
        K: float,
        tau: float,
        sigma: float,
        r: float,
        option_type: str,
        model: str = 'gbm',
        current_variance: float = None,
        parameters: Dict[str, float] = None,
    ) -> Tuple[float, float, float, float]:
        """Legacy wrapper for Greeks. Prefer _compute_greeks_model."""
        params = parameters or {}
        variance_t = current_variance if current_variance is not None else None
        pricer = HestonAnalyticalPricer(risk_free_rate=r) if model == 'heston' else None
        return BacktestingService._compute_greeks_model(
            model=model,
            S=S,
            K=K,
            tau=tau,
            params=params,
            r=r,
            option_type=option_type,
            variance_t=variance_t,
            pricer=pricer,
            heston_pricer="analytical",
        )

    @staticmethod
    def _calculate_final_pnl_distribution(
        paths: np.ndarray,
        variance_paths: Optional[np.ndarray],
        option_type: str,
        strike: float,
        quantity: float,
        maturity_time: float,
        time_grid: np.ndarray,
        hedging_strategy: str,
        parameters: Dict[str, float],
        model: str,
        dt: float,
        risk_free_rate: float,
        rebalance_threshold: float,
        transaction_cost_bps: float,
        hedge_option_specs: Optional[List[Dict[str, Any]]] = None,
        heston_pricer: str = "mgf",
    ) -> np.ndarray:
        """Calculate final P&L for all paths using consistent hedging logic."""
        n_paths = paths.shape[0]
        final_pnls = np.zeros(n_paths)
        heston_pricer_obj = (
            HestonAnalyticalPricer(risk_free_rate=risk_free_rate)
            if model == 'heston' and heston_pricer != "mgf"
            else None
        )

        for path_idx in range(n_paths):
            path = paths[path_idx]
            variance_path = None
            if model == 'heston' and variance_paths is not None:
                variance_path = variance_paths[path_idx]

            final_pnls[path_idx] = BacktestingService._simulate_hedged_path(
                path=path,
                variance_path=variance_path,
                option_type=option_type,
                strike=strike,
                quantity=quantity,
                maturity_time=maturity_time,
                time_grid=time_grid,
                hedging_strategy=hedging_strategy,
                parameters=parameters,
                model=model,
                risk_free_rate=risk_free_rate,
                dt=dt,
                rebalance_threshold=rebalance_threshold,
                transaction_cost_bps=transaction_cost_bps,
                hedge_option_specs=hedge_option_specs,
                pricer=heston_pricer_obj,
                heston_pricer=heston_pricer,
            )

        return final_pnls

    @staticmethod
    def _simulate_hedged_path(
        path: np.ndarray,
        variance_path: Optional[np.ndarray],
        option_type: str,
        strike: float,
        quantity: float,
        maturity_time: float,
        time_grid: np.ndarray,
        hedging_strategy: str,
        parameters: Dict[str, float],
        model: str,
        risk_free_rate: float,
        dt: float,
        rebalance_threshold: float,
        transaction_cost_bps: float,
        hedge_option_specs: Optional[List[Dict[str, Any]]],
        pricer: Optional[HestonAnalyticalPricer],
        heston_pricer: str = "mgf",
    ) -> float:
        n_steps = len(path) - 1
        hedge_positions = 0.0
        hedge_option_positions = 0.0
        cash = 0.0

        S0 = path[0]
        tau0 = maturity_time
        v0 = variance_path[0] if variance_path is not None else None

        option_value = BacktestingService._price_option_model(
            model=model,
            S=S0,
            K=strike,
            tau=tau0,
            params=parameters,
            r=risk_free_rate,
            option_type=option_type,
            variance_t=v0,
            pricer=pricer,
            heston_pricer=heston_pricer,
        )
        delta0, gamma0, vega0, _ = BacktestingService._compute_greeks_model(
            model=model,
            S=S0,
            K=strike,
            tau=tau0,
            params=parameters,
            r=risk_free_rate,
            option_type=option_type,
            variance_t=v0,
            pricer=pricer,
            heston_pricer=heston_pricer,
        )

        hedge_option_specs = hedge_option_specs or []
        n_hedge_options = len(hedge_option_specs)
        hedge_option_positions = np.zeros(n_hedge_options)
        hedge_option_values = np.zeros(n_hedge_options)
        hedge_deltas0 = np.zeros(n_hedge_options)
        hedge_gammas0 = np.zeros(n_hedge_options)
        hedge_vegas0 = np.zeros(n_hedge_options)

        if n_hedge_options > 0:
            for i, spec in enumerate(hedge_option_specs):
                hedge_option_values[i] = BacktestingService._price_option_model(
                    model=model,
                    S=S0,
                    K=spec['strike'],
                    tau=spec['maturity_days'] / 365.0,
                    params=parameters,
                    r=risk_free_rate,
                    option_type=spec['option_type'],
                    variance_t=v0,
                    pricer=pricer,
                    heston_pricer=heston_pricer,
                )
                hedge_deltas0[i], hedge_gammas0[i], hedge_vegas0[i], _ = BacktestingService._compute_greeks_model(
                    model=model,
                    S=S0,
                    K=spec['strike'],
                    tau=spec['maturity_days'] / 365.0,
                    params=parameters,
                    r=risk_free_rate,
                    option_type=spec['option_type'],
                    variance_t=v0,
                    pricer=pricer,
                    heston_pricer=heston_pricer,
                )

            target_hedge_options = np.zeros(n_hedge_options)
            if hedging_strategy == 'delta_gamma_hedge' and abs(hedge_gammas0[0]) > 1e-8:
                target_hedge_options[0] = -quantity * gamma0 / hedge_gammas0[0]
            elif hedging_strategy == 'delta_vega_hedge' and abs(hedge_vegas0[0]) > 1e-8:
                target_hedge_options[0] = -quantity * vega0 / hedge_vegas0[0]
            elif hedging_strategy == 'delta_gamma_vega_hedge':
                if abs(hedge_gammas0[0]) > 1e-8:
                    target_hedge_options[0] = -quantity * gamma0 / hedge_gammas0[0]
                if n_hedge_options > 1 and abs(hedge_vegas0[1]) > 1e-8:
                    target_hedge_options[1] = -quantity * vega0 / hedge_vegas0[1]

            hedge_option_positions = target_hedge_options

        hedge_delta_exposure = float(np.dot(hedge_option_positions, hedge_deltas0)) if n_hedge_options > 0 else 0.0
        if hedging_strategy in ['delta_hedge', 'delta_gamma_hedge', 'delta_vega_hedge', 'delta_gamma_vega_hedge']:
            hedge_positions = -quantity * delta0 - hedge_delta_exposure

        cash = -quantity * option_value - hedge_positions * S0 - float(np.dot(hedge_option_positions, hedge_option_values))
        last_hedge_delta = hedge_positions

        for t in range(1, n_steps + 1):
            S_t = path[t]
            tau_t = max(maturity_time - time_grid[t], 0.0)
            v_t = variance_path[t] if variance_path is not None else None

            option_value = BacktestingService._price_option_model(
                model=model,
                S=S_t,
                K=strike,
                tau=tau_t,
                params=parameters,
                r=risk_free_rate,
                option_type=option_type,
                variance_t=v_t,
                pricer=pricer,
                heston_pricer=heston_pricer,
            )

            delta_t, gamma_t, vega_t, _ = BacktestingService._compute_greeks_model(
                model=model,
                S=S_t,
                K=strike,
                tau=tau_t,
                params=parameters,
                r=risk_free_rate,
                option_type=option_type,
                variance_t=v_t,
                pricer=pricer,
                heston_pricer=heston_pricer,
            )

            target_hedge = 0.0
            target_hedge_options = np.zeros(n_hedge_options)
            hedge_deltas_t = np.zeros(n_hedge_options)
            hedge_gammas_t = np.zeros(n_hedge_options)
            hedge_vegas_t = np.zeros(n_hedge_options)
            hedge_option_values = np.zeros(n_hedge_options)

            if hedging_strategy == 'delta_hedge':
                target_hedge = -quantity * delta_t
            elif hedging_strategy in ['delta_gamma_hedge', 'delta_vega_hedge', 'delta_gamma_vega_hedge']:
                for i, spec in enumerate(hedge_option_specs):
                    tau_remaining = max(spec['maturity_days'] / 365.0 - time_grid[t], 0.0)
                    hedge_option_values[i] = BacktestingService._price_option_model(
                        model=model,
                        S=S_t,
                        K=spec['strike'],
                        tau=tau_remaining,
                        params=parameters,
                        r=risk_free_rate,
                        option_type=spec['option_type'],
                        variance_t=v_t,
                        pricer=pricer,
                        heston_pricer=heston_pricer,
                    )
                    hedge_deltas_t[i], hedge_gammas_t[i], hedge_vegas_t[i], _ = BacktestingService._compute_greeks_model(
                        model=model,
                        S=S_t,
                        K=spec['strike'],
                        tau=tau_remaining,
                        params=parameters,
                        r=risk_free_rate,
                        option_type=spec['option_type'],
                        variance_t=v_t,
                        pricer=pricer,
                        heston_pricer=heston_pricer,
                    )

                if hedging_strategy == 'delta_gamma_hedge' and abs(hedge_gammas_t[0]) > 1e-8:
                    target_hedge_options[0] = -quantity * gamma_t / hedge_gammas_t[0]
                elif hedging_strategy == 'delta_vega_hedge' and abs(hedge_vegas_t[0]) > 1e-8:
                    target_hedge_options[0] = -quantity * vega_t / hedge_vegas_t[0]
                elif hedging_strategy == 'delta_gamma_vega_hedge':
                    if abs(hedge_gammas_t[0]) > 1e-8:
                        target_hedge_options[0] = -quantity * gamma_t / hedge_gammas_t[0]
                    if n_hedge_options > 1 and abs(hedge_vegas_t[1]) > 1e-8:
                        target_hedge_options[1] = -quantity * vega_t / hedge_vegas_t[1]

                hedge_delta_exposure_t = float(np.dot(target_hedge_options, hedge_deltas_t))
                target_hedge = -quantity * delta_t - hedge_delta_exposure_t

            delta_drift = abs(target_hedge - last_hedge_delta)
            if hedging_strategy in ['delta_gamma_hedge', 'delta_vega_hedge', 'delta_gamma_vega_hedge']:
                should_rebalance = True
            else:
                should_rebalance = delta_drift >= rebalance_threshold or tau_t <= 0.0

            if should_rebalance and hedging_strategy != 'no_hedge':
                shares_to_trade = target_hedge - hedge_positions
                transaction_cost = abs(shares_to_trade) * S_t * (transaction_cost_bps / 10000.0)

                hedge_option_trade = target_hedge_options - hedge_option_positions
                hedge_option_cost = np.sum(np.abs(hedge_option_trade) * hedge_option_values * (transaction_cost_bps / 10000.0))
                hedge_option_cash = float(np.dot(hedge_option_trade, hedge_option_values))

                cash = (
                    cash * np.exp(risk_free_rate * dt)
                    - shares_to_trade * S_t
                    - transaction_cost
                    - hedge_option_cash
                    - hedge_option_cost
                )
                hedge_positions = target_hedge
                hedge_option_positions = target_hedge_options
                last_hedge_delta = target_hedge
            else:
                cash = cash * np.exp(risk_free_rate * dt)

        portfolio_value = (
            cash
            + hedge_positions * path[-1]
            + float(np.dot(hedge_option_positions, hedge_option_values))
            + quantity * option_value
        )
        return float(portfolio_value)

    @staticmethod
    def _generate_option_chain(
        S: float,
        time_step: int,
        time_grid: np.ndarray,
        risk_free_rate: float,
        model: str,
        parameters: Dict[str, float],
        variance_t: Optional[float] = None,
        pricer: Optional[HestonAnalyticalPricer] = None,
        heston_pricer: str = "mgf",
    ) -> Dict[str, Any]:
        """
        Generate option chain at current state using Heston MGF pricing (same as HestonEnv).

        Creates a grid of options across multiple strikes and maturities using the
        SyntheticEquityOptionChainGenerator, which ensures realistic IV smile/skew
        and no-arbitrage conditions.

        Args:
            S: Current spot price
            time_step: Current time step index
            time_grid: Full time grid
            risk_free_rate: Risk-free rate
            parameters: Model parameters
            model: Model name ('heston', 'gbm', etc.)
            heston_pricer: Heston pricing method ('mgf' or 'analytical')

        Returns:
            Dict with option chain data
        """
        # Define option grid (matching HestonEnv default adaptive grid)
        moneyness_by_maturity = {
            7:  [0.95, 0.975, 1.0, 1.025, 1.05],              # Very tight: ±5%
            14: [0.95, 0.975, 1.0, 1.025, 1.05],              # Tight: ±5%
            30: [0.90, 0.95, 0.975, 1.0, 1.025, 1.05, 1.10],  # Medium: ±10%
            60: [0.90, 0.95, 1.0, 1.05, 1.10],                # Standard: ±10%
            90: [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15],    # Wide: ±15%
        }
        maturity_days_grid = [7, 14, 30, 60, 90]

        current_time = time_grid[time_step]
        current_variance = float(variance_t) if variance_t is not None else float(parameters.get('v0', parameters.get('sigma', 0.2) ** 2))
        current_vol = float(np.sqrt(max(current_variance, 1e-8)))

        # For Heston model, use HestonVolatilityProfile and SyntheticEquityOptionChainGenerator
        if model == 'heston':
            # Create Heston volatility profile with current variance
            vol_profile = HestonVolatilityProfile(
                kappa=parameters.get('kappa', 2.0),
                theta=parameters.get('theta', 0.04),
                xi=parameters.get('xi', parameters.get('sigma_v', 0.3)),
                rho=parameters.get('rho', 0.0),
                v0=current_variance,
                atm_iv=current_vol,
            )

            # Create option chain generator
            generator = SyntheticEquityOptionChainGenerator(
                risk_free_rate=risk_free_rate,
                dividend_yield=0.0,
                maturities_days=maturity_days_grid,
                moneyness_by_maturity=moneyness_by_maturity,
                add_noise=False,
                random_seed=42,
            )

            # Generate chain at current state
            reference_date = date(2024, 1, 1) + timedelta(days=int(current_time * 365))
            option_chain = generator.generate_single_chain(
                reference_date=reference_date,
                spot_price=S,
                vol_profile=vol_profile,
            )

            # Extract options and convert to frontend format
            options = []
            for opt in option_chain.options:
                maturity_days = (opt.expiry - reference_date).days
                moneyness = opt.strike / S

                options.append({
                    'strike': float(opt.strike),
                    'maturity_days': maturity_days,
                    'option_type': opt.option_type,
                    'price': float(opt.mid),
                    'moneyness': float(moneyness),
                    'implied_volatility': float(opt.implied_volatility),
                })

        else:
            # For non-Heston models (GBM, OU, etc.), fall back to Black-Scholes pricing
            options = []
            for maturity_days in maturity_days_grid:
                tau = maturity_days / 365.0

                # Skip if maturity is in the past
                if tau < current_time:
                    continue

                # Adjust remaining time to maturity
                tau_remaining = tau - current_time
                if tau_remaining <= 0:
                    continue

                moneyness_list = moneyness_by_maturity.get(maturity_days, [0.90, 0.95, 1.0, 1.05, 1.10])
                for moneyness in moneyness_list:
                    strike = S * moneyness

                    # Price call and put
                    for option_type in ['call', 'put']:
                        price = BacktestingService._price_option_model(
                            model=model,
                            S=S,
                            K=strike,
                            tau=tau_remaining,
                            params=parameters,
                            r=risk_free_rate,
                            option_type=option_type,
                            variance_t=variance_t,
                            pricer=pricer,
                            heston_pricer=heston_pricer,
                        )

                        # Compute IV (for visualization, assume it equals current_vol)
                        implied_vol = current_vol

                        options.append({
                            'strike': float(strike),
                            'maturity_days': maturity_days,
                            'option_type': option_type,
                            'price': float(price),
                            'moneyness': float(moneyness),
                            'implied_volatility': float(implied_vol),
                        })

        return {
            'time_step': time_step,
            'time': float(current_time),
            'spot': float(S),
            'volatility': float(current_vol),
            'options': options,
        }

    @staticmethod
    def _compute_iv_surface(
        option_chains: List[Dict[str, Any]],
    ) -> Dict[str, List[float]]:
        """
        Compute IV surface data from option chains for 3D visualization.

        Aggregates all options across timesteps into flat lists for Plotly 3D scatter.

        Args:
            option_chains: List of option chain dicts

        Returns:
            Dict with flattened arrays for 3D plotting:
                - moneyness: List of moneyness values
                - ttm: List of time-to-maturity in days
                - iv: List of implied volatilities
                - option_type: List of 'call' or 'put'
                - time_step: List of time step indices
        """
        moneyness_list = []
        ttm_list = []
        iv_list = []
        option_type_list = []
        time_step_list = []
        spot_list = []

        # Sample timesteps to avoid too much data (every 5th step or last step)
        sample_indices = list(range(0, len(option_chains), max(1, len(option_chains) // 10)))
        if len(option_chains) - 1 not in sample_indices:
            sample_indices.append(len(option_chains) - 1)

        for idx in sample_indices:
            chain = option_chains[idx]
            for opt in chain['options']:
                moneyness_list.append(opt['moneyness'])
                ttm_list.append(opt['maturity_days'])
                iv_list.append(opt['implied_volatility'] * 100)  # Convert to %
                option_type_list.append(opt['option_type'])
                time_step_list.append(chain['time_step'])
                spot_list.append(chain['spot'])

        return {
            'moneyness': moneyness_list,
            'ttm': ttm_list,
            'iv': iv_list,
            'option_type': option_type_list,
            'time_step': time_step_list,
            'spot': spot_list,
        }
