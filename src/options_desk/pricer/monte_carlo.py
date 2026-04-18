"""
Monte Carlo pricing engine for derivatives

author: Yunian Pan
email: yp1170@nyu.edu
"""
import copy
import numpy as np
import time
from typing import Optional, Union

from .base import Pricer, PricingResult
from options_desk.derivatives.base import PathIndependentDerivative, PathDependentDerivative


class MonteCarloPricer(Pricer):
    """
    Monte Carlo pricing engine for any derivative/process combination

    Features:
    - Works with any stochastic process (GBM, Heston, jump-diffusion, etc.)
    - Supports path-independent and path-dependent derivatives
    - Variance reduction via antithetic variates (built into SimulationConfig)
    - Greeks estimation via finite differences
    - Convergence diagnostics
    """

    def __init__(self, risk_free_rate: float = 0.0, dividend_yield: float = 0.0):
        """
        Initialize Monte Carlo pricer

        Args:
            risk_free_rate: Risk-free interest rate for discounting
            dividend_yield: Continuous dividend yield (for equity options)
        """
        super().__init__(name="MonteCarlo")
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield

    def price(
        self,
        derivative,
        process,
        X0: Union[float, np.ndarray],
        config: Optional[object] = None,
        scheme: str = "euler",
        compute_greeks: bool = False,
        greek_bump: float = 0.01,
    ) -> PricingResult:
        """
        Price derivative via Monte Carlo simulation

        Args:
            derivative: Derivative contract (from derivatives module)
            process: Stochastic process (from processes module)
            X0: Initial value(s) - float for single asset, array for multiple
            config: SimulationConfig object (if None, uses default)
            scheme: Simulation scheme ('euler', 'milstein', 'exact')
            compute_greeks: Whether to compute Greeks via finite differences
            greek_bump: Bump size for finite difference (relative, e.g., 0.01 = 1%)

        Returns:
            PricingResult with price, std error, and optionally Greeks
        """
        start_time = time.time()

        # Convert X0 to array if needed
        X0_array = np.atleast_1d(np.array(X0, dtype=float))

        # Use default config if not provided
        if config is None:
            from options_desk.processes.base import SimulationConfig
            config = SimulationConfig(n_paths=10000, n_steps=252)

        # If the process exposes a drift parameter, align it to the risk-neutral drift (r - q)
        # Use a shallow copy so the caller's process object is never mutated.
        sim_process = process
        if hasattr(process, "mu"):
            sim_process = copy.copy(process)
            target_mu = self.risk_free_rate - self.dividend_yield
            sim_process.mu = (
                float(target_mu)
                if np.ndim(process.mu) == 0
                else np.full_like(process.mu, target_mu, dtype=float)
            )

        # Simulate process
        t_grid, paths = sim_process.simulate(X0_array, derivative.maturity, config, scheme)

        # Calculate payoffs
        if isinstance(derivative, PathIndependentDerivative):
            # Path-independent: only need terminal values
            S_T = paths[-1, :, :]  # Shape: (n_paths, dim)

            # Flatten if single asset
            if S_T.shape[1] == 1:
                S_T = S_T[:, 0]

            payoffs = derivative.payoff(S_T)

        elif isinstance(derivative, PathDependentDerivative):
            # Path-dependent: need full path
            payoffs = derivative.payoff(paths)

        else:
            raise ValueError(f"Unknown derivative type: {type(derivative)}")

        # Discount payoffs
        discount_factor = np.exp(-self.risk_free_rate * derivative.maturity)
        discounted_payoffs = discount_factor * payoffs

        # Calculate price and statistics
        price = discounted_payoffs.mean()
        std_error = discounted_payoffs.std() / np.sqrt(len(discounted_payoffs))

        # 95% confidence interval
        z_score = 1.96
        ci_lower = price - z_score * std_error
        ci_upper = price + z_score * std_error

        computation_time = time.time() - start_time

        # Compute Greeks if requested
        greeks = None
        if compute_greeks:
            greeks = self._compute_greeks(
                derivative, process, X0_array, config, scheme, greek_bump, price
            )

        # Metadata
        metadata = {
            "scheme": scheme,
            "n_steps": config.n_steps,
            "antithetic": config.antithetic,
            "contract_type": derivative.contract_type,
            "process": process.name,
        }

        return PricingResult(
            price=price,
            std_error=std_error,
            confidence_interval=(ci_lower, ci_upper),
            n_paths=config.n_paths,
            computation_time=computation_time,
            greeks=greeks,
            metadata=metadata,
        )

    def _compute_greeks(
        self,
        derivative,
        process,
        X0: np.ndarray,
        config,
        scheme: str,
        bump: float,
        base_price: float,
    ) -> dict:
        """
        Compute Greeks via finite differences

        Args:
            derivative: Derivative contract
            process: Stochastic process
            X0: Initial value
            config: Simulation config
            scheme: Simulation scheme
            bump: Relative bump size
            base_price: Already computed base price

        Returns:
            Dictionary of Greeks
        """
        greeks = {}

        # Delta: dV/dS (first derivative w.r.t. spot)
        S0 = X0[0]
        X0_up = X0.copy()
        X0_up[0] = S0 * (1 + bump)

        # Price with bumped spot
        result_up = self.price(derivative, process, X0_up, config, scheme, compute_greeks=False)
        greeks['delta'] = (result_up.price - base_price) / (S0 * bump)

        # Gamma: d²V/dS² (second derivative w.r.t. spot)
        X0_down = X0.copy()
        X0_down[0] = S0 * (1 - bump)
        result_down = self.price(derivative, process, X0_down, config, scheme, compute_greeks=False)

        greeks['gamma'] = (result_up.price - 2 * base_price + result_down.price) / ((S0 * bump) ** 2)

        # Vega: dV/dσ (derivative w.r.t. volatility) - only for processes with sigma
        if hasattr(process, 'sigma'):
            bumped_process = copy.copy(process)
            bumped_process.sigma = process.sigma * (1 + bump)
            result_vega = self.price(derivative, bumped_process, X0, config, scheme, compute_greeks=False)
            greeks['vega'] = (result_vega.price - base_price) / (process.sigma * bump)

        # Theta: -dV/dt (derivative w.r.t. time to maturity)
        maturity_original = derivative.maturity
        if maturity_original > 0:
            dt = min(1.0 / 365.0, maturity_original * 0.5)
            bumped_derivative = copy.copy(derivative)
            bumped_derivative.maturity = max(maturity_original - dt, 0.0)
            result_theta = self.price(bumped_derivative, process, X0, config, scheme, compute_greeks=False)
            greeks['theta'] = -(result_theta.price - base_price) / dt
        else:
            greeks['theta'] = 0.0

        # Rho: dV/dr (derivative w.r.t. risk-free rate)
        bumped_pricer = copy.copy(self)
        bumped_pricer.risk_free_rate = self.risk_free_rate + 0.01
        result_rho = bumped_pricer.price(derivative, process, X0, config, scheme, compute_greeks=False)
        greeks['rho'] = (result_rho.price - base_price) / 0.01

        return greeks

    def convergence_analysis(
        self,
        derivative,
        process,
        X0: Union[float, np.ndarray],
        path_counts: list = None,
        scheme: str = "euler",
        n_steps: int = 252,
    ) -> dict:
        """
        Analyze convergence of Monte Carlo estimator

        Args:
            derivative: Derivative contract
            process: Stochastic process
            X0: Initial value
            path_counts: List of path counts to test (default: [1k, 5k, 10k, 50k, 100k])
            scheme: Simulation scheme
            n_steps: Number of time steps

        Returns:
            Dictionary with convergence statistics
        """
        from options_desk.processes.base import SimulationConfig

        if path_counts is None:
            path_counts = [1000, 5000, 10000, 50000, 100000]

        results = []
        for n_paths in path_counts:
            config = SimulationConfig(n_paths=n_paths, n_steps=n_steps)
            result = self.price(derivative, process, X0, config, scheme)
            results.append({
                'n_paths': n_paths,
                'price': result.price,
                'std_error': result.std_error,
                'time': result.computation_time,
            })

        return {
            'path_counts': path_counts,
            'results': results,
        }

    def control_variate_price(
        self,
        derivative,
        process,
        X0: Union[float, np.ndarray],
        control_derivative,
        control_price: float,
        config: Optional[object] = None,
        scheme: str = "euler",
    ) -> PricingResult:
        """
        Price with control variate variance reduction

        Args:
            derivative: Target derivative to price
            process: Stochastic process
            X0: Initial value
            control_derivative: Control derivative with known price
            control_price: Known analytical price of control derivative
            config: Simulation config
            scheme: Simulation scheme

        Returns:
            PricingResult with reduced variance
        """
        start_time = time.time()

        X0_array = np.atleast_1d(np.array(X0, dtype=float))

        if config is None:
            from options_desk.processes.base import SimulationConfig
            config = SimulationConfig(n_paths=10000, n_steps=252)

        # Simulate process once for both derivatives
        t_grid, paths = process.simulate(X0_array, derivative.maturity, config, scheme)

        # Calculate payoffs for target derivative
        if isinstance(derivative, PathIndependentDerivative):
            S_T = paths[-1, :, 0] if paths.shape[2] == 1 else paths[-1, :, :]
            target_payoffs = derivative.payoff(S_T)
        else:
            target_payoffs = derivative.payoff(paths)

        # Calculate payoffs for control derivative
        if isinstance(control_derivative, PathIndependentDerivative):
            S_T = paths[-1, :, 0] if paths.shape[2] == 1 else paths[-1, :, :]
            control_payoffs = control_derivative.payoff(S_T)
        else:
            control_payoffs = control_derivative.payoff(paths)

        # Discount
        discount_factor = np.exp(-self.risk_free_rate * derivative.maturity)
        target_pv = discount_factor * target_payoffs
        control_pv = discount_factor * control_payoffs

        # Control variate adjustment
        # V_cv = V_target - β * (V_control - E[V_control])
        # Optimal β = Cov(V_target, V_control) / Var(V_control)
        cov = np.cov(target_pv, control_pv)[0, 1]
        var_control = np.var(control_pv)
        beta = cov / var_control if var_control > 0 else 0

        adjusted_payoffs = target_pv - beta * (control_pv - control_price)

        # Statistics
        price = adjusted_payoffs.mean()
        std_error = adjusted_payoffs.std() / np.sqrt(len(adjusted_payoffs))

        z_score = 1.96
        ci_lower = price - z_score * std_error
        ci_upper = price + z_score * std_error

        computation_time = time.time() - start_time

        # Variance reduction ratio
        variance_reduction = np.var(target_pv) / np.var(adjusted_payoffs) if np.var(adjusted_payoffs) > 0 else 1.0

        metadata = {
            "scheme": scheme,
            "control_variate": control_derivative.contract_type,
            "beta": beta,
            "variance_reduction": variance_reduction,
        }

        return PricingResult(
            price=price,
            std_error=std_error,
            confidence_interval=(ci_lower, ci_upper),
            n_paths=config.n_paths,
            computation_time=computation_time,
            metadata=metadata,
        )
