"""
Heston model calibration.

Calibrate Heston stochastic volatility model to option prices.

author: Yunian Pan
email: yp1170@nyu.edu
"""

import numpy as np
from scipy import optimize
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Callable
import time

from ..data.data_provider import OptionChain
from ...pricer import COSPricer
from ...processes import Heston
from ...derivatives import EuropeanCall, EuropeanPut


@dataclass
class CalibrationResult:
    """Result of model calibration."""
    
    # Calibrated parameters
    kappa: float      # Mean reversion speed
    theta: float      # Long-term variance
    xi: float         # Vol of vol
    rho: float        # Correlation
    v0: float         # Initial variance
    
    # Fit quality
    rmse: float                    # Root mean squared error
    rmse_iv: Optional[float]       # RMSE in IV terms
    max_error: float               # Maximum absolute error
    
    # Optimization info
    n_iterations: int
    computation_time: float
    success: bool
    message: str
    
    # Additional diagnostics
    feller_satisfied: bool         # 2*kappa*theta > xi^2
    
    @property
    def params(self) -> Dict[str, float]:
        """Return parameters as dictionary."""
        return {
            'kappa': self.kappa,
            'theta': self.theta,
            'xi': self.xi,
            'rho': self.rho,
            'v0': self.v0,
        }
    
    def __repr__(self) -> str:
        lines = [
            "HestonCalibrationResult:",
            f"  κ (kappa) = {self.kappa:.4f}  (mean reversion)",
            f"  θ (theta) = {self.theta:.4f}  (long-term var, σ∞ ≈ {np.sqrt(self.theta)*100:.1f}%)",
            f"  ξ (xi)    = {self.xi:.4f}  (vol of vol)",
            f"  ρ (rho)   = {self.rho:.4f}  (correlation)",
            f"  v₀        = {self.v0:.4f}  (initial var, σ₀ ≈ {np.sqrt(self.v0)*100:.1f}%)",
            f"",
            f"  Feller: {'✓' if self.feller_satisfied else '✗'} (2κθ {'>' if self.feller_satisfied else '<'} ξ²)",
            f"  RMSE: {self.rmse:.6f}",
            f"  Time: {self.computation_time:.2f}s",
        ]
        return "\n".join(lines)


class HestonCalibrator:
    """
    Calibrate Heston model to option prices.
    
    Uses characteristic function-based pricing (COS method) for speed.
    
    Example:
        calibrator = HestonCalibrator(pricer)
        result = calibrator.calibrate(option_chain, spot, rate)
        print(result)
    """
    
    # Parameter bounds: (lower, upper)
    DEFAULT_BOUNDS = {
        'kappa': (0.1, 10.0),
        'theta': (0.001, 1.0),
        'xi': (0.01, 2.0),
        'rho': (-0.99, 0.99),
        'v0': (0.001, 1.0),
    }
    
    def __init__(
        self,
        pricer=None,
        weighting: str = 'vega',
        feller_penalty: float = 100.0,
        use_cos: bool = True,
    ):
        """
        Initialize calibrator.

        Args:
            pricer: Pricer object with price() method (e.g., COSPricer)
                   If None, creates default COSPricer (fast) or uses Simpson integration
            weighting: Weighting scheme for errors:
                      'uniform' - equal weights
                      'vega' - weight by option vega
                      'oi' - weight by open interest
            feller_penalty: Penalty for violating Feller condition
            use_cos: If True and pricer is None, use COSPricer (fast). Otherwise use Simpson.
        """
        self.use_cos = use_cos
        if pricer is None and use_cos:
            # Create default COSPricer with good parameters
            self.pricer = COSPricer(N=256, L=10.0)
        else:
            self.pricer = pricer
        self.weighting = weighting
        self.feller_penalty = feller_penalty
    
    def calibrate(
        self,
        chain: OptionChain,
        spot: Optional[float] = None,
        rate: Optional[float] = None,
        initial_guess: Optional[Dict[str, float]] = None,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        method: str = 'differential_evolution',
        maxiter: int = 1000,
        tol: float = 1e-6,
    ) -> 'CalibrationResult':
        """
        Calibrate Heston model to option chain.
        
        Args:
            chain: OptionChain with market data
            spot: Spot price (defaults to chain.spot_price)
            rate: Risk-free rate (defaults to chain.risk_free_rate)
            initial_guess: Initial parameter values
            bounds: Parameter bounds
            method: Optimization method ('differential_evolution', 'L-BFGS-B', 'SLSQP')
            maxiter: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            CalibrationResult with calibrated parameters
        """
        start_time = time.time()
        
        # Extract market data
        S0 = spot or chain.spot_price
        r = rate or chain.risk_free_rate
        
        # Prepare calibration targets
        targets = self._prepare_targets(chain, S0)
        
        if len(targets['strikes']) == 0:
            raise ValueError("No valid options for calibration")
        
        # Set up bounds
        param_bounds = bounds or self.DEFAULT_BOUNDS
        bounds_array = [
            param_bounds['kappa'],
            param_bounds['theta'],
            param_bounds['xi'],
            param_bounds['rho'],
            param_bounds['v0'],
        ]
        
        # Initial guess
        if initial_guess is None:
            # Use ATM vol to estimate v0 and theta
            atm_idx = np.argmin(np.abs(targets['moneyness'] - 1.0))
            atm_iv = targets['ivs'][atm_idx] if targets['ivs'][atm_idx] else 0.2
            
            initial_guess = {
                'kappa': 2.0,
                'theta': atm_iv**2,
                'xi': 0.5,
                'rho': -0.7,
                'v0': atm_iv**2,
            }
        
        x0 = [
            initial_guess['kappa'],
            initial_guess['theta'],
            initial_guess['xi'],
            initial_guess['rho'],
            initial_guess['v0'],
        ]
        
        # Define objective function
        def objective(params):
            return self._objective(params, targets, S0, r)
        
        # Optimize
        if method == 'differential_evolution':
            result = optimize.differential_evolution(
                objective,
                bounds=bounds_array,
                maxiter=maxiter,
                tol=tol,
                seed=42,
                polish=True,
            )
        else:
            result = optimize.minimize(
                objective,
                x0=x0,
                method=method,
                bounds=bounds_array,
                options={'maxiter': maxiter, 'ftol': tol},
            )
        
        # Extract results
        kappa, theta, xi, rho, v0 = result.x
        
        # Calculate fit metrics
        model_prices = self._compute_prices(result.x, targets, S0, r)
        errors = model_prices - targets['prices']
        rmse = np.sqrt(np.mean(errors**2))
        max_error = np.max(np.abs(errors))
        
        # RMSE in IV terms (approximate)
        rmse_iv = None  # TODO: implement proper IV conversion
        
        computation_time = time.time() - start_time
        
        return CalibrationResult(
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=rho,
            v0=v0,
            rmse=rmse,
            rmse_iv=rmse_iv,
            max_error=max_error,
            n_iterations=result.nit if hasattr(result, 'nit') else -1,
            computation_time=computation_time,
            success=result.success,
            message=result.message if hasattr(result, 'message') else "",
            feller_satisfied=(2 * kappa * theta > xi**2),
        )
    
    def _prepare_targets(
        self,
        chain: OptionChain,
        S0: float,
    ) -> Dict:
        """Extract calibration targets from option chain."""
        strikes = []
        maturities = []
        prices = []
        ivs = []
        is_calls = []
        weights = []
        
        for opt in chain.options:
            # Use mid price as target
            if opt.mid <= 0:
                continue
                
            T = (opt.expiry - chain.reference_date).days / 365.0
            if T <= 0:
                continue
            
            strikes.append(opt.strike)
            maturities.append(T)
            prices.append(opt.mid)
            ivs.append(opt.implied_volatility)
            is_calls.append(opt.is_call)
            
            # Compute weight
            if self.weighting == 'uniform':
                w = 1.0
            elif self.weighting == 'vega':
                # Approximate vega weight
                w = np.sqrt(T) * opt.mid
            elif self.weighting == 'oi':
                w = np.sqrt(opt.open_interest + 1)
            else:
                w = 1.0
            weights.append(w)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        return {
            'strikes': np.array(strikes),
            'maturities': np.array(maturities),
            'prices': np.array(prices),
            'ivs': np.array(ivs),
            'is_calls': np.array(is_calls),
            'weights': weights,
            'moneyness': np.array(strikes) / S0,
        }
    
    def _objective(
        self,
        params: np.ndarray,
        targets: Dict,
        S0: float,
        r: float,
    ) -> float:
        """Compute calibration objective (weighted sum of squared errors)."""
        kappa, theta, xi, rho, v0 = params
        
        # Feller condition penalty
        feller_violation = max(0, xi**2 - 2 * kappa * theta)
        penalty = self.feller_penalty * feller_violation
        
        # Compute model prices
        model_prices = self._compute_prices(params, targets, S0, r)
        
        # Weighted squared errors
        errors = (model_prices - targets['prices'])**2
        weighted_errors = np.sum(targets['weights'] * errors)
        
        return weighted_errors + penalty
    
    def _compute_prices(
        self,
        params: np.ndarray,
        targets: Dict,
        S0: float,
        r: float,
    ) -> np.ndarray:
        """
        Compute option prices under Heston model.

        Uses COSPricer if available (fast), otherwise falls back to Simpson integration.
        """
        kappa, theta, xi, rho, v0 = params

        prices = np.zeros(len(targets['strikes']))

        if self.pricer is not None:
            # Use fast COSPricer
            # Create Heston process with current parameters
            # Note: Heston process uses 'sigma_v' not 'xi', and needs 'mu' (drift = r for Q-measure)
            heston = Heston(
                mu=r,  # For Q-measure, drift = risk-free rate
                kappa=kappa,
                theta=theta,
                sigma_v=xi,  # Calibrator uses 'xi' but process uses 'sigma_v'
                rho=rho,
                v0=v0,
            )

            # Update pricer's risk-free rate
            self.pricer.risk_free_rate = r

            for i in range(len(targets['strikes'])):
                K = targets['strikes'][i]
                T = targets['maturities'][i]
                is_call = targets['is_calls'][i]

                # Create derivative
                if is_call:
                    option = EuropeanCall(strike=K, maturity=T)
                else:
                    option = EuropeanPut(strike=K, maturity=T)

                # Price using COSPricer
                try:
                    result = self.pricer.price(option, heston, X0=np.array([S0, v0]))
                    prices[i] = max(result.price, 0.0)
                except Exception as e:
                    # Fallback to intrinsic value if pricing fails
                    if is_call:
                        prices[i] = max(S0 - K * np.exp(-r * T), 0)
                    else:
                        prices[i] = max(K * np.exp(-r * T) - S0, 0)
        else:
            # Fallback to hand-coded Simpson integration
            for i in range(len(targets['strikes'])):
                K = targets['strikes'][i]
                T = targets['maturities'][i]
                is_call = targets['is_calls'][i]

                price = self._heston_price_simpson(S0, K, T, r, kappa, theta, xi, rho, v0, is_call)
                prices[i] = price

        return prices
    
    def _heston_price_simpson(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
        v0: float,
        is_call: bool,
    ) -> float:
        """
        Price European option under Heston using CF integration (Simpson's rule).

        Legacy fallback method - slower than COSPricer.
        """
        # Integration bounds
        N = 64
        u_max = 50
        
        # Simpson's rule integration
        du = u_max / N
        
        # P1 and P2 integrands
        integral1 = 0.0
        integral2 = 0.0
        
        for j in range(1, N + 1):
            u = j * du
            
            # Characteristic function
            phi1 = self._heston_cf(u - 1j, S0, K, T, r, kappa, theta, xi, rho, v0)
            phi2 = self._heston_cf(u, S0, K, T, r, kappa, theta, xi, rho, v0)
            
            # Integrands
            integrand1 = np.real(np.exp(-1j * u * np.log(K)) * phi1 / (1j * u * S0 * np.exp(r * T)))
            integrand2 = np.real(np.exp(-1j * u * np.log(K)) * phi2 / (1j * u))
            
            # Simpson weights
            if j == 1 or j == N:
                w = 1
            elif j % 2 == 0:
                w = 4
            else:
                w = 2
                
            integral1 += w * integrand1
            integral2 += w * integrand2
        
        integral1 *= du / 3
        integral2 *= du / 3
        
        # Probabilities
        P1 = 0.5 + integral1 / np.pi
        P2 = 0.5 + integral2 / np.pi
        
        # Call price
        call_price = S0 * P1 - K * np.exp(-r * T) * P2
        
        if is_call:
            return max(call_price, 0)
        else:
            # Put via put-call parity
            return max(call_price - S0 + K * np.exp(-r * T), 0)
    
    def _heston_cf(
        self,
        u: complex,
        S0: float,
        K: float,
        T: float,
        r: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
        v0: float,
    ) -> complex:
        """Heston characteristic function."""
        # Log-Heston CF
        x0 = np.log(S0)
        
        d = np.sqrt((rho * xi * 1j * u - kappa)**2 + xi**2 * (1j * u + u**2))
        g = (kappa - rho * xi * 1j * u - d) / (kappa - rho * xi * 1j * u + d)
        
        C = r * 1j * u * T + (kappa * theta / xi**2) * (
            (kappa - rho * xi * 1j * u - d) * T 
            - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))
        )
        
        D = ((kappa - rho * xi * 1j * u - d) / xi**2) * (
            (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
        )
        
        return np.exp(C + D * v0 + 1j * u * x0)
