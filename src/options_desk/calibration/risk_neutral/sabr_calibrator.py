"""
SABR model calibration using Hagan's approximation formula.

Calibrate SABR stochastic volatility model to implied volatility surface.

author: Yunian Pan
email: yp1170@nyu.edu
"""

import numpy as np
from scipy import optimize
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import time

from ..data.data_provider import OptionChain


@dataclass
class SABRCalibrationResult:
    """Result of SABR calibration."""

    # Calibrated parameters
    alpha: float      # Vol of vol
    beta: float       # CEV exponent
    rho: float        # Correlation
    nu: float         # Initial volatility (ATM vol at T=0)

    # Fit quality
    rmse_iv: float              # RMSE in IV terms
    max_error_iv: float         # Maximum absolute IV error
    mean_error_iv: float        # Mean IV error

    # Optimization info
    n_iterations: int
    computation_time: float
    success: bool
    message: str

    @property
    def params(self) -> Dict[str, float]:
        """Return parameters as dictionary."""
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'rho': self.rho,
            'nu': self.nu,
        }

    def __repr__(self) -> str:
        lines = [
            "SABRCalibrationResult:",
            f"  α (alpha) = {self.alpha:.4f}  (vol of vol)",
            f"  β (beta)  = {self.beta:.4f}  (CEV exponent)",
            f"  ρ (rho)   = {self.rho:.4f}  (correlation)",
            f"  ν (nu)    = {self.nu:.4f}  (ATM vol)",
            f"",
            f"  RMSE (IV): {self.rmse_iv:.6f}",
            f"  Max Error: {self.max_error_iv:.6f}",
            f"  Time: {self.computation_time:.2f}s",
        ]
        return "\n".join(lines)


class SABRCalibrator:
    """
    Calibrate SABR model to implied volatility surface.

    Uses Hagan's analytical approximation formula for fast calibration.

    The SABR model:
        dF_t = σ_t F_t^β dW_t^F
        dσ_t = α σ_t dW_t^σ
        E[dW^F dW^σ] = ρ dt

    Parameters:
        - α (alpha): volatility of volatility
        - β (beta): CEV exponent (0=Normal, 0.5=popular, 1=Lognormal)
        - ρ (rho): correlation between forward and vol
        - ν (nu): ATM volatility

    Reference:
        Hagan et al. (2002) "Managing Smile Risk"

    Example:
        calibrator = SABRCalibrator(beta=0.5)  # Fix beta
        result = calibrator.calibrate(option_chain, forward=100.0)
        print(result)
    """

    # Parameter bounds
    DEFAULT_BOUNDS = {
        'alpha': (0.001, 5.0),
        'beta': (0.0, 1.0),
        'rho': (-0.999, 0.999),
        'nu': (0.001, 2.0),
    }

    def __init__(
        self,
        beta: Optional[float] = None,
        weighting: str = 'vega',
    ):
        """
        Initialize SABR calibrator.

        Args:
            beta: Fixed beta value (None to calibrate). Common choices: 0, 0.5, 1.0
            weighting: Weighting scheme ('uniform', 'vega', 'oi')
        """
        self.fixed_beta = beta
        self.weighting = weighting

    def calibrate(
        self,
        chain: OptionChain,
        forward: Optional[float] = None,
        maturity: Optional[float] = None,
        initial_guess: Optional[Dict[str, float]] = None,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        method: str = 'L-BFGS-B',
    ) -> SABRCalibrationResult:
        """
        Calibrate SABR to option chain.

        Args:
            chain: OptionChain with market data
            forward: Forward price (defaults to spot * exp(r*T))
            maturity: Option maturity in years (uses single expiry from chain if not provided)
            initial_guess: Initial parameter values
            bounds: Parameter bounds
            method: Optimization method ('L-BFGS-B', 'SLSQP', 'differential_evolution')

        Returns:
            SABRCalibrationResult with calibrated parameters
        """
        start_time = time.time()

        # Determine maturity (use single expiry for SABR)
        if maturity is None:
            # Get the most common or nearest expiry
            expiries = [(opt.expiry - chain.reference_date).days / 365.0 for opt in chain.options]
            maturity = np.median(expiries) if expiries else 0.25

        # Determine forward price
        if forward is None:
            forward = chain.spot_price * np.exp(chain.risk_free_rate * maturity)

        # Prepare calibration targets
        targets = self._prepare_targets(chain, forward, maturity)

        if len(targets['strikes']) == 0:
            raise ValueError("No valid options for calibration")

        # Set up bounds
        param_bounds = bounds or self.DEFAULT_BOUNDS.copy()

        # Build parameter vector and bounds based on whether beta is fixed
        if self.fixed_beta is not None:
            # Calibrate (alpha, rho, nu) with fixed beta
            bounds_array = [
                param_bounds['alpha'],
                param_bounds['rho'],
                param_bounds['nu'],
            ]

            # Initial guess
            if initial_guess is None:
                atm_iv = np.median(targets['ivs'])
                initial_guess = {
                    'alpha': 0.3,
                    'rho': -0.3,
                    'nu': atm_iv,
                }

            x0 = [
                initial_guess['alpha'],
                initial_guess['rho'],
                initial_guess['nu'],
            ]

            # Objective function
            def objective(params):
                alpha, rho, nu = params
                return self._objective(alpha, self.fixed_beta, rho, nu, targets, forward, maturity)
        else:
            # Calibrate all 4 parameters
            bounds_array = [
                param_bounds['alpha'],
                param_bounds['beta'],
                param_bounds['rho'],
                param_bounds['nu'],
            ]

            # Initial guess
            if initial_guess is None:
                atm_iv = np.median(targets['ivs'])
                initial_guess = {
                    'alpha': 0.3,
                    'beta': 0.5,
                    'rho': -0.3,
                    'nu': atm_iv,
                }

            x0 = [
                initial_guess['alpha'],
                initial_guess['beta'],
                initial_guess['rho'],
                initial_guess['nu'],
            ]

            # Objective function
            def objective(params):
                alpha, beta, rho, nu = params
                return self._objective(alpha, beta, rho, nu, targets, forward, maturity)

        # Optimize
        if method == 'differential_evolution':
            result = optimize.differential_evolution(
                objective,
                bounds=bounds_array,
                maxiter=500,
                seed=42,
                polish=True,
            )
        else:
            result = optimize.minimize(
                objective,
                x0=x0,
                method=method,
                bounds=bounds_array,
                options={'maxiter': 1000, 'ftol': 1e-8},
            )

        # Extract parameters
        if self.fixed_beta is not None:
            alpha, rho, nu = result.x
            beta = self.fixed_beta
        else:
            alpha, beta, rho, nu = result.x

        # Calculate fit metrics
        model_ivs = self._compute_ivs(alpha, beta, rho, nu, targets, forward, maturity)
        errors = model_ivs - targets['ivs']
        rmse_iv = np.sqrt(np.mean(errors**2))
        max_error_iv = np.max(np.abs(errors))
        mean_error_iv = np.mean(errors)

        computation_time = time.time() - start_time

        return SABRCalibrationResult(
            alpha=alpha,
            beta=beta,
            rho=rho,
            nu=nu,
            rmse_iv=rmse_iv,
            max_error_iv=max_error_iv,
            mean_error_iv=mean_error_iv,
            n_iterations=result.nit if hasattr(result, 'nit') else -1,
            computation_time=computation_time,
            success=result.success if hasattr(result, 'success') else True,
            message=result.message if hasattr(result, 'message') else "",
        )

    def _prepare_targets(
        self,
        chain: OptionChain,
        forward: float,
        target_maturity: float,
    ) -> Dict:
        """Extract calibration targets from option chain."""
        strikes = []
        ivs = []
        weights = []

        for opt in chain.options:
            # Skip if no valid IV
            if opt.implied_volatility is None or opt.implied_volatility <= 0:
                continue

            # Calculate maturity
            T = (opt.expiry - chain.reference_date).days / 365.0

            # Only use options near target maturity (within 7 days)
            if abs(T - target_maturity) > 7.0 / 365.0:
                continue

            strikes.append(opt.strike)
            ivs.append(opt.implied_volatility)

            # Compute weight
            if self.weighting == 'uniform':
                w = 1.0
            elif self.weighting == 'vega':
                # Approximate vega weight (sqrt(T) * ATM-ness)
                moneyness = opt.strike / forward
                w = np.sqrt(T) * np.exp(-0.5 * (np.log(moneyness))**2)
            elif self.weighting == 'oi':
                w = np.sqrt(opt.open_interest + 1)
            else:
                w = 1.0
            weights.append(w)

        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights

        return {
            'strikes': np.array(strikes),
            'ivs': np.array(ivs),
            'weights': weights,
        }

    def _objective(
        self,
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
        targets: Dict,
        forward: float,
        maturity: float,
    ) -> float:
        """Compute calibration objective (weighted sum of squared IV errors)."""
        # Compute model IVs
        model_ivs = self._compute_ivs(alpha, beta, rho, nu, targets, forward, maturity)

        # Weighted squared errors
        errors = (model_ivs - targets['ivs'])**2
        weighted_errors = np.sum(targets['weights'] * errors)

        return weighted_errors

    def _compute_ivs(
        self,
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
        targets: Dict,
        forward: float,
        maturity: float,
    ) -> np.ndarray:
        """Compute implied volatilities using Hagan's formula."""
        ivs = np.zeros(len(targets['strikes']))

        for i, K in enumerate(targets['strikes']):
            ivs[i] = self._sabr_hagan_iv(forward, K, maturity, alpha, beta, rho, nu)

        return ivs

    def _sabr_hagan_iv(
        self,
        F: float,
        K: float,
        T: float,
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
    ) -> float:
        """
        Hagan's SABR implied volatility approximation formula.

        Reference: Hagan et al. (2002), Equation 2.17a
        """
        # Handle ATM case separately
        eps = 1e-7
        if abs(F - K) < eps:
            # ATM formula (Equation 2.17b from Hagan)
            FK_mid = (F + K) / 2.0
            FK_beta = FK_mid ** (1 - beta)

            term1 = alpha / (FK_beta * (1 + ((1 - beta)**2 / 24) * (np.log(FK_mid))**2))

            term2 = 1 + T * (
                ((1 - beta)**2 / 24) * (alpha**2 / FK_beta**2) +
                (rho * beta * nu * alpha) / (4 * FK_beta) +
                ((2 - 3 * rho**2) / 24) * nu**2
            )

            return term1 * term2

        # Non-ATM formula
        # Log-moneyness
        log_FK = np.log(F / K)

        # FK_mid term
        FK_mid = (F * K) ** ((1 - beta) / 2)

        # z parameter
        z = (nu / alpha) * FK_mid * log_FK

        # x(z) function - handle small z
        if abs(z) < eps:
            x_z = 1.0
        else:
            # x(z) = log((sqrt(1 - 2*rho*z + z^2) + z - rho) / (1 - rho))
            sqrt_term = np.sqrt(1 - 2 * rho * z + z**2)
            x_z = np.log((sqrt_term + z - rho) / (1 - rho))

        # First term: α / (FK_mid * denominator)
        denominator1 = 1 + ((1 - beta)**2 / 24) * log_FK**2 + ((1 - beta)**4 / 1920) * log_FK**4
        term1 = alpha / (FK_mid * denominator1)

        # Second term: z / x(z)
        if abs(x_z) < eps:
            term2 = 1.0
        else:
            term2 = z / x_z

        # Third term: correction for time
        FK_sum_beta = ((F * K) ** ((1 - beta) / 2)) ** 2  # = (FK)^(1-beta)

        term3 = 1 + T * (
            ((1 - beta)**2 / 24) * (alpha**2 / FK_sum_beta) +
            (rho * beta * nu * alpha) / (4 * FK_mid**2) +
            ((2 - 3 * rho**2) / 24) * nu**2
        )

        iv = term1 * term2 * term3

        # Ensure positive IV
        return max(iv, 1e-6)
