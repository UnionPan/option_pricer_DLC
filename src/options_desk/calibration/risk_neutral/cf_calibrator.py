"""
Generic characteristic-function-based Q-measure calibrator.

Works with any process that implements characteristic_function(u, X0, T).
Uses COS method for fast option pricing during calibration.

Supports: MertonJD, KouJD, VarianceGamma, NIG, CEV (via CF), and any
future process with a characteristic function.

author: Yunian Pan
email: yp1170@nyu.edu
"""

import numpy as np
from scipy import optimize
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
import time

from ..data.data_provider import OptionChain
from ...pricer.fourier import COSPricer
from ...derivatives.vanilla import EuropeanCall, EuropeanPut


@dataclass
class CFCalibrationResult:
    """Result of characteristic-function-based calibration."""
    model_name: str
    params: Dict[str, float]
    rmse_price: float
    rmse_iv: float
    max_error_iv: float
    n_options: int
    n_iterations: int
    computation_time: float
    success: bool
    message: str

    def __repr__(self) -> str:
        lines = [
            f"{self.model_name} Calibration Result:",
            "  Parameters:",
        ]
        for k, v in self.params.items():
            lines.append(f"    {k:12s} = {v:.6f}")
        lines.extend([
            f"  RMSE (price): {self.rmse_price:.6f}",
            f"  RMSE (IV):    {self.rmse_iv * 100:.2f}%",
            f"  Max err (IV): {self.max_error_iv * 100:.2f}%",
            f"  n_options:    {self.n_options}",
            f"  Time:         {self.computation_time:.2f}s",
            f"  Success:      {self.success}",
        ])
        return "\n".join(lines)


class CFCalibrator:
    """
    Generic calibrator for any process with a characteristic function.

    Usage:
        # Define how to build a process from a parameter vector
        def build_merton(params_vec):
            sigma, lam, mu_J, sigma_J = params_vec
            return MertonJD(mu=0.0, sigma=sigma, lambda_jump=lam,
                           mu_J=mu_J, sigma_J=sigma_J)

        calibrator = CFCalibrator(
            model_name='MertonJD',
            build_process_fn=build_merton,
            param_names=['sigma', 'lambda', 'mu_J', 'sigma_J'],
            param_bounds=[(0.01, 1.0), (0.01, 10), (-0.5, 0.5), (0.01, 1.0)],
        )
        result = calibrator.calibrate(chain, spot=100)
    """

    def __init__(
        self,
        model_name: str,
        build_process_fn: Callable,
        param_names: List[str],
        param_bounds: List[Tuple[float, float]],
        cos_N: int = 256,
        cos_L: float = 12.0,
    ):
        """
        Args:
            model_name: Name for display
            build_process_fn: Callable(param_vector) -> process with characteristic_function
            param_names: List of parameter names (for results)
            param_bounds: List of (lower, upper) bounds per parameter
            cos_N: Number of COS terms
            cos_L: COS truncation range
        """
        self.model_name = model_name
        self.build_process_fn = build_process_fn
        self.param_names = param_names
        self.param_bounds = param_bounds
        self.cos_N = cos_N
        self.cos_L = cos_L

    def calibrate(
        self,
        chain: OptionChain,
        spot: Optional[float] = None,
        rate: Optional[float] = None,
        initial_guess: Optional[np.ndarray] = None,
        method: str = 'differential_evolution',
        maxiter: int = 500,
    ) -> CFCalibrationResult:
        """
        Calibrate model to option chain.

        Args:
            chain: OptionChain with market prices/IVs
            spot: Spot price (default: chain.spot_price)
            rate: Risk-free rate (default: chain.risk_free_rate)
            initial_guess: Initial parameter vector
            method: 'differential_evolution' or 'L-BFGS-B'
            maxiter: Max iterations

        Returns:
            CFCalibrationResult
        """
        start_time = time.time()

        S0 = spot or chain.spot_price
        r = rate if rate is not None else chain.risk_free_rate

        # Extract calibration targets
        strikes, maturities, market_prices, option_types, market_ivs = \
            self._extract_targets(chain, S0)

        if len(strikes) == 0:
            raise ValueError("No valid options for calibration")

        pricer = COSPricer(risk_free_rate=r, N=self.cos_N, L=self.cos_L)

        def objective(params_vec):
            try:
                process = self.build_process_fn(params_vec)
            except (ValueError, RuntimeError):
                return 1e6

            total_error = 0.0
            for i in range(len(strikes)):
                K = strikes[i]
                T = maturities[i]
                is_call = option_types[i] == 'call'

                if is_call:
                    deriv = EuropeanCall(strike=K, maturity=T)
                else:
                    deriv = EuropeanPut(strike=K, maturity=T)

                try:
                    X0 = self._get_X0(process, S0)
                    result = pricer.price(deriv, process, X0)
                    model_price = result.price
                except Exception:
                    model_price = 0.0

                diff = model_price - market_prices[i]
                total_error += diff ** 2

            return total_error / len(strikes)

        # Optimize
        if method == 'differential_evolution':
            result = optimize.differential_evolution(
                objective, self.param_bounds,
                maxiter=maxiter, tol=1e-8, seed=42,
                polish=True, workers=1,
            )
            best_params = result.x
            n_iter = result.nit
            success = result.success
            message = result.message
        else:
            if initial_guess is None:
                initial_guess = np.array([
                    (lo + hi) / 2 for lo, hi in self.param_bounds
                ])
            result = optimize.minimize(
                objective, initial_guess,
                bounds=self.param_bounds, method=method,
                options={'maxiter': maxiter},
            )
            best_params = result.x
            n_iter = result.nit if hasattr(result, 'nit') else 0
            success = result.success
            message = result.message

        # Compute final fit quality
        process = self.build_process_fn(best_params)
        model_prices = []
        for i in range(len(strikes)):
            K, T = strikes[i], maturities[i]
            is_call = option_types[i] == 'call'
            deriv = EuropeanCall(strike=K, maturity=T) if is_call else EuropeanPut(strike=K, maturity=T)
            try:
                X0 = self._get_X0(process, S0)
                model_prices.append(pricer.price(deriv, process, X0).price)
            except Exception:
                model_prices.append(0.0)

        model_prices = np.array(model_prices)
        price_errors = model_prices - market_prices
        rmse_price = float(np.sqrt(np.mean(price_errors ** 2)))

        # IV errors (where market IVs are available)
        valid_iv = np.isfinite(market_ivs) & (market_ivs > 0.001)
        if valid_iv.any():
            iv_errors = np.zeros_like(market_ivs)
            for i in np.where(valid_iv)[0]:
                K, T = strikes[i], maturities[i]
                try:
                    model_iv = self._price_to_iv(
                        model_prices[i], S0, K, T, r, option_types[i] == 'call'
                    )
                    iv_errors[i] = model_iv - market_ivs[i]
                except Exception:
                    iv_errors[i] = 0.0
            rmse_iv = float(np.sqrt(np.mean(iv_errors[valid_iv] ** 2)))
            max_error_iv = float(np.max(np.abs(iv_errors[valid_iv])))
        else:
            rmse_iv = 0.0
            max_error_iv = 0.0

        computation_time = time.time() - start_time

        params_dict = dict(zip(self.param_names, best_params))

        return CFCalibrationResult(
            model_name=self.model_name,
            params=params_dict,
            rmse_price=rmse_price,
            rmse_iv=rmse_iv,
            max_error_iv=max_error_iv,
            n_options=len(strikes),
            n_iterations=n_iter,
            computation_time=computation_time,
            success=success,
            message=str(message),
        )

    def _extract_targets(self, chain, S0):
        """Extract arrays of strikes, maturities, prices, types, IVs from OptionChain."""
        strikes = []
        maturities = []
        market_prices = []
        option_types = []
        market_ivs = []

        for opt in chain.options:
            T = (opt.expiry - chain.reference_date).days / 365.0
            if T < 1 / 365.0:
                continue
            if opt.mid <= 0:
                continue

            strikes.append(opt.strike)
            maturities.append(T)
            market_prices.append(opt.mid)
            option_types.append(opt.option_type)
            market_ivs.append(opt.implied_volatility if opt.implied_volatility else np.nan)

        return (
            np.array(strikes),
            np.array(maturities),
            np.array(market_prices),
            option_types,
            np.array(market_ivs),
        )

    def _get_X0(self, process, S0):
        """Build initial state vector for a process."""
        if process.dim == 1:
            return np.array([S0])
        else:
            return np.array([S0] + [0.0] * (process.dim - 1))

    def _price_to_iv(self, price, S, K, T, r, is_call):
        """Invert BS price to implied volatility."""
        from scipy.stats import norm
        from scipy.optimize import brentq

        def bs(sigma):
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            if is_call:
                return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return brentq(lambda s: bs(s) - price, 0.001, 5.0)


# ============================================================================
# Pre-built calibrators for specific models
# ============================================================================
def make_merton_calibrator(**kwargs) -> CFCalibrator:
    """Create a calibrator for Merton jump-diffusion."""
    from ...processes.merton import MertonJD

    def build(p):
        # p = [sigma, lambda, mu_J, sigma_J]
        # mu is set to 0 for Q-measure calibration
        return MertonJD(mu=0.0, sigma=p[0], lambda_jump=p[1],
                        mu_J=p[2], sigma_J=p[3])

    return CFCalibrator(
        model_name='MertonJD',
        build_process_fn=build,
        param_names=['sigma', 'lambda', 'mu_J', 'sigma_J'],
        param_bounds=[(0.01, 1.0), (0.01, 10.0), (-0.5, 0.5), (0.01, 1.0)],
        **kwargs,
    )


def make_vg_calibrator(**kwargs) -> CFCalibrator:
    """Create a calibrator for Variance Gamma."""
    from ...processes.variance_gamma import VarianceGamma

    def build(p):
        # p = [theta, sigma, nu]
        return VarianceGamma(theta=p[0], sigma=p[1], nu=p[2])

    return CFCalibrator(
        model_name='VarianceGamma',
        build_process_fn=build,
        param_names=['theta', 'sigma', 'nu'],
        param_bounds=[(-0.5, 0.5), (0.01, 1.0), (0.01, 5.0)],
        **kwargs,
    )


def make_nig_calibrator(**kwargs) -> CFCalibrator:
    """Create a calibrator for Normal Inverse Gaussian."""
    from ...processes.nig import NIG

    def build(p):
        # p = [alpha, beta, delta, mu]
        return NIG(alpha=p[0], beta=p[1], delta=p[2], mu=p[3])

    return CFCalibrator(
        model_name='NIG',
        build_process_fn=build,
        param_names=['alpha', 'beta', 'delta', 'mu'],
        param_bounds=[(1.0, 50.0), (-20.0, 20.0), (0.01, 5.0), (-0.5, 0.5)],
        **kwargs,
    )


def make_kou_calibrator(**kwargs) -> CFCalibrator:
    """Create a calibrator for Kou jump-diffusion."""
    from ...processes.kou import KouJD

    def build(p):
        # p = [sigma, lambda, p_up, eta_up, eta_down]
        return KouJD(mu=0.0, sigma=p[0], lambda_jump=p[1],
                     p=p[2], eta_up=p[3], eta_down=p[4])

    return CFCalibrator(
        model_name='KouJD',
        build_process_fn=build,
        param_names=['sigma', 'lambda', 'p', 'eta_up', 'eta_down'],
        param_bounds=[
            (0.01, 1.0), (0.01, 10.0), (0.01, 0.99),
            (1.0, 50.0), (1.0, 50.0),
        ],
        **kwargs,
    )


def make_bates_calibrator(**kwargs) -> CFCalibrator:
    """Create a calibrator for Bates stochastic vol + jumps."""
    from ...processes.bates import Bates

    def build(p):
        # p = [kappa, theta, sigma_v, rho, v0, lambda_j, mu_J, sigma_J]
        # mu=0 for Q-measure
        return Bates(
            mu=0.0, kappa=p[0], theta=p[1], sigma_v=p[2], rho=p[3],
            lambda_j=p[4], mu_J=p[5], sigma_J=p[6], v0=p[7],
        )

    return CFCalibrator(
        model_name='Bates',
        build_process_fn=build,
        param_names=['kappa', 'theta', 'sigma_v', 'rho',
                     'lambda_j', 'mu_J', 'sigma_J', 'v0'],
        param_bounds=[
            (0.1, 10.0), (0.001, 1.0), (0.01, 2.0), (-0.99, 0.99),
            (0.01, 10.0), (-0.5, 0.5), (0.01, 1.0), (0.001, 1.0),
        ],
        **kwargs,
    )
