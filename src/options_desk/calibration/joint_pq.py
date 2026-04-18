"""
Joint P/Q Calibration Framework

Simultaneously calibrate physical (P) and risk-neutral (Q) parameters
for stochastic volatility models, enforcing consistency constraints
on diffusion parameters and extracting market prices of risk.

Key insight: some parameters (sigma_v, rho for Heston) are properties
of the diffusion and must be identical under P and Q.  Only the drift
parameters change across measures via the market price of risk.

    kappa_Q = kappa_P + lambda_v
    theta_Q = kappa_P * theta_P / kappa_Q

References:
    - Broadie, M., Chernov, M., Johannes, M. (2007) "Model Specification
      and Risk Premia: Evidence from Futures Options"
    - Aït-Sahalia, Y., Kimmel, R. (2007) "Maximum Likelihood Estimation
      of Stochastic Volatility Models"

author: Yunian Pan
email: yp1170@nyu.edu
"""

import numpy as np
from scipy import optimize
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class JointPQResult:
    """Result of joint P/Q calibration."""

    # P-measure params (from time series)
    p_params: Dict[str, float]

    # Q-measure params (from option prices)
    q_params: Dict[str, float]

    # Market price of risk (connects P and Q)
    market_price_of_risk: Dict[str, float]

    # Shared params that must be consistent across measures
    shared_params: Dict[str, float]

    # Fit quality
    p_log_likelihood: float
    q_loss: float
    joint_loss: float

    # Consistency diagnostics
    sigma_v_consistent: bool
    rho_consistent: bool

    computation_time: float

    def __repr__(self) -> str:
        lines = [
            "JointPQCalibrationResult:",
            f"  P-measure: mu={self.p_params.get('mu', 0):.4f}, "
            f"kappa_P={self.p_params.get('kappa', 0):.4f}, "
            f"theta_P={self.p_params.get('theta', 0):.6f}",
            f"  Q-measure: kappa_Q={self.q_params.get('kappa', 0):.4f}, "
            f"theta_Q={self.q_params.get('theta', 0):.6f}",
            f"  Shared: sigma_v={self.shared_params.get('sigma_v', 0):.4f}, "
            f"rho={self.shared_params.get('rho', 0):.4f}, "
            f"v0={self.shared_params.get('v0', 0):.6f}",
            f"  Risk premia: lambda_v={self.market_price_of_risk.get('lambda_v', 0):.4f}, "
            f"lambda_s={self.market_price_of_risk.get('lambda_s', 0):.4f}",
            f"  P log-lik={self.p_log_likelihood:.2f}, Q RMSE={self.q_loss:.6f}",
            f"  Consistent: sigma_v={'yes' if self.sigma_v_consistent else 'NO'}, "
            f"rho={'yes' if self.rho_consistent else 'NO'}",
            f"  Time: {self.computation_time:.2f}s",
        ]
        return "\n".join(lines)


def extract_market_price_of_risk(
    p_params: Dict[str, float],
    q_params: Dict[str, float],
    rate: float = 0.05,
) -> Dict[str, float]:
    """
    Extract market prices of risk from P and Q parameters.

    Under the Heston model:
        lambda_v = kappa_Q - kappa_P    (variance risk premium)
        lambda_s = (mu_P - r) / sqrt(v_bar)  (equity risk premium)

    A positive lambda_v means the market charges a premium for variance
    risk (kappa_Q > kappa_P, faster mean reversion under Q).

    Args:
        p_params: Physical parameters with keys mu, kappa, theta.
        q_params: Risk-neutral parameters with keys kappa, theta.
        rate: Risk-free rate for equity premium calculation.

    Returns:
        Dict with lambda_v, lambda_s, and descriptive fields.
    """
    kappa_P = p_params['kappa']
    kappa_Q = q_params['kappa']
    theta_P = p_params.get('theta', q_params.get('theta', 0.04))
    mu_P = p_params.get('mu', rate)

    lambda_v = kappa_Q - kappa_P

    v_bar = max(theta_P, 1e-8)
    lambda_s = (mu_P - rate) / np.sqrt(v_bar)

    return {
        'lambda_v': lambda_v,
        'lambda_s': lambda_s,
        'variance_risk_premium_sign': 'positive' if lambda_v > 0 else 'negative',
        'equity_risk_premium_annualized': mu_P - rate,
    }


class JointPQCalibrator:
    """
    Joint calibration of physical and risk-neutral parameters.

    Approach (sequential, default):
        1. Calibrate Q-params from option chain (well-posed inverse problem)
        2. Fix sigma_v, rho from Q-calibration (diffusion params are
           measure-invariant)
        3. Calibrate P-params (kappa_P, theta_P, mu) from time series
           with sigma_v and rho constrained
        4. Extract market prices of risk

    Approach (joint, optional):
        Minimize a weighted combination of P-likelihood and Q-option-fit
        with shared sigma_v, rho, v0 across measures.

    Example::

        calibrator = JointPQCalibrator()
        result = calibrator.calibrate(
            prices=historical_prices,
            option_chain=chain,
            rate=0.05,
        )
        print(result.market_price_of_risk)
    """

    def __init__(self, model_type: str = 'heston'):
        """
        Args:
            model_type: Model to calibrate. Currently only 'heston'.
        """
        if model_type != 'heston':
            raise ValueError(f"Unsupported model: {model_type}. Use 'heston'.")
        self.model_type = model_type

    def calibrate(
        self,
        prices: np.ndarray,
        option_chain,
        spot: Optional[float] = None,
        rate: float = 0.05,
        dividend_yield: float = 0.0,
        dt: float = 1 / 252,
        joint_optimization: bool = False,
        p_weight: float = 0.3,
        q_weight: float = 0.7,
    ) -> JointPQResult:
        """
        Run joint P/Q calibration.

        Args:
            prices: Historical prices array, shape (T,).
            option_chain: OptionChain for Q-calibration.
            spot: Current spot (defaults to last price).
            rate: Risk-free rate.
            dividend_yield: Continuous dividend yield.
            dt: Time step for P-calibration (1/252 for daily).
            joint_optimization: If True, run expensive joint fit.
            p_weight: Weight on P-likelihood in joint objective.
            q_weight: Weight on Q-option fit in joint objective.

        Returns:
            JointPQResult with all calibrated parameters.
        """
        start_time = time.time()
        S0 = spot or float(prices[-1])

        # Step 1: Q-calibration from options
        q_params, q_loss = self._calibrate_q(option_chain, S0, rate, dividend_yield)
        logger.info("Q-calibration done: kappa=%.4f, theta=%.6f, RMSE=%.6f",
                     q_params['kappa'], q_params['theta'], q_loss)

        # Step 2: P-calibration from time series with shared diffusion params
        sigma_v_shared = q_params['sigma_v']
        rho_shared = q_params['rho']
        v0_shared = q_params['v0']

        p_params, p_loglik = self._calibrate_p(
            prices, rate, dt, sigma_v_shared, rho_shared, v0_shared,
        )
        logger.info("P-calibration done: mu=%.4f, kappa_P=%.4f, log_lik=%.2f",
                     p_params['mu'], p_params['kappa'], p_loglik)

        # Step 3: Extract risk premia
        mpr = extract_market_price_of_risk(p_params, q_params, rate)

        shared = {
            'sigma_v': sigma_v_shared,
            'rho': rho_shared,
            'v0': v0_shared,
        }

        joint_loss = p_weight * (-p_loglik) + q_weight * q_loss

        # Step 4 (optional): joint re-optimization
        if joint_optimization:
            logger.info("Running joint optimization...")
            q_params, p_params, shared, q_loss, p_loglik, joint_loss = (
                self._joint_optimize(
                    prices, option_chain, S0, rate, dividend_yield, dt,
                    q_params, p_params, shared,
                    p_weight, q_weight,
                )
            )
            mpr = extract_market_price_of_risk(p_params, q_params, rate)

        computation_time = time.time() - start_time

        return JointPQResult(
            p_params=p_params,
            q_params=q_params,
            market_price_of_risk=mpr,
            shared_params=shared,
            p_log_likelihood=p_loglik,
            q_loss=q_loss,
            joint_loss=joint_loss,
            sigma_v_consistent=True,
            rho_consistent=True,
            computation_time=computation_time,
        )

    # ------------------------------------------------------------------
    # Q-calibration (from options)
    # ------------------------------------------------------------------
    def _calibrate_q(
        self,
        chain,
        S0: float,
        rate: float,
        q: float,
    ) -> Tuple[Dict[str, float], float]:
        """Calibrate Q-params using HestonCalibrator."""
        from .risk_neutral.heston_calibrator import HestonCalibrator

        cal = HestonCalibrator()
        result = cal.calibrate(chain, spot=S0, rate=rate, dividend_yield=q)

        params = {
            'kappa': result.kappa,
            'theta': result.theta,
            'sigma_v': result.xi,
            'rho': result.rho,
            'v0': result.v0,
        }
        return params, result.rmse

    # ------------------------------------------------------------------
    # P-calibration (from time series)
    # ------------------------------------------------------------------
    def _calibrate_p(
        self,
        prices: np.ndarray,
        rate: float,
        dt: float,
        sigma_v: float,
        rho: float,
        v0: float,
    ) -> Tuple[Dict[str, float], float]:
        """
        Calibrate P-params via quasi-MLE on log-returns.

        Uses realized variance from squared returns as a proxy for v_t,
        then maximizes the Gaussian log-likelihood of returns conditional
        on the variance proxy.
        """
        log_returns = np.diff(np.log(prices))
        n = len(log_returns)

        # Realized variance proxy (EWMA with 20-day halflife)
        decay = np.exp(-np.log(2) / 20)
        rv = np.zeros(n)
        rv[0] = log_returns[0] ** 2 / dt
        for i in range(1, n):
            rv[i] = decay * rv[i - 1] + (1 - decay) * log_returns[i] ** 2 / dt

        # Floor variance proxy
        rv = np.maximum(rv, 1e-8)

        def neg_log_lik(params):
            mu_P, kappa_P, theta_P = params

            # Conditional Gaussian log-likelihood
            # log(S_{t+1}/S_t) ~ N((mu - 0.5*v_t)*dt, v_t*dt)
            mean = (mu_P - 0.5 * rv[:-1]) * dt
            var = rv[:-1] * dt
            var = np.maximum(var, 1e-12)

            residuals = log_returns[1:] - mean
            ll = -0.5 * np.sum(np.log(2 * np.pi * var) + residuals ** 2 / var)

            # Penalise extreme kappa/theta to aid convergence
            penalty = 1e-4 * (kappa_P ** 2 + (theta_P - np.mean(rv)) ** 2)
            return -ll + penalty

        x0 = [
            np.mean(log_returns) / dt + 0.5 * np.mean(rv),  # mu
            2.0,   # kappa_P
            np.mean(rv),  # theta_P
        ]
        bounds = [(-1.0, 2.0), (0.01, 20.0), (1e-6, 2.0)]

        res = optimize.minimize(neg_log_lik, x0, bounds=bounds, method='L-BFGS-B')
        mu_P, kappa_P, theta_P = res.x

        params = {
            'mu': mu_P,
            'kappa': kappa_P,
            'theta': theta_P,
            'sigma_v': sigma_v,
            'rho': rho,
            'v0': v0,
        }
        return params, -res.fun

    # ------------------------------------------------------------------
    # Joint optimisation
    # ------------------------------------------------------------------
    def _joint_optimize(
        self,
        prices, chain, S0, rate, q, dt,
        q_init, p_init, shared_init,
        p_weight, q_weight,
    ) -> Tuple[Dict, Dict, Dict, float, float, float]:
        """
        Joint optimisation with shared sigma_v, rho, v0.

        Uses differential_evolution for robustness.
        """
        from .risk_neutral.heston_calibrator import HestonCalibrator

        log_returns = np.diff(np.log(prices))
        n = len(log_returns)

        # Realised variance proxy
        decay = np.exp(-np.log(2) / 20)
        rv = np.zeros(n)
        rv[0] = log_returns[0] ** 2 / dt
        for i in range(1, n):
            rv[i] = decay * rv[i - 1] + (1 - decay) * log_returns[i] ** 2 / dt
        rv = np.maximum(rv, 1e-8)

        cal_q = HestonCalibrator()

        def objective(x):
            mu_P, kappa_P, theta_P, kappa_Q, theta_Q, sigma_v, rho, v0 = x

            # P log-likelihood
            mean = (mu_P - 0.5 * rv[:-1]) * dt
            var = np.maximum(rv[:-1] * dt, 1e-12)
            residuals = log_returns[1:] - mean
            p_ll = -0.5 * np.sum(np.log(2 * np.pi * var) + residuals ** 2 / var)

            # Q option RMSE (quick evaluation via CF pricer)
            try:
                q_rmse = cal_q._evaluate_rmse(
                    chain, S0, rate, q, kappa_Q, theta_Q, sigma_v, rho, v0,
                )
            except Exception:
                q_rmse = 1e6

            return p_weight * (-p_ll / n) + q_weight * q_rmse

        bounds = [
            (-0.5, 1.5),     # mu_P
            (0.01, 20.0),    # kappa_P
            (1e-6, 1.0),     # theta_P
            (0.01, 20.0),    # kappa_Q
            (1e-6, 1.0),     # theta_Q
            (0.01, 3.0),     # sigma_v (shared)
            (-0.99, 0.0),    # rho (shared)
            (1e-6, 1.0),     # v0 (shared)
        ]

        x0 = [
            p_init['mu'], p_init['kappa'], p_init['theta'],
            q_init['kappa'], q_init['theta'],
            shared_init['sigma_v'], shared_init['rho'], shared_init['v0'],
        ]

        try:
            res = optimize.differential_evolution(
                objective, bounds, x0=x0, maxiter=200,
                seed=42, tol=1e-6, polish=True,
            )
            mu_P, kappa_P, theta_P, kappa_Q, theta_Q, sigma_v, rho, v0 = res.x
        except Exception:
            logger.warning("Joint optimization failed, using sequential results")
            return (q_init, p_init, shared_init,
                    q_init.get('rmse', 0), 0, 0)

        q_out = {'kappa': kappa_Q, 'theta': theta_Q,
                 'sigma_v': sigma_v, 'rho': rho, 'v0': v0}
        p_out = {'mu': mu_P, 'kappa': kappa_P, 'theta': theta_P,
                 'sigma_v': sigma_v, 'rho': rho, 'v0': v0}
        shared_out = {'sigma_v': sigma_v, 'rho': rho, 'v0': v0}

        # Recompute individual losses at the optimum
        mean = (mu_P - 0.5 * rv[:-1]) * dt
        var = np.maximum(rv[:-1] * dt, 1e-12)
        residuals = log_returns[1:] - mean
        p_ll = -0.5 * np.sum(np.log(2 * np.pi * var) + residuals ** 2 / var)

        try:
            q_rmse = cal_q._evaluate_rmse(
                chain, S0, rate, q, kappa_Q, theta_Q, sigma_v, rho, v0,
            )
        except Exception:
            q_rmse = float('nan')

        joint = p_weight * (-p_ll / n) + q_weight * q_rmse

        return q_out, p_out, shared_out, q_rmse, p_ll, joint
