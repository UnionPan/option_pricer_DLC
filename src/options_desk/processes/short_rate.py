"""
Short-Rate Models: Vasicek, CIR, Hull-White

Single-factor models for the instantaneous short rate r_t, with
analytical zero-coupon bond pricing and (for Vasicek) bond option pricing.

    Vasicek:    dr = kappa*(theta - r)*dt + sigma*dW
    CIR:        dr = kappa*(theta - r)*dt + sigma*sqrt(r)*dW
    Hull-White:  dr = (theta(t) - a*r)*dt + sigma*dW

author: Yunian Pan
email: yp1170@nyu.edu
"""

import numpy as np
from typing import Optional, Callable
from .base import DriftDiffusionProcess
from ._structured_inputs import StructuredTimeCurve


class Vasicek(DriftDiffusionProcess):
    """
    Vasicek short-rate model.

        dr_t = kappa * (theta - r_t) dt + sigma * dW_t

    Gaussian OU process for the short rate.  Admits negative rates,
    which is acceptable for many currencies.

    Analytical results:
        - Zero-coupon bond price P(r, T)
        - Bond option price (Jamshidian)
        - Yield curve
    """

    def __init__(
        self,
        kappa: float,
        theta: float,
        sigma: float,
        name: str = "Vasicek",
    ):
        super().__init__(name=name)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma = float(sigma)

        self.params['kappa'] = self.kappa
        self.params['theta'] = self.theta
        self.params['sigma'] = self.sigma

    def _build_jax_spec(self):
        from ._process_defs import OUParams, ou_drift, ou_diffusion
        return {
            'drift_fn': ou_drift,
            'diffusion_fn': ou_diffusion,
            'params': OUParams(theta=self.kappa, mu=self.theta, sigma=self.sigma),
            'dim': 1,
        }

    def drift(self, X: np.ndarray, t: float) -> np.ndarray:
        return self.kappa * (self.theta - X)

    def diffusion(self, X: np.ndarray, t: float) -> np.ndarray:
        return self.sigma * np.ones_like(X)

    def _exact_simulation(self, X0, T, dt, t_grid, config):
        n_paths = config.n_paths
        if config.antithetic:
            n_paths = n_paths // 2

        paths = np.zeros((len(t_grid), n_paths, self.dim))
        paths[0] = X0

        for i in range(1, len(t_grid)):
            t = t_grid[i]
            exp_kt = np.exp(-self.kappa * t)
            mean = X0 * exp_kt + self.theta * (1.0 - exp_kt)
            var = (self.sigma ** 2 / (2.0 * self.kappa)) * (1.0 - np.exp(-2.0 * self.kappa * t))
            Z = np.random.normal(0, 1, (n_paths, self.dim))
            paths[i] = mean + np.sqrt(var) * Z

        if config.antithetic:
            paths_anti = np.zeros((len(t_grid), n_paths, self.dim))
            paths_anti[0] = X0
            if config.random_seed is not None:
                np.random.seed(config.random_seed)
            for i in range(1, len(t_grid)):
                t = t_grid[i]
                exp_kt = np.exp(-self.kappa * t)
                mean = X0 * exp_kt + self.theta * (1.0 - exp_kt)
                var = (self.sigma ** 2 / (2.0 * self.kappa)) * (1.0 - np.exp(-2.0 * self.kappa * t))
                Z = np.random.normal(0, 1, (n_paths, self.dim))
                paths_anti[i] = mean - np.sqrt(var) * Z
            paths = np.concatenate([paths, paths_anti], axis=1)

        return t_grid, paths

    def expectation(self, X0: np.ndarray, t: float) -> np.ndarray:
        X0 = np.atleast_1d(X0)
        exp_kt = np.exp(-self.kappa * t)
        return X0 * exp_kt + self.theta * (1.0 - exp_kt)

    def variance(self, t: float) -> float:
        return (self.sigma ** 2 / (2.0 * self.kappa)) * (1.0 - np.exp(-2.0 * self.kappa * t))

    # ------------------------------------------------------------------
    # Bond pricing
    # ------------------------------------------------------------------
    def _B(self, T: float) -> float:
        """Vasicek B(T) = (1 - exp(-kappa*T)) / kappa."""
        return (1.0 - np.exp(-self.kappa * T)) / self.kappa

    def _A(self, T: float) -> float:
        """Vasicek A(T)."""
        B = self._B(T)
        exponent = (self.theta - self.sigma ** 2 / (2.0 * self.kappa ** 2)) * (B - T) \
                   - self.sigma ** 2 * B ** 2 / (4.0 * self.kappa)
        return np.exp(exponent)

    def zero_coupon_bond(self, r: float, T: float) -> float:
        """
        Analytical zero-coupon bond price P(r, T).

        P(r, T) = A(T) * exp(-B(T) * r)

        Args:
            r: Current short rate.
            T: Time to maturity.

        Returns:
            Bond price.
        """
        return self._A(T) * np.exp(-self._B(T) * r)

    def yield_curve(self, r: float, maturities: np.ndarray) -> np.ndarray:
        """
        Compute continuously compounded yield curve.

        y(T) = -log(P(r,T)) / T

        Args:
            r: Current short rate.
            maturities: Array of maturities.

        Returns:
            Array of yields.
        """
        maturities = np.asarray(maturities)
        prices = np.array([self.zero_coupon_bond(r, T) for T in maturities])
        return -np.log(prices) / maturities

    def bond_option_price(
        self, r: float, T_option: float, T_bond: float,
        K: float, call: bool = True,
    ) -> float:
        """
        Jamshidian closed-form bond option price.

        Args:
            r: Current short rate.
            T_option: Option expiry.
            T_bond: Bond maturity (T_bond > T_option).
            K: Strike price.
            call: True for call, False for put.

        Returns:
            Option price.
        """
        from scipy.stats import norm

        P_T = self.zero_coupon_bond(r, T_option)
        P_S = self.zero_coupon_bond(r, T_bond)

        sigma_p = self.sigma * self._B(T_bond - T_option) * \
            np.sqrt((1.0 - np.exp(-2.0 * self.kappa * T_option)) / (2.0 * self.kappa))

        d1 = np.log(P_S / (K * P_T)) / sigma_p + 0.5 * sigma_p
        d2 = d1 - sigma_p

        if call:
            return P_S * norm.cdf(d1) - K * P_T * norm.cdf(d2)
        else:
            return K * P_T * norm.cdf(-d2) - P_S * norm.cdf(-d1)


class CIR(DriftDiffusionProcess):
    """
    Cox-Ingersoll-Ross short-rate model.

        dr_t = kappa * (theta - r_t) dt + sigma * sqrt(r_t) dW_t

    Square-root diffusion ensures r_t >= 0 when Feller condition holds:
        2 * kappa * theta >= sigma^2

    Analytical results:
        - Zero-coupon bond price P(r, T)
        - Yield curve
    """

    def __init__(
        self,
        kappa: float,
        theta: float,
        sigma: float,
        name: str = "CIR",
    ):
        super().__init__(name=name)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma = float(sigma)

        self.feller_condition = 2.0 * self.kappa * self.theta >= self.sigma ** 2

        self.params['kappa'] = self.kappa
        self.params['theta'] = self.theta
        self.params['sigma'] = self.sigma
        self.params['feller_satisfied'] = self.feller_condition

    def _build_jax_spec(self):
        from ._process_defs import (
            CIRParams, cir_drift, cir_diffusion,
            cir_diffusion_deriv, cir_post_step,
        )
        return {
            'drift_fn': cir_drift,
            'diffusion_fn': cir_diffusion,
            'params': CIRParams(kappa=self.kappa, theta=self.theta, sigma=self.sigma),
            'dim': 1,
            'diffusion_deriv_fn': cir_diffusion_deriv,
            'post_step_fn': cir_post_step,
        }

    def drift(self, X: np.ndarray, t: float) -> np.ndarray:
        return self.kappa * (self.theta - X)

    def diffusion(self, X: np.ndarray, t: float) -> np.ndarray:
        return self.sigma * np.sqrt(np.maximum(X, 0.0))

    def diffusion_derivative(self, X: np.ndarray, t: float) -> np.ndarray:
        r_safe = np.maximum(X, 1e-12)
        return self.sigma / (2.0 * np.sqrt(r_safe))

    def _euler_maruyama(self, X0, T, dt, t_grid, config):
        t_grid, paths = super()._euler_maruyama(X0, T, dt, t_grid, config)
        paths = np.maximum(paths, 0.0)
        return t_grid, paths

    def expectation(self, X0: np.ndarray, t: float) -> np.ndarray:
        X0 = np.atleast_1d(X0)
        exp_kt = np.exp(-self.kappa * t)
        return X0 * exp_kt + self.theta * (1.0 - exp_kt)

    # ------------------------------------------------------------------
    # Bond pricing (CIR analytical)
    # ------------------------------------------------------------------
    def _gamma(self) -> float:
        return np.sqrt(self.kappa ** 2 + 2.0 * self.sigma ** 2)

    def _B_cir(self, T: float) -> float:
        gamma = self._gamma()
        num = 2.0 * (np.exp(gamma * T) - 1.0)
        den = (gamma + self.kappa) * (np.exp(gamma * T) - 1.0) + 2.0 * gamma
        return num / den

    def _A_cir(self, T: float) -> float:
        gamma = self._gamma()
        num = 2.0 * gamma * np.exp((self.kappa + gamma) * T / 2.0)
        den = (gamma + self.kappa) * (np.exp(gamma * T) - 1.0) + 2.0 * gamma
        exponent = 2.0 * self.kappa * self.theta / self.sigma ** 2
        return (num / den) ** exponent

    def zero_coupon_bond(self, r: float, T: float) -> float:
        """
        Analytical CIR zero-coupon bond price.

        P(r, T) = A(T) * exp(-B(T) * r)
        """
        return self._A_cir(T) * np.exp(-self._B_cir(T) * r)

    def yield_curve(self, r: float, maturities: np.ndarray) -> np.ndarray:
        maturities = np.asarray(maturities)
        prices = np.array([self.zero_coupon_bond(r, T) for T in maturities])
        return -np.log(prices) / maturities


class HullWhite(DriftDiffusionProcess):
    """
    Hull-White (extended Vasicek) short-rate model.

        dr_t = (theta(t) - a * r_t) dt + sigma * dW_t

    where theta(t) is calibrated to fit the initial yield curve exactly.

    If no yield curve is supplied, theta(t) defaults to a constant
    (reducing to the Vasicek model).
    """

    def __init__(
        self,
        a: float,
        sigma: float,
        theta_fn: Optional[Callable[[float], float]] = None,
        theta_const: float = 0.05,
        theta_times: Optional[np.ndarray] = None,
        theta_values: Optional[np.ndarray] = None,
        name: str = "HullWhite",
    ):
        """
        Args:
            a: Mean reversion speed.
            sigma: Volatility.
            theta_fn: Time-dependent theta(t). If None, uses a*theta_const.
            theta_const: Constant long-term mean when theta_fn is not provided.
            theta_times: Optional time grid for a structured theta(t) representation.
            theta_values: Optional theta values on theta_times for JAX-compatible simulation.
        """
        super().__init__(name=name)
        self.a = float(a)
        self.sigma = float(sigma)
        self.theta_const = float(theta_const)
        self.theta_curve = None

        if (theta_times is None) != (theta_values is None):
            raise ValueError("theta_times and theta_values must be provided together")

        if theta_times is not None and theta_values is not None:
            self.theta_curve = StructuredTimeCurve(theta_times, theta_values)
        elif theta_fn is None:
            self.theta_curve = StructuredTimeCurve.constant(self.a * self.theta_const)

        self.theta_times = None if self.theta_curve is None else self.theta_curve.times
        self.theta_values = None if self.theta_curve is None else self.theta_curve.values

        if theta_fn is not None:
            self.theta_fn = theta_fn
        elif self.theta_curve is not None:
            self.theta_fn = self.theta_curve.as_numpy_callable()
        else:
            self.theta_fn = lambda t: self.a * self.theta_const

        self.params['a'] = self.a
        self.params['sigma'] = self.sigma
        if self.theta_curve is not None:
            self.params['theta_times'] = self.theta_curve.times
            self.params['theta_values'] = self.theta_curve.values

    def _build_jax_spec(self):
        if self.theta_curve is None:
            return None

        from ._process_defs import HullWhiteParams, hull_white_drift, hull_white_diffusion
        return {
            'drift_fn': hull_white_drift,
            'diffusion_fn': hull_white_diffusion,
            'params': HullWhiteParams(
                a=self.a,
                sigma=self.sigma,
                theta_times=self.theta_curve.times,
                theta_values=self.theta_curve.values,
            ),
            'dim': 1,
        }

    def drift(self, X: np.ndarray, t: float) -> np.ndarray:
        theta_t = self.theta_fn(t)
        return (theta_t - self.a * X)

    def diffusion(self, X: np.ndarray, t: float) -> np.ndarray:
        return self.sigma * np.ones_like(X)

    def expectation(self, X0: np.ndarray, t: float) -> np.ndarray:
        X0 = np.atleast_1d(X0)
        exp_at = np.exp(-self.a * t)
        return X0 * exp_at + self.theta_const * (1.0 - exp_at)

    # ------------------------------------------------------------------
    # Bond pricing (uses Vasicek-like formulas for constant theta)
    # ------------------------------------------------------------------
    def _B(self, T: float) -> float:
        return (1.0 - np.exp(-self.a * T)) / self.a

    def zero_coupon_bond(self, r: float, T: float) -> float:
        """
        Approximate ZCB price using constant-theta Vasicek formula.

        For exact pricing with time-dependent theta, use numerical
        integration or tree methods.
        """
        B = self._B(T)
        exponent = (self.theta_const - self.sigma ** 2 / (2.0 * self.a ** 2)) * (B - T) \
                   - self.sigma ** 2 * B ** 2 / (4.0 * self.a)
        return np.exp(exponent - B * r)

    @staticmethod
    def from_yield_curve(
        a: float,
        sigma: float,
        maturities: np.ndarray,
        yields: np.ndarray,
    ) -> 'HullWhite':
        """
        Calibrate theta(t) to match an observed yield curve.

        Uses the relationship:
            theta(t) = f'(0,t) + a*f(0,t) + sigma^2/(2*a)*(1-exp(-2*a*t))

        where f(0, t) is the instantaneous forward rate.

        Args:
            a: Mean reversion speed.
            sigma: Volatility.
            maturities: Observed maturities.
            yields: Observed continuously-compounded yields.

        Returns:
            HullWhite instance with calibrated theta(t).
        """
        from scipy.interpolate import CubicSpline

        maturities = np.asarray(maturities, dtype=np.float64)
        yields = np.asarray(yields, dtype=np.float64)

        # Instantaneous forward rates via finite differences on -T*y(T)
        discount_log = -maturities * yields
        cs = CubicSpline(maturities, discount_log)
        forward_rates = -cs(maturities, 1)  # f(0, t) = -d/dt log P(0,t)

        # theta(t)
        forward_deriv = -cs(maturities, 2)  # f'(0, t)
        theta_vals = forward_deriv + a * forward_rates + \
            sigma ** 2 / (2.0 * a) * (1.0 - np.exp(-2.0 * a * maturities))

        theta_spline = CubicSpline(maturities, theta_vals, extrapolate=True)

        return HullWhite(
            a=a,
            sigma=sigma,
            theta_fn=theta_spline,
            theta_const=yields[0],
            theta_times=maturities,
            theta_values=theta_vals,
        )
