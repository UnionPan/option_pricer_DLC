"""
Analytical pricing formulas (closed-form solutions)

author: Yunian Pan
email: yp1170@nyu.edu
"""
import numpy as np
from scipy import stats
import time
from typing import Union

from .base import Pricer, PricingResult


class BlackScholesPricer(Pricer):
    """
    Black-Scholes analytical pricing for European options under GBM

    Closed-form formulas for:
    - European Call/Put
    - Greeks (Delta, Gamma, Vega, Theta, Rho)
    - Digital Call/Put

    Assumptions:
    - Process is GBM (Geometric Brownian Motion)
    - Derivative is European-style (path-independent)
    - No dividends (or continuous dividend yield)
    """

    def __init__(self, risk_free_rate: float = 0.0, dividend_yield: float = 0.0):
        """
        Initialize Black-Scholes pricer

        Args:
            risk_free_rate: Risk-free interest rate (annualized)
            dividend_yield: Continuous dividend yield
        """
        super().__init__(name="BlackScholes")
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield

    def price(
        self,
        derivative,
        process,
        X0: Union[float, np.ndarray],
        compute_greeks: bool = True,
        **kwargs
    ) -> PricingResult:
        """
        Price European option using Black-Scholes formula

        Args:
            derivative: European derivative (EuropeanCall, EuropeanPut, etc.)
            process: GBM process (must have mu and sigma attributes)
            X0: Initial spot price
            compute_greeks: Whether to compute Greeks
            **kwargs: Additional parameters (ignored)

        Returns:
            PricingResult with price and Greeks
        """
        start_time = time.time()

        # Validate inputs
        if not hasattr(process, 'sigma') or not hasattr(process, 'mu'):
            raise ValueError("Process must be GBM with mu and sigma attributes")

        S0 = float(X0) if np.isscalar(X0) else float(X0[0])
        K = derivative.strike
        T = derivative.maturity
        sigma = process.sigma
        r = self.risk_free_rate
        q = self.dividend_yield

        # Determine option type
        contract_type = derivative.contract_type

        # Price the option
        if "call" in contract_type and "digital" not in contract_type:
            price = self._bs_call(S0, K, T, r, q, sigma)
        elif "put" in contract_type and "digital" not in contract_type:
            price = self._bs_put(S0, K, T, r, q, sigma)
        elif contract_type == "digital_call":
            price = self._digital_call(S0, K, T, r, q, sigma)
        elif contract_type == "digital_put":
            price = self._digital_put(S0, K, T, r, q, sigma)
        else:
            raise ValueError(f"Unsupported derivative type for Black-Scholes: {contract_type}")

        # Compute Greeks if requested
        greeks = None
        if compute_greeks:
            greeks = self._compute_greeks(S0, K, T, r, q, sigma, contract_type)

        computation_time = time.time() - start_time

        metadata = {
            "formula": "Black-Scholes",
            "contract_type": contract_type,
            "spot": S0,
            "strike": K,
            "maturity": T,
            "volatility": sigma,
            "risk_free_rate": r,
        }

        return PricingResult(
            price=price,
            std_error=0.0,  # Analytical, no error
            confidence_interval=None,
            n_paths=None,
            computation_time=computation_time,
            greeks=greeks,
            metadata=metadata,
        )

    def _bs_call(self, S0: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        """Black-Scholes call option price"""
        d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        call_price = S0 * np.exp(-q * T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        return call_price

    def _bs_put(self, S0: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        """Black-Scholes put option price"""
        d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        put_price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S0 * np.exp(-q * T) * stats.norm.cdf(-d1)
        return put_price

    def _digital_call(self, S0: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        """Digital call option price (cash-or-nothing)"""
        d2 = (np.log(S0 / K) + (r - q - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return np.exp(-r * T) * stats.norm.cdf(d2)

    def _digital_put(self, S0: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        """Digital put option price (cash-or-nothing)"""
        d2 = (np.log(S0 / K) + (r - q - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return np.exp(-r * T) * stats.norm.cdf(-d2)

    def _compute_greeks(
        self, S0: float, K: float, T: float, r: float, q: float, sigma: float, contract_type: str
    ) -> dict:
        """
        Compute Greeks analytically

        Greeks:
        - Delta: ∂V/∂S
        - Gamma: ∂²V/∂S²
        - Vega: ∂V/∂σ
        - Theta: -∂V/∂t
        - Rho: ∂V/∂r
        """
        greeks = {}

        d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Common terms
        pdf_d1 = stats.norm.pdf(d1)
        cdf_d1 = stats.norm.cdf(d1)
        cdf_d2 = stats.norm.cdf(d2)

        if "call" in contract_type and "digital" not in contract_type:
            # Call Greeks
            greeks['delta'] = np.exp(-q * T) * cdf_d1
            greeks['gamma'] = np.exp(-q * T) * pdf_d1 / (S0 * sigma * np.sqrt(T))
            greeks['vega'] = S0 * np.exp(-q * T) * pdf_d1 * np.sqrt(T)
            greeks['theta'] = (
                -S0 * pdf_d1 * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
                - r * K * np.exp(-r * T) * cdf_d2
                + q * S0 * np.exp(-q * T) * cdf_d1
            )
            greeks['rho'] = K * T * np.exp(-r * T) * cdf_d2

        elif "put" in contract_type and "digital" not in contract_type:
            # Put Greeks
            greeks['delta'] = -np.exp(-q * T) * stats.norm.cdf(-d1)
            greeks['gamma'] = np.exp(-q * T) * pdf_d1 / (S0 * sigma * np.sqrt(T))
            greeks['vega'] = S0 * np.exp(-q * T) * pdf_d1 * np.sqrt(T)
            greeks['theta'] = (
                -S0 * pdf_d1 * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
                + r * K * np.exp(-r * T) * stats.norm.cdf(-d2)
                - q * S0 * np.exp(-q * T) * stats.norm.cdf(-d1)
            )
            greeks['rho'] = -K * T * np.exp(-r * T) * stats.norm.cdf(-d2)

        # Normalize vega (typically reported per 1% change in volatility)
        if 'vega' in greeks:
            greeks['vega'] = greeks['vega'] / 100

        # Normalize theta (typically reported per day)
        if 'theta' in greeks:
            greeks['theta'] = greeks['theta'] / 365

        # Normalize rho (typically reported per 1% change in rate)
        if 'rho' in greeks:
            greeks['rho'] = greeks['rho'] / 100

        return greeks

    def implied_volatility(
        self,
        derivative,
        market_price: float,
        S0: float,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> float:
        """
        Calculate implied volatility using Newton-Raphson method

        Args:
            derivative: European option derivative
            market_price: Observed market price
            S0: Current spot price
            max_iterations: Maximum iterations for Newton-Raphson
            tolerance: Convergence tolerance

        Returns:
            Implied volatility (annualized)
        """
        K = derivative.strike
        T = derivative.maturity
        r = self.risk_free_rate
        q = self.dividend_yield
        contract_type = derivative.contract_type

        # Initial guess: ATM volatility approximation
        sigma = np.sqrt(2 * np.pi / T) * market_price / S0

        for i in range(max_iterations):
            # Price and vega at current sigma
            if "call" in contract_type:
                price = self._bs_call(S0, K, T, r, q, sigma)
            else:
                price = self._bs_put(S0, K, T, r, q, sigma)

            # Vega
            d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            vega = S0 * np.exp(-q * T) * stats.norm.pdf(d1) * np.sqrt(T)

            # Newton-Raphson update
            price_diff = price - market_price

            if abs(price_diff) < tolerance:
                return sigma

            if vega < 1e-10:
                raise ValueError("Vega too small, cannot converge")

            sigma = sigma - price_diff / vega

            # Keep sigma positive
            sigma = max(sigma, 1e-6)

        raise ValueError(f"Implied volatility did not converge after {max_iterations} iterations")


class HestonAnalyticalPricer(Pricer):
    """
    Heston model analytical pricing via characteristic function

    Uses semi-closed form solution for European options
    Based on Heston (1993) formula
    """

    def __init__(self, risk_free_rate: float = 0.0):
        super().__init__(name="HestonAnalytical")
        self.risk_free_rate = risk_free_rate

    def price(self, derivative, process, X0: Union[float, np.ndarray], **kwargs) -> PricingResult:
        """
        Price European option under Heston model

        Args:
            derivative: European call or put
            process: Heston process with parameters
            X0: Initial [S0, V0] where V0 is initial variance
            **kwargs: Additional parameters

        Returns:
            PricingResult with price
        """
        start_time = time.time()

        # Extract parameters
        if not hasattr(process, 'kappa'):
            raise ValueError("Process must be Heston model with kappa, theta, xi, rho")

        S0 = float(X0[0]) if len(X0) > 1 else float(X0)
        V0 = float(X0[1]) if len(X0) > 1 else process.theta
        K = derivative.strike
        T = derivative.maturity

        kappa = process.kappa
        theta = process.theta
        xi = process.xi
        rho = process.rho
        r = self.risk_free_rate

        # Determine call or put
        is_call = "call" in derivative.contract_type

        # Heston formula via characteristic function
        price = self._heston_price(S0, K, V0, T, r, kappa, theta, xi, rho, is_call)

        computation_time = time.time() - start_time

        metadata = {
            "formula": "Heston",
            "contract_type": derivative.contract_type,
        }

        return PricingResult(
            price=price,
            std_error=0.0,
            computation_time=computation_time,
            metadata=metadata,
        )

    def _heston_price(
        self, S0: float, K: float, V0: float, T: float, r: float,
        kappa: float, theta: float, xi: float, rho: float, is_call: bool
    ) -> float:
        """
        Heston semi-closed form formula

        P_call = S0 * P1 - K * exp(-rT) * P2
        where P1 and P2 are probabilities computed via Fourier inversion
        """
        # Integration bounds
        integration_limit = 100
        n_points = 1000

        # Compute P1 and P2 via numerical integration
        P1 = 0.5 + (1 / np.pi) * self._heston_integral(
            1, S0, K, V0, T, r, kappa, theta, xi, rho, integration_limit, n_points
        )

        P2 = 0.5 + (1 / np.pi) * self._heston_integral(
            2, S0, K, V0, T, r, kappa, theta, xi, rho, integration_limit, n_points
        )

        if is_call:
            return S0 * P1 - K * np.exp(-r * T) * P2
        else:
            # Put-call parity
            call_price = S0 * P1 - K * np.exp(-r * T) * P2
            return call_price - S0 + K * np.exp(-r * T)

    def _heston_integral(
        self, j: int, S0: float, K: float, V0: float, T: float, r: float,
        kappa: float, theta: float, xi: float, rho: float, limit: float, n_points: int
    ) -> float:
        """
        Numerical integration for Heston probabilities

        Simplified implementation - production code should use adaptive quadrature
        """
        x = np.log(S0 / K)
        du = limit / n_points
        integral_sum = 0.0

        for i in range(1, n_points + 1):
            u = i * du
            integrand = np.real(
                np.exp(-1j * u * x) * self._heston_char_func(u, V0, T, r, kappa, theta, xi, rho, j)
                / (1j * u)
            )
            integral_sum += integrand

        return integral_sum * du

    def _heston_char_func(
        self, u: float, V0: float, T: float, r: float,
        kappa: float, theta: float, xi: float, rho: float, j: int
    ) -> complex:
        """
        Heston characteristic function

        Simplified version - full implementation requires careful handling of branch cuts
        """
        if j == 1:
            u_j = 0.5
            b_j = kappa - rho * xi
        else:
            u_j = -0.5
            b_j = kappa

        d = np.sqrt((rho * xi * u * 1j - b_j)**2 - xi**2 * (2 * u_j * u * 1j - u**2))
        g = (b_j - rho * xi * u * 1j + d) / (b_j - rho * xi * u * 1j - d)

        C = r * u * 1j * T + (kappa * theta / xi**2) * (
            (b_j - rho * xi * u * 1j + d) * T - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g))
        )

        D = (b_j - rho * xi * u * 1j + d) / xi**2 * (1 - np.exp(d * T)) / (1 - g * np.exp(d * T))

        return np.exp(C + D * V0)
