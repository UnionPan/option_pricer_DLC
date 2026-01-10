"""
Fourier-based pricing methods (COS, FFT, Lewis)

author: Yunian Pan
email: yp1170@nyu.edu
"""
import numpy as np
import time
from typing import Union

from .base import Pricer, PricingResult


class COSPricer(Pricer):
    """
    COS method for European option pricing

    The COS (Fourier-cosine) method is extremely fast and accurate for
    European options when the characteristic function is known.

    References:
    - Fang & Oosterlee (2008) "A Novel Pricing Method for European Options
      Based on Fourier-Cosine Series Expansions"

    Works with processes that have characteristic_function() method:
    - GBM
    - Merton Jump-Diffusion
    - Kou Jump-Diffusion
    - Variance Gamma
    - NIG
    - Heston (with modifications)
    """

    def __init__(self, risk_free_rate: float = 0.0, N: int = 128, L: float = 10.0):
        """
        Initialize COS pricer

        Args:
            risk_free_rate: Risk-free rate for discounting
            N: Number of Fourier terms (power of 2, typically 128-512)
            L: Truncation range for log-price (typically 8-12)
        """
        super().__init__(name="COS")
        self.risk_free_rate = risk_free_rate
        self.N = N
        self.L = L

    def price(
        self,
        derivative,
        process,
        X0: Union[float, np.ndarray],
        **kwargs
    ) -> PricingResult:
        """
        Price European option using COS method

        Args:
            derivative: European call or put
            process: Process with characteristic_function() method
            X0: Initial value(s)
            **kwargs: Additional parameters

        Returns:
            PricingResult with price
        """
        start_time = time.time()

        # Validate process has characteristic function
        if not hasattr(process, 'characteristic_function'):
            raise ValueError("Process must have characteristic_function() method for COS pricing")

        S0 = float(X0) if np.isscalar(X0) else float(X0[0])
        K = derivative.strike
        T = derivative.maturity
        r = self.risk_free_rate

        # Determine option type
        is_call = "call" in derivative.contract_type

        # COS method
        price = self._cos_price(process, S0, K, T, r, X0, is_call)

        computation_time = time.time() - start_time

        metadata = {
            "method": "COS",
            "N_terms": self.N,
            "L_truncation": self.L,
            "contract_type": derivative.contract_type,
        }

        return PricingResult(
            price=price,
            std_error=0.0,  # Deterministic method
            computation_time=computation_time,
            metadata=metadata,
        )

    def _cos_price(
        self, process, S0: float, K: float, T: float, r: float, X0, is_call: bool
    ) -> float:
        """
        COS method implementation

        The method expands the risk-neutral density in a Fourier-cosine series
        and computes option value via efficient summation.
        """
        # Truncation range [a, b] for log-price
        # Typically based on cumulants, here we use simple approximation
        log_S0 = np.log(S0)

        # Estimate range based on variance (simplified)
        if hasattr(process, 'variance'):
            try:
                var = process.variance(X0, T)
                var_val = var[0] if isinstance(var, np.ndarray) else var
                c1 = np.log(S0) + (r - 0.5 * var_val / T) * T
                c2 = var_val
            except:
                # Fallback
                c1 = np.log(S0) + r * T
                c2 = 0.25 * T  # Assume 50% annual vol
        else:
            # Fallback for processes without variance
            c1 = np.log(S0) + r * T
            c2 = 0.25 * T

        a = c1 - self.L * np.sqrt(c2)
        b = c1 + self.L * np.sqrt(c2)

        # Fourier coefficients for payoff
        k = np.arange(self.N)
        k_pi_over_ba = k * np.pi / (b - a)

        if is_call:
            # Call option payoff coefficients
            chi_k = self._chi_k_call(k, a, b, K)
            psi_k = self._psi_k_call(k, a, b, K)
        else:
            # Put option payoff coefficients
            chi_k = self._chi_k_put(k, a, b, K)
            psi_k = self._psi_k_put(k, a, b, K)

        # Characteristic function evaluations
        u = k_pi_over_ba
        char_vals = np.zeros(self.N, dtype=complex)

        for i, u_val in enumerate(u):
            try:
                char_vals[i] = process.characteristic_function(u_val, X0, T)
            except:
                # Some processes may not support complex characteristic functions
                char_vals[i] = 0.0 + 0.0j

        # Adjust for integration range
        char_vals = char_vals * np.exp(-1j * u * a)

        # COS formula
        U_k = 2 / (b - a) * np.real(char_vals * (chi_k - psi_k))
        U_k[0] = U_k[0] / 2  # k=0 term has weight 1/2

        # Option value
        price = np.exp(-r * T) * K * np.sum(U_k)

        return price

    def _chi_k_call(self, k: np.ndarray, a: float, b: float, K: float) -> np.ndarray:
        """Chi coefficients for call option"""
        log_K = np.log(K)

        if log_K < a or log_K > b:
            # Strike outside truncation range
            if log_K < a:
                return 1.0 - self._psi_k(k, a, b, a)
            else:
                return 0.0

        chi = 1.0 / (1.0 + (k * np.pi / (b - a))**2) * (
            np.cos(k * np.pi * (log_K - a) / (b - a)) * np.exp(log_K)
            - self._psi_k(k, a, b, log_K)
            + k * np.pi / (b - a) * np.sin(k * np.pi * (log_K - a) / (b - a)) * np.exp(log_K)
        )

        # Handle k=0 case
        chi = np.where(k == 0, b - log_K, chi)

        return chi

    def _psi_k_call(self, k: np.ndarray, a: float, b: float, K: float) -> np.ndarray:
        """Psi coefficients for call option"""
        log_K = np.log(K)

        if log_K < a:
            return b - a
        elif log_K > b:
            return 0.0

        return self._psi_k(k, a, b, log_K) - self._psi_k(k, a, b, a)

    def _chi_k_put(self, k: np.ndarray, a: float, b: float, K: float) -> np.ndarray:
        """Chi coefficients for put option"""
        log_K = np.log(K)

        if log_K < a:
            return 0.0
        elif log_K > b:
            return -1.0 + self._psi_k(k, a, b, b)

        chi = -1.0 / (1.0 + (k * np.pi / (b - a))**2) * (
            np.cos(k * np.pi * (log_K - a) / (b - a)) * np.exp(log_K)
            - self._psi_k(k, a, b, log_K)
            + k * np.pi / (b - a) * np.sin(k * np.pi * (log_K - a) / (b - a)) * np.exp(log_K)
        )

        # Handle k=0 case
        chi = np.where(k == 0, -log_K + a, chi)

        return chi

    def _psi_k_put(self, k: np.ndarray, a: float, b: float, K: float) -> np.ndarray:
        """Psi coefficients for put option"""
        log_K = np.log(K)

        if log_K < a:
            return 0.0
        elif log_K > b:
            return b - a

        return self._psi_k(k, a, b, b) - self._psi_k(k, a, b, log_K)

    def _psi_k(self, k: np.ndarray, a: float, b: float, x: float) -> np.ndarray:
        """Helper function for Fourier coefficients"""
        psi = (b - a) / (k * np.pi) * np.sin(k * np.pi * (x - a) / (b - a))

        # Handle k=0 case
        psi = np.where(k == 0, x - a, psi)

        return psi


class CarrMadanPricer(Pricer):
    """
    Carr-Madan FFT method for option pricing

    Fast Fourier Transform (FFT) method for computing option prices
    across a range of strikes simultaneously.

    Reference:
    - Carr & Madan (1999) "Option Valuation Using the Fast Fourier Transform"

    Particularly efficient when pricing many strikes at once.
    """

    def __init__(self, risk_free_rate: float = 0.0, N: int = 4096, alpha: float = 1.5):
        """
        Initialize Carr-Madan pricer

        Args:
            risk_free_rate: Risk-free rate
            N: Number of FFT points (power of 2)
            alpha: Dampening factor (typically 1.0-2.0 for calls)
        """
        super().__init__(name="CarrMadan")
        self.risk_free_rate = risk_free_rate
        self.N = N
        self.alpha = alpha

    def price(
        self,
        derivative,
        process,
        X0: Union[float, np.ndarray],
        **kwargs
    ) -> PricingResult:
        """
        Price European option using Carr-Madan FFT

        Args:
            derivative: European option
            process: Process with characteristic_function
            X0: Initial value
            **kwargs: Additional parameters

        Returns:
            PricingResult with price
        """
        start_time = time.time()

        if not hasattr(process, 'characteristic_function'):
            raise ValueError("Process must have characteristic_function() method")

        S0 = float(X0) if np.isscalar(X0) else float(X0[0])
        K = derivative.strike
        T = derivative.maturity
        r = self.risk_free_rate

        is_call = "call" in derivative.contract_type

        # FFT grid
        eta = 0.25  # Grid spacing in log-strike
        lambda_val = 2 * np.pi / (self.N * eta)

        # Grid points
        v = np.arange(self.N) * eta
        k_u = -self.N * lambda_val / 2 + np.arange(self.N) * lambda_val

        # Characteristic function evaluations
        char_vals = np.zeros(self.N, dtype=complex)
        for i, v_i in enumerate(v):
            u = v_i - (self.alpha + 1) * 1j
            try:
                char_vals[i] = process.characteristic_function(u, X0, T)
            except:
                char_vals[i] = 0.0 + 0.0j

        # Modified characteristic function
        psi_vals = np.exp(-r * T) * char_vals / (self.alpha**2 + self.alpha - v**2 + 1j * v * (2 * self.alpha + 1))

        # Simpson's rule weights
        weights = np.ones(self.N)
        weights[0] = 0.5
        weights[-1] = 0.5
        weights[1::2] = 4
        weights[2::2] = 2
        weights = weights * eta / 3

        # FFT
        fft_input = np.exp(1j * v * np.log(S0)) * psi_vals * weights
        fft_output = np.fft.fft(fft_input)

        # Extract price for strike K
        log_K = np.log(K)
        strikes = np.exp(k_u)

        # Interpolate to find price at strike K
        prices = np.real(fft_output * np.exp(-self.alpha * k_u)) / np.pi

        # Find closest strike
        idx = np.argmin(np.abs(strikes - K))
        price = prices[idx]

        computation_time = time.time() - start_time

        metadata = {
            "method": "Carr-Madan FFT",
            "N_points": self.N,
            "alpha": self.alpha,
        }

        return PricingResult(
            price=price,
            std_error=0.0,
            computation_time=computation_time,
            metadata=metadata,
        )
