"""
Vanilla (European-style) derivatives

author: Yunian Pan
email: yp1170@nyu.edu
"""
import numpy as np
from .base import PathIndependentDerivative


class EuropeanCall(PathIndependentDerivative):
    """European call option"""

    def __init__(self, strike: float, maturity: float):
        super().__init__(maturity)
        self.strike = float(strike)

    def payoff(self, S_T: np.ndarray) -> np.ndarray:
        """Max(S_T - K, 0)"""
        return np.maximum(S_T - self.strike, 0.0)

    @property
    def contract_type(self) -> str:
        return "european_call"

    def __repr__(self) -> str:
        return f"EuropeanCall(K={self.strike}, T={self.maturity})"


class EuropeanPut(PathIndependentDerivative):
    """European put option"""

    def __init__(self, strike: float, maturity: float):
        super().__init__(maturity)
        self.strike = float(strike)

    def payoff(self, S_T: np.ndarray) -> np.ndarray:
        """Max(K - S_T, 0)"""
        return np.maximum(self.strike - S_T, 0.0)

    @property
    def contract_type(self) -> str:
        return "european_put"

    def __repr__(self) -> str:
        return f"EuropeanPut(K={self.strike}, T={self.maturity})"


class DigitalCall(PathIndependentDerivative):
    """Cash-or-nothing digital call"""

    def __init__(self, strike: float, maturity: float, payout: float = 1.0):
        super().__init__(maturity)
        self.strike = float(strike)
        self.payout = float(payout)

    def payoff(self, S_T: np.ndarray) -> np.ndarray:
        """Payout if S_T > K, else 0"""
        return self.payout * (S_T > self.strike).astype(float)

    @property
    def contract_type(self) -> str:
        return "digital_call"

    def __repr__(self) -> str:
        return f"DigitalCall(K={self.strike}, payout={self.payout}, T={self.maturity})"


class DigitalPut(PathIndependentDerivative):
    """Cash-or-nothing digital put"""

    def __init__(self, strike: float, maturity: float, payout: float = 1.0):
        super().__init__(maturity)
        self.strike = float(strike)
        self.payout = float(payout)

    def payoff(self, S_T: np.ndarray) -> np.ndarray:
        """Payout if S_T < K, else 0"""
        return self.payout * (S_T < self.strike).astype(float)

    @property
    def contract_type(self) -> str:
        return "digital_put"

    def __repr__(self) -> str:
        return f"DigitalPut(K={self.strike}, payout={self.payout}, T={self.maturity})"


class Straddle(PathIndependentDerivative):
    """Long call + long put at same strike"""

    def __init__(self, strike: float, maturity: float):
        super().__init__(maturity)
        self.strike = float(strike)

    def payoff(self, S_T: np.ndarray) -> np.ndarray:
        """Call payoff + put payoff = |S_T - K|"""
        return np.abs(S_T - self.strike)

    @property
    def contract_type(self) -> str:
        return "straddle"

    def __repr__(self) -> str:
        return f"Straddle(K={self.strike}, T={self.maturity})"


class Strangle(PathIndependentDerivative):
    """Long call at K_call + long put at K_put (K_put < K_call)"""

    def __init__(self, strike_put: float, strike_call: float, maturity: float):
        super().__init__(maturity)
        self.strike_put = float(strike_put)
        self.strike_call = float(strike_call)

        if self.strike_put >= self.strike_call:
            raise ValueError("strike_put must be less than strike_call")

    def payoff(self, S_T: np.ndarray) -> np.ndarray:
        """Max(S_T - K_call, 0) + Max(K_put - S_T, 0)"""
        call_payoff = np.maximum(S_T - self.strike_call, 0.0)
        put_payoff = np.maximum(self.strike_put - S_T, 0.0)
        return call_payoff + put_payoff

    @property
    def contract_type(self) -> str:
        return "strangle"

    def __repr__(self) -> str:
        return f"Strangle(K_put={self.strike_put}, K_call={self.strike_call}, T={self.maturity})"


class ButterflySpread(PathIndependentDerivative):
    """Butterfly spread: Long 1 call at K1, short 2 calls at K2, long 1 call at K3"""

    def __init__(self, strike_low: float, strike_mid: float, strike_high: float, maturity: float):
        super().__init__(maturity)
        self.strike_low = float(strike_low)
        self.strike_mid = float(strike_mid)
        self.strike_high = float(strike_high)

        if not (self.strike_low < self.strike_mid < self.strike_high):
            raise ValueError("Strikes must be ordered: K_low < K_mid < K_high")

    def payoff(self, S_T: np.ndarray) -> np.ndarray:
        """
        +1 Call(K1) - 2 Call(K2) + 1 Call(K3)
        """
        payoff_low = np.maximum(S_T - self.strike_low, 0.0)
        payoff_mid = np.maximum(S_T - self.strike_mid, 0.0)
        payoff_high = np.maximum(S_T - self.strike_high, 0.0)

        return payoff_low - 2 * payoff_mid + payoff_high

    @property
    def contract_type(self) -> str:
        return "butterfly_spread"

    def __repr__(self) -> str:
        return f"ButterflySpread(K=[{self.strike_low}, {self.strike_mid}, {self.strike_high}], T={self.maturity})"
