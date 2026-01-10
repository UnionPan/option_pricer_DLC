"""
Interest rate derivatives

author: Yunian Pan
email: yp1170@nyu.edu
"""
import numpy as np
from .base import Derivative, PathIndependentDerivative, PathDependentDerivative


class Caplet(PathIndependentDerivative):
    """
    Single-period interest rate cap (call on rate)
    Payoff = max(rate_T - strike, 0) * accrual * notional
    """

    def __init__(self, strike: float, maturity: float,
                 notional: float = 1.0, accrual: float = 0.25):
        """
        Args:
            strike: Cap rate (e.g., 0.05 for 5%)
            maturity: Time to maturity
            notional: Notional principal
            accrual: Accrual period (e.g., 0.25 for quarterly)
        """
        super().__init__(maturity)
        self.strike = float(strike)
        self.notional = float(notional)
        self.accrual = float(accrual)

    def payoff(self, rate_T: np.ndarray) -> np.ndarray:
        """
        Caplet payoff

        Args:
            rate_T: Terminal rate, shape (n_paths,)
        """
        if rate_T.ndim > 1:
            rate_T = rate_T[:, 0]  # Take first rate if multi-dimensional

        return self.notional * self.accrual * np.maximum(rate_T - self.strike, 0.0)

    @property
    def contract_type(self) -> str:
        return "caplet"

    def __repr__(self) -> str:
        return f"Caplet(K={self.strike:.4f}, T={self.maturity}, notional={self.notional})"


class Floorlet(PathIndependentDerivative):
    """
    Single-period interest rate floor (put on rate)
    Payoff = max(strike - rate_T, 0) * accrual * notional
    """

    def __init__(self, strike: float, maturity: float,
                 notional: float = 1.0, accrual: float = 0.25):
        """
        Args:
            strike: Floor rate (e.g., 0.02 for 2%)
            maturity: Time to maturity
            notional: Notional principal
            accrual: Accrual period (e.g., 0.25 for quarterly)
        """
        super().__init__(maturity)
        self.strike = float(strike)
        self.notional = float(notional)
        self.accrual = float(accrual)

    def payoff(self, rate_T: np.ndarray) -> np.ndarray:
        """
        Floorlet payoff

        Args:
            rate_T: Terminal rate, shape (n_paths,)
        """
        if rate_T.ndim > 1:
            rate_T = rate_T[:, 0]  # Take first rate if multi-dimensional

        return self.notional * self.accrual * np.maximum(self.strike - rate_T, 0.0)

    @property
    def contract_type(self) -> str:
        return "floorlet"

    def __repr__(self) -> str:
        return f"Floorlet(K={self.strike:.4f}, T={self.maturity}, notional={self.notional})"


class InterestRateSwap(PathDependentDerivative):
    """
    Plain vanilla interest rate swap
    Pay fixed, receive floating (payer swap) or vice versa (receiver swap)
    """

    def __init__(self, fixed_rate: float, maturity: float,
                 notional: float = 1.0, payment_freq: float = 0.25,
                 is_payer: bool = True):
        """
        Args:
            fixed_rate: Fixed rate (e.g., 0.05 for 5%)
            maturity: Swap maturity
            notional: Notional principal
            payment_freq: Payment frequency (0.25 = quarterly)
            is_payer: True for payer swap (pay fixed), False for receiver
        """
        super().__init__(maturity)
        self.fixed_rate = float(fixed_rate)
        self.notional = float(notional)
        self.payment_freq = float(payment_freq)
        self.is_payer = is_payer

    def payoff(self, rate_path: np.ndarray) -> np.ndarray:
        """
        NPV of swap cashflows (simplified, no discounting)

        Args:
            rate_path: Shape (n_steps, n_paths) - floating rate path
        """
        if rate_path.ndim == 3:
            rate_path = rate_path[:, :, 0]  # Extract single rate

        n_steps, n_paths = rate_path.shape

        # Payment dates (approximate based on steps)
        n_payments = int(self.maturity / self.payment_freq)
        payment_indices = np.linspace(0, n_steps - 1, n_payments, dtype=int)

        # Floating leg: sum of floating rates at payment dates
        floating_rates = rate_path[payment_indices, :]  # (n_payments, n_paths)
        floating_leg = (floating_rates * self.payment_freq).sum(axis=0)

        # Fixed leg: fixed rate at all payment dates
        fixed_leg = self.fixed_rate * self.payment_freq * n_payments

        # Payer swap: receive floating - pay fixed
        # Receiver swap: receive fixed - pay floating
        if self.is_payer:
            swap_value = floating_leg - fixed_leg
        else:
            swap_value = fixed_leg - floating_leg

        return self.notional * swap_value

    @property
    def contract_type(self) -> str:
        return "payer_swap" if self.is_payer else "receiver_swap"

    def __repr__(self) -> str:
        swap_type = "payer" if self.is_payer else "receiver"
        return f"InterestRateSwap({swap_type}, fixed={self.fixed_rate:.4f}, T={self.maturity})"


class Swaption(Derivative):
    """
    Option to enter an interest rate swap
    European-style option on a forward-starting swap
    """

    def __init__(self, swap_rate: float, option_maturity: float,
                 swap_maturity: float, notional: float = 1.0,
                 payment_freq: float = 0.25, is_payer: bool = True):
        """
        Args:
            swap_rate: Fixed rate of underlying swap (strike)
            option_maturity: Time to option expiry
            swap_maturity: Maturity of underlying swap (from option expiry)
            notional: Notional principal
            payment_freq: Payment frequency (0.25 = quarterly)
            is_payer: True for payer swaption, False for receiver
        """
        super().__init__(option_maturity)
        self.swap_rate = float(swap_rate)
        self.swap_maturity = float(swap_maturity)
        self.notional = float(notional)
        self.payment_freq = float(payment_freq)
        self.is_payer = is_payer

    def payoff(self, rate_at_expiry: np.ndarray, discount_factor: np.ndarray = None) -> np.ndarray:
        """
        Swaption payoff (simplified)

        At option expiry, the swaption pays the NPV of the underlying swap
        if it's in the money.

        Args:
            rate_at_expiry: Market rate at option expiry, shape (n_paths,)
            discount_factor: Optional discount factors for NPV (if None, no discounting)
        """
        if rate_at_expiry.ndim > 1:
            rate_at_expiry = rate_at_expiry[:, 0]

        # Calculate swap annuity (simplified)
        n_payments = int(self.swap_maturity / self.payment_freq)
        annuity = self.payment_freq * n_payments  # Simplified: ignores discounting

        # Swap value = annuity * (market_rate - swap_rate)
        swap_value = annuity * (rate_at_expiry - self.swap_rate)

        # Swaption payoff
        if self.is_payer:
            # Payer swaption: right to pay fixed at swap_rate
            # In the money when market rate > swap rate
            payoff = np.maximum(swap_value, 0.0)
        else:
            # Receiver swaption: right to receive fixed at swap_rate
            # In the money when market rate < swap rate
            payoff = np.maximum(-swap_value, 0.0)

        return self.notional * payoff

    @property
    def contract_type(self) -> str:
        return "payer_swaption" if self.is_payer else "receiver_swaption"

    def __repr__(self) -> str:
        swaption_type = "payer" if self.is_payer else "receiver"
        return f"Swaption({swaption_type}, K={self.swap_rate:.4f}, T_opt={self.maturity}, T_swap={self.swap_maturity})"


class YieldCurveOption(PathIndependentDerivative):
    """
    Option on yield curve spread between two maturities
    Useful for trading curve steepness/flatness
    """

    def __init__(self, strike: float, maturity: float,
                 call_or_put: str = "call", notional: float = 1.0):
        """
        Args:
            strike: Strike on spread
            maturity: Option maturity
            call_or_put: "call" (bet on steepening) or "put" (bet on flattening)
            notional: Notional principal
        """
        super().__init__(maturity)
        self.strike = float(strike)
        self.call_or_put = call_or_put.lower()
        self.notional = float(notional)

        if self.call_or_put not in ["call", "put"]:
            raise ValueError(f"call_or_put must be 'call' or 'put'")

    def payoff(self, rates_T: np.ndarray) -> np.ndarray:
        """
        Payoff on yield curve spread

        Args:
            rates_T: Shape (n_paths, 2) - [short_rate, long_rate]
        """
        if rates_T.shape[1] != 2:
            raise ValueError(f"YieldCurveOption requires 2 rates, got {rates_T.shape[1]}")

        # Spread = long_rate - short_rate
        spread = rates_T[:, 1] - rates_T[:, 0]

        # Apply call or put payoff
        if self.call_or_put == "call":
            payoff = np.maximum(spread - self.strike, 0.0)
        else:  # put
            payoff = np.maximum(self.strike - spread, 0.0)

        return self.notional * payoff

    @property
    def contract_type(self) -> str:
        return f"yield_curve_{self.call_or_put}"

    def __repr__(self) -> str:
        return f"YieldCurveOption({self.call_or_put}, K={self.strike:.4f}, T={self.maturity})"


class BondOption(PathIndependentDerivative):
    """
    European option on a zero-coupon bond
    Call/put on bond price
    """

    def __init__(self, strike: float, maturity: float,
                 call_or_put: str = "call", bond_maturity: float = None):
        """
        Args:
            strike: Strike price
            maturity: Option maturity
            call_or_put: "call" or "put"
            bond_maturity: Maturity of underlying bond (from option expiry)
        """
        super().__init__(maturity)
        self.strike = float(strike)
        self.call_or_put = call_or_put.lower()
        self.bond_maturity = float(bond_maturity) if bond_maturity else maturity

        if self.call_or_put not in ["call", "put"]:
            raise ValueError(f"call_or_put must be 'call' or 'put'")

    def payoff(self, bond_price_T: np.ndarray) -> np.ndarray:
        """
        Payoff on bond price

        Args:
            bond_price_T: Bond price at option expiry, shape (n_paths,)
        """
        if bond_price_T.ndim > 1:
            bond_price_T = bond_price_T[:, 0]

        # Standard European option payoff
        if self.call_or_put == "call":
            return np.maximum(bond_price_T - self.strike, 0.0)
        else:  # put
            return np.maximum(self.strike - bond_price_T, 0.0)

    @property
    def contract_type(self) -> str:
        return f"bond_{self.call_or_put}"

    def __repr__(self) -> str:
        return f"BondOption({self.call_or_put}, K={self.strike}, T={self.maturity})"
