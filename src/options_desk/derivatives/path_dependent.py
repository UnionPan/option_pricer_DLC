"""
Path-dependent derivatives

author: Yunian Pan
email: yp1170@nyu.edu
"""
import numpy as np
from .base import PathDependentDerivative


class AsianOption(PathDependentDerivative):
    """Asian option with arithmetic or geometric average"""

    def __init__(self, strike: float, maturity: float,
                 call_or_put: str = "call", average_type: str = "arithmetic"):
        super().__init__(maturity)
        self.strike = float(strike)
        self.call_or_put = call_or_put.lower()
        self.average_type = average_type.lower()

        if self.call_or_put not in ["call", "put"]:
            raise ValueError(f"call_or_put must be 'call' or 'put', got {call_or_put}")
        if self.average_type not in ["arithmetic", "geometric"]:
            raise ValueError(f"average_type must be 'arithmetic' or 'geometric'")

    def payoff(self, path: np.ndarray) -> np.ndarray:
        """
        Args:
            path: Shape (n_steps, n_paths, 1) or (n_steps, n_paths)
        """
        # Extract prices
        if path.ndim == 3:
            prices = path[:, :, 0]
        else:
            prices = path

        # Calculate average (path logic - same for call and put)
        if self.average_type == "arithmetic":
            avg_price = prices.mean(axis=0)
        else:  # geometric
            avg_price = np.exp(np.log(prices).mean(axis=0))

        # Apply call or put payoff
        if self.call_or_put == "call":
            return np.maximum(avg_price - self.strike, 0.0)
        else:  # put
            return np.maximum(self.strike - avg_price, 0.0)

    @property
    def contract_type(self) -> str:
        return f"asian_{self.call_or_put}_{self.average_type}"

    def __repr__(self) -> str:
        return f"AsianOption({self.call_or_put}, K={self.strike}, avg={self.average_type}, T={self.maturity})"


class BarrierOption(PathDependentDerivative):
    """Barrier option (knock-in/knock-out, up/down)"""

    def __init__(self, strike: float, barrier: float, maturity: float,
                 call_or_put: str = "call", barrier_type: str = "up_and_out"):
        super().__init__(maturity)
        self.strike = float(strike)
        self.barrier = float(barrier)
        self.call_or_put = call_or_put.lower()
        self.barrier_type = barrier_type.lower()

        valid_barriers = ["up_and_out", "up_and_in", "down_and_out", "down_and_in"]
        if self.barrier_type not in valid_barriers:
            raise ValueError(f"barrier_type must be one of {valid_barriers}")

    def payoff(self, path: np.ndarray) -> np.ndarray:
        if path.ndim == 3:
            prices = path[:, :, 0]
        else:
            prices = path

        S_T = prices[-1]

        # European payoff
        if self.call_or_put == "call":
            european_payoff = np.maximum(S_T - self.strike, 0.0)
        else:  # put
            european_payoff = np.maximum(self.strike - S_T, 0.0)

        # Barrier logic (path logic - same for call and put!)
        if self.barrier_type == "up_and_out":
            knocked_out = (prices >= self.barrier).any(axis=0)
            return european_payoff * (~knocked_out)

        elif self.barrier_type == "up_and_in":
            knocked_in = (prices >= self.barrier).any(axis=0)
            return european_payoff * knocked_in

        elif self.barrier_type == "down_and_out":
            knocked_out = (prices <= self.barrier).any(axis=0)
            return european_payoff * (~knocked_out)

        else:  # down_and_in
            knocked_in = (prices <= self.barrier).any(axis=0)
            return european_payoff * knocked_in

    @property
    def contract_type(self) -> str:
        return f"barrier_{self.call_or_put}_{self.barrier_type}"

    def __repr__(self) -> str:
        return f"BarrierOption({self.call_or_put}, K={self.strike}, B={self.barrier}, type={self.barrier_type}, T={self.maturity})"


class LookbackOption(PathDependentDerivative):
    """Lookback option (floating or fixed strike)"""

    def __init__(self, maturity: float, call_or_put: str = "call",
                 strike: float = None, lookback_type: str = "floating"):
        """
        Args:
            strike: For fixed strike lookback. None for floating strike.
            lookback_type: "floating" or "fixed"
        """
        super().__init__(maturity)
        self.call_or_put = call_or_put.lower()
        self.strike = float(strike) if strike is not None else None
        self.lookback_type = lookback_type.lower()

        if self.lookback_type not in ["floating", "fixed"]:
            raise ValueError(f"lookback_type must be 'floating' or 'fixed'")

    def payoff(self, path: np.ndarray) -> np.ndarray:
        if path.ndim == 3:
            prices = path[:, :, 0]
        else:
            prices = path

        S_T = prices[-1]
        S_max = prices.max(axis=0)
        S_min = prices.min(axis=0)

        if self.lookback_type == "floating":
            # Floating strike lookback
            if self.call_or_put == "call":
                # S_T - min(S): buy at lowest, sell at final
                return S_T - S_min
            else:  # put
                # max(S) - S_T: buy at final, sell at highest
                return S_max - S_T

        else:  # fixed strike
            if self.strike is None:
                raise ValueError("Fixed strike lookback requires strike parameter")

            if self.call_or_put == "call":
                # max(S) - K: best possible call payoff
                return np.maximum(S_max - self.strike, 0.0)
            else:  # put
                # K - min(S): best possible put payoff
                return np.maximum(self.strike - S_min, 0.0)

    @property
    def contract_type(self) -> str:
        return f"lookback_{self.call_or_put}_{self.lookback_type}"

    def __repr__(self) -> str:
        if self.lookback_type == "floating":
            return f"LookbackOption({self.call_or_put}, floating, T={self.maturity})"
        else:
            return f"LookbackOption({self.call_or_put}, fixed, K={self.strike}, T={self.maturity})"


class CliquetOption(PathDependentDerivative):
    """
    Cliquet (ratchet) option

    Locks in gains at predetermined reset dates.
    Payoff = sum of capped returns over each period.
    """

    def __init__(self, maturity: float, reset_dates: np.ndarray,
                 local_floor: float = 0.0, local_cap: float = None,
                 global_floor: float = None, global_cap: float = None):
        """
        Args:
            reset_dates: Array of reset dates (fraction of maturity)
            local_floor: Minimum return per period (e.g., 0.0 for no negative returns)
            local_cap: Maximum return per period (e.g., 0.1 for 10% cap)
            global_floor: Minimum total return
            global_cap: Maximum total return
        """
        super().__init__(maturity)
        self.reset_dates = np.array(reset_dates)
        self.local_floor = float(local_floor) if local_floor is not None else None
        self.local_cap = float(local_cap) if local_cap is not None else None
        self.global_floor = float(global_floor) if global_floor is not None else None
        self.global_cap = float(global_cap) if global_cap is not None else None

    def payoff(self, path: np.ndarray) -> np.ndarray:
        """
        Sum of capped returns over reset periods
        """
        if path.ndim == 3:
            prices = path[:, :, 0]
        else:
            prices = path

        n_steps, n_paths = prices.shape

        # Find reset indices
        reset_indices = (self.reset_dates * (n_steps - 1)).astype(int)
        reset_indices = np.concatenate([[0], reset_indices, [n_steps - 1]])
        reset_indices = np.unique(reset_indices)

        # Calculate returns over each period
        total_return = np.zeros(n_paths)

        for i in range(len(reset_indices) - 1):
            start_idx = reset_indices[i]
            end_idx = reset_indices[i + 1]

            period_return = (prices[end_idx] - prices[start_idx]) / prices[start_idx]

            # Apply local floor and cap
            if self.local_floor is not None:
                period_return = np.maximum(period_return, self.local_floor)
            if self.local_cap is not None:
                period_return = np.minimum(period_return, self.local_cap)

            total_return += period_return

        # Apply global floor and cap
        if self.global_floor is not None:
            total_return = np.maximum(total_return, self.global_floor)
        if self.global_cap is not None:
            total_return = np.minimum(total_return, self.global_cap)

        # Return payoff (usually notional * total_return, here notional=1)
        return total_return

    @property
    def contract_type(self) -> str:
        return "cliquet"

    def __repr__(self) -> str:
        return f"CliquetOption(resets={len(self.reset_dates)}, T={self.maturity})"
