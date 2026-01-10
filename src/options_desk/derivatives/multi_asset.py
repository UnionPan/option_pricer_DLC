"""
Multi-asset derivatives

author: Yunian Pan
email: yp1170@nyu.edu
"""
import numpy as np
from .base import PathIndependentDerivative


class BasketOption(PathIndependentDerivative):
    """Option on weighted basket of multiple assets"""

    def __init__(self, strike: float, maturity: float,
                 call_or_put: str = "call", weights: np.ndarray = None):
        """
        Args:
            strike: Strike price
            maturity: Time to maturity
            call_or_put: "call" or "put"
            weights: Basket weights (must sum to 1). If None, equal weights.
        """
        super().__init__(maturity)
        self.strike = float(strike)
        self.call_or_put = call_or_put.lower()
        self.weights = weights

        if self.call_or_put not in ["call", "put"]:
            raise ValueError(f"call_or_put must be 'call' or 'put', got {call_or_put}")

    def payoff(self, S_T: np.ndarray) -> np.ndarray:
        """
        Payoff on weighted basket

        Args:
            S_T: Shape (n_paths, n_assets) or (n_paths,) for single asset
        """
        if S_T.ndim == 1:
            S_T = S_T.reshape(-1, 1)

        # Calculate basket value
        if self.weights is None:
            basket_value = S_T.mean(axis=1)
        else:
            weights = np.array(self.weights)
            if not np.isclose(weights.sum(), 1.0):
                raise ValueError(f"Weights must sum to 1, got {weights.sum()}")
            basket_value = (S_T * weights).sum(axis=1)

        # Apply call or put payoff
        if self.call_or_put == "call":
            return np.maximum(basket_value - self.strike, 0.0)
        else:  # put
            return np.maximum(self.strike - basket_value, 0.0)

    @property
    def contract_type(self) -> str:
        return f"basket_{self.call_or_put}"

    def __repr__(self) -> str:
        weights_str = "equal" if self.weights is None else f"{len(self.weights)} assets"
        return f"BasketOption({self.call_or_put}, K={self.strike}, weights={weights_str}, T={self.maturity})"


class SpreadOption(PathIndependentDerivative):
    """Option on spread between two assets (S1 - S2)"""

    def __init__(self, strike: float, maturity: float, call_or_put: str = "call"):
        """
        Args:
            strike: Strike on the spread
            maturity: Time to maturity
            call_or_put: "call" or "put"
        """
        super().__init__(maturity)
        self.strike = float(strike)
        self.call_or_put = call_or_put.lower()

        if self.call_or_put not in ["call", "put"]:
            raise ValueError(f"call_or_put must be 'call' or 'put', got {call_or_put}")

    def payoff(self, S_T: np.ndarray) -> np.ndarray:
        """
        Payoff on spread between two assets

        Args:
            S_T: Shape (n_paths, 2) - exactly 2 assets required
        """
        if S_T.shape[1] != 2:
            raise ValueError(f"SpreadOption requires exactly 2 assets, got {S_T.shape[1]}")

        spread = S_T[:, 0] - S_T[:, 1]

        # Apply call or put payoff
        if self.call_or_put == "call":
            return np.maximum(spread - self.strike, 0.0)
        else:  # put
            return np.maximum(self.strike - spread, 0.0)

    @property
    def contract_type(self) -> str:
        return f"spread_{self.call_or_put}"

    def __repr__(self) -> str:
        return f"SpreadOption({self.call_or_put}, K={self.strike}, T={self.maturity})"


class RainbowOption(PathIndependentDerivative):
    """Option on maximum or minimum of multiple assets (best-of/worst-of)"""

    def __init__(self, strike: float, maturity: float,
                 call_or_put: str = "call", rainbow_type: str = "best_of"):
        """
        Args:
            strike: Strike price
            maturity: Time to maturity
            call_or_put: "call" or "put"
            rainbow_type: "best_of" (max) or "worst_of" (min)
        """
        super().__init__(maturity)
        self.strike = float(strike)
        self.call_or_put = call_or_put.lower()
        self.rainbow_type = rainbow_type.lower()

        if self.call_or_put not in ["call", "put"]:
            raise ValueError(f"call_or_put must be 'call' or 'put', got {call_or_put}")
        if self.rainbow_type not in ["best_of", "worst_of"]:
            raise ValueError(f"rainbow_type must be 'best_of' or 'worst_of', got {rainbow_type}")

    def payoff(self, S_T: np.ndarray) -> np.ndarray:
        """
        Payoff on max or min of assets

        Args:
            S_T: Shape (n_paths, n_assets)
        """
        if S_T.ndim == 1:
            S_T = S_T.reshape(-1, 1)

        # Select asset based on rainbow type
        if self.rainbow_type == "best_of":
            asset_value = S_T.max(axis=1)
        else:  # worst_of
            asset_value = S_T.min(axis=1)

        # Apply call or put payoff
        if self.call_or_put == "call":
            return np.maximum(asset_value - self.strike, 0.0)
        else:  # put
            return np.maximum(self.strike - asset_value, 0.0)

    @property
    def contract_type(self) -> str:
        return f"rainbow_{self.rainbow_type}_{self.call_or_put}"

    def __repr__(self) -> str:
        return f"RainbowOption({self.call_or_put}, {self.rainbow_type}, K={self.strike}, T={self.maturity})"


class ExchangeOption(PathIndependentDerivative):
    """
    Margrabe exchange option: option to exchange one asset for another
    Payoff = max(S1_T - S2_T, 0) or max(S2_T - S1_T, 0)
    """

    def __init__(self, maturity: float, exchange_direction: str = "1_for_2"):
        """
        Args:
            maturity: Time to maturity
            exchange_direction: "1_for_2" (exchange asset 2 for 1) or "2_for_1"
        """
        super().__init__(maturity)
        self.exchange_direction = exchange_direction.lower()

        if self.exchange_direction not in ["1_for_2", "2_for_1"]:
            raise ValueError(f"exchange_direction must be '1_for_2' or '2_for_1'")

    def payoff(self, S_T: np.ndarray) -> np.ndarray:
        """
        Exchange option payoff (strike = 0)

        Args:
            S_T: Shape (n_paths, 2) - exactly 2 assets required
        """
        if S_T.shape[1] != 2:
            raise ValueError(f"ExchangeOption requires exactly 2 assets, got {S_T.shape[1]}")

        if self.exchange_direction == "1_for_2":
            # Exchange asset 2 for asset 1: max(S1 - S2, 0)
            return np.maximum(S_T[:, 0] - S_T[:, 1], 0.0)
        else:  # 2_for_1
            # Exchange asset 1 for asset 2: max(S2 - S1, 0)
            return np.maximum(S_T[:, 1] - S_T[:, 0], 0.0)

    @property
    def contract_type(self) -> str:
        return f"exchange_{self.exchange_direction}"

    def __repr__(self) -> str:
        return f"ExchangeOption({self.exchange_direction}, T={self.maturity})"


class QuantoOption(PathIndependentDerivative):
    """
    Quanto option: foreign asset with domestic currency payoff
    Payoff in domestic currency at fixed exchange rate
    """

    def __init__(self, strike: float, maturity: float,
                 call_or_put: str = "call", quanto_rate: float = 1.0):
        """
        Args:
            strike: Strike in foreign currency
            maturity: Time to maturity
            call_or_put: "call" or "put"
            quanto_rate: Fixed exchange rate for payoff conversion
        """
        super().__init__(maturity)
        self.strike = float(strike)
        self.call_or_put = call_or_put.lower()
        self.quanto_rate = float(quanto_rate)

        if self.call_or_put not in ["call", "put"]:
            raise ValueError(f"call_or_put must be 'call' or 'put'")

    def payoff(self, S_T: np.ndarray) -> np.ndarray:
        """
        Quanto payoff: foreign option payoff converted at fixed rate

        Args:
            S_T: Shape (n_paths,) - foreign asset price
        """
        if S_T.ndim > 1:
            S_T = S_T[:, 0]  # Take first asset if multi-dimensional

        # Calculate foreign currency payoff
        if self.call_or_put == "call":
            foreign_payoff = np.maximum(S_T - self.strike, 0.0)
        else:  # put
            foreign_payoff = np.maximum(self.strike - S_T, 0.0)

        # Convert to domestic currency at fixed rate
        return self.quanto_rate * foreign_payoff

    @property
    def contract_type(self) -> str:
        return f"quanto_{self.call_or_put}"

    def __repr__(self) -> str:
        return f"QuantoOption({self.call_or_put}, K={self.strike}, rate={self.quanto_rate}, T={self.maturity})"
