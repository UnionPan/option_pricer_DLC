"""
Core option contract data structures.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class OptionType(Enum):
    """Option type enumeration."""

    CALL = "call"
    PUT = "put"


class ExerciseStyle(Enum):
    """Exercise style enumeration."""

    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass
class OptionContract:
    """
    Represents a single option contract.

    Attributes:
        symbol: Underlying symbol
        strike: Strike price
        expiry: Expiration date
        option_type: Call or Put
        exercise_style: European or American
        contract_size: Number of shares per contract (default 100)
    """

    symbol: str
    strike: float
    expiry: datetime
    option_type: OptionType
    exercise_style: ExerciseStyle = ExerciseStyle.AMERICAN
    contract_size: int = 100

    def __post_init__(self):
        """Validate option contract parameters."""
        if self.strike <= 0:
            raise ValueError("Strike price must be positive")
        if self.contract_size <= 0:
            raise ValueError("Contract size must be positive")

    @property
    def is_call(self) -> bool:
        """Check if option is a call."""
        return self.option_type == OptionType.CALL

    @property
    def is_put(self) -> bool:
        """Check if option is a put."""
        return self.option_type == OptionType.PUT

    def time_to_expiry(self, current_date: datetime) -> float:
        """
        Calculate time to expiry in years.

        Args:
            current_date: Current date/time

        Returns:
            Time to expiry in years
        """
        time_diff = self.expiry - current_date
        return time_diff.total_seconds() / (365.25 * 24 * 3600)


@dataclass
class OptionQuote:
    """
    Represents market quote for an option.

    Attributes:
        contract: The option contract
        bid: Bid price
        ask: Ask price
        last: Last traded price
        volume: Trading volume
        open_interest: Open interest
        implied_vol: Implied volatility (if available)
    """

    contract: OptionContract
    bid: float
    ask: float
    last: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    implied_vol: Optional[float] = None

    @property
    def mid_price(self) -> float:
        """Calculate mid price from bid and ask."""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        """Calculate bid-ask spread as percentage of mid."""
        mid = self.mid_price
        if mid == 0:
            return 0
        return self.spread / mid
