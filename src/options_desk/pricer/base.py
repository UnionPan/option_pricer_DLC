"""
Base classes for derivative pricing

author: Yunian Pan
email: yp1170@nyu.edu
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class PricingResult:
    """
    Container for pricing results with statistics

    Attributes:
        price: Estimated option price
        std_error: Standard error of the estimate
        confidence_interval: (lower, upper) 95% confidence interval
        n_paths: Number of Monte Carlo paths used
        computation_time: Time taken for pricing (seconds)
        greeks: Dictionary of Greeks (delta, gamma, vega, etc.)
        metadata: Additional pricing information
    """
    price: float
    std_error: Optional[float] = None
    confidence_interval: Optional[tuple] = None
    n_paths: Optional[int] = None
    computation_time: Optional[float] = None
    greeks: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:
        lines = [
            f"Price: {self.price:.6f}",
        ]

        if self.std_error is not None:
            lines.append(f"Std Error: {self.std_error:.6f}")

        if self.confidence_interval is not None:
            lower, upper = self.confidence_interval
            lines.append(f"95% CI: [{lower:.6f}, {upper:.6f}]")

        if self.n_paths is not None:
            lines.append(f"Paths: {self.n_paths:,}")

        if self.computation_time is not None:
            lines.append(f"Time: {self.computation_time:.4f}s")

        if self.greeks:
            lines.append("Greeks:")
            for greek, value in self.greeks.items():
                lines.append(f"  {greek}: {value:.6f}")

        return "\n".join(lines)


class Pricer(ABC):
    """
    Abstract base class for derivative pricers

    All pricing methods should inherit from this class and implement
    the price() method for their specific approach (Monte Carlo, analytical, etc.)
    """

    def __init__(self, name: str = "Pricer"):
        self.name = name

    @abstractmethod
    def price(self, derivative, process, **kwargs) -> PricingResult:
        """
        Price a derivative contract under a stochastic process

        Args:
            derivative: Derivative contract (from derivatives module)
            process: Stochastic process (from processes module)
            **kwargs: Additional pricing parameters

        Returns:
            PricingResult object with price and statistics
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
