"""
Utility Functions for Portfolio Optimization and Market Making

Implements common utility functions:
- CARA (Constant Absolute Risk Aversion)
- CRRA (Constant Relative Risk Aversion) - future
- Mean-variance - future

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

import numpy as np
from typing import Union


class CARAUtility:
    """
    Constant Absolute Risk Aversion (CARA) utility function.

    U(W) = -exp(-γ * W)  for γ > 0

    where:
        W: wealth
        γ: risk aversion coefficient

    Properties:
        - Absolute risk aversion: A(W) = -U''(W)/U'(W) = γ (constant)
        - More risk averse as γ increases
        - γ → 0: risk neutral
        - γ → ∞: infinitely risk averse

    Common in market making literature (Avellaneda-Stoikov, etc.) because:
        - Tractable optimal controls
        - Closed-form solutions for inventory management
        - Certainty equivalent has simple form

    Usage:
        util = CARAUtility(gamma=0.01)

        # Utility of $1000 wealth
        u = util.utility(1000.0)  # -exp(-10) ≈ -4.5e-5

        # Compare two wealth distributions
        wealth_A = 1000.0
        wealth_B = 900.0 + 200 * np.random.randn()  # Risky

        E_U_B = np.mean([util.utility(w) for w in wealth_B])
        CE_B = util.certainty_equivalent(E_U_B)

        if CE_B > wealth_A:
            print("Prefer risky B despite lower expected value")
    """

    def __init__(self, gamma: float):
        """
        Initialize CARA utility.

        Args:
            gamma: Risk aversion coefficient (γ > 0)
                   Typical values:
                   - 0.001-0.01: Low risk aversion (aggressive market maker)
                   - 0.01-0.1: Moderate risk aversion
                   - 0.1-1.0: High risk aversion (conservative)

        Raises:
            ValueError: If gamma <= 0
        """
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")

        self.gamma = gamma

    def utility(self, wealth: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute CARA utility of wealth.

        U(W) = -exp(-γ * W)

        Args:
            wealth: Wealth value or array of values

        Returns:
            Utility U(W) ∈ (-∞, 0]
            - Higher (closer to 0) is better
            - U(W) → 0 as W → ∞
            - U(W) → -∞ as W → -∞

        Examples:
            >>> util = CARAUtility(gamma=0.01)
            >>> util.utility(100.0)
            -0.36787944117144233  # -exp(-1)
            >>> util.utility([100.0, 200.0])
            array([-0.36787944, -0.13533528])
        """
        return -np.exp(-self.gamma * wealth)

    def certainty_equivalent(
        self,
        expected_utility: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Compute certainty equivalent wealth from expected utility.

        CE: The certain wealth that gives same utility as risky wealth
        U(CE) = E[U(W)] → CE = -(1/γ) * log(-E[U(W)])

        Args:
            expected_utility: E[U(W)] where W is risky wealth
                              Must be negative (since U < 0 always)

        Returns:
            Certainty equivalent wealth

        Notes:
            - CE < E[W] for risk-averse agents (γ > 0)
            - Difference E[W] - CE is risk premium
            - For normal W ~ N(μ, σ²): CE ≈ μ - (γ/2) * σ²

        Examples:
            >>> util = CARAUtility(gamma=0.01)
            >>> # Risky wealth: 50% chance of 100, 50% chance of 200
            >>> E_U = 0.5 * util.utility(100) + 0.5 * util.utility(200)
            >>> CE = util.certainty_equivalent(E_U)
            >>> CE  # ≈ 138.6 (less than E[W] = 150 due to risk)
        """
        if np.any(expected_utility > 0):
            raise ValueError(
                "expected_utility must be negative (CARA utility is always negative)"
            )

        return -np.log(-expected_utility) / self.gamma

    def certainty_equivalent_normal(
        self,
        mean: float,
        variance: float,
    ) -> float:
        """
        Compute certainty equivalent for normally distributed wealth.

        For W ~ N(μ, σ²), closed-form CE:
            CE = μ - (γ/2) * σ²

        This is exact for CARA utility with normal wealth.

        Args:
            mean: Expected wealth E[W]
            variance: Variance of wealth Var[W]

        Returns:
            Certainty equivalent

        Notes:
            - Risk premium = μ - CE = (γ/2) * σ²
            - Risk premium increases with γ and σ²
            - Independent of mean (constant absolute risk aversion)

        Examples:
            >>> util = CARAUtility(gamma=0.01)
            >>> # Wealth: mean 1000, std 100
            >>> CE = util.certainty_equivalent_normal(1000, 100**2)
            >>> CE  # = 1000 - 0.01/2 * 10000 = 950
            >>> # Risk premium = 50 (willing to pay $50 to avoid risk)
        """
        return mean - (self.gamma / 2) * variance

    def risk_premium_normal(self, variance: float) -> float:
        """
        Compute risk premium for normally distributed wealth.

        Risk Premium = E[W] - CE = (γ/2) * σ²

        Args:
            variance: Variance of wealth

        Returns:
            Risk premium (always non-negative for γ > 0)
        """
        return (self.gamma / 2) * variance

    def marginal_utility(
        self,
        wealth: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Compute marginal utility U'(W).

        U'(W) = γ * exp(-γ * W) > 0

        Args:
            wealth: Wealth value or array

        Returns:
            Marginal utility (always positive)

        Notes:
            - U'(W) > 0: utility increases with wealth
            - U'(W) decreases as W increases (diminishing marginal utility)
        """
        return self.gamma * np.exp(-self.gamma * wealth)

    def absolute_risk_aversion(self, wealth: Union[float, np.ndarray]) -> float:
        """
        Compute absolute risk aversion coefficient.

        A(W) = -U''(W) / U'(W) = γ  (constant)

        Args:
            wealth: Wealth value (not used, included for interface consistency)

        Returns:
            Absolute risk aversion = γ
        """
        return self.gamma

    def __repr__(self) -> str:
        return f"CARAUtility(gamma={self.gamma})"

    def __call__(self, wealth: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Allow using instance as function: util(W) same as util.utility(W)."""
        return self.utility(wealth)
