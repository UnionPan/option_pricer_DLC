"""
American-style derivatives (early exercise allowed)

American options can be exercised at any time before maturity,
unlike European options which can only be exercised at maturity.

Key differences from European:
- Higher value due to early exercise optionality
- No closed-form solution for most cases (except perpetual American)
- Require numerical methods: PDE, LSM, Binomial trees

author: Yunian Pan
email: yp1170@nyu.edu
"""
import numpy as np
from .base import PathDependentDerivative


class AmericanOption(PathDependentDerivative):
    """
    Base class for American options

    American options can be exercised at any time t ∈ [0, T].
    The holder chooses the optimal exercise time τ* to maximize:
        V = E[e^{-rτ} * payoff(S_τ)]

    This creates a free boundary problem - the exercise boundary is unknown.
    """

    def __init__(self, strike: float, maturity: float, call_or_put: str = "call"):
        """
        Initialize American option

        Args:
            strike: Strike price K
            maturity: Time to maturity T
            call_or_put: "call" or "put"
        """
        super().__init__(maturity)
        self.strike = float(strike)
        self.call_or_put = call_or_put.lower()

        if self.call_or_put not in ["call", "put"]:
            raise ValueError(f"call_or_put must be 'call' or 'put', got {call_or_put}")

    def payoff(self, path: np.ndarray) -> np.ndarray:
        """
        For American options, we need the full path to determine optimal exercise

        However, the actual optimal exercise decision depends on the pricing method:
        - PDE methods: Check exercise at each time step during backward induction
        - LSM (Longstaff-Schwartz): Regression-based continuation value estimation
        - Binomial trees: Backward induction on tree

        This payoff method returns the terminal payoff (exercise at maturity).
        The pricer is responsible for checking early exercise.

        Args:
            path: Shape (n_steps, n_paths, dim) or (n_steps, n_paths)

        Returns:
            Terminal payoff (exercise at T), shape (n_paths,)
        """
        # Extract terminal values
        if path.ndim == 3:
            S_T = path[-1, :, 0]
        else:
            S_T = path[-1, :]

        # Standard European payoff at maturity
        if self.call_or_put == "call":
            return np.maximum(S_T - self.strike, 0.0)
        else:  # put
            return np.maximum(self.strike - S_T, 0.0)

    def exercise_value(self, S: np.ndarray) -> np.ndarray:
        """
        Immediate exercise value at stock price S

        This is used by pricing methods to check early exercise:
        V(S,t) = max(continuation_value(S,t), exercise_value(S))

        Args:
            S: Stock price(s), shape (n_paths,) or scalar

        Returns:
            Exercise value, same shape as S
        """
        S = np.atleast_1d(S)

        if self.call_or_put == "call":
            return np.maximum(S - self.strike, 0.0)
        else:  # put
            return np.maximum(self.strike - S, 0.0)

    @property
    def contract_type(self) -> str:
        return f"american_{self.call_or_put}"

    def __repr__(self) -> str:
        return f"AmericanOption({self.call_or_put}, K={self.strike}, T={self.maturity})"


class AmericanCall(AmericanOption):
    """American call option: right to buy at strike K any time before T"""

    def __init__(self, strike: float, maturity: float):
        super().__init__(strike, maturity, call_or_put="call")

    def __repr__(self) -> str:
        return f"AmericanCall(K={self.strike}, T={self.maturity})"


class AmericanPut(AmericanOption):
    """American put option: right to sell at strike K any time before T"""

    def __init__(self, strike: float, maturity: float):
        super().__init__(strike, maturity, call_or_put="put")

    def __repr__(self) -> str:
        return f"AmericanPut(K={self.strike}, T={self.maturity})"


class PerpetualAmericanOption(AmericanOption):
    """
    Perpetual American option (T = ∞)

    Has closed-form solution! This is the only American option with analytical formula.

    Optimal exercise boundary for put:
        S* = K * β / (β - 1)
    where β is the negative root of the fundamental quadratic.

    Value:
        V(S) = {
            K - S                           if S ≤ S*
            (K - S*)(S/S*)^β               if S > S*
        }

    Reference: Merton (1973), "Theory of Rational Option Pricing"
    """

    def __init__(self, strike: float, call_or_put: str = "put"):
        """
        Initialize perpetual American option

        Args:
            strike: Strike price
            call_or_put: "call" or "put"

        Note: Perpetual American call on non-dividend stock is never optimal to exercise
              (worth more alive than dead). Only put and dividend-paying call make sense.
        """
        # Use very large maturity to represent infinity
        super().__init__(strike, maturity=1e6, call_or_put=call_or_put)
        self.is_perpetual = True

    def analytical_price(self, S0: float, r: float, sigma: float, q: float = 0.0) -> float:
        """
        Analytical formula for perpetual American option

        Args:
            S0: Current stock price
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield

        Returns:
            Option value
        """
        if self.call_or_put == "put":
            return self._perpetual_put_price(S0, self.strike, r, sigma, q)
        else:
            if q == 0:
                # Perpetual American call on non-dividend stock = stock price
                # Never optimal to exercise
                return S0
            else:
                return self._perpetual_call_price(S0, self.strike, r, sigma, q)

    def _perpetual_put_price(self, S: float, K: float, r: float, sigma: float, q: float) -> float:
        """Perpetual American put closed-form solution"""
        # Solve fundamental quadratic: ½σ²β(β-1) + (r-q)β - r = 0
        a = 0.5 * sigma**2
        b = r - q - 0.5 * sigma**2
        c = -r

        # Quadratic formula: β = (-b ± √(b²-4ac)) / 2a
        discriminant = b**2 - 4*a*c
        beta_minus = (-b - np.sqrt(discriminant)) / (2*a)  # Negative root

        # Optimal exercise boundary
        S_star = K * beta_minus / (beta_minus - 1)

        # Option value
        if S <= S_star:
            # Immediate exercise
            return K - S
        else:
            # Continuation region
            return (K - S_star) * (S / S_star)**beta_minus

    def _perpetual_call_price(self, S: float, K: float, r: float, sigma: float, q: float) -> float:
        """Perpetual American call closed-form solution (with dividends)"""
        # Solve fundamental quadratic
        a = 0.5 * sigma**2
        b = r - q - 0.5 * sigma**2
        c = -r

        discriminant = b**2 - 4*a*c
        beta_plus = (-b + np.sqrt(discriminant)) / (2*a)  # Positive root

        # Optimal exercise boundary
        S_star = K * beta_plus / (beta_plus - 1)

        # Option value
        if S >= S_star:
            # Immediate exercise
            return S - K
        else:
            # Continuation region
            return (S_star - K) * (S / S_star)**beta_plus

    def optimal_exercise_boundary(self, r: float, sigma: float, q: float = 0.0) -> float:
        """
        Optimal exercise boundary S* for perpetual option

        Args:
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield

        Returns:
            Optimal exercise boundary S*
        """
        a = 0.5 * sigma**2
        b = r - q - 0.5 * sigma**2
        c = -r

        discriminant = b**2 - 4*a*c

        if self.call_or_put == "put":
            beta = (-b - np.sqrt(discriminant)) / (2*a)  # Negative root
        else:
            beta = (-b + np.sqrt(discriminant)) / (2*a)  # Positive root

        return self.strike * beta / (beta - 1)

    def __repr__(self) -> str:
        return f"PerpetualAmerican{self.call_or_put.capitalize()}(K={self.strike})"


class BermudanOption(PathDependentDerivative):
    """
    Bermudan option: can be exercised at discrete dates before maturity

    Interpolates between European (1 exercise date) and American (continuous exercise).

    Examples:
    - Bermudan swaption: exercise on coupon payment dates
    - Quarterly exercise dates
    - Monthly exercise dates

    Pricing:
    - Backward induction on tree (easy)
    - LSM with restricted exercise times
    - PDE with exercise checks only at allowed dates
    """

    def __init__(
        self,
        strike: float,
        maturity: float,
        exercise_dates: np.ndarray,
        call_or_put: str = "call"
    ):
        """
        Initialize Bermudan option

        Args:
            strike: Strike price
            maturity: Final maturity
            exercise_dates: Array of allowed exercise times (fractions of maturity)
                           e.g., [0.25, 0.5, 0.75, 1.0] for quarterly + maturity
            call_or_put: "call" or "put"
        """
        super().__init__(maturity)
        self.strike = float(strike)
        self.exercise_dates = np.sort(np.array(exercise_dates))
        self.call_or_put = call_or_put.lower()

        if self.call_or_put not in ["call", "put"]:
            raise ValueError(f"call_or_put must be 'call' or 'put'")

        # Validate exercise dates
        if np.any(self.exercise_dates < 0) or np.any(self.exercise_dates > 1):
            raise ValueError("Exercise dates must be in [0, 1]")

    def payoff(self, path: np.ndarray) -> np.ndarray:
        """
        Terminal payoff (exercise at maturity)

        Optimal exercise is determined by pricing method.
        """
        if path.ndim == 3:
            S_T = path[-1, :, 0]
        else:
            S_T = path[-1, :]

        if self.call_or_put == "call":
            return np.maximum(S_T - self.strike, 0.0)
        else:
            return np.maximum(self.strike - S_T, 0.0)

    def exercise_value(self, S: np.ndarray) -> np.ndarray:
        """Exercise value at price S"""
        S = np.atleast_1d(S)

        if self.call_or_put == "call":
            return np.maximum(S - self.strike, 0.0)
        else:
            return np.maximum(self.strike - S, 0.0)

    def can_exercise(self, t: float, tolerance: float = 1e-6) -> bool:
        """
        Check if exercise is allowed at time t

        Args:
            t: Current time (fraction of maturity)
            tolerance: Tolerance for matching exercise dates

        Returns:
            True if t is an exercise date
        """
        # Check if t matches any exercise date (within tolerance)
        return np.any(np.abs(self.exercise_dates - t) < tolerance)

    @property
    def contract_type(self) -> str:
        return f"bermudan_{self.call_or_put}"

    def __repr__(self) -> str:
        n_dates = len(self.exercise_dates)
        return f"BermudanOption({self.call_or_put}, K={self.strike}, {n_dates} exercise dates, T={self.maturity})"
