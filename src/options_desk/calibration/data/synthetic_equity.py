"""
Synthetic Equity Option Chain Generator

Equity-specific characteristics:
- Standard monthly expirations (30, 60, 90, 120, 180 days)
- Tighter strike spacing around ATM (97.5%, 100%, 102.5% + wider OTM)
- Market hours (9:30-16:00 ET, closed weekends)
- Dividends and interest rates matter
- Tighter bid-ask spreads than crypto

Uses Heston model for pricing to generate realistic IV smile/skew.

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
from scipy.stats import norm
from scipy.integrate import quad
import warnings

from .data_provider import OptionChain, OptionQuote
from options_desk.pricer.heston_mgf_pricer import heston_price_vanilla, heston_price_slice


def get_default_moneyness_by_maturity() -> Dict[int, List[float]]:
    """
    Default adaptive moneyness grid: tighter for short maturities, wider for long.

    Rationale:
    - Short-dated options: Less time to move → tighter strikes around ATM
    - Long-dated options: More time to move → wider strikes for hedging

    Returns:
        Dictionary mapping maturity (days) to moneyness list
    """
    return {
        10:  [0.95, 0.975, 1.0, 1.025, 1.05],              # Very tight: ±5%
        20:  [0.95, 0.975, 1.0, 1.025, 1.05],              # Tight: ±5%
        30:  [0.90, 0.95, 0.975, 1.0, 1.025, 1.05, 1.10],  # Medium: ±10%
        60:  [0.90, 0.95, 1.0, 1.05, 1.10],                # Standard: ±10%
        90:  [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15],    # Wide: ±15%
        120: [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15],    # Wide: ±15%
    }


@dataclass
class HestonVolatilityProfile:
    """Heston volatility parameters for option pricing."""

    # Heston parameters
    kappa: float      # Mean reversion speed
    theta: float      # Long-run variance
    xi: float         # Vol of vol (sigma_v)
    rho: float        # Correlation (spot-vol)
    v0: float         # Current variance

    # Reference
    atm_iv: float     # ATM implied vol (for reference)

    # Bounds for IV backout (lowered significantly to allow proper smile formation)
    min_iv: float = 0.001  # 0.1% floor to allow smile to form properly
    max_iv: float = 2.0

    def __post_init__(self):
        """Validate Feller condition."""
        feller_satisfied = 2 * self.kappa * self.theta > self.xi**2
        if not feller_satisfied:
            warnings.warn(
                f"Feller condition violated: 2κθ = {2*self.kappa*self.theta:.4f} "
                f"< ξ² = {self.xi**2:.4f}. Variance process may hit zero."
            )


class SyntheticEquityOptionChainGenerator:
    """
    Generate synthetic equity option chains using Heston model.

    Equity-specific features:
    - Monthly expirations (standard cycle)
    - Tighter strike spacing
    - Dividends and interest rates
    - Tighter bid-ask spreads
    - Market hours considerations

    Example:
        >>> profile = HestonVolatilityProfile(
        ...     kappa=4.0, theta=0.04, xi=0.5, rho=-0.7, v0=0.04, atm_iv=0.20
        ... )
        >>> generator = SyntheticEquityOptionChainGenerator()
        >>> chain = generator.generate_single_chain(
        ...     reference_date=date(2024, 1, 1),
        ...     spot_price=100.0,
        ...     vol_profile=profile,
        ... )
    """

    def __init__(
        self,
        risk_free_rate: float = 0.03,
        dividend_yield: float = 0.01,
        # Equity-specific: standard expiries (10-120 days)
        maturities_days: List[int] = [10, 20, 30, 60, 90, 120],
        # Equity-specific: adaptive strike spacing by maturity
        # Can be a single list (same for all maturities) or dict mapping maturity to moneyness
        moneyness_range: List[float] = None,
        moneyness_by_maturity: Dict[int, List[float]] = None,
        # Tighter spreads than crypto
        atm_spread_pct: float = 0.002,
        otm_spread_pct: float = 0.01,
        min_spread_pct: float = 0.001,
        max_spread_pct: float = 0.05,
        absolute_min_spread: float = 0.01,
        add_noise: bool = False,
        noise_level: float = 0.002,
        price_floor: float = 0.0001,  # Very low floor to not interfere with smile
        enforce_intrinsic: bool = True,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize equity option chain generator.

        Args:
            risk_free_rate: Risk-free rate (annualized)
            dividend_yield: Dividend yield (annualized)
            maturities_days: Option maturities in days
            moneyness_range: Single list of strike/spot ratios (same for all maturities)
            moneyness_by_maturity: Dict mapping maturity (days) to moneyness list (adaptive)
            *_spread_pct: Spread model parameters
            add_noise: Add small symmetric noise to mid prices
            noise_level: Noise std as fraction of price
            price_floor: Absolute minimum price
            enforce_intrinsic: Ensure price >= max(intrinsic, floor)
            random_seed: For reproducibility
        """
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.maturities_days = sorted(maturities_days)

        # Setup moneyness grid (adaptive by maturity or uniform)
        # Priority: moneyness_by_maturity > moneyness_range > default adaptive
        if moneyness_by_maturity is not None:
            # Use provided maturity-specific grid
            self.moneyness_by_maturity = {
                maturity: sorted(moneyness_list)
                for maturity, moneyness_list in moneyness_by_maturity.items()
            }
        elif moneyness_range is not None:
            # Use single list for all maturities
            uniform_moneyness = sorted(moneyness_range)
            self.moneyness_by_maturity = {
                maturity: uniform_moneyness
                for maturity in self.maturities_days
            }
        else:
            # Use default adaptive grid
            self.moneyness_by_maturity = get_default_moneyness_by_maturity()

        # Ensure all maturities have a grid (use ATM if missing)
        for maturity in self.maturities_days:
            if maturity not in self.moneyness_by_maturity:
                # Fallback: use grid from closest maturity, or ATM only
                warnings.warn(
                    f"No moneyness grid for {maturity}d maturity. Using [1.0] (ATM only)."
                )
                self.moneyness_by_maturity[maturity] = [1.0]

        self.atm_spread_pct = atm_spread_pct
        self.otm_spread_pct = otm_spread_pct
        self.min_spread_pct = min_spread_pct
        self.max_spread_pct = max_spread_pct
        self.absolute_min_spread = absolute_min_spread
        self.add_noise = add_noise
        self.noise_level = noise_level
        self.price_floor = price_floor
        self.enforce_intrinsic = enforce_intrinsic

        self._rng = np.random.RandomState(random_seed)

    def generate_single_chain(
        self,
        reference_date: date,
        spot_price: float,
        vol_profile: HestonVolatilityProfile,
    ) -> OptionChain:
        """
        Generate single option chain using Heston MGF pricing.

        Args:
            reference_date: Current date
            spot_price: Current spot price
            vol_profile: Heston volatility parameters

        Returns:
            OptionChain with synthetic options
        """
        options = []

        for maturity_days in self.maturities_days:
            expiry = reference_date + timedelta(days=maturity_days)
            T = maturity_days / 365.0

            # Prepare all strikes and option types for this maturity
            # Use maturity-specific moneyness grid (adaptive)
            strikes_maturity = []
            types_maturity = []
            moneyness_maturity = []

            moneyness_list = self.moneyness_by_maturity[maturity_days]
            for moneyness in moneyness_list:
                strike = spot_price * moneyness
                for is_call in [True, False]:
                    strikes_maturity.append(strike)
                    types_maturity.append('call' if is_call else 'put')
                    moneyness_maturity.append(moneyness)

            # Price all options at this maturity using MGF grid (efficient)
            strikes_array = np.array(strikes_maturity)
            types_array = np.array(types_maturity)

            prices_mid = heston_price_slice(
                S=spot_price,
                strikes=strikes_array,
                T=T,
                r=self.risk_free_rate,
                q=self.dividend_yield,
                v0=vol_profile.v0,
                theta=vol_profile.theta,
                kappa=vol_profile.kappa,
                volvol=vol_profile.xi,
                rho=vol_profile.rho,
                option_types=types_array
            )

            # Process each option
            for i, (strike, opt_type, moneyness, price_mid) in enumerate(
                zip(strikes_maturity, types_maturity, moneyness_maturity, prices_mid)
            ):
                is_call = (opt_type == 'call')

                # Add noise (optional)
                if self.add_noise:
                    noise = self._rng.normal(0, self.noise_level * max(price_mid, 1.0))
                    price_mid = max(price_mid + noise, self.price_floor)

                # Back out IV
                iv = self._black_scholes_iv(
                    S=spot_price,
                    K=strike,
                    T=T,
                    r=self.risk_free_rate,
                    q=self.dividend_yield,
                    price=price_mid,
                    is_call=is_call,
                    vol_profile=vol_profile,
                )

                # Bid/ask spread
                spread = self._compute_bid_ask_spread(price_mid, moneyness, T)
                bid = max(price_mid - spread / 2, self.price_floor)
                ask = price_mid + spread / 2

                # Volume/OI
                volume = self._generate_volume(moneyness, T)
                oi = self._generate_open_interest(moneyness, T)

                option = OptionQuote(
                    strike=strike,
                    expiry=expiry,
                    option_type=opt_type,
                    bid=bid,
                    ask=ask,
                    mid=price_mid,
                    last=price_mid * self._rng.uniform(0.99, 1.01),
                    volume=volume,
                    open_interest=oi,
                    implied_volatility=iv,
                )

                options.append(option)

        return OptionChain(
            underlying='SPY',
            spot_price=spot_price,
            reference_date=reference_date,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=self.dividend_yield,
            options=options,
        )

    def _heston_price_cf(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        vol_profile: HestonVolatilityProfile,
        is_call: bool,
    ) -> float:
        """
        Heston (1993) call option pricing via CF inversion.

        Reference: Heston, S. (1993). A Closed-Form Solution for Options with
        Stochastic Volatility. Review of Financial Studies, 6(2), 327-343.

        Uses Albrecher et al. (2007) formulation for numerical stability.
        """
        if T <= 0:
            intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0) if is_call else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
            return max(intrinsic, self.price_floor)

        kappa = vol_profile.kappa
        theta = vol_profile.theta
        sigma = vol_profile.xi  # vol-of-vol
        rho = vol_profile.rho
        v0 = vol_profile.v0

        # Log-moneyness
        x = np.log(S / K)

        def char_func(phi, j):
            """
            Heston characteristic function for probability P_j.

            Uses Albrecher et al. (2007) formulation with branch cut handling.
            j=1: For delta (S*P1 term)
            j=2: For risk-neutral probability (K*P2 term)
            """
            if phi == 0:
                return 1.0  # CF at zero is always 1

            if j == 1:
                u = 0.5
                b = kappa - rho * sigma
            else:  # j == 2
                u = -0.5
                b = kappa

            # Characteristic function exponents with careful branch cut handling
            # d = sqrt((rho*sigma*i*phi - b)^2 - sigma^2*(2*u*i*phi - phi^2))
            discriminant = (rho * sigma * 1j * phi - b)**2 - sigma**2 * (2*u*1j*phi - phi**2)
            d = np.sqrt(discriminant)

            # Ensure Re(d) >= 0 for numerical stability (Albrecher branch selection)
            if np.real(d) < 0:
                d = -d

            # g = (b - rho*sigma*i*phi - d) / (b - rho*sigma*i*phi + d)
            g = (b - rho * sigma * 1j * phi - d) / (b - rho * sigma * 1j * phi + d)

            # Avoid log(0) issues
            exp_dT = np.exp(-d * T)

            # Numerator and denominator for log term
            log_num = 1 - g * exp_dT
            log_den = 1 - g

            # Handle potential division by zero or log of negative
            if abs(log_den) < 1e-15:
                log_term = -d * T
            elif abs(log_num) < 1e-15:
                log_term = np.log(1e-15 / log_den)
            else:
                log_term = np.log(log_num / log_den)

            # C = (r-q)*i*phi*T + (kappa*theta/sigma^2)*((b - rho*sigma*i*phi - d)*T - 2*log(...))
            C = (r - q) * 1j * phi * T + (kappa * theta / sigma**2) * (
                (b - rho * sigma * 1j * phi - d) * T - 2 * log_term
            )

            # D = (b - rho*sigma*i*phi - d) / sigma^2 * (1 - exp(-dT)) / (1 - g*exp(-dT))
            D_num = (b - rho * sigma * 1j * phi - d) * (1 - exp_dT)
            D_den = sigma**2 * (1 - g * exp_dT)

            if abs(D_den) < 1e-15:
                D = 0.0
            else:
                D = D_num / D_den

            # CF = exp(C + D*v0 + i*phi*x)
            return np.exp(C + D * v0 + 1j * phi * x)

        def integrand(phi, j):
            """Integrand for P_j probability."""
            if phi == 0:
                return 0
            cf = char_func(phi, j)
            return np.real(cf / (1j * phi))

        # For very short maturities (< 5 days) + far OTM, CF integration becomes unstable
        # Use Black-Scholes approximation with current variance as fallback
        log_moneyness = abs(np.log(S / K))
        if T < 0.014 and log_moneyness > 0.10:  # < 5 days and > 10% OTM
            # Use BS with current vol as approximation for far OTM very short-dated options
            bs_price, _ = self._black_scholes_price_and_vega(
                S, K, T, r, q, np.sqrt(v0), is_call
            )
            return max(bs_price, self.price_floor)

        try:
            # Adaptive integration parameters based on maturity and moneyness
            # Very short maturities need tighter control
            if T < 0.02:  # < 7 days
                # Use tighter limits for very short maturities
                int_limit = 80
                subdivisions = 400
                abs_tol = 1e-10
                rel_tol = 1e-8
            elif T < 0.05:  # < 18 days
                int_limit = 100
                subdivisions = 300
                abs_tol = 1e-12
                rel_tol = 1e-10
            else:
                int_limit = 100
                subdivisions = 200
                abs_tol = 1e-12
                rel_tol = 1e-10

            # Compute P1 and P2 via integration with error checking
            I1, err1 = quad(lambda phi: integrand(phi, 1), 0, int_limit,
                           limit=subdivisions, epsabs=abs_tol, epsrel=rel_tol)
            I2, err2 = quad(lambda phi: integrand(phi, 2), 0, int_limit,
                           limit=subdivisions, epsabs=abs_tol, epsrel=rel_tol)

            P1 = 0.5 + I1 / np.pi
            P2 = 0.5 + I2 / np.pi

            # Sanity check probabilities with tolerance
            if P1 < -0.05 or P1 > 1.05 or P2 < -0.05 or P2 > 1.05:
                # Probabilities significantly out of range - use BS approximation
                bs_price, _ = self._black_scholes_price_and_vega(
                    S, K, T, r, q, np.sqrt(v0), is_call
                )
                return max(bs_price, self.price_floor)

            # Clip probabilities to valid range (small violations are numerical noise)
            P1 = np.clip(P1, 0.0, 1.0)
            P2 = np.clip(P2, 0.0, 1.0)

            # Heston call price
            call_price = S * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2

            if is_call:
                price = call_price
            else:
                # Put via put-call parity
                price = call_price - S * np.exp(-q * T) + K * np.exp(-r * T)

            # Enforce intrinsic value
            if self.enforce_intrinsic:
                intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0) if is_call else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
                price = max(price, intrinsic)

            # Apply price floor
            price = max(price, self.price_floor)

            return price

        except Exception as e:
            warnings.warn(f"Heston CF integration failed: {e}. Returning intrinsic.")
            intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0) if is_call else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
            return max(intrinsic, self.price_floor)

    def _heston_cf(
        self,
        u: complex,
        T: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
        v0: float,
        r: float,
        q: float,
    ) -> complex:
        """
        Heston characteristic function.

        Uses the formulation from Albrecher et al. (2007) with proper
        branch cut handling for numerical stability.
        """
        mu = r - q

        # Compute d with proper branch cut (Albrecher formulation)
        # d = sqrt((rho*xi*i*u - kappa)^2 + xi^2*(i*u + u^2))
        #   = sqrt(-xi^2*u^2 + i*u*(2*kappa*rho*xi) + (kappa^2 - xi^2))
        discriminant = (rho * xi * 1j * u - kappa)**2 + xi**2 * (1j * u + u**2)

        # Use sqrt with principal branch
        d = np.sqrt(discriminant)

        # Ensure Re(d) > 0 for stability (branch selection)
        if np.real(d) < 0:
            d = -d

        # Compute g (use Albrecher's stable formulation)
        # g = (kappa - rho*xi*i*u - d) / (kappa - rho*xi*i*u + d)
        g = (kappa - rho * xi * 1j * u - d) / (kappa - rho * xi * 1j * u + d)

        # Characteristic exponent C
        # C = mu*i*u*T + (kappa*theta/xi^2) * ((kappa - rho*xi*i*u - d)*T - 2*log((1 - g*exp(-dT))/(1 - g)))
        exp_dT = np.exp(-d * T)
        log_term = np.log((1 - g * exp_dT) / (1 - g))

        C = mu * 1j * u * T + (kappa * theta / xi**2) * (
            (kappa - rho * xi * 1j * u - d) * T - 2 * log_term
        )

        # Characteristic exponent D
        # D = ((kappa - rho*xi*i*u - d) / xi^2) * (1 - exp(-dT)) / (1 - g*exp(-dT))
        D = ((kappa - rho * xi * 1j * u - d) / xi**2) * (
            (1 - exp_dT) / (1 - g * exp_dT)
        )

        # Characteristic function: phi(u) = exp(C + D*v0)
        return np.exp(C + D * v0)

    def _black_scholes_price_and_vega(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        iv: float,
        is_call: bool,
    ) -> tuple:
        """BS price and vega."""
        d1 = (np.log(S / K) + (r - q + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
        d2 = d1 - iv * np.sqrt(T)

        if is_call:
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        return price, vega

    def _black_scholes_iv(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        price: float,
        is_call: bool,
        vol_profile: HestonVolatilityProfile,
    ) -> float:
        """Robust IV inversion using Newton-Raphson with bisection fallback."""
        if T <= 0 or T > 10:
            return vol_profile.atm_iv

        # Only return min_iv if price is extremely close to floor
        if price <= self.price_floor * 1.0001:
            return vol_profile.min_iv

        # Check intrinsic - but allow some tolerance for time value
        intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0) if is_call else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
        # Don't immediately give up if price is close to intrinsic
        if intrinsic > 0 and price < intrinsic * 0.999:
            return vol_profile.min_iv  # Price below intrinsic is invalid

        iv_low = vol_profile.min_iv
        iv_high = vol_profile.max_iv

        # Initial guess
        time_value = price - intrinsic
        if time_value < 0.01 * S:
            iv = vol_profile.min_iv
        else:
            iv = vol_profile.atm_iv

        iv = np.clip(iv, iv_low, iv_high)

        # Newton-Raphson
        converged = False
        for iteration in range(50):
            model_price, vega = self._black_scholes_price_and_vega(S, K, T, r, q, iv, is_call)
            price_diff = model_price - price

            if abs(price_diff) < 1e-8 or abs(price_diff / max(price, 1e-8)) < 1e-6:
                converged = True
                break

            vega_threshold = 1e-6 * S
            if vega < vega_threshold:
                break

            newton_step = -price_diff / vega
            damping = 1.0
            iv_new = iv + newton_step

            if iv_new < iv_low or iv_new > iv_high:
                damping = 0.5
                iv_new = iv + damping * newton_step

            while (iv_new < iv_low or iv_new > iv_high) and damping > 0.01:
                damping *= 0.5
                iv_new = iv + damping * newton_step

            iv_new = np.clip(iv_new, iv_low, iv_high)

            if abs(iv_new - iv) < 1e-12:
                break

            if model_price < price:
                iv_low = max(iv_low, iv)
            else:
                iv_high = min(iv_high, iv)

            iv = iv_new

        if converged:
            return np.clip(iv, vol_profile.min_iv, vol_profile.max_iv)

        # Bisection fallback
        iv_low = vol_profile.min_iv
        iv_high = vol_profile.max_iv

        price_low, _ = self._black_scholes_price_and_vega(S, K, T, r, q, iv_low, is_call)
        price_high, _ = self._black_scholes_price_and_vega(S, K, T, r, q, iv_high, is_call)

        if price < price_low:
            return vol_profile.min_iv
        if price > price_high:
            return vol_profile.max_iv

        for iteration in range(100):
            iv_mid = 0.5 * (iv_low + iv_high)
            price_mid, _ = self._black_scholes_price_and_vega(S, K, T, r, q, iv_mid, is_call)

            if abs(price_mid - price) < 1e-8:
                return iv_mid

            if price_mid < price:
                iv_low = iv_mid
            else:
                iv_high = iv_mid

            if iv_high - iv_low < 1e-10:
                return 0.5 * (iv_low + iv_high)

        return 0.5 * (iv_low + iv_high)

    def _compute_bid_ask_spread(
        self,
        mid_price: float,
        moneyness: float,
        T: float,
    ) -> float:
        """Spread model for equity options (tighter than crypto)."""
        log_moneyness = abs(np.log(moneyness))

        if log_moneyness < 0.1:
            base_spread_pct = self.atm_spread_pct
        elif log_moneyness < 0.2:
            base_spread_pct = self.atm_spread_pct + \
                (self.otm_spread_pct - self.atm_spread_pct) * (log_moneyness / 0.2)
        else:
            base_spread_pct = self.otm_spread_pct

        # Tenor multiplier
        T_days = T * 365
        if T_days <= 30:
            tenor_mult = 1.0
        elif T_days <= 90:
            tenor_mult = 1.1
        else:
            tenor_mult = 1.3

        spread_pct = base_spread_pct * tenor_mult
        spread_pct = np.clip(spread_pct, self.min_spread_pct, self.max_spread_pct)

        dollar_spread = spread_pct * mid_price
        return max(dollar_spread, self.absolute_min_spread)

    def _generate_volume(self, moneyness: float, T: float) -> int:
        """Equity volume generation."""
        atm_distance = abs(moneyness - 1.0)
        atm_factor = np.exp(-10 * atm_distance**2)

        T_days = T * 365
        if T_days <= 30:
            maturity_factor = 2.5
        elif T_days <= 60:
            maturity_factor = 2.0
        elif T_days <= 90:
            maturity_factor = 1.5
        else:
            maturity_factor = 1.0

        base = 1000
        random_factor = self._rng.uniform(0.8, 1.2)

        volume = base * atm_factor * maturity_factor * random_factor
        return int(max(volume, 10))

    def _generate_open_interest(self, moneyness: float, T: float) -> int:
        """Equity open interest generation."""
        volume = self._generate_volume(moneyness, T)

        T_days = T * 365
        if T_days <= 30:
            oi_volume_ratio = self._rng.uniform(15, 25)
        elif T_days <= 90:
            oi_volume_ratio = self._rng.uniform(20, 35)
        else:
            oi_volume_ratio = self._rng.uniform(25, 50)

        atm_distance = abs(moneyness - 1.0)
        if atm_distance < 0.05:
            oi_boost = 1.5
        else:
            oi_boost = 1.0

        oi = volume * oi_volume_ratio * oi_boost
        return int(max(oi, 100))
