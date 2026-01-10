"""
Synthetic Crypto Option Chain Generator - Clean Model-Driven Approach

Crypto-specific characteristics:
- Weekly expirations (7, 14, 21, 28d) + monthly (56, 84d)
- Liquidity-concentrated strikes (round percentages: 85%, 90%, 95%, 100%, 105%, 110%, 115%, 120%)
- Realistic volume/OI patterns (peak at ATM, round strikes, near expiry)
- 24/7 market (no time-of-day effects)

Philosophy:
- Use a single coherent model (Heston) to price all options
- Invert prices to implied volatilities (this IS the smile)
- No post-hoc arbitrage enforcement (trust the model)
- Minimal floors to avoid distorting the surface

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import List, Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass
from scipy.stats import norm
import warnings

from .data_provider import OptionChain, OptionQuote

if TYPE_CHECKING:
    from ..risk_neutral.regime_switching_heston_calibrator import RegimeSwitchingHestonResult


@dataclass
class RegimeVolatilityProfile:
    """Heston volatility parameters for a regime."""

    regime_id: int
    regime_name: str

    # Heston parameters
    kappa: float      # Mean reversion speed
    theta: float      # Long-term variance
    xi: float         # Vol of vol
    rho: float        # Correlation (spot-vol)
    v0: float         # Initial variance

    # Reference levels
    atm_iv: float     # ATM implied vol (for reference)

    # Bounds for IV backout
    min_iv: float = 0.05
    max_iv: float = 2.0

    def __post_init__(self):
        """Validate Feller condition."""
        feller_satisfied = 2 * self.kappa * self.theta > self.xi**2
        if not feller_satisfied:
            warnings.warn(
                f"Regime {self.regime_id} ({self.regime_name}): "
                f"Feller condition violated: 2κθ = {2*self.kappa*self.theta:.4f} "
                f"< ξ² = {self.xi**2:.4f}. Variance process may hit zero."
            )


class SyntheticOptionChainGenerator:
    """
    Generate synthetic option chains using a clean model-driven approach:
    
    1. Price with Heston characteristic function (the model)
    2. Invert to BS implied volatility (creates the smile)
    3. Add bid/ask spreads
    4. Done - no post-hoc arbitrage enforcement
    
    The smile shape comes from Heston parameters (κ, θ, ξ, ρ, v₀), not heuristics.
    """

    def __init__(
        self,
        regime_vol_profiles: Optional[Dict[int, RegimeVolatilityProfile]] = None,
        calibration_result: Optional['RegimeSwitchingHestonResult'] = None,
        risk_free_rate: float = 0.03,
        # Crypto-specific: weekly expiries + monthly
        maturities_days: List[int] = [7, 14, 21, 28, 56, 84],
        # Crypto-specific: liquidity concentrates at round percentages
        moneyness_range: List[float] = [0.85, 0.90, 0.95, 0.975, 1.0, 1.025, 1.05, 1.10, 1.15, 1.20],
        atm_spread_pct: float = 0.005,
        otm_spread_pct: float = 0.03,
        min_spread_pct: float = 0.002,
        max_spread_pct: float = 0.10,
        absolute_min_spread: float = 0.01,
        add_noise: bool = False,           # Symmetric noise on mid
        noise_level: float = 0.005,         # 0.5% of price
        price_floor: float = 0.01,          # Minimal floor
        enforce_intrinsic: bool = True,     # Enforce price >= intrinsic
        random_seed: Optional[int] = None,
    ):
        """
        Initialize generator.

        Args:
            regime_vol_profiles: Heston parameters per regime (optional if calibration_result provided)
            calibration_result: RegimeSwitchingHestonResult from calibrator (optional)
            risk_free_rate: Risk-free rate
            maturities_days: Option maturities
            moneyness_range: Strike/spot ratios
            *_spread_pct: Spread model parameters
            add_noise: Add small symmetric noise to mid prices
            noise_level: Noise std as fraction of price
            price_floor: Absolute minimum price (small to avoid distortion)
            enforce_intrinsic: Ensure price >= max(intrinsic, floor)
            random_seed: For reproducibility

        Note:
            Either regime_vol_profiles OR calibration_result must be provided.
            If calibration_result is provided, it takes precedence and regime_vol_profiles is ignored.
            If neither is provided, uses default Heston parameters.
        """
        # Extract regime_vol_profiles from calibration_result if provided
        if calibration_result is not None:
            self.regime_vol_profiles = self._convert_calibration_result(calibration_result)
        elif regime_vol_profiles is not None:
            self.regime_vol_profiles = regime_vol_profiles
        else:
            # Use default profiles
            self.regime_vol_profiles = self._get_default_profiles()
        self.risk_free_rate = risk_free_rate
        self.maturities_days = sorted(maturities_days)
        self.moneyness_range = sorted(moneyness_range)
        self.atm_spread_pct = atm_spread_pct
        self.otm_spread_pct = otm_spread_pct
        self.min_spread_pct = min_spread_pct
        self.max_spread_pct = max_spread_pct
        self.absolute_min_spread = absolute_min_spread
        self.add_noise = add_noise
        self.noise_level = noise_level
        self.price_floor = price_floor
        self.enforce_intrinsic = enforce_intrinsic

        # Local RNG (don't pollute global state)
        self._rng = np.random.RandomState(random_seed)

    def _convert_calibration_result(
        self,
        calibration_result: 'RegimeSwitchingHestonResult',
    ) -> Dict[int, RegimeVolatilityProfile]:
        """
        Convert RegimeSwitchingHestonResult to regime_vol_profiles.

        Args:
            calibration_result: Calibration result from RegimeSwitchingHestonCalibrator

        Returns:
            Dictionary mapping regime_id to RegimeVolatilityProfile
        """
        regime_vol_profiles = {}

        for regime_id, params in calibration_result.regime_params.items():
            regime_vol_profiles[regime_id] = RegimeVolatilityProfile(
                regime_id=params.regime_id,
                regime_name=params.regime_name,
                kappa=params.kappa,
                theta=params.theta,
                xi=params.xi,
                rho=params.rho,
                v0=params.v0,
                atm_iv=params.avg_atm_iv,
                min_iv=0.05,  # Reasonable bounds for crypto
                max_iv=2.0,
            )

        return regime_vol_profiles

    def _get_default_profiles(self) -> Dict[int, RegimeVolatilityProfile]:
        """
        Get default Heston parameters (single regime, moderate volatility).

        Returns:
            Dictionary with a single regime (regime_id=0)
        """
        return {
            0: RegimeVolatilityProfile(
                regime_id=0,
                regime_name='Default',
                kappa=4.0,
                theta=0.09,  # 30% long-term vol (typical for crypto)
                xi=0.5,      # Moderate vol-of-vol
                rho=-0.7,    # Negative correlation (leverage effect)
                v0=0.09,     # Start at long-term variance
                atm_iv=0.30, # 30% ATM IV
                min_iv=0.05,
                max_iv=2.0,
            )
        }

    def generate_from_regime_result(
        self,
        regime_result,
        underlying_prices: np.ndarray,
        dates: np.ndarray,
        sample_frequency: int = 5,
    ) -> List[OptionChain]:
        """Generate option chains from regime result."""
        regime_labels = regime_result.regime_labels
        option_chains = []

        for i in range(0, len(dates), sample_frequency):
            reference_date = dates[i] if isinstance(dates[i], date) else pd.to_datetime(dates[i]).date()
            spot = underlying_prices[i]
            current_regime = regime_labels[i]

            vol_profile = self.regime_vol_profiles.get(current_regime)
            if vol_profile is None:
                continue

            chain = self.generate_single_chain(
                reference_date=reference_date,
                spot_price=spot,
                vol_profile=vol_profile,
            )

            option_chains.append(chain)

        return option_chains

    def generate_single_chain(
        self,
        reference_date: date,
        spot_price: float,
        vol_profile: RegimeVolatilityProfile,
    ) -> OptionChain:
        """
        Generate a single option chain using clean model-driven approach.
        
        Steps:
        1. Price all options with Heston CF
        2. Add symmetric noise (optional)
        3. Back out IV from each price
        4. Compute bid/ask spreads
        5. Done (no arbitrage enforcement)
        """
        options = []

        for maturity_days in self.maturities_days:
            expiry = reference_date + timedelta(days=maturity_days)
            T = maturity_days / 365.0

            for moneyness in self.moneyness_range:
                strike = spot_price * moneyness

                for is_call in [True, False]:
                    # 1. Price with Heston CF (the model)
                    price_mid = self._heston_price_cf(
                        S=spot_price,
                        K=strike,
                        T=T,
                        r=self.risk_free_rate,
                        vol_profile=vol_profile,
                        is_call=is_call,
                    )

                    # 2. Add symmetric noise (optional, realistic)
                    if self.add_noise:
                        noise = self._rng.normal(0, self.noise_level * max(price_mid, 1.0))
                        price_mid = max(price_mid + noise, self.price_floor)

                    # 3. Back out IV from Heston price (this IS the smile)
                    iv = self._black_scholes_iv(
                        S=spot_price,
                        K=strike,
                        T=T,
                        r=self.risk_free_rate,
                        price=price_mid,
                        is_call=is_call,
                        vol_profile=vol_profile,
                    )

                    # 4. Compute bid/ask spread
                    spread = self._compute_bid_ask_spread(price_mid, moneyness, T)
                    bid = max(price_mid - spread / 2, self.price_floor)
                    ask = price_mid + spread / 2

                    # 5. Generate volume/OI (ad-hoc)
                    volume = self._generate_volume(moneyness, T)
                    oi = self._generate_open_interest(moneyness, T)

                    option = OptionQuote(
                        strike=strike,
                        expiry=expiry,
                        option_type='call' if is_call else 'put',
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
            underlying='BTC',
            spot_price=spot_price,
            reference_date=reference_date,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=0.0,
            options=options,
        )

    def _heston_price_cf(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        vol_profile: RegimeVolatilityProfile,
        is_call: bool,
    ) -> float:
        """
        Heston pricing via characteristic function.
        
        Uses semi-analytic formula with trapezoidal integration.
        """
        if T <= 0:
            intrinsic = max(S - K, 0) if is_call else max(K - S, 0)
            return max(intrinsic, self.price_floor)

        kappa = vol_profile.kappa
        theta = vol_profile.theta
        xi = vol_profile.xi
        rho = vol_profile.rho
        v0 = vol_profile.v0

        x = np.log(S / K)

        # Integration parameters
        u_max = 100
        N = 128

        def integrand_P1(u):
            phi = self._heston_cf(u - 1j, T, kappa, theta, xi, rho, v0, r)
            return np.real(np.exp(-1j * u * x) * phi / (1j * u))

        def integrand_P2(u):
            phi = self._heston_cf(u, T, kappa, theta, xi, rho, v0, r)
            return np.real(np.exp(-1j * u * x) * phi / (1j * u))

        # Trapezoidal integration
        u_grid = np.linspace(1e-10, u_max, N)
        du = u_grid[1] - u_grid[0]

        try:
            I1 = np.sum([integrand_P1(u) for u in u_grid]) * du
            I2 = np.sum([integrand_P2(u) for u in u_grid]) * du

            P1 = 0.5 + I1 / np.pi
            P2 = 0.5 + I2 / np.pi

            # Call price
            call_price = S * P1 - K * np.exp(-r * T) * P2

            if is_call:
                price = call_price
            else:
                # Put via put-call parity
                price = call_price - S + K * np.exp(-r * T)

            # Enforce intrinsic value floor if requested
            if self.enforce_intrinsic:
                intrinsic = max(S - K, 0) if is_call else max(K - S, 0)
                price = max(price, intrinsic)

            # Absolute floor
            price = max(price, self.price_floor)

            return price

        except Exception as e:
            warnings.warn(f"Heston CF integration failed: {e}. Returning intrinsic.")
            intrinsic = max(S - K, 0) if is_call else max(K - S, 0)
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
    ) -> complex:
        """Heston characteristic function (log-price)."""
        d = np.sqrt((rho * xi * 1j * u - kappa)**2 + xi**2 * (1j * u + u**2))
        g = (kappa - rho * xi * 1j * u - d) / (kappa - rho * xi * 1j * u + d)

        C = r * 1j * u * T + (kappa * theta / xi**2) * (
            (kappa - rho * xi * 1j * u - d) * T
            - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))
        )

        D = ((kappa - rho * xi * 1j * u - d) / xi**2) * (
            (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
        )

        return np.exp(C + D * v0)

    def _black_scholes_price_and_vega(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        iv: float,
        is_call: bool,
    ) -> tuple[float, float]:
        """Compute BS price and vega together (efficient)."""
        d1 = (np.log(S / K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
        d2 = d1 - iv * np.sqrt(T)

        if is_call:
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        vega = S * norm.pdf(d1) * np.sqrt(T)
        return price, vega

    def _black_scholes_iv(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        price: float,
        is_call: bool,
        vol_profile: RegimeVolatilityProfile,
    ) -> float:
        """
        Robust IV inversion using Newton-Raphson with bisection fallback.

        Handles edge cases:
        - Deep OTM/ITM where vega → 0
        - Very short/long maturities
        - Options near intrinsic value

        Strategy:
        1. Check boundary cases (floor, intrinsic)
        2. Try Newton-Raphson with adaptive damping
        3. Fall back to bisection if Newton fails/stalls
        """
        if T <= 0 or T > 10:  # T > 10 years is unrealistic
            return vol_profile.atm_iv

        # Check if at floor
        if price <= self.price_floor * 1.01:
            return vol_profile.min_iv

        # Check if deep ITM at intrinsic
        intrinsic = max(S - K, 0) if is_call else max(K - S, 0)
        if intrinsic > 0 and price <= intrinsic * 1.005:
            return vol_profile.min_iv

        # Bounds for bracketing
        iv_low = vol_profile.min_iv
        iv_high = vol_profile.max_iv

        # Better initial guess: use intrinsic value to estimate
        # If price ≈ intrinsic, start with min_iv
        # If price >> intrinsic, start with atm_iv
        time_value = price - intrinsic
        if time_value < 0.01 * S:  # Very little time value
            iv = vol_profile.min_iv
        else:
            iv = vol_profile.atm_iv

        iv = np.clip(iv, iv_low, iv_high)

        # Try Newton-Raphson with adaptive damping
        converged = False
        for iteration in range(50):
            model_price, vega = self._black_scholes_price_and_vega(S, K, T, r, iv, is_call)
            price_diff = model_price - price

            # Check convergence
            if abs(price_diff) < 1e-8 or abs(price_diff / max(price, 1e-8)) < 1e-6:
                converged = True
                break

            # Adaptive damping based on vega
            # If vega is small, use heavy damping
            vega_threshold = 1e-6 * S  # Scaled by spot
            if vega < vega_threshold:
                # Vega too small, fall back to bisection
                break

            # Newton step
            newton_step = -price_diff / vega

            # Adaptive damping: reduce step size if it would overshoot bounds
            damping = 1.0
            iv_new = iv + newton_step

            # If step overshoots bounds, reduce damping
            if iv_new < iv_low or iv_new > iv_high:
                damping = 0.5
                iv_new = iv + damping * newton_step

            # Further reduce if still overshooting
            while (iv_new < iv_low or iv_new > iv_high) and damping > 0.01:
                damping *= 0.5
                iv_new = iv + damping * newton_step

            # Clip to bounds
            iv_new = np.clip(iv_new, iv_low, iv_high)

            # Check if we're making progress
            if abs(iv_new - iv) < 1e-12:
                # Stalled, try bisection
                break

            # Update bracketing bounds
            if model_price < price:
                iv_low = max(iv_low, iv)
            else:
                iv_high = min(iv_high, iv)

            iv = iv_new

        if converged:
            return np.clip(iv, vol_profile.min_iv, vol_profile.max_iv)

        # Bisection fallback (guaranteed convergence)
        iv_low = vol_profile.min_iv
        iv_high = vol_profile.max_iv

        # Check that price is bracketed
        price_low, _ = self._black_scholes_price_and_vega(S, K, T, r, iv_low, is_call)
        price_high, _ = self._black_scholes_price_and_vega(S, K, T, r, iv_high, is_call)

        if price < price_low:
            return vol_profile.min_iv
        if price > price_high:
            return vol_profile.max_iv

        # Bisection
        for iteration in range(100):
            iv_mid = 0.5 * (iv_low + iv_high)
            price_mid, _ = self._black_scholes_price_and_vega(S, K, T, r, iv_mid, is_call)

            if abs(price_mid - price) < 1e-8:
                return iv_mid

            if price_mid < price:
                iv_low = iv_mid
            else:
                iv_high = iv_mid

            if iv_high - iv_low < 1e-10:
                return 0.5 * (iv_low + iv_high)

        # Fallback: return midpoint
        return 0.5 * (iv_low + iv_high)

    def _compute_bid_ask_spread(
        self,
        mid_price: float,
        moneyness: float,
        T: float,
    ) -> float:
        """Spread varies by moneyness and tenor."""
        log_moneyness = abs(np.log(moneyness))

        # Interpolate ATM → OTM
        if log_moneyness < 0.2:
            base_spread_pct = self.atm_spread_pct + \
                (self.otm_spread_pct - self.atm_spread_pct) * (log_moneyness / 0.2)
        else:
            base_spread_pct = self.otm_spread_pct

        # Tenor multiplier
        T_days = T * 365
        if T_days <= 30:
            tenor_mult = 1.0
        elif T_days <= 90:
            tenor_mult = 1.2
        else:
            tenor_mult = 1.5

        spread_pct = base_spread_pct * tenor_mult
        spread_pct = np.clip(spread_pct, self.min_spread_pct, self.max_spread_pct)

        dollar_spread = spread_pct * mid_price
        return max(dollar_spread, self.absolute_min_spread)

    def _generate_volume(self, moneyness: float, T: float) -> int:
        """
        Crypto-realistic volume generation.

        Volume characteristics:
        - Peaks at ATM (moneyness = 1.0)
        - Concentrates at round strikes (0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15, 1.20)
        - Higher for near-expiry options (weekly options dominate)
        - Random variation ±30%
        """
        # ATM factor: gaussian peak around moneyness = 1.0
        atm_distance = abs(moneyness - 1.0)
        atm_factor = np.exp(-8 * atm_distance**2)

        # Round strike bonus: liquidity concentrates at round percentages
        # Check if close to 5% intervals (0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20)
        round_strikes = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]
        dist_to_round = min(abs(moneyness - rs) for rs in round_strikes)
        if dist_to_round < 0.01:  # Within 1% of a round strike
            round_bonus = 2.0
        elif dist_to_round < 0.02:  # Within 2%
            round_bonus = 1.5
        else:
            round_bonus = 1.0

        # Maturity factor: volume decays as expiry approaches longer dates
        # Front week (7d): highest volume
        # Monthly (28d+): lower volume
        T_days = T * 365
        if T_days <= 7:
            maturity_factor = 3.0  # Front week very active
        elif T_days <= 14:
            maturity_factor = 2.5
        elif T_days <= 28:
            maturity_factor = 2.0
        elif T_days <= 56:
            maturity_factor = 1.2
        else:
            maturity_factor = 0.8  # Longer-dated less active

        # Base volume (scales with typical crypto option activity)
        base = 500

        # Random variation ±30%
        random_factor = self._rng.uniform(0.7, 1.3)

        volume = base * atm_factor * round_bonus * maturity_factor * random_factor
        return int(max(volume, 5))

    def _generate_open_interest(self, moneyness: float, T: float) -> int:
        """
        Crypto-realistic open interest generation.

        OI characteristics:
        - OI > volume (positions build over days)
        - Higher for longer-dated options (traders hold positions longer)
        - OI/volume ratio: 10-50x typical for crypto
        - Concentrates at liquid strikes
        """
        # Get volume for this strike
        volume = self._generate_volume(moneyness, T)

        # OI/volume multiplier increases with time to expiry
        # (longer-dated options accumulate more OI)
        T_days = T * 365
        if T_days <= 7:
            oi_volume_ratio = self._rng.uniform(8, 15)    # Front week: lower ratio
        elif T_days <= 14:
            oi_volume_ratio = self._rng.uniform(12, 20)
        elif T_days <= 28:
            oi_volume_ratio = self._rng.uniform(15, 30)
        elif T_days <= 56:
            oi_volume_ratio = self._rng.uniform(20, 40)
        else:
            oi_volume_ratio = self._rng.uniform(25, 50)   # Long-dated: highest ratio

        # ATM options have higher OI (more hedging activity)
        atm_distance = abs(moneyness - 1.0)
        if atm_distance < 0.05:  # Very close to ATM
            oi_boost = 1.5
        elif atm_distance < 0.10:
            oi_boost = 1.2
        else:
            oi_boost = 1.0

        oi = volume * oi_volume_ratio * oi_boost
        return int(max(oi, 50))


def quick_generate_option_chains(
    underlying_prices: np.ndarray,
    dates: np.ndarray,
    regime_labels: np.ndarray,
    regime_names: Dict[int, str],
    spot_symbol: str = 'BTC',
    atm_iv_per_regime: Optional[Dict[int, float]] = None,
    random_seed: Optional[int] = 42,
) -> List[OptionChain]:
    """Quick helper to generate Heston-driven chains."""
    
    # Auto-estimate ATM IV from historical vol
    if atm_iv_per_regime is None:
        atm_iv_per_regime = {}
        for regime_id in np.unique(regime_labels):
            regime_mask = regime_labels == regime_id
            regime_prices = underlying_prices[1:][regime_mask[:-1]]

            if len(regime_prices) > 10:
                returns = np.diff(np.log(regime_prices))
                vol = np.std(returns) * np.sqrt(252)
                atm_iv_per_regime[regime_id] = np.clip(vol, 0.10, 2.0)
            else:
                atm_iv_per_regime[regime_id] = 0.25

    # Create Heston profiles (Feller-satisfying)
    vol_profiles = {}
    for regime_id, regime_name in regime_names.items():
        atm_iv = atm_iv_per_regime.get(regime_id, 0.25)

        kappa = 4.0
        theta = atm_iv**2
        xi = 0.25

        vol_profiles[regime_id] = RegimeVolatilityProfile(
            regime_id=regime_id,
            regime_name=regime_name,
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=-0.7,
            v0=theta,
            atm_iv=atm_iv,
            min_iv=0.05,
            max_iv=2.0,
        )

    # Mock regime result
    from types import SimpleNamespace
    regime_result = SimpleNamespace(
        regime_labels=regime_labels,
        regime_names=regime_names,
    )

    # Generate
    generator = SyntheticOptionChainGenerator(
        regime_vol_profiles=vol_profiles,
        add_noise=False,          # Clean Heston prices
        random_seed=random_seed,
    )

    return generator.generate_from_regime_result(
        regime_result=regime_result,
        underlying_prices=underlying_prices,
        dates=dates,
        sample_frequency=5,
    )
