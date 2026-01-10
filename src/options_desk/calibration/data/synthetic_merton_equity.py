"""
Synthetic Equity Option Chain Generator - Merton Jump-Diffusion
"""

import numpy as np
from datetime import date, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
import warnings

from .data_provider import OptionChain, OptionQuote
from options_desk.pricer.merton_mgf_pricer import merton_price_slice


def get_default_moneyness_by_maturity() -> Dict[int, List[float]]:
    """Default adaptive moneyness grid by maturity (days)."""
    return {
        10:  [0.95, 0.975, 1.0, 1.025, 1.05],
        20:  [0.95, 0.975, 1.0, 1.025, 1.05],
        30:  [0.90, 0.95, 0.975, 1.0, 1.025, 1.05, 1.10],
        60:  [0.90, 0.95, 1.0, 1.05, 1.10],
        90:  [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15],
        120: [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15],
    }


@dataclass
class MertonVolatilityProfile:
    """Merton jump-diffusion parameters for pricing."""
    sigma: float
    lambda_jump: float
    mu_J: float
    sigma_J: float
    atm_iv: float

    min_iv: float = 0.001
    max_iv: float = 2.0

    def __post_init__(self):
        if self.sigma <= 0:
            warnings.warn("Sigma should be positive.")


class SyntheticMertonOptionChainGenerator:
    """
    Generate synthetic equity option chains using Merton jump-diffusion.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.03,
        dividend_yield: float = 0.01,
        maturities_days: List[int] = None,
        moneyness_range: Optional[List[float]] = None,
        moneyness_by_maturity: Optional[Dict[int, List[float]]] = None,
        atm_spread_pct: float = 0.002,
        otm_spread_pct: float = 0.01,
        min_spread_pct: float = 0.001,
        max_spread_pct: float = 0.05,
        absolute_min_spread: float = 0.01,
        add_noise: bool = False,
        noise_level: float = 0.002,
        price_floor: float = 0.0001,
        enforce_intrinsic: bool = True,
        random_seed: Optional[int] = None,
    ):
        if maturities_days is None:
            maturities_days = [10, 20, 30, 60, 90, 120]
        if moneyness_range is None:
            moneyness_range = [0.9, 0.95, 1.0, 1.05, 1.1]

        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.maturities_days = sorted(maturities_days)
        self.moneyness_by_maturity = (
            {int(k): sorted(v) for k, v in moneyness_by_maturity.items()}
            if moneyness_by_maturity
            else {m: moneyness_range for m in self.maturities_days}
        )
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
        vol_profile: MertonVolatilityProfile,
    ) -> OptionChain:
        options = []

        for maturity_days in self.maturities_days:
            expiry = reference_date + timedelta(days=maturity_days)
            T = maturity_days / 365.0

            moneyness_list = self.moneyness_by_maturity.get(maturity_days, [])
            strikes = []
            types = []
            moneyness_maturity = []
            for moneyness in moneyness_list:
                strike = spot_price * moneyness
                strikes.append(strike)
                types.append('call')
                moneyness_maturity.append(moneyness)
                strikes.append(strike)
                types.append('put')
                moneyness_maturity.append(moneyness)

            prices_mid = merton_price_slice(
                S=spot_price,
                strikes=np.array(strikes),
                T=T,
                r=self.risk_free_rate,
                q=self.dividend_yield,
                sigma=vol_profile.sigma,
                lambda_jump=vol_profile.lambda_jump,
                mu_J=vol_profile.mu_J,
                sigma_J=vol_profile.sigma_J,
                option_types=np.array(types),
            )

            for strike, opt_type, moneyness, price_mid in zip(strikes, types, moneyness_maturity, prices_mid):
                is_call = opt_type == 'call'

                if self.add_noise:
                    noise = self._rng.normal(0, self.noise_level * max(price_mid, 1.0))
                    price_mid = max(price_mid + noise, self.price_floor)

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

                spread = self._compute_bid_ask_spread(price_mid, moneyness, T)
                bid = max(price_mid - spread / 2, self.price_floor)
                ask = price_mid + spread / 2

                volume = self._generate_volume(moneyness, T)
                oi = self._generate_open_interest(moneyness, T)

                options.append(OptionQuote(
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
                ))

        return OptionChain(
            underlying='SPY',
            spot_price=spot_price,
            reference_date=reference_date,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=self.dividend_yield,
            options=options,
        )

    def _black_scholes_iv(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        price: float,
        is_call: bool,
        vol_profile: MertonVolatilityProfile,
    ) -> float:
        if T <= 0 or T > 10:
            return vol_profile.atm_iv

        if price <= self.price_floor * 1.01:
            return vol_profile.min_iv

        intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0) if is_call else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
        if intrinsic > 0 and price <= intrinsic * 1.005:
            return vol_profile.min_iv

        iv_low = vol_profile.min_iv
        iv_high = vol_profile.max_iv
        iv = np.clip(vol_profile.atm_iv, iv_low, iv_high)

        for _ in range(50):
            model_price, vega = self._black_scholes_price_and_vega(S, K, T, r, q, iv, is_call)
            price_diff = model_price - price
            if abs(price_diff) < 1e-8 or abs(price_diff / max(price, 1e-8)) < 1e-6:
                return iv
            if vega < 1e-8:
                break
            iv = np.clip(iv - price_diff / vega, iv_low, iv_high)

        # Bisection fallback
        low, high = iv_low, iv_high
        for _ in range(60):
            mid = 0.5 * (low + high)
            price_mid, _ = self._black_scholes_price_and_vega(S, K, T, r, q, mid, is_call)
            if price_mid > price:
                high = mid
            else:
                low = mid
        return 0.5 * (low + high)

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
        if T <= 0:
            intrinsic = max(S - K, 0) if is_call else max(K - S, 0)
            return intrinsic, 0.0

        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * iv**2) * T) / (iv * sqrt_T)
        d2 = d1 - iv * sqrt_T

        from scipy.stats import norm
        if is_call:
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

        vega = S * np.exp(-q * T) * norm.pdf(d1) * sqrt_T
        return price, vega

    def _compute_bid_ask_spread(self, price: float, moneyness: float, T: float) -> float:
        # Wider spreads for deep OTM and short maturity
        distance = abs(moneyness - 1.0)
        spread_pct = self.atm_spread_pct + distance * (self.otm_spread_pct - self.atm_spread_pct)
        spread_pct = np.clip(spread_pct, self.min_spread_pct, self.max_spread_pct)
        return max(price * spread_pct, self.absolute_min_spread)

    def _generate_volume(self, moneyness: float, T: float) -> int:
        base = 500
        decay = np.exp(-3 * abs(moneyness - 1.0))
        time_decay = np.exp(-2 * T)
        return int(base * decay * time_decay + self._rng.randint(0, 20))

    def _generate_open_interest(self, moneyness: float, T: float) -> int:
        base = 2000
        decay = np.exp(-2 * abs(moneyness - 1.0))
        time_decay = np.exp(-1.5 * T)
        return int(base * decay * time_decay + self._rng.randint(0, 50))
