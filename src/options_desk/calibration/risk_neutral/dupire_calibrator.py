"""
Dupire local volatility calibration.

Extract local volatility surface from option prices using Dupire's formula.

author: Yunian Pan
email: yp1170@nyu.edu
"""

import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Callable
import time

from ..data.data_provider import OptionChain


@dataclass
class DupireResult:
    """Result of Dupire local volatility extraction."""

    # Local volatility surface
    strikes: np.ndarray         # Strike grid
    maturities: np.ndarray      # Maturity grid
    local_vols: np.ndarray      # Local vol surface (maturities x strikes)

    # Interpolation function
    lv_function: Callable[[float, float], float]  # σ_LV(K, T)

    # Fit quality
    n_options: int
    min_strike: float
    max_strike: float
    min_maturity: float
    max_maturity: float

    # Computation info
    computation_time: float
    smoothing_applied: bool

    def __repr__(self) -> str:
        lines = [
            "DupireLocalVolatilityResult:",
            f"  Strike range: [{self.min_strike:.2f}, {self.max_strike:.2f}]",
            f"  Maturity range: [{self.min_maturity:.4f}, {self.max_maturity:.4f}] years",
            f"  Grid size: {len(self.maturities)} x {len(self.strikes)}",
            f"  Options used: {self.n_options}",
            f"  Smoothing: {'Yes' if self.smoothing_applied else 'No'}",
            f"  Time: {self.computation_time:.2f}s",
        ]
        return "\n".join(lines)


class DupireCalibrator:
    """
    Extract local volatility surface using Dupire's formula.

    Dupire's formula (1994):
        σ_LV²(K,T) = (∂C/∂T + rK∂C/∂K) / (½K²∂²C/∂K²)

    where C(K,T) is the call option price.

    The local volatility surface is extracted by:
    1. Interpolating option prices to create smooth surface C(K,T)
    2. Computing numerical derivatives ∂C/∂T, ∂C/∂K, ∂²C/∂K²
    3. Applying Dupire's formula
    4. Applying smoothing and arbitrage-free constraints

    Reference:
        Dupire, B. (1994) "Pricing with a Smile"

    Example:
        calibrator = DupireCalibrator(smoothing=True)
        result = calibrator.calibrate(option_chain)
        local_vol = result.lv_function(strike=100, maturity=0.5)
    """

    def __init__(
        self,
        smoothing: bool = True,
        smoothing_sigma: float = 1.0,
        interpolation: str = 'rbf',
    ):
        """
        Initialize Dupire calibrator.

        Args:
            smoothing: Apply Gaussian smoothing to reduce noise
            smoothing_sigma: Standard deviation for Gaussian filter
            interpolation: Interpolation method ('rbf', 'linear', 'cubic')
        """
        self.smoothing = smoothing
        self.smoothing_sigma = smoothing_sigma
        self.interpolation = interpolation

    def calibrate(
        self,
        chain: OptionChain,
        spot: Optional[float] = None,
        rate: Optional[float] = None,
        dividend_yield: float = 0.0,
        n_strikes: int = 50,
        n_maturities: int = 20,
    ) -> DupireResult:
        """
        Extract local volatility surface from option chain.

        Args:
            chain: OptionChain with market data
            spot: Spot price (defaults to chain.spot_price)
            rate: Risk-free rate (defaults to chain.risk_free_rate)
            dividend_yield: Dividend yield
            n_strikes: Number of strike points in output grid
            n_maturities: Number of maturity points in output grid

        Returns:
            DupireResult with local volatility surface
        """
        start_time = time.time()

        S0 = spot or chain.spot_price
        r = rate or chain.risk_free_rate
        q = dividend_yield

        # Extract call options and prices
        call_data = self._extract_call_prices(chain, S0, r, q)

        if len(call_data['strikes']) < 10:
            raise ValueError("Need at least 10 call options for local vol extraction")

        # Create interpolated price surface
        price_surface = self._build_price_surface(call_data, S0, r, q)

        # Define strike and maturity grids
        K_min = max(call_data['strikes'].min(), S0 * 0.5)
        K_max = min(call_data['strikes'].max(), S0 * 2.0)
        T_min = call_data['maturities'].min()
        T_max = call_data['maturities'].max()

        strikes = np.linspace(K_min, K_max, n_strikes)
        maturities = np.linspace(T_min, T_max, n_maturities)

        # Compute local volatility on grid
        local_vols = np.zeros((n_maturities, n_strikes))

        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                local_vols[i, j] = self._dupire_local_vol(
                    K, T, price_surface, S0, r, q
                )

        # Apply smoothing if requested
        if self.smoothing:
            local_vols = gaussian_filter(local_vols, sigma=self.smoothing_sigma)

        # Ensure positive and reasonable bounds
        local_vols = np.clip(local_vols, 0.01, 3.0)

        # Create interpolation function
        lv_interp = interpolate.RectBivariateSpline(
            maturities, strikes, local_vols, kx=2, ky=2
        )

        def lv_function(K: float, T: float) -> float:
            """Local volatility function σ_LV(K, T)"""
            # Extrapolate using nearest values for out-of-bounds
            K_clip = np.clip(K, K_min, K_max)
            T_clip = np.clip(T, T_min, T_max)
            return float(lv_interp(T_clip, K_clip)[0, 0])

        computation_time = time.time() - start_time

        return DupireResult(
            strikes=strikes,
            maturities=maturities,
            local_vols=local_vols,
            lv_function=lv_function,
            n_options=len(call_data['strikes']),
            min_strike=K_min,
            max_strike=K_max,
            min_maturity=T_min,
            max_maturity=T_max,
            computation_time=computation_time,
            smoothing_applied=self.smoothing,
        )

    def _extract_call_prices(
        self,
        chain: OptionChain,
        S0: float,
        r: float,
        q: float,
    ) -> Dict:
        """Extract call option prices from chain."""
        strikes = []
        maturities = []
        prices = []

        for opt in chain.options:
            # Calculate maturity
            T = (opt.expiry - chain.reference_date).days / 365.0
            if T <= 0 or opt.mid <= 0:
                continue

            K = opt.strike

            # Convert puts to calls via put-call parity if needed
            if opt.is_call:
                call_price = opt.mid
            else:
                # Put-call parity: C = P + S*e^(-qT) - K*e^(-rT)
                put_price = opt.mid
                call_price = put_price + S0 * np.exp(-q * T) - K * np.exp(-r * T)

            # Sanity check: call price should satisfy bounds
            intrinsic = max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0)
            time_value_bound = S0 * np.exp(-q * T)

            if call_price < intrinsic * 0.9 or call_price > time_value_bound * 1.1:
                continue  # Skip suspicious prices

            strikes.append(K)
            maturities.append(T)
            prices.append(call_price)

        return {
            'strikes': np.array(strikes),
            'maturities': np.array(maturities),
            'prices': np.array(prices),
        }

    def _build_price_surface(
        self,
        call_data: Dict,
        S0: float,
        r: float,
        q: float,
    ) -> Callable:
        """Build interpolated call price surface C(K, T)."""
        strikes = call_data['strikes']
        maturities = call_data['maturities']
        prices = call_data['prices']

        # Use RBF interpolation for smooth surface
        if self.interpolation == 'rbf':
            # Normalize inputs for better RBF performance
            K_norm = strikes / S0
            T_norm = maturities

            rbf = interpolate.Rbf(
                K_norm, T_norm, prices,
                function='multiquadric',
                smooth=0.01,
            )

            def price_surface(K: float, T: float) -> float:
                K_n = K / S0
                return float(rbf(K_n, T))

        elif self.interpolation == 'linear':
            # Linear interpolation
            from scipy.interpolate import LinearNDInterpolator
            interp = LinearNDInterpolator(
                list(zip(strikes, maturities)), prices
            )

            def price_surface(K: float, T: float) -> float:
                result = interp(K, T)
                if np.isnan(result):
                    # Fallback to nearest neighbor
                    idx = np.argmin((strikes - K)**2 + (maturities - T)**2)
                    return prices[idx]
                return float(result)

        else:  # cubic
            from scipy.interpolate import CloughTocher2DInterpolator
            interp = CloughTocher2DInterpolator(
                list(zip(strikes, maturities)), prices
            )

            def price_surface(K: float, T: float) -> float:
                result = interp(K, T)
                if np.isnan(result):
                    idx = np.argmin((strikes - K)**2 + (maturities - T)**2)
                    return prices[idx]
                return float(result)

        return price_surface

    def _dupire_local_vol(
        self,
        K: float,
        T: float,
        price_surface: Callable,
        S0: float,
        r: float,
        q: float,
    ) -> float:
        """
        Compute local volatility using Dupire's formula.

        σ_LV²(K,T) = (∂C/∂T + (r-q)K∂C/∂K + qC) / (½K²∂²C/∂K²)
        """
        # Numerical derivatives using finite differences
        h_T = max(T * 0.01, 1/365)  # Time step (1 day minimum)
        h_K = K * 0.01               # Strike step (1%)

        # Price at (K, T)
        C = price_surface(K, T)

        # ∂C/∂T (forward difference)
        if T + h_T < 5.0:  # Don't extrapolate too far
            C_T_plus = price_surface(K, T + h_T)
            dC_dT = (C_T_plus - C) / h_T
        else:
            # Backward difference
            C_T_minus = price_surface(K, T - h_T)
            dC_dT = (C - C_T_minus) / h_T

        # ∂C/∂K and ∂²C/∂K² (central differences)
        C_K_plus = price_surface(K + h_K, T)
        C_K_minus = price_surface(K - h_K, T)

        dC_dK = (C_K_plus - C_K_minus) / (2 * h_K)
        d2C_dK2 = (C_K_plus - 2 * C + C_K_minus) / (h_K**2)

        # Apply Dupire's formula
        # σ_LV² = (∂C/∂T + (r-q)K∂C/∂K + qC) / (½K²∂²C/∂K²)

        numerator = dC_dT + (r - q) * K * dC_dK + q * C
        denominator = 0.5 * K**2 * d2C_dK2

        # Avoid division by zero or negative variance
        if denominator <= 1e-10:
            # Fallback: use implied vol or a reasonable default
            return 0.2

        sigma_LV_squared = numerator / denominator

        # Ensure positive variance
        if sigma_LV_squared <= 0:
            return 0.1  # Floor

        sigma_LV = np.sqrt(sigma_LV_squared)

        # Clip to reasonable range [1%, 300%]
        return np.clip(sigma_LV, 0.01, 3.0)
