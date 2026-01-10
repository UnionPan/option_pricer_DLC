"""
Avellaneda-Stoikov Calibration for Market Making Game

Two-part calibration:
1. Regime calibration: 2-regime volatility model + drift bounds (from OHLCV data)
2. Microstructure calibration: Order arrival rates + spread sensitivity (from tick data)

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, asdict
import warnings
from pathlib import Path

try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not available, regime calibration will use simple threshold")


@dataclass
class ASCalibrationResult:
    """
    Complete calibration results for Avellaneda-Stoikov market making game.

    Attributes:
        # Regime parameters
        sigma_stable: Volatility in stable regime (annualized)
        sigma_volatile: Volatility in volatile regime (annualized)
        mu_stable: Drift in stable regime (annualized)
        mu_volatile: Drift in volatile regime (annualized)
        base_transition_rate: Base regime switching rate (per day)

        # Predator bounds
        xi_estimate: Estimated predator cost coefficient
        max_drift_bound: Maximum observed |drift| for bounds

        # Microstructure parameters
        lambda_0: Base order arrival intensity (annualized)
        kappa: Spread sensitivity parameter
        avg_trade_size: Average trade size (BTC)
        trades_per_day: Average number of trades per day

        # Metadata
        data_source: Description of data used
        calibration_period: Date range
        n_observations: Number of data points used
    """
    # Regime parameters
    sigma_stable: float
    sigma_volatile: float
    mu_stable: float
    mu_volatile: float
    base_transition_rate: float

    # Predator bounds
    xi_estimate: float
    max_drift_bound: float

    # Microstructure parameters
    lambda_0: float
    kappa: float
    avg_trade_size: float
    trades_per_day: float

    # Metadata
    data_source: str
    calibration_period: str
    n_observations: int

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    def to_env_config(self) -> Dict:
        """
        Export parameters for make_market_making_env().

        Returns:
            Dict with keys: gamma, kappa, xi, btc_process params, etc.
        """
        return {
            # Regime params
            'sigma_stable': self.sigma_stable,
            'sigma_volatile': self.sigma_volatile,
            'mu_stable': self.mu_stable,
            'mu_volatile': self.mu_volatile,
            'base_transition_rate': self.base_transition_rate,

            # Microstructure
            'lambda_0': self.lambda_0,
            'kappa': self.kappa,

            # Predator
            'xi': self.xi_estimate,
        }

    def summary(self) -> str:
        """Generate summary string."""
        s = "=" * 80 + "\n"
        s += "AVELLANEDA-STOIKOV CALIBRATION RESULTS\n"
        s += "=" * 80 + "\n\n"

        s += "REGIME PARAMETERS\n"
        s += "-" * 80 + "\n"
        s += f"Stable Regime:\n"
        s += f"  Volatility (σ₀):  {self.sigma_stable:7.2%} annualized\n"
        s += f"  Drift (μ₀):       {self.mu_stable:+7.2%} annualized\n"
        s += f"\nVolatile Regime:\n"
        s += f"  Volatility (σ₁):  {self.sigma_volatile:7.2%} annualized\n"
        s += f"  Drift (μ₁):       {self.mu_volatile:+7.2%} annualized\n"
        s += f"\nRegime Switching:\n"
        s += f"  Base rate:        {self.base_transition_rate:.3f} per day\n"
        s += f"  Avg holding time: {1/self.base_transition_rate:.1f} days\n"

        s += f"\nPREDATOR PARAMETERS\n"
        s += "-" * 80 + "\n"
        s += f"  ξ (cost coeff):   {self.xi_estimate:.4f}\n"
        s += f"  Max drift bound:  {self.max_drift_bound:+.6f}\n"

        s += f"\nMICROSTRUCTURE PARAMETERS\n"
        s += "-" * 80 + "\n"
        s += f"  λ₀ (base arrival): {self.lambda_0:,.0f} per year\n"
        s += f"  κ (sensitivity):   {self.kappa:.3f}\n"
        s += f"  Avg trade size:    {self.avg_trade_size:.4f} BTC\n"
        s += f"  Trades per day:    {self.trades_per_day:,.0f}\n"

        s += f"\nMETADATA\n"
        s += "-" * 80 + "\n"
        s += f"  Data source:       {self.data_source}\n"
        s += f"  Period:            {self.calibration_period}\n"
        s += f"  Observations:      {self.n_observations:,}\n"

        s += "=" * 80 + "\n"

        return s

    def __repr__(self) -> str:
        return self.summary()


class RegimeCalibrator:
    """
    Calibrate 2-regime model from OHLCV data.

    Uses rolling volatility to identify high/low vol regimes,
    then estimates drift and transition rates.
    """

    def __init__(
        self,
        rolling_window: int = 20,  # Days for rolling vol
        annual_factor: float = 252,  # Trading days per year
    ):
        """
        Args:
            rolling_window: Window size for rolling volatility (in observations)
            annual_factor: Annualization factor (252 for daily, 252*24*12 for 5min, etc.)
        """
        self.rolling_window = rolling_window
        self.annual_factor = annual_factor

    def calibrate(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
    ) -> Dict:
        """
        Calibrate regime parameters from OHLCV data.

        Args:
            df: DataFrame with OHLCV data (indexed by datetime)
            price_col: Column name for price

        Returns:
            Dict with regime parameters
        """
        # Compute returns
        prices = df[price_col].values
        returns = np.diff(np.log(prices))

        # Rolling volatility
        vol_series = pd.Series(returns).rolling(
            window=self.rolling_window,
            min_periods=max(1, self.rolling_window // 2)
        ).std() * np.sqrt(self.annual_factor)

        vol_series = vol_series.dropna()

        # Identify 2 regimes using K-means or simple threshold
        if HAS_SKLEARN and len(vol_series) > 10:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(vol_series.values.reshape(-1, 1))

            # Sort regimes by volatility (0 = low, 1 = high)
            centers = kmeans.cluster_centers_.flatten()
            if centers[0] > centers[1]:
                labels = 1 - labels  # Flip
        else:
            # Simple threshold at median
            threshold = vol_series.median()
            labels = (vol_series.values > threshold).astype(int)

        # Compute regime statistics
        returns_extended = np.pad(returns, (len(prices) - len(returns), 0), constant_values=0)
        valid_idx = min(len(labels), len(returns_extended))

        regime_0_mask = labels[:valid_idx] == 0
        regime_1_mask = labels[:valid_idx] == 1

        # Volatilities
        sigma_0 = np.std(returns_extended[:valid_idx][regime_0_mask]) * np.sqrt(self.annual_factor)
        sigma_1 = np.std(returns_extended[:valid_idx][regime_1_mask]) * np.sqrt(self.annual_factor)

        # Drifts
        mu_0 = np.mean(returns_extended[:valid_idx][regime_0_mask]) * self.annual_factor
        mu_1 = np.mean(returns_extended[:valid_idx][regime_1_mask]) * self.annual_factor

        # Transition rates (from regime persistence)
        transitions = np.diff(labels)
        n_transitions = np.sum(transitions != 0)
        avg_holding_time = len(labels) / max(n_transitions, 1)
        transition_rate = 1.0 / max(avg_holding_time, 1)  # Per observation

        # Convert to per-day rate (depends on data frequency)
        # Assume data is daily if not specified
        transition_rate_per_day = transition_rate

        return {
            'sigma_stable': float(sigma_0),
            'sigma_volatile': float(sigma_1),
            'mu_stable': float(mu_0),
            'mu_volatile': float(mu_1),
            'base_transition_rate': float(transition_rate_per_day),
            'regime_labels': labels,
            'regime_vols': vol_series.values,
        }


class MicrostructureCalibrator:
    """
    Calibrate microstructure parameters from tick-level trade data.

    Estimates:
    - Base order arrival intensity λ₀
    - Spread sensitivity κ
    - Average trade size
    """

    def __init__(
        self,
        annual_factor: float = 252 * 24 * 60 * 60,  # Seconds in trading year
    ):
        """
        Args:
            annual_factor: Factor to annualize rates (default: seconds in year)
        """
        self.annual_factor = annual_factor

    def calibrate(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        price_col: str = 'price',
        amount_col: str = 'amount',
        type_col: Optional[str] = 'type',
    ) -> Dict:
        """
        Calibrate microstructure parameters from tick data.

        Args:
            df: DataFrame with tick-level trades
            timestamp_col: Column name for timestamp (unix seconds)
            price_col: Column name for trade price
            amount_col: Column name for trade size
            type_col: Column name for buy/sell indicator (optional)

        Returns:
            Dict with microstructure parameters
        """
        # Convert string columns to numeric if needed
        if df[price_col].dtype == 'object':
            df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
        if df[amount_col].dtype == 'object':
            df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')

        df = df.dropna(subset=[price_col, amount_col])

        # Compute inter-arrival times
        timestamps = df[timestamp_col].values
        inter_arrival_times = np.diff(timestamps)  # In seconds

        # Base arrival rate
        avg_inter_arrival = np.mean(inter_arrival_times)
        lambda_raw = 1.0 / avg_inter_arrival  # Trades per second
        lambda_0_annual = lambda_raw * self.annual_factor  # Annualized

        # Average trade size
        avg_trade_size = df[amount_col].mean()

        # Trades per day
        time_span = (timestamps[-1] - timestamps[0]) / (60 * 60 * 24)  # Days
        trades_per_day = len(df) / max(time_span, 1)

        # Estimate spread sensitivity κ
        # Use mid-price changes and trade flow imbalance
        prices = df[price_col].values
        mid_prices = prices  # Assume trades are at mid for simplicity

        # Compute realized spread (distance from mid)
        # For traded prices, estimate how far from "true" mid
        # Simple approach: use rolling median as proxy for mid
        rolling_mid = pd.Series(prices).rolling(window=20, min_periods=1).median().values
        spreads = np.abs(prices - rolling_mid) / rolling_mid  # Relative spreads

        # Estimate κ from spread distribution
        # In AS model: λ(δ) = λ₀ * exp(-κδ)
        # Higher κ → faster decay → more sensitive to spread

        # Use median spread to estimate κ
        median_spread = np.median(spreads[spreads > 0])

        # Heuristic: κ ≈ 1 / (median_spread)
        # This ensures λ(median_spread) ≈ λ₀ / e
        kappa_estimate = 1.0 / max(median_spread, 1e-6)

        # Clamp to reasonable range
        kappa_estimate = np.clip(kappa_estimate, 0.5, 10.0)

        return {
            'lambda_0': float(lambda_0_annual),
            'kappa': float(kappa_estimate),
            'avg_trade_size': float(avg_trade_size),
            'trades_per_day': float(trades_per_day),
            'median_spread': float(median_spread),
            'avg_inter_arrival_sec': float(avg_inter_arrival),
        }


def calibrate_from_csv_and_parquet(
    ohlcv_path: str,
    tick_path: str,
    gamma: float = 0.01,
    **kwargs
) -> ASCalibrationResult:
    """
    Complete calibration from CSV (OHLCV) and parquet (tick data).

    Args:
        ohlcv_path: Path to CSV file with OHLCV data
        tick_path: Path to parquet file with tick trades
        gamma: Market maker risk aversion (for ξ estimation)
        **kwargs: Additional args for calibrators

    Returns:
        ASCalibrationResult with all calibrated parameters

    Example:
        result = calibrate_from_csv_and_parquet(
            ohlcv_path='data/btc_ohlcv_5min.csv',
            tick_path='notebooks/gemini_btcusd_trades.parquet',
            gamma=0.01,
        )
        print(result.summary())
    """
    # 1. Load OHLCV data
    df_ohlcv = pd.read_csv(ohlcv_path, index_col=0, parse_dates=True)

    # 2. Load tick data
    df_tick = pd.read_parquet(tick_path)

    # 3. Calibrate regimes
    regime_cal = RegimeCalibrator(**kwargs.get('regime_kwargs', {}))
    regime_params = regime_cal.calibrate(df_ohlcv)

    # 4. Calibrate microstructure
    micro_cal = MicrostructureCalibrator(**kwargs.get('micro_kwargs', {}))
    micro_params = micro_cal.calibrate(df_tick)

    # 5. Estimate predator cost ξ
    # Use relationship: ξ ≈ (σ_high - σ_low) / γ
    # This makes predatory risk comparable to regime vol difference
    sigma_diff = regime_params['sigma_volatile'] - regime_params['sigma_stable']
    xi_estimate = sigma_diff / max(gamma, 1e-6)

    # Bound ξ to reasonable range
    xi_estimate = np.clip(xi_estimate, 0.001, 0.1)

    # 6. Drift bounds (from extreme returns)
    df_ohlcv_copy = df_ohlcv.copy()
    returns = np.diff(np.log(df_ohlcv_copy['Close'].values))
    max_drift = np.abs(np.percentile(returns, [1, 99])).max() * 252  # Annualized

    # 7. Metadata
    data_source = f"OHLCV: {Path(ohlcv_path).name}, Tick: {Path(tick_path).name}"
    period_start = df_ohlcv.index[0].strftime('%Y-%m-%d') if hasattr(df_ohlcv.index[0], 'strftime') else 'unknown'
    period_end = df_ohlcv.index[-1].strftime('%Y-%m-%d') if hasattr(df_ohlcv.index[-1], 'strftime') else 'unknown'
    calibration_period = f"{period_start} to {period_end}"

    return ASCalibrationResult(
        # Regime
        sigma_stable=regime_params['sigma_stable'],
        sigma_volatile=regime_params['sigma_volatile'],
        mu_stable=regime_params['mu_stable'],
        mu_volatile=regime_params['mu_volatile'],
        base_transition_rate=regime_params['base_transition_rate'],

        # Predator
        xi_estimate=xi_estimate,
        max_drift_bound=max_drift,

        # Microstructure
        lambda_0=micro_params['lambda_0'],
        kappa=micro_params['kappa'],
        avg_trade_size=micro_params['avg_trade_size'],
        trades_per_day=micro_params['trades_per_day'],

        # Metadata
        data_source=data_source,
        calibration_period=calibration_period,
        n_observations=len(df_ohlcv) + len(df_tick),
    )


def quick_calibrate_from_yfinance(
    ticker: str = 'BTC-USD',
    start_date: str = '2023-01-01',
    end_date: str = '2024-01-01',
    interval: str = '1h',  # 1h, 1d, 5m, etc.
    gamma: float = 0.01,
    lambda_0_default: float = 100.0 * 252,  # Default if no tick data
    kappa_default: float = 1.5,
) -> ASCalibrationResult:
    """
    Quick calibration using yfinance (no tick data required).

    Args:
        ticker: Yahoo Finance ticker
        start_date: Start date
        end_date: End date
        interval: Data interval
        gamma: Risk aversion
        lambda_0_default: Default lambda if no tick data
        kappa_default: Default kappa if no tick data

    Returns:
        ASCalibrationResult (with default microstructure params)
    """
    import sys
    sys.path.insert(0, 'src')
    from calibration.data.yfinance_fetcher import YFinanceFetcher

    # Fetch data
    fetcher = YFinanceFetcher()
    df = fetcher.get_history(ticker, start_date, end_date, interval=interval)

    # Calibrate regimes
    regime_cal = RegimeCalibrator()
    regime_params = regime_cal.calibrate(df)

    # Estimate ξ
    sigma_diff = regime_params['sigma_volatile'] - regime_params['sigma_stable']
    xi_estimate = np.clip(sigma_diff / max(gamma, 1e-6), 0.001, 0.1)

    # Drift bounds
    returns = np.diff(np.log(df['Close'].values))
    max_drift = np.abs(np.percentile(returns, [1, 99])).max() * 252

    # Use defaults for microstructure
    avg_trade_size = 0.1  # Typical retail BTC trade
    trades_per_day = lambda_0_default / 252  # Rough estimate

    return ASCalibrationResult(
        sigma_stable=regime_params['sigma_stable'],
        sigma_volatile=regime_params['sigma_volatile'],
        mu_stable=regime_params['mu_stable'],
        mu_volatile=regime_params['mu_volatile'],
        base_transition_rate=regime_params['base_transition_rate'],
        xi_estimate=xi_estimate,
        max_drift_bound=max_drift,
        lambda_0=lambda_0_default,
        kappa=kappa_default,
        avg_trade_size=avg_trade_size,
        trades_per_day=trades_per_day,
        data_source=f"YFinance {ticker} ({interval})",
        calibration_period=f"{start_date} to {end_date}",
        n_observations=len(df),
    )
