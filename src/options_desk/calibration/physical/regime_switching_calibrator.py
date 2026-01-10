"""
Regime-Switching Model Calibration

Calibrate regime-switching models for assets with clear regime changes
(bull markets, bear markets, sideways consolidation).

Perfect for:
- Bitcoin and cryptocurrencies (obvious regime changes)
- Commodities during supply shocks
- Equities during crisis periods

Methodology:
1. Identify regimes (manual or HMM)
2. Calibrate per-regime parameters (μ, σ)
3. Estimate transition probabilities
4. Simulate counterfactual scenarios

author: Yunian Pan
email: yp1170@nyu.edu
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
import warnings
from datetime import date

from .gbm_calibrator import GBMCalibrator
from .data_utils import compute_returns


@dataclass
class RegimeParameters:
    """Parameters for a single regime."""

    regime_id: int
    regime_name: str

    # GBM parameters
    mu: float              # Drift (annualized)
    sigma: float           # Volatility (annualized)

    # Statistical properties
    mu_stderr: float
    sigma_stderr: float
    n_observations: int

    # Regime characteristics
    avg_duration_days: float      # Average time spent in this regime
    total_occurrences: int         # How many times entered this regime

    # Summary stats
    mean_return: float
    median_return: float
    skewness: float
    kurtosis: float

    def __repr__(self) -> str:
        return (
            f"Regime {self.regime_id} ({self.regime_name}):\n"
            f"  μ = {self.mu:7.2%}   σ = {self.sigma:6.2%}\n"
            f"  Duration: {self.avg_duration_days:.0f} days\n"
            f"  Observations: {self.n_observations}"
        )


@dataclass
class RegimeSwitchingCalibrationResult:
    """Result of regime-switching model calibration."""

    # Regime parameters
    regime_params: Dict[int, RegimeParameters]

    # Transition matrix
    transition_matrix: np.ndarray  # P[i,j] = prob(regime i → regime j)

    # Regime labels
    regime_labels: np.ndarray      # Time series of regime assignments
    dates: np.ndarray              # Corresponding dates

    # Metadata
    n_regimes: int
    regime_names: Dict[int, str]

    # Diagnostics
    method: str                    # 'manual', 'hmm', 'threshold'

    def summary(self) -> str:
        """Pretty-printed summary."""
        lines = [
            "=" * 70,
            "Regime-Switching Model Calibration Result",
            "=" * 70,
            "",
            f"Method: {self.method}",
            f"Number of regimes: {self.n_regimes}",
            f"Time period: {self.dates[0]} to {self.dates[-1]}",
            f"Total observations: {len(self.regime_labels)}",
            "",
            "=" * 70,
            "REGIME PARAMETERS",
            "=" * 70,
        ]

        for regime_id, params in self.regime_params.items():
            lines.append("")
            lines.append(str(params))

        lines.append("")
        lines.append("=" * 70)
        lines.append("TRANSITION MATRIX")
        lines.append("=" * 70)
        lines.append("")

        # Format transition matrix
        header = "         " + "".join(f"  R{i}    " for i in range(self.n_regimes))
        lines.append(header)

        for i in range(self.n_regimes):
            row_name = f"R{i} ({self.regime_names[i][:6]:6s})"
            row_vals = "  ".join(f"{self.transition_matrix[i,j]:6.2%}" for j in range(self.n_regimes))
            lines.append(f"{row_name}  {row_vals}")

        lines.append("")
        lines.append("Expected holding times (days):")
        for i in range(self.n_regimes):
            holding_time = self.expected_holding_time(i)
            lines.append(f"  {self.regime_names[i]:12s}: {holding_time:6.1f} days")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def expected_holding_time(self, regime_id: int) -> float:
        """
        Expected time until leaving this regime.

        E[T] = 1 / (1 - P[stay in regime])
        """
        stay_prob = self.transition_matrix[regime_id, regime_id]
        if stay_prob >= 1.0:
            return np.inf
        return 1.0 / (1.0 - stay_prob)

    def long_run_probabilities(self) -> np.ndarray:
        """
        Compute stationary distribution of regimes.

        Solves: π * P = π, where π is stationary distribution

        Returns:
            Array of long-run probabilities for each regime
        """
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)

        # Find eigenvector corresponding to eigenvalue 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / np.sum(stationary)

        return stationary


class RegimeSwitchingCalibrator:
    """
    Calibrate regime-switching models from price data.

    Example:
        >>> # Manual regime labeling
        >>> calibrator = RegimeSwitchingCalibrator()
        >>>
        >>> # Define regimes based on visual inspection
        >>> def label_bitcoin_regimes(prices, dates):
        ...     regimes = np.zeros(len(prices), dtype=int)
        ...     regimes[dates < '2018-01'] = 2  # Bull
        ...     regimes[(dates >= '2018-01') & (dates < '2020-03')] = 0  # Bear
        ...     regimes[(dates >= '2020-03') & (dates < '2021-11')] = 2  # Bull
        ...     regimes[(dates >= '2021-11') & (dates < '2023-01')] = 0  # Bear
        ...     regimes[dates >= '2023-01'] = 1  # Sideways
        ...     return regimes
        >>>
        >>> result = calibrator.fit_manual(
        ...     prices=btc_prices,
        ...     dates=btc_dates,
        ...     regime_labeler=label_bitcoin_regimes,
        ...     regime_names={0: 'Bear', 1: 'Sideways', 2: 'Bull'},
        ... )
        >>>
        >>> print(result.summary())
    """

    def __init__(self, dt: float = 1.0/365.0):
        """
        Initialize calibrator.

        Args:
            dt: Time increment in years (default: 1/365 for daily data)
        """
        self.dt = dt
        self.gbm_calibrator = GBMCalibrator()

    def fit_manual(
        self,
        prices: np.ndarray,
        dates: np.ndarray,
        regime_labeler: Callable,
        regime_names: Dict[int, str],
    ) -> RegimeSwitchingCalibrationResult:
        """
        Calibrate with manual regime labels.

        Args:
            prices: Price time series
            dates: Corresponding dates
            regime_labeler: Function(prices, dates) -> regime_labels
            regime_names: Dict mapping regime_id to name

        Returns:
            Calibration result with per-regime parameters and transitions
        """
        prices = np.asarray(prices)
        dates = np.asarray(dates)

        # Get regime labels
        regime_labels = regime_labeler(prices, dates)
        regime_labels = np.asarray(regime_labels, dtype=int)

        if len(regime_labels) != len(prices):
            raise ValueError("regime_labeler must return same length as prices")

        # Calibrate
        return self._fit_from_labels(
            prices=prices,
            dates=dates,
            regime_labels=regime_labels,
            regime_names=regime_names,
            method='manual',
        )

    def fit_threshold(
        self,
        prices: np.ndarray,
        dates: np.ndarray,
        bear_threshold: float = -0.20,
        bull_threshold: float = 0.20,
        regime_names: Dict[int, str] = None,
    ) -> RegimeSwitchingCalibrationResult:
        """
        Automatic regime labeling based on drawdown/rally thresholds.

        Args:
            prices: Price time series
            dates: Corresponding dates
            bear_threshold: Drawdown from peak to enter bear (e.g., -0.20 = -20%)
            bull_threshold: Rally from trough to enter bull (e.g., 0.20 = +20%)
            regime_names: Optional regime names

        Returns:
            Calibration result

        Algorithm:
            - Bear: Price down >20% from recent peak
            - Bull: Price up >20% from recent trough
            - Sideways: Neither condition met
        """
        prices = np.asarray(prices)
        regime_labels = np.zeros(len(prices), dtype=int)  # Default: sideways

        # Compute rolling max (peak) and min (trough)
        window = 60  # 60-day window for peak/trough

        for i in range(window, len(prices)):
            lookback = prices[max(0, i-window):i+1]
            peak = np.max(lookback)
            trough = np.min(lookback)
            current = prices[i]

            # Drawdown from peak
            drawdown = (current - peak) / peak

            # Rally from trough
            rally = (current - trough) / trough if trough > 0 else 0

            if drawdown <= bear_threshold:
                regime_labels[i] = 0  # Bear
            elif rally >= bull_threshold:
                regime_labels[i] = 2  # Bull
            else:
                regime_labels[i] = 1  # Sideways

        if regime_names is None:
            regime_names = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}

        return self._fit_from_labels(
            prices=prices,
            dates=dates,
            regime_labels=regime_labels,
            regime_names=regime_names,
            method='threshold',
        )

    def _fit_from_labels(
        self,
        prices: np.ndarray,
        dates: np.ndarray,
        regime_labels: np.ndarray,
        regime_names: Dict[int, str],
        method: str,
    ) -> RegimeSwitchingCalibrationResult:
        """
        Core calibration logic given regime labels.
        """
        # Compute returns
        returns = compute_returns(prices, method='log', remove_nan=False)

        # Remove NaN returns and corresponding regime labels
        valid_mask = ~np.isnan(returns)
        returns = returns[valid_mask]
        regime_labels_returns = regime_labels[1:][valid_mask]  # Align and filter

        if len(returns) < len(regime_labels) - 1:
            n_removed = (len(regime_labels) - 1) - len(returns)
            warnings.warn(
                f"Removed {n_removed} NaN returns (possibly from zero/negative prices). "
                "Ensure price data is strictly positive."
            )

        unique_regimes = np.unique(regime_labels)
        n_regimes = len(unique_regimes)

        # Calibrate per-regime parameters
        regime_params = {}

        for regime_id in unique_regimes:
            # Get returns in this regime
            mask = regime_labels_returns == regime_id
            regime_returns = returns[mask]

            if len(regime_returns) < 10:
                warnings.warn(
                    f"Regime {regime_id} has only {len(regime_returns)} observations. "
                    "Parameters may be unreliable."
                )

            # Estimate parameters
            mu = np.mean(regime_returns) / self.dt
            sigma = np.std(regime_returns, ddof=1) / np.sqrt(self.dt)

            # Standard errors
            n = len(regime_returns)
            mu_stderr = sigma / np.sqrt(n) if n > 0 else np.nan
            sigma_stderr = sigma / np.sqrt(2 * n) if n > 0 else np.nan

            # Regime duration analysis
            duration_stats = self._analyze_regime_duration(regime_labels, regime_id)

            # Summary statistics
            mean_return = np.mean(regime_returns) * (365 / self.dt)
            median_return = np.median(regime_returns) * (365 / self.dt)
            skew = stats.skew(regime_returns) if len(regime_returns) > 3 else np.nan
            kurt = stats.kurtosis(regime_returns) if len(regime_returns) > 3 else np.nan

            regime_params[regime_id] = RegimeParameters(
                regime_id=regime_id,
                regime_name=regime_names.get(regime_id, f"Regime{regime_id}"),
                mu=mu,
                sigma=sigma,
                mu_stderr=mu_stderr,
                sigma_stderr=sigma_stderr,
                n_observations=n,
                avg_duration_days=duration_stats['avg_duration'],
                total_occurrences=duration_stats['n_occurrences'],
                mean_return=mean_return,
                median_return=median_return,
                skewness=skew,
                kurtosis=kurt,
            )

        # Estimate transition matrix
        transition_matrix = self._estimate_transition_matrix(regime_labels)

        return RegimeSwitchingCalibrationResult(
            regime_params=regime_params,
            transition_matrix=transition_matrix,
            regime_labels=regime_labels,
            dates=dates,
            n_regimes=n_regimes,
            regime_names=regime_names,
            method=method,
        )

    def _analyze_regime_duration(
        self,
        regime_labels: np.ndarray,
        regime_id: int,
    ) -> Dict:
        """
        Analyze how long the asset stays in this regime.
        """
        durations = []
        n_occurrences = 0

        in_regime = False
        start_idx = 0

        for i, label in enumerate(regime_labels):
            if label == regime_id and not in_regime:
                # Entering regime
                start_idx = i
                in_regime = True
                n_occurrences += 1
            elif label != regime_id and in_regime:
                # Exiting regime
                duration = i - start_idx
                durations.append(duration)
                in_regime = False

        # Handle case where still in regime at end
        if in_regime:
            durations.append(len(regime_labels) - start_idx)

        avg_duration = np.mean(durations) if durations else 0

        return {
            'avg_duration': avg_duration,
            'n_occurrences': n_occurrences,
            'durations': durations,
        }

    def _estimate_transition_matrix(
        self,
        regime_labels: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate transition probability matrix.

        P[i,j] = probability of moving from regime i to regime j
        """
        unique_regimes = np.unique(regime_labels)
        n_regimes = len(unique_regimes)

        # Create mapping from regime_id to matrix index
        regime_to_idx = {regime_id: idx for idx, regime_id in enumerate(unique_regimes)}

        # Count transitions
        transitions = np.zeros((n_regimes, n_regimes))

        for t in range(len(regime_labels) - 1):
            i = regime_to_idx[regime_labels[t]]
            j = regime_to_idx[regime_labels[t + 1]]
            transitions[i, j] += 1

        # Normalize to probabilities
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_probs = transitions / row_sums

        return transition_probs


class RegimeSwitchingSimulator:
    """
    Simulate price paths with regime switching.

    Example:
        >>> simulator = RegimeSwitchingSimulator(calibration_result)
        >>>
        >>> # Baseline scenario
        >>> paths_baseline = simulator.simulate(
        ...     S0=50000,
        ...     T=0.5,
        ...     n_paths=1000,
        ...     scenario='baseline',
        ... )
        >>>
        >>> # Bear market scenario
        >>> paths_bear = simulator.simulate(
        ...     S0=50000,
        ...     T=0.5,
        ...     n_paths=1000,
        ...     scenario='force_bear',
        ... )
    """

    def __init__(
        self,
        calibration_result: RegimeSwitchingCalibrationResult,
        dt: float = 1.0/365.0,
    ):
        """
        Initialize simulator.

        Args:
            calibration_result: Result from RegimeSwitchingCalibrator
            dt: Time step in years (default: daily)
        """
        self.result = calibration_result
        self.dt = dt

        # Build regime_id to index mapping
        self.regime_ids = sorted(calibration_result.regime_params.keys())
        self.regime_to_idx = {rid: idx for idx, rid in enumerate(self.regime_ids)}
        self.idx_to_regime = {idx: rid for idx, rid in enumerate(self.regime_ids)}

    def simulate(
        self,
        S0: float,
        T: float,
        n_paths: int,
        scenario: str = 'baseline',
        initial_regime: Optional[int] = None,
        vol_multiplier: float = 1.0,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate price paths with regime switching.

        Args:
            S0: Initial price
            T: Time horizon in years
            n_paths: Number of paths to simulate
            scenario: Simulation scenario
                - 'baseline': Normal regime switching
                - 'force_bear': Stay in bear regime
                - 'force_sideways': Stay in sideways regime
                - 'force_bull': Stay in bull regime
                - 'freeze_regime': Stay in initial regime (no switching)
            initial_regime: Starting regime (default: detect from current data)
            vol_multiplier: Multiply all σ by this factor (stress testing)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (paths, regime_labels)
            - paths: shape (n_paths, n_steps+1)
            - regime_labels: shape (n_paths, n_steps+1)
        """
        if seed is not None:
            np.random.seed(seed)

        n_steps = int(T / self.dt)

        # Initialize arrays
        paths = np.zeros((n_paths, n_steps + 1))
        regime_labels = np.zeros((n_paths, n_steps + 1), dtype=int)

        # Set initial price
        paths[:, 0] = S0

        # Set initial regime
        if initial_regime is None:
            initial_regime = self._detect_current_regime()
        regime_labels[:, 0] = initial_regime

        # Simulate each path
        for path_idx in range(n_paths):
            current_regime = initial_regime

            for t in range(n_steps):
                # Apply scenario
                if scenario == 'force_bear':
                    current_regime = self._get_regime_by_name('Bear')
                elif scenario == 'force_sideways':
                    current_regime = self._get_regime_by_name('Sideways')
                elif scenario == 'force_bull':
                    current_regime = self._get_regime_by_name('Bull')
                elif scenario == 'freeze_regime':
                    pass  # Keep current_regime
                # else: scenario == 'baseline', allow switching

                # Get regime parameters
                params = self.result.regime_params[current_regime]
                mu = params.mu
                sigma = params.sigma * vol_multiplier

                # Simulate GBM step
                dW = np.random.normal(0, np.sqrt(self.dt))
                S_t = paths[path_idx, t]
                S_next = S_t * np.exp((mu - 0.5 * sigma**2) * self.dt + sigma * dW)

                paths[path_idx, t + 1] = S_next

                # Regime transition (only for baseline)
                if scenario == 'baseline':
                    current_regime = self._sample_next_regime(current_regime)

                regime_labels[path_idx, t + 1] = current_regime

        return paths, regime_labels

    def _sample_next_regime(self, current_regime: int) -> int:
        """Sample next regime from transition matrix."""
        regime_idx = self.regime_to_idx[current_regime]
        probs = self.result.transition_matrix[regime_idx]

        # Sample from categorical distribution
        next_idx = np.random.choice(len(probs), p=probs)
        next_regime = self.idx_to_regime[next_idx]

        return next_regime

    def _detect_current_regime(self) -> int:
        """Detect most recent regime from calibration data."""
        return self.result.regime_labels[-1]

    def _get_regime_by_name(self, name: str) -> int:
        """Get regime ID by name."""
        for regime_id, regime_name in self.result.regime_names.items():
            if regime_name.lower() == name.lower():
                return regime_id

        # Fallback: return first regime
        return self.regime_ids[0]

    def scenario_analysis(
        self,
        S0: float,
        T: float,
        n_paths: int = 1000,
        scenarios: List[str] = None,
    ) -> Dict[str, Dict]:
        """
        Run multiple scenarios and return summary statistics.

        Args:
            S0: Initial price
            T: Time horizon
            n_paths: Paths per scenario
            scenarios: List of scenario names (default: all scenarios)

        Returns:
            Dict mapping scenario name to statistics:
            - 'paths': Simulated paths
            - 'mean_final': Mean final price
            - 'median_final': Median final price
            - 'std_final': Std dev of final price
            - 'var_95': 5th percentile (Value at Risk)
            - 'regime_distribution': % time in each regime
        """
        if scenarios is None:
            scenarios = ['baseline', 'force_bear', 'force_sideways', 'force_bull']

        results = {}

        for scenario in scenarios:
            paths, regimes = self.simulate(S0, T, n_paths, scenario=scenario)

            final_prices = paths[:, -1]

            # Regime distribution
            regime_dist = {}
            for regime_id in self.regime_ids:
                regime_dist[self.result.regime_names[regime_id]] = (
                    np.mean(regimes == regime_id) * 100
                )

            results[scenario] = {
                'paths': paths,
                'regimes': regimes,
                'mean_final': np.mean(final_prices),
                'median_final': np.median(final_prices),
                'std_final': np.std(final_prices),
                'var_95': np.percentile(final_prices, 5),
                'cvar_95': np.mean(final_prices[final_prices <= np.percentile(final_prices, 5)]),
                'regime_distribution': regime_dist,
            }

        return results
