"""
Regime-Switching Heston Model Calibration

Calibrate Heston stochastic volatility model with regime switching.

Each regime has its own Heston parameters (kappa, theta, xi, rho, v0).
Regimes are identified from volatility surface changes or manual labeling.

Use cases:
- Volatility clustering (high vol vs low vol regimes)
- Market stress detection (calm vs crisis regimes)
- Term structure shifts (steep vs flat vol surface)
- Counterfactual scenario analysis with stochastic volatility

author: Yunian Pan
email: yp1170@nyu.edu
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable, Tuple
from datetime import datetime, date
import warnings

from .heston_calibrator import HestonCalibrator, CalibrationResult


@dataclass
class RegimeHestonParameters:
    """Heston parameters for a single regime."""

    regime_id: int
    regime_name: str

    # Heston parameters
    kappa: float      # Mean reversion speed
    theta: float      # Long-term variance
    xi: float         # Vol of vol
    rho: float        # Correlation
    v0: float         # Initial variance

    # Fit quality
    rmse: float
    max_error: float
    feller_satisfied: bool

    # Regime statistics
    n_observations: int
    duration_days: int
    avg_atm_iv: float
    avg_term_structure_slope: float

    def summary(self) -> str:
        """Return formatted summary."""
        lines = [
            f"Regime {self.regime_id} ({self.regime_name}):",
            f"  κ = {self.kappa:.4f}  (mean reversion)",
            f"  θ = {self.theta:.4f}  (long-term var, σ∞ ≈ {np.sqrt(self.theta)*100:.1f}%)",
            f"  ξ = {self.xi:.4f}  (vol of vol)",
            f"  ρ = {self.rho:.4f}  (correlation)",
            f"  v₀ = {self.v0:.4f}  (initial var, σ₀ ≈ {np.sqrt(self.v0)*100:.1f}%)",
            f"  Feller: {'✓' if self.feller_satisfied else '✗'}",
            f"  RMSE: {self.rmse:.6f}",
            f"  Observations: {self.n_observations}",
            f"  Avg ATM IV: {self.avg_atm_iv*100:.2f}%",
        ]
        return "\n".join(lines)


@dataclass
class RegimeSwitchingHestonResult:
    """Result of regime-switching Heston calibration."""

    # Regime information
    n_regimes: int
    regime_names: Dict[int, str]
    regime_params: Dict[int, RegimeHestonParameters]

    # Transition matrix
    transition_matrix: np.ndarray

    # Regime labels
    regime_labels: np.ndarray
    dates: np.ndarray

    # Calibration method
    method: str  # 'manual', 'iv_threshold', 'term_structure'

    # Time period
    start_date: datetime
    end_date: datetime

    # Global fit quality
    overall_rmse: float

    def summary(self) -> str:
        """Return formatted summary."""
        lines = [
            "=" * 70,
            "Regime-Switching Heston Model Calibration Result",
            "=" * 70,
            "",
            f"Method: {self.method}",
            f"Number of regimes: {self.n_regimes}",
            f"Time period: {self.start_date} to {self.end_date}",
            f"Total observations: {len(self.regime_labels)}",
            "",
            "=" * 70,
            "REGIME PARAMETERS",
            "=" * 70,
            "",
        ]

        # Per-regime parameters
        for regime_id in sorted(self.regime_params.keys()):
            params = self.regime_params[regime_id]
            lines.append(params.summary())
            lines.append("")

        # Transition matrix
        lines.extend([
            "=" * 70,
            "TRANSITION MATRIX",
            "=" * 70,
            "",
        ])

        # Header
        header = "           " + "".join(f"  R{i:<5d}" for i in range(self.n_regimes))
        lines.append(header)

        # Matrix rows
        for i in range(self.n_regimes):
            regime_name = self.regime_names[i][:6]  # Truncate long names
            row = f"R{i} ({regime_name:6s})"
            for j in range(self.n_regimes):
                row += f"  {self.transition_matrix[i, j]*100:5.2f}%"
            lines.append(row)

        lines.append("")

        # Expected holding times
        lines.append("Expected holding times (days):")
        for i in range(self.n_regimes):
            regime_name = self.regime_names[i]
            p_stay = self.transition_matrix[i, i]
            expected_holding = 1 / (1 - p_stay) if p_stay < 1 else float('inf')
            lines.append(f"  {regime_name:12s}: {expected_holding:6.1f} days")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def long_run_probabilities(self) -> np.ndarray:
        """
        Compute stationary distribution of regimes.

        Returns:
            Array of long-run probabilities for each regime
        """
        # Find stationary distribution: π = π * P
        # Solve (P^T - I) π = 0 with constraint sum(π) = 1

        P = self.transition_matrix
        n = len(P)

        # Set up system: (P^T - I) π = 0
        A = P.T - np.eye(n)

        # Replace last equation with normalization constraint
        A[-1, :] = 1
        b = np.zeros(n)
        b[-1] = 1

        # Solve
        pi = np.linalg.solve(A, b)

        return pi


class RegimeSwitchingHestonCalibrator:
    """
    Calibrate regime-switching Heston model.

    Workflow:
    1. Identify regimes from volatility surface changes or manual labels
    2. Calibrate Heston parameters per regime
    3. Estimate transition probabilities
    4. Validate model fit

    Example:
        >>> calibrator = RegimeSwitchingHestonCalibrator()
        >>>
        >>> # Method 1: Manual regime labeling
        >>> def label_regimes(dates, atm_ivs):
        ...     # 0 = low vol, 1 = high vol
        ...     return (atm_ivs > 0.25).astype(int)
        >>>
        >>> result = calibrator.fit_manual(
        ...     option_chains=daily_chains,
        ...     regime_labeler=label_regimes,
        ...     regime_names={0: 'Low Vol', 1: 'High Vol'},
        ... )
        >>>
        >>> # Method 2: Automatic IV threshold
        >>> result = calibrator.fit_iv_threshold(
        ...     option_chains=daily_chains,
        ...     low_vol_threshold=0.15,
        ...     high_vol_threshold=0.30,
        ... )
        >>>
        >>> print(result.summary())
    """

    def __init__(
        self,
        pricer=None,
        weighting: str = 'vega',
        verbose: bool = True,
    ):
        """
        Initialize calibrator.

        Args:
            pricer: Pricer for Heston model (optional)
            weighting: Weighting scheme ('vega', 'uniform', 'oi')
            verbose: Print progress messages
        """
        self.pricer = pricer
        self.weighting = weighting
        self.verbose = verbose

    def fit_manual(
        self,
        option_chains: List,  # List of OptionChain objects
        regime_labeler: Callable,
        regime_names: Dict[int, str],
        spot_prices: Optional[np.ndarray] = None,
        risk_free_rates: Optional[np.ndarray] = None,
    ) -> RegimeSwitchingHestonResult:
        """
        Calibrate with manual regime labels.

        Args:
            option_chains: List of OptionChain objects (one per day)
            regime_labeler: Function(dates, atm_ivs) -> regime_labels
            regime_names: Mapping from regime_id to regime_name
            spot_prices: Optional array of spot prices (one per chain)
            risk_free_rates: Optional array of risk-free rates (one per chain)

        Returns:
            RegimeSwitchingHestonResult
        """
        if self.verbose:
            print(f"Calibrating regime-switching Heston model (manual labels)...")

        # Extract dates and ATM IVs for regime labeling
        dates, atm_ivs = self._extract_market_features(option_chains)

        # Get regime labels from user function
        regime_labels = regime_labeler(dates, atm_ivs)

        # Calibrate per-regime
        return self._fit_from_labels(
            option_chains=option_chains,
            regime_labels=regime_labels,
            regime_names=regime_names,
            method='manual',
            spot_prices=spot_prices,
            risk_free_rates=risk_free_rates,
        )

    def fit_iv_threshold(
        self,
        option_chains: List,
        low_vol_threshold: float = 0.15,
        high_vol_threshold: float = 0.30,
        spot_prices: Optional[np.ndarray] = None,
        risk_free_rates: Optional[np.ndarray] = None,
    ) -> RegimeSwitchingHestonResult:
        """
        Automatic regime detection based on ATM IV thresholds.

        Regimes:
        - 0 (Low Vol): ATM IV < low_vol_threshold
        - 1 (Normal Vol): low_vol_threshold <= ATM IV < high_vol_threshold
        - 2 (High Vol): ATM IV >= high_vol_threshold

        Args:
            option_chains: List of OptionChain objects
            low_vol_threshold: IV threshold for low vol regime (e.g., 0.15 = 15%)
            high_vol_threshold: IV threshold for high vol regime (e.g., 0.30 = 30%)
            spot_prices: Optional spot prices
            risk_free_rates: Optional risk-free rates

        Returns:
            RegimeSwitchingHestonResult
        """
        if self.verbose:
            print(f"Detecting regimes based on ATM IV thresholds...")
            print(f"  Low vol:  IV < {low_vol_threshold*100:.1f}%")
            print(f"  Normal:   {low_vol_threshold*100:.1f}% <= IV < {high_vol_threshold*100:.1f}%")
            print(f"  High vol: IV >= {high_vol_threshold*100:.1f}%")

        # Extract ATM IVs
        dates, atm_ivs = self._extract_market_features(option_chains)

        # Label regimes based on IV
        regime_labels = np.zeros(len(atm_ivs), dtype=int)
        regime_labels[atm_ivs < low_vol_threshold] = 0  # Low vol
        regime_labels[(atm_ivs >= low_vol_threshold) & (atm_ivs < high_vol_threshold)] = 1  # Normal
        regime_labels[atm_ivs >= high_vol_threshold] = 2  # High vol

        regime_names = {
            0: 'Low Vol',
            1: 'Normal Vol',
            2: 'High Vol',
        }

        return self._fit_from_labels(
            option_chains=option_chains,
            regime_labels=regime_labels,
            regime_names=regime_names,
            method='iv_threshold',
            spot_prices=spot_prices,
            risk_free_rates=risk_free_rates,
        )

    def _extract_market_features(
        self,
        option_chains: List,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract market features for regime detection.

        Returns:
            dates: Array of dates
            atm_ivs: Array of ATM implied volatilities
        """
        dates = []
        atm_ivs = []

        for chain in option_chains:
            dates.append(chain.reference_date)

            # Find ATM option
            spot = chain.spot_price
            atm_option = None
            min_moneyness_diff = float('inf')

            for opt in chain.options:
                if opt.implied_volatility is None or opt.implied_volatility <= 0:
                    continue

                moneyness_diff = abs(opt.strike / spot - 1.0)
                if moneyness_diff < min_moneyness_diff:
                    min_moneyness_diff = moneyness_diff
                    atm_option = opt

            if atm_option:
                atm_ivs.append(atm_option.implied_volatility)
            else:
                # Fallback: use median IV
                valid_ivs = [opt.implied_volatility for opt in chain.options
                            if opt.implied_volatility is not None and opt.implied_volatility > 0]
                atm_ivs.append(np.median(valid_ivs) if valid_ivs else 0.20)

        return np.array(dates), np.array(atm_ivs)

    def _fit_from_labels(
        self,
        option_chains: List,
        regime_labels: np.ndarray,
        regime_names: Dict[int, str],
        method: str,
        spot_prices: Optional[np.ndarray] = None,
        risk_free_rates: Optional[np.ndarray] = None,
    ) -> RegimeSwitchingHestonResult:
        """
        Core calibration logic given regime labels.

        Args:
            option_chains: List of OptionChain objects
            regime_labels: Array of regime labels (one per chain)
            regime_names: Mapping from regime_id to name
            method: Calibration method name
            spot_prices: Optional spot prices
            risk_free_rates: Optional risk-free rates

        Returns:
            RegimeSwitchingHestonResult
        """
        dates = np.array([chain.reference_date for chain in option_chains])

        # Validate
        if len(regime_labels) != len(option_chains):
            raise ValueError(
                f"regime_labels length ({len(regime_labels)}) must match "
                f"option_chains length ({len(option_chains)})"
            )

        unique_regimes = np.unique(regime_labels)
        n_regimes = len(unique_regimes)

        if self.verbose:
            print(f"\nFound {n_regimes} regimes")

        # Calibrate per-regime Heston parameters
        regime_params = {}
        all_rmses = []

        for regime_id in unique_regimes:
            regime_name = regime_names.get(regime_id, f'Regime {regime_id}')

            if self.verbose:
                print(f"\nCalibrating {regime_name}...")

            # Get chains for this regime
            regime_mask = regime_labels == regime_id
            regime_chains = [option_chains[i] for i in range(len(option_chains)) if regime_mask[i]]

            if len(regime_chains) == 0:
                warnings.warn(f"No data for regime {regime_id}, skipping")
                continue

            # Calibrate Heston for this regime
            params = self._calibrate_regime(
                regime_chains,
                regime_id,
                regime_name,
                spot_prices[regime_mask] if spot_prices is not None else None,
                risk_free_rates[regime_mask] if risk_free_rates is not None else None,
            )

            regime_params[regime_id] = params
            all_rmses.append(params.rmse)

            if self.verbose:
                print(f"  κ={params.kappa:.3f}, θ={params.theta:.4f}, ξ={params.xi:.3f}, "
                      f"ρ={params.rho:.3f}, v₀={params.v0:.4f}")
                print(f"  RMSE: {params.rmse:.6f}")

        # Estimate transition matrix
        transition_matrix = self._estimate_transition_matrix(regime_labels, regime_names)

        # Overall RMSE
        overall_rmse = np.mean(all_rmses) if all_rmses else 0.0

        return RegimeSwitchingHestonResult(
            n_regimes=n_regimes,
            regime_names=regime_names,
            regime_params=regime_params,
            transition_matrix=transition_matrix,
            regime_labels=regime_labels,
            dates=dates,
            method=method,
            start_date=dates[0],
            end_date=dates[-1],
            overall_rmse=overall_rmse,
        )

    def _calibrate_regime(
        self,
        regime_chains: List,
        regime_id: int,
        regime_name: str,
        spot_prices: Optional[np.ndarray],
        risk_free_rates: Optional[np.ndarray],
    ) -> RegimeHestonParameters:
        """
        Calibrate Heston parameters for a single regime.

        Strategy: Pool all options from all chains in this regime,
        then calibrate one set of Heston parameters.

        Args:
            regime_chains: List of OptionChain objects in this regime
            regime_id: Regime identifier
            regime_name: Regime name
            spot_prices: Optional spot prices per chain
            risk_free_rates: Optional risk-free rates per chain

        Returns:
            RegimeHestonParameters
        """
        # Pool all options from this regime
        # For simplicity, use first chain as representative
        # In production, you might pool options or use time-weighted average

        representative_chain = regime_chains[len(regime_chains) // 2]  # Middle chain

        # Use provided spot/rate or defaults
        spot = spot_prices[len(spot_prices) // 2] if spot_prices is not None else None
        rate = risk_free_rates[len(risk_free_rates) // 2] if risk_free_rates is not None else None

        # Calibrate Heston
        calibrator = HestonCalibrator(
            pricer=self.pricer,
            weighting=self.weighting,
        )

        try:
            result = calibrator.calibrate(
                chain=representative_chain,
                spot=spot,
                rate=rate,
                maxiter=500,  # Faster for regime-switching
            )
        except Exception as e:
            warnings.warn(f"Calibration failed for {regime_name}: {e}")
            # Return default parameters
            return RegimeHestonParameters(
                regime_id=regime_id,
                regime_name=regime_name,
                kappa=2.0,
                theta=0.04,
                xi=0.5,
                rho=-0.7,
                v0=0.04,
                rmse=999.0,
                max_error=999.0,
                feller_satisfied=False,
                n_observations=len(regime_chains),
                duration_days=0,
                avg_atm_iv=0.20,
                avg_term_structure_slope=0.0,
            )

        # Compute regime statistics
        dates = [chain.reference_date for chain in regime_chains]
        duration_days = (max(dates) - min(dates)).days if len(dates) > 1 else 0

        # Average ATM IV
        atm_ivs = []
        for chain in regime_chains:
            spot = chain.spot_price
            for opt in chain.options:
                if opt.implied_volatility and abs(opt.strike / spot - 1.0) < 0.05:
                    atm_ivs.append(opt.implied_volatility)
        avg_atm_iv = np.mean(atm_ivs) if atm_ivs else 0.20

        # Term structure slope (rough estimate)
        avg_term_structure_slope = 0.0  # TODO: compute from option chain

        return RegimeHestonParameters(
            regime_id=regime_id,
            regime_name=regime_name,
            kappa=result.kappa,
            theta=result.theta,
            xi=result.xi,
            rho=result.rho,
            v0=result.v0,
            rmse=result.rmse,
            max_error=result.max_error,
            feller_satisfied=result.feller_satisfied,
            n_observations=len(regime_chains),
            duration_days=duration_days,
            avg_atm_iv=avg_atm_iv,
            avg_term_structure_slope=avg_term_structure_slope,
        )

    def _estimate_transition_matrix(
        self,
        regime_labels: np.ndarray,
        regime_names: Dict[int, str],
    ) -> np.ndarray:
        """
        Estimate transition probability matrix from regime sequence.

        P[i, j] = probability of transitioning from regime i to regime j

        Args:
            regime_labels: Array of regime labels over time
            regime_names: Regime names

        Returns:
            Transition matrix (n_regimes x n_regimes)
        """
        unique_regimes = sorted(regime_names.keys())
        n_regimes = len(unique_regimes)

        # Map regime IDs to indices
        regime_to_idx = {regime_id: i for i, regime_id in enumerate(unique_regimes)}

        # Count transitions
        transitions = np.zeros((n_regimes, n_regimes))

        for t in range(len(regime_labels) - 1):
            current_regime = regime_labels[t]
            next_regime = regime_labels[t + 1]

            i = regime_to_idx[current_regime]
            j = regime_to_idx[next_regime]

            transitions[i, j] += 1

        # Normalize to probabilities
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_probs = transitions / row_sums

        return transition_probs


class RegimeSwitchingHestonSimulator:
    """
    Simulate paths under regime-switching Heston model.

    Example:
        >>> simulator = RegimeSwitchingHestonSimulator(calibration_result)
        >>> paths = simulator.simulate(S0=100, v0=0.04, T=1.0, n_paths=1000)
    """

    def __init__(
        self,
        result: RegimeSwitchingHestonResult,
        dt: float = 1/252,
    ):
        """
        Initialize simulator.

        Args:
            result: Calibration result with regime parameters
            dt: Time step (default 1/252 = 1 trading day)
        """
        self.result = result
        self.dt = dt

    def simulate(
        self,
        S0: float,
        v0: float,
        T: float,
        n_paths: int = 1000,
        scenario: str = 'baseline',
        initial_regime: Optional[int] = None,
        r: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate price and variance paths with regime switching.

        Args:
            S0: Initial price
            v0: Initial variance
            T: Time horizon (years)
            n_paths: Number of paths
            scenario: 'baseline', 'force_low_vol', 'force_high_vol', etc.
            initial_regime: Starting regime (if None, sample from stationary dist)
            r: Risk-free rate

        Returns:
            S_paths: Price paths (n_paths x n_steps)
            v_paths: Variance paths (n_paths x n_steps)
        """
        n_steps = int(T / self.dt)
        dt = self.dt

        # Initialize paths
        S_paths = np.zeros((n_paths, n_steps + 1))
        v_paths = np.zeros((n_paths, n_steps + 1))
        S_paths[:, 0] = S0
        v_paths[:, 0] = v0

        # Determine scenario regime forcing
        regime_ids = sorted(self.result.regime_params.keys())

        if scenario == 'force_low_vol':
            # Find regime with lowest theta
            forced_regime = min(regime_ids, key=lambda r: self.result.regime_params[r].theta)
        elif scenario == 'force_high_vol':
            # Find regime with highest theta
            forced_regime = min(regime_ids, key=lambda r: -self.result.regime_params[r].theta)
        else:
            forced_regime = None

        # Simulate each path
        for path_idx in range(n_paths):
            # Initial regime
            if initial_regime is not None:
                current_regime = initial_regime
            else:
                # Sample from stationary distribution
                pi = self.result.long_run_probabilities()
                current_regime = np.random.choice(regime_ids, p=pi)

            # Simulate path
            for t in range(n_steps):
                # Apply scenario forcing
                if forced_regime is not None:
                    current_regime = forced_regime

                # Get regime parameters
                params = self.result.regime_params[current_regime]
                kappa = params.kappa
                theta = params.theta
                xi = params.xi
                rho = params.rho

                # Current state
                S_t = S_paths[path_idx, t]
                v_t = max(v_paths[path_idx, t], 0)  # Ensure non-negative variance

                # Correlated Brownian motions
                dW_S = np.random.randn() * np.sqrt(dt)
                dW_v = rho * dW_S + np.sqrt(1 - rho**2) * np.random.randn() * np.sqrt(dt)

                # Heston dynamics (Euler-Maruyama discretization)
                # dS = r * S * dt + sqrt(v) * S * dW_S
                # dv = kappa * (theta - v) * dt + xi * sqrt(v) * dW_v

                S_next = S_t * np.exp((r - 0.5 * v_t) * dt + np.sqrt(v_t) * dW_S)
                v_next = v_t + kappa * (theta - v_t) * dt + xi * np.sqrt(v_t) * dW_v
                v_next = max(v_next, 0)  # Truncation scheme

                S_paths[path_idx, t + 1] = S_next
                v_paths[path_idx, t + 1] = v_next

                # Sample next regime (if not forcing)
                if forced_regime is None and t < n_steps - 1:
                    regime_idx = regime_ids.index(current_regime)
                    transition_probs = self.result.transition_matrix[regime_idx, :]
                    current_regime = np.random.choice(regime_ids, p=transition_probs)

        return S_paths, v_paths

    def scenario_analysis(
        self,
        S0: float,
        v0: float,
        T: float,
        n_paths: int,
        scenarios: List[str],
        r: float = 0.0,
    ) -> Dict:
        """
        Run multiple scenarios and compute statistics.

        Args:
            S0: Initial price
            v0: Initial variance
            T: Time horizon
            n_paths: Number of paths per scenario
            scenarios: List of scenario names
            r: Risk-free rate

        Returns:
            Dictionary mapping scenario to statistics
        """
        results = {}

        for scenario in scenarios:
            S_paths, v_paths = self.simulate(S0, v0, T, n_paths, scenario, r=r)

            final_prices = S_paths[:, -1]
            final_variances = v_paths[:, -1]

            results[scenario] = {
                'mean_final': np.mean(final_prices),
                'median_final': np.median(final_prices),
                'std_final': np.std(final_prices),
                'var_95': np.percentile(final_prices, 5),
                'cvar_95': np.mean(final_prices[final_prices <= np.percentile(final_prices, 5)]),
                'mean_final_variance': np.mean(final_variances),
                'mean_final_volatility': np.mean(np.sqrt(final_variances)),
            }

        return results
