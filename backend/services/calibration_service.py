"""
Calibration service for P-measure and Q-measure models
"""
import sys
from pathlib import Path

# Add src to path
src_path = str(Path(__file__).parent.parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import List, Dict, Tuple, Optional

# Import P-measure calibrators
from options_desk.calibration.physical import (
    GBMCalibrator,
    OUCalibrator,
    RegimeSwitchingCalibrator,
    RoughBergomiCalibrator,
    MertonJumpCalibrator,
    GARCHCalibrator,
    HestonParticleFilter,
    RoughBergomiParticleFilter,
)
from options_desk.processes import Heston, RoughBergomi, SimulationConfig
from options_desk.processes.gbm import GBM
from options_desk.processes.ornstein_uhlenbeck import OrnsteinUhlenbeck

# Import Q-measure calibrators and data providers
from options_desk.calibration.risk_neutral import (
    HestonCalibrator,
    CalibrationResult as HestonCalibrationResult,
    SABRCalibrator,
    SABRCalibrationResult,
    DupireCalibrator,
    DupireResult,
)
from options_desk.calibration.data import (
    YFinanceFetcher,
    OptionChain,
    MarketData,
)


class CalibrationService:
    """Service for fetching data and calibrating P-measure and Q-measure models"""

    @staticmethod
    def simulate_model(
        model: str,
        parameters: Dict[str, float],
        s0: float,
        n_steps: int,
        n_paths: int,
        dt: float,
        max_paths_return: int = 20,
        random_seed: Optional[int] = None,
    ) -> Dict:
        """Simulate counterfactual spot paths using calibrated parameters."""
        if s0 <= 0:
            raise ValueError("s0 must be positive")
        if n_steps < 1 or n_paths < 1:
            raise ValueError("n_steps and n_paths must be >= 1")

        model = model.lower()
        time_grid = np.linspace(0.0, n_steps * dt, n_steps + 1)

        if random_seed is not None:
            np.random.seed(random_seed)

        if model == "gbm":
            mu = parameters.get("mu", 0.0)
            sigma = parameters["sigma"]
            process = GBM(mu=mu, sigma=sigma)
            config = SimulationConfig(n_paths=n_paths, n_steps=n_steps, random_seed=random_seed)
            _, paths = process.simulate(np.array([s0]), n_steps * dt, config, scheme="exact")
            spot_paths = paths[:, :, 0]

        elif model == "ou":
            mu = parameters.get("mu", 0.0)
            kappa = parameters["kappa"]
            sigma = parameters["sigma"]
            process = OrnsteinUhlenbeck(theta=kappa, mu=mu, sigma=sigma)
            config = SimulationConfig(n_paths=n_paths, n_steps=n_steps, random_seed=random_seed)
            _, paths = process.simulate(np.array([s0]), n_steps * dt, config, scheme="exact")
            spot_paths = paths[:, :, 0]

        elif model == "heston":
            mu = parameters.get("mu", 0.0)
            rho = parameters.get("rho", 0.0)
            v0 = parameters.get("v0", parameters["theta"])
            process = Heston(
                mu=mu,
                kappa=parameters["kappa"],
                theta=parameters["theta"],
                sigma_v=parameters["xi"],
                rho=rho,
                v0=v0,
            )
            config = SimulationConfig(n_paths=n_paths, n_steps=n_steps, random_seed=random_seed)
            _, paths = process.simulate(np.array([s0, v0]), n_steps * dt, config, scheme="milstein")
            spot_paths = paths[:, :, 0]

        elif model == "rough_bergomi":
            mu = parameters.get("mu", 0.0)
            process = RoughBergomi(
                mu=mu,
                xi0=parameters["xi0"],
                eta=parameters["eta"],
                rho=parameters["rho"],
                H=parameters["H"],
            )
            config = SimulationConfig(n_paths=n_paths, n_steps=n_steps, random_seed=random_seed)
            _, paths = process.simulate(np.array([s0, parameters["xi0"]]), n_steps * dt, config)
            spot_paths = paths[:, :, 0]

        elif model == "merton_jump":
            mu = parameters.get("mu", 0.0)
            sigma = parameters["sigma"]
            lambda_ = parameters["lambda"]
            mu_j = parameters["mu_j"]
            sigma_j = parameters["sigma_j"]
            kappa = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
            drift = (mu - 0.5 * sigma**2 - lambda_ * kappa) * dt

            log_s = np.zeros((n_steps + 1, n_paths))
            log_s[0] = np.log(s0)
            for t in range(n_steps):
                z = np.random.normal(0.0, 1.0, size=n_paths)
                n_jump = np.random.poisson(lambda_ * dt, size=n_paths)
                jump = n_jump * mu_j + np.sqrt(n_jump) * sigma_j * np.random.normal(0.0, 1.0, size=n_paths)
                log_s[t + 1] = log_s[t] + drift + sigma * np.sqrt(dt) * z + jump
            spot_paths = np.exp(log_s)

        elif model == "garch":
            mu = parameters.get("mu", 0.0)
            omega = parameters["omega"]
            alpha = parameters["alpha"]
            beta = parameters["beta"]
            var0 = omega / max(1.0 - alpha - beta, 1e-6)

            returns = np.zeros((n_steps, n_paths))
            sigma2 = np.full(n_paths, var0)

            for t in range(n_steps):
                z = np.random.normal(0.0, 1.0, size=n_paths)
                returns[t] = mu + np.sqrt(sigma2) * z
                sigma2 = omega + alpha * (returns[t] - mu) ** 2 + beta * sigma2
                sigma2 = np.maximum(sigma2, 1e-12)

            log_s = np.zeros((n_steps + 1, n_paths))
            log_s[0] = np.log(s0)
            log_s[1:] = log_s[0] + np.cumsum(returns, axis=0)
            spot_paths = np.exp(log_s)

        elif model == "regime_switching_gbm":
            mu0 = parameters["mu_regime0"]
            mu1 = parameters["mu_regime1"]
            sigma0 = parameters["sigma_regime0"]
            sigma1 = parameters["sigma_regime1"]
            p00 = parameters.get("p_00", 0.95)
            p01 = parameters.get("p_01", 1.0 - p00)
            p10 = parameters.get("p_10", 0.1)
            p11 = parameters.get("p_11", 1.0 - p10)

            log_s = np.zeros((n_steps + 1, n_paths))
            log_s[0] = np.log(s0)
            regime = np.zeros(n_paths, dtype=int)

            for t in range(n_steps):
                probs = np.where(regime == 0, p01, p10)
                switch = np.random.rand(n_paths) < probs
                regime = np.where(switch, 1 - regime, regime)

                mu = np.where(regime == 0, mu0, mu1)
                sigma = np.where(regime == 0, sigma0, sigma1)
                z = np.random.normal(0.0, 1.0, size=n_paths)
                log_s[t + 1] = log_s[t] + (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z

            spot_paths = np.exp(log_s)

        else:
            raise ValueError(f"Unsupported model for simulation: {model}")

        final = spot_paths[-1]
        stats = {
            "mean": float(np.mean(final)),
            "std": float(np.std(final)),
            "p5": float(np.percentile(final, 5)),
            "p50": float(np.percentile(final, 50)),
            "p95": float(np.percentile(final, 95)),
            "min": float(np.min(final)),
            "max": float(np.max(final)),
        }

        mean_path = np.mean(spot_paths, axis=1)
        n_return = min(max_paths_return, n_paths)
        sample_paths = [spot_paths[:, i].tolist() for i in range(n_return)]

        return {
            "model": model,
            "n_steps": n_steps,
            "n_paths": n_paths,
            "dt": dt,
            "time_grid": time_grid.tolist(),
            "mean_path": mean_path.tolist(),
            "sample_paths": sample_paths,
            "stats": stats,
        }

    @staticmethod
    def fetch_ohlcv_data(tickers: List[str], start_date: str, end_date: str) -> List[Dict]:
        """
        Fetch OHLCV data from yfinance

        Args:
            tickers: List of ticker symbols
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)

        Returns:
            List of OHLCV data dictionaries
        """
        results = []

        for ticker in tickers:
            try:
                # Fetch data from yfinance
                data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)

                if data.empty:
                    raise ValueError(f"No data available for {ticker}")

                # Flatten multi-index columns if necessary
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)

                # Convert to dict format
                ohlcv = {
                    "ticker": ticker,
                    "dates": [d.strftime('%Y-%m-%d') for d in data.index],
                    "open": data['Open'].values.flatten().tolist(),
                    "high": data['High'].values.flatten().tolist(),
                    "low": data['Low'].values.flatten().tolist(),
                    "close": data['Close'].values.flatten().tolist(),
                    "volume": data['Volume'].values.flatten().tolist(),
                }

                results.append(ohlcv)

            except Exception as e:
                raise ValueError(f"Failed to fetch data for {ticker}: {str(e)}")

        return results

    @staticmethod
    def calibrate_model(
        ticker: str,
        prices: np.ndarray,
        dates: List[str],
        model: str,
        method: str,
        include_drift: bool
    ) -> Tuple[Dict[str, float], Dict]:
        """
        Calibrate a P-measure model to price data

        Args:
            ticker: Ticker symbol
            prices: Array of prices
            dates: List of date strings
            model: Model name
            method: Calibration method
            include_drift: Whether to include drift

        Returns:
            Tuple of (parameters, diagnostics)
        """
        # Calculate returns
        returns = np.diff(np.log(prices))

        # Time step (assuming daily data)
        dt = 1/252

        # Calibrate based on model type
        if model == 'gbm':
            # GBM: Direct MLE (closed-form)
            return CalibrationService._calibrate_gbm(returns, dt, include_drift)

        elif model == 'ou':
            # OU: Direct MLE (closed-form)
            return CalibrationService._calibrate_ou(prices, dt, include_drift)

        elif model == 'heston':
            # Heston: Particle filter for P-measure
            if method == 'particle_filter':
                return CalibrationService._calibrate_heston_particle_filter(prices, dt, include_drift)
            else:
                raise ValueError(f"Heston P-measure only supports 'particle_filter' method")

        elif model == 'rough_bergomi':
            # Rough Bergomi: Moment matching or particle filter
            if method == 'moment_matching':
                return CalibrationService._calibrate_rough_bergomi_moments(prices, dt, include_drift)
            elif method == 'particle_filter':
                return CalibrationService._calibrate_rough_bergomi_particle_filter(prices, dt, include_drift)
            else:
                raise ValueError(f"Rough Bergomi supports 'moment_matching' or 'particle_filter' methods")

        elif model == 'regime_switching_gbm':
            # Regime-switching: EM algorithm
            return CalibrationService._calibrate_regime_switching(returns, dt, include_drift)

        elif model == 'merton_jump':
            # Merton jump-diffusion: MLE
            return CalibrationService._calibrate_merton_jump(prices, dt, include_drift)

        elif model == 'garch':
            # GARCH(1,1): QMLE
            return CalibrationService._calibrate_garch(prices, dt, include_drift)

        else:
            raise ValueError(f"Unsupported model: {model}")

    @staticmethod
    def _calibrate_gbm(returns: np.ndarray, dt: float, include_drift: bool) -> Tuple[Dict[str, float], Dict]:
        """Calibrate GBM model"""
        calibrator = GBMCalibrator()
        result = calibrator.calibrate(returns, dt)

        parameters = {
            "sigma": result.sigma,
        }

        if include_drift:
            parameters["mu"] = result.mu

        diagnostics = {
            "logLikelihood": result.log_likelihood if hasattr(result, 'log_likelihood') else None,
            "aic": result.aic if hasattr(result, 'aic') else None,
            "bic": result.bic if hasattr(result, 'bic') else None,
            "errorMetrics": {},
        }

        return parameters, diagnostics

    @staticmethod
    def _calibrate_heston_particle_filter(prices: np.ndarray, dt: float, include_drift: bool) -> Tuple[Dict[str, float], Dict]:
        """Calibrate Heston model using particle filter"""
        # First get initial guesses from simple volatility estimation
        returns = np.diff(np.log(prices))
        mu_init = np.mean(returns) / dt if include_drift else 0.0
        v0_init = np.var(returns)

        # Estimate rho (correlation) from empirical returns and volatility changes
        # Use rolling window to estimate realized volatility
        window = min(20, max(5, len(returns) // 10))
        realized_vol = np.array([np.std(returns[max(0, i-window):i+1]) if i >= 1 else np.std(returns[:2])
                                 for i in range(len(returns))])

        # Compute changes in realized vol (proxy for vol innovations)
        vol_changes = np.diff(realized_vol)

        # Correlation between returns and subsequent vol changes (leverage effect)
        if len(vol_changes) > 5:
            # Align: returns[:-1] vs vol_changes
            rho_init = float(np.corrcoef(returns[:-1], vol_changes)[0, 1])
            # Clip to reasonable range
            rho_init = np.clip(rho_init, -0.95, 0.95)
            # Handle NaN
            if np.isnan(rho_init):
                rho_init = -0.5  # Default negative correlation (typical for equities)
        else:
            rho_init = -0.5

        # Initial parameter guesses
        kappa_init = 2.0
        theta_init = v0_init
        xi_init = 0.5

        # Run particle filter
        pf = HestonParticleFilter(n_particles=2000)
        result = pf.filter(
            prices=prices,
            dt=dt,
            mu=mu_init,
            kappa=kappa_init,
            theta=theta_init,
            xi=xi_init,
            v0=v0_init,
        )

        parameters = {
            'kappa': kappa_init,
            'theta': theta_init,
            'xi': xi_init,
            'v0': v0_init,
            'rho': rho_init,
        }

        if include_drift:
            parameters['mu'] = mu_init

        diagnostics = {
            'logLikelihood': result.log_likelihood,
            'mean_ess': float(np.mean(result.effective_sample_size)),
            'n_particles': result.n_particles,
            'errorMetrics': {},
            'note': 'Particle filter-based estimation with empirical rho from return-volatility correlation.',
        }

        return parameters, diagnostics

    @staticmethod
    def _calibrate_rough_bergomi_moments(prices: np.ndarray, dt: float, include_drift: bool) -> Tuple[Dict[str, float], Dict]:
        """Calibrate rough Bergomi using moment matching (variogram)"""
        calibrator = RoughBergomiCalibrator(window=20, max_lag=10)
        result = calibrator.fit(prices, dt)

        parameters = {
            'xi0': result.xi0,
            'eta': result.eta,
            'rho': result.rho,
            'H': result.H,
        }

        if include_drift:
            parameters['mu'] = result.mu

        diagnostics = {
            'variogram_r2': result.diagnostics.get('variogram_r2'),
            'mean_return': result.diagnostics.get('mean_return'),
            'mean_var': result.diagnostics.get('mean_var'),
            'n_observations': result.n_observations,
            'window': result.window,
            'max_lag': result.max_lag,
            'errorMetrics': {},
        }

        return parameters, diagnostics

    @staticmethod
    def _calibrate_rough_bergomi_particle_filter(prices: np.ndarray, dt: float, include_drift: bool) -> Tuple[Dict[str, float], Dict]:
        """Calibrate rough Bergomi using particle filter"""
        # First get moment-based initial guesses
        calibrator = RoughBergomiCalibrator(window=20, max_lag=10)
        moment_result = calibrator.fit(prices, dt)

        # Run particle filter with these parameters
        pf = RoughBergomiParticleFilter(n_particles=2000)
        pf_result = pf.filter(
            prices=prices,
            dt=dt,
            mu=moment_result.mu,
            xi0=moment_result.xi0,
            eta=moment_result.eta,
            H=moment_result.H,
            rho=moment_result.rho,
        )

        parameters = {
            'xi0': moment_result.xi0,
            'eta': moment_result.eta,
            'rho': moment_result.rho,
            'H': moment_result.H,
        }

        if include_drift:
            parameters['mu'] = moment_result.mu

        diagnostics = {
            'logLikelihood': pf_result.log_likelihood,
            'mean_ess': float(np.mean(pf_result.effective_sample_size)),
            'n_particles': pf_result.n_particles,
            'variogram_r2': moment_result.diagnostics.get('variogram_r2'),
            'errorMetrics': {},
            'note': 'Particle filter evaluation using moment-based initialization',
        }

        return parameters, diagnostics

    @staticmethod
    def _calibrate_ou(prices: np.ndarray, dt: float, include_drift: bool) -> Tuple[Dict[str, float], Dict]:
        """Calibrate Ornstein-Uhlenbeck model"""
        calibrator = OUCalibrator()
        result = calibrator.calibrate(prices, dt)

        parameters = {
            "kappa": result.kappa,
            "theta": result.theta,
            "sigma": result.sigma,
        }

        diagnostics = {
            "logLikelihood": result.log_likelihood if hasattr(result, 'log_likelihood') else None,
            "aic": result.aic if hasattr(result, 'aic') else None,
            "bic": result.bic if hasattr(result, 'bic') else None,
            "errorMetrics": {},
        }

        return parameters, diagnostics

    @staticmethod
    def _calibrate_regime_switching(returns: np.ndarray, dt: float, include_drift: bool) -> Tuple[Dict[str, float], Dict]:
        """Calibrate regime-switching GBM"""
        try:
            calibrator = RegimeSwitchingCalibrator(n_regimes=2)
            result = calibrator.calibrate(returns, dt)

            parameters = {}
            for i in range(2):
                parameters[f"mu_regime{i}"] = result.parameters[i]['mu'] if include_drift else 0.0
                parameters[f"sigma_regime{i}"] = result.parameters[i]['sigma']

            # Transition matrix
            if hasattr(result, 'transition_matrix'):
                for i in range(2):
                    for j in range(2):
                        parameters[f"p_{i}{j}"] = result.transition_matrix[i, j]

            diagnostics = {
                "logLikelihood": result.log_likelihood if hasattr(result, 'log_likelihood') else None,
                "aic": result.aic if hasattr(result, 'aic') else None,
                "bic": result.bic if hasattr(result, 'bic') else None,
                "errorMetrics": {},
            }

            return parameters, diagnostics

        except Exception as e:
            # Fallback to simple GBM if regime switching fails
            return CalibrationService._calibrate_gbm(returns, dt, include_drift)

    @staticmethod
    def _calibrate_merton_jump(prices: np.ndarray, dt: float, include_drift: bool) -> Tuple[Dict[str, float], Dict]:
        """Calibrate Merton jump-diffusion model"""
        calibrator = MertonJumpCalibrator(k_max=5)
        result = calibrator.fit(prices, dt)

        parameters = {
            "sigma": result.sigma,
            "lambda": result.lambda_,
            "mu_j": result.mu_j,
            "sigma_j": result.sigma_j,
        }
        if include_drift:
            parameters["mu"] = result.mu

        diagnostics = {
            "logLikelihood": result.log_likelihood,
            "aic": result.aic,
            "bic": result.bic,
            "errorMetrics": {},
        }

        return parameters, diagnostics

    @staticmethod
    def _calibrate_garch(prices: np.ndarray, dt: float, include_drift: bool) -> Tuple[Dict[str, float], Dict]:
        """Calibrate GARCH(1,1) model"""
        calibrator = GARCHCalibrator()
        result = calibrator.fit(prices, dt)

        parameters = {
            "omega": result.omega,
            "alpha": result.alpha,
            "beta": result.beta,
        }
        if include_drift:
            parameters["mu"] = result.mu

        diagnostics = {
            "logLikelihood": result.log_likelihood,
            "aic": result.aic,
            "bic": result.bic,
            "errorMetrics": {},
        }

        return parameters, diagnostics

    # ========== Q-measure (Risk-Neutral) Calibration Methods ==========

    @staticmethod
    def fetch_option_chain(
        ticker: str,
        reference_date: Optional[str] = None,
        risk_free_rate: float = 0.05,
        expiry: Optional[str] = None,
    ) -> Dict:
        """
        Fetch option chain from yfinance

        Args:
            ticker: Ticker symbol
            reference_date: Reference date (YYYY-MM-DD), defaults to today
            risk_free_rate: Risk-free rate for pricing
            expiry: Specific expiry date (YYYY-MM-DD) or None for all

        Returns:
            Dictionary with option chain data
        """
        try:
            # Parse reference date
            if reference_date:
                ref_date = datetime.strptime(reference_date, '%Y-%m-%d').date()
            else:
                ref_date = date.today()

            # Try to fetch dividend yield from yfinance
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                div_yield_raw = info.get('dividendYield', 0.0) or 0.0
                # yfinance returns dividend yield in percentage form (e.g., 1.07 for 1.07%)
                # Convert to decimal if it's greater than 1
                dividend_yield = div_yield_raw / 100.0 if div_yield_raw > 1 else div_yield_raw
            except:
                dividend_yield = 0.0

            # Initialize fetcher
            fetcher = YFinanceFetcher(
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
            )

            # Fetch option chain
            chain = fetcher.get_option_chain(ticker, reference_date=ref_date, expiry=expiry)

            # Convert to dict format
            options_list = []
            for opt in chain.options:
                options_list.append({
                    'strike': opt.strike,
                    'expiry': opt.expiry.strftime('%Y-%m-%d'),
                    'option_type': opt.option_type,
                    'bid': opt.bid,
                    'ask': opt.ask,
                    'mid': opt.mid,
                    'last': opt.last,
                    'volume': opt.volume,
                    'open_interest': opt.open_interest,
                    'implied_volatility': opt.implied_volatility,
                    'moneyness': opt.strike / chain.spot_price,
                })

            # Get available expiries
            expiries = sorted(set(opt.expiry.strftime('%Y-%m-%d') for opt in chain.options))

            return {
                'ticker': ticker,
                'spot_price': chain.spot_price,
                'reference_date': chain.reference_date.strftime('%Y-%m-%d'),
                'risk_free_rate': chain.risk_free_rate,
                'dividend_yield': chain.dividend_yield,
                'options': options_list,
                'expiries': expiries,
                'n_options': len(options_list),
            }

        except Exception as e:
            raise ValueError(f"Failed to fetch option chain for {ticker}: {str(e)}")

    @staticmethod
    def calibrate_qmeasure_model(
        ticker: str,
        model: str,
        reference_date: Optional[str] = None,
        risk_free_rate: float = 0.05,
        filter_params: Optional[Dict] = None,
        calibration_method: str = 'differential_evolution',
        maxiter: int = 1000,
    ) -> Tuple[Dict[str, float], Dict]:
        """
        Calibrate a Q-measure model to option prices

        Args:
            ticker: Ticker symbol
            model: Model name ('heston', 'sabr', 'dupire')
            reference_date: Reference date (YYYY-MM-DD)
            risk_free_rate: Risk-free rate
            filter_params: Parameters for filtering options (min_volume, min_oi, etc.)
            calibration_method: Optimization method
            maxiter: Maximum iterations

        Returns:
            Tuple of (parameters, diagnostics)
        """
        try:
            # Parse reference date
            if reference_date:
                ref_date = datetime.strptime(reference_date, '%Y-%m-%d').date()
            else:
                ref_date = date.today()

            # Try to fetch dividend yield from yfinance
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                div_yield_raw = info.get('dividendYield', 0.0) or 0.0
                # yfinance returns dividend yield in percentage form (e.g., 1.07 for 1.07%)
                # Convert to decimal if it's greater than 1
                dividend_yield = div_yield_raw / 100.0 if div_yield_raw > 1 else div_yield_raw
            except:
                dividend_yield = 0.0

            # Initialize fetcher
            fetcher = YFinanceFetcher(
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
            )

            # Fetch option chain
            chain = fetcher.get_option_chain(ticker, reference_date=ref_date)

            # Apply filters if provided
            if filter_params:
                chain = chain.filter(
                    min_volume=filter_params.get('min_volume', 0),
                    min_open_interest=filter_params.get('min_open_interest', 0),
                    max_spread_pct=filter_params.get('max_spread_pct', 1.0),
                    moneyness_range=tuple(filter_params.get('moneyness_range', [0.8, 1.2])),
                )

            if len(chain.options) == 0:
                raise ValueError(f"No valid options found for {ticker} after filtering")

            # Calibrate based on model type
            if model == 'heston':
                return CalibrationService._calibrate_heston_qmeasure(
                    chain=chain,
                    method=calibration_method,
                    maxiter=maxiter,
                )
            elif model == 'sabr':
                return CalibrationService._calibrate_sabr_qmeasure(
                    chain=chain,
                    method=calibration_method,
                    dividend_yield=dividend_yield,
                )
            elif model == 'dupire':
                return CalibrationService._calibrate_dupire_qmeasure(
                    chain=chain,
                    dividend_yield=dividend_yield,
                )
            else:
                raise ValueError(f"Unsupported Q-measure model: {model}. Supported: 'heston', 'sabr', 'dupire'")

        except Exception as e:
            raise ValueError(f"Q-measure calibration failed for {ticker}: {str(e)}")

    @staticmethod
    def _calibrate_heston_qmeasure(
        chain: OptionChain,
        method: str = 'differential_evolution',
        maxiter: int = 1000,
    ) -> Tuple[Dict[str, float], Dict]:
        """Calibrate Heston model to option chain (Q-measure)"""

        # Initialize calibrator
        calibrator = HestonCalibrator(
            pricer=None,  # Uses built-in characteristic function pricing
            weighting='vega',  # Weight by vega
            feller_penalty=100.0,
        )

        # Run calibration
        result: HestonCalibrationResult = calibrator.calibrate(
            chain=chain,
            spot=chain.spot_price,
            rate=chain.risk_free_rate,
            method=method,
            maxiter=maxiter,
        )

        # Extract parameters (convert numpy types to Python types for JSON serialization)
        parameters = {
            'kappa': float(result.kappa),
            'theta': float(result.theta),
            'xi': float(result.xi),
            'rho': float(result.rho),
            'v0': float(result.v0),
        }

        # Extract diagnostics (convert numpy types to Python types)
        diagnostics = {
            'rmse': float(result.rmse),
            'rmse_iv': float(result.rmse_iv) if result.rmse_iv is not None else None,
            'max_error': float(result.max_error),
            'n_iterations': int(result.n_iterations),
            'computation_time': float(result.computation_time),
            'success': bool(result.success),
            'message': str(result.message),
            'feller_satisfied': bool(result.feller_satisfied),
            'feller_condition': f"2*kappa*theta = {2*float(result.kappa)*float(result.theta):.4f}, xi^2 = {float(result.xi)**2:.4f}",
            'errorMetrics': {
                'rmse': float(result.rmse),
                'max_error': float(result.max_error),
            },
        }

        return parameters, diagnostics

    @staticmethod
    def _calibrate_sabr_qmeasure(
        chain: OptionChain,
        method: str = 'L-BFGS-B',
        dividend_yield: float = 0.0,
    ) -> Tuple[Dict[str, float], Dict]:
        """Calibrate SABR model to option chain (Q-measure)"""

        # Initialize calibrator with beta=0.5 (popular choice)
        # User can also set beta=None to calibrate it
        calibrator = SABRCalibrator(
            beta=0.5,  # Fix beta for faster calibration
            weighting='vega',
        )

        # Run calibration
        result: SABRCalibrationResult = calibrator.calibrate(
            chain=chain,
            forward=None,  # Will compute from spot and rate
            maturity=None,  # Will use median maturity
            method=method,
        )

        # Extract parameters (convert to Python types)
        parameters = {
            'alpha': float(result.alpha),
            'beta': float(result.beta),
            'rho': float(result.rho),
            'nu': float(result.nu),
        }

        # Extract diagnostics (convert to Python types)
        diagnostics = {
            'rmse_iv': float(result.rmse_iv),
            'max_error_iv': float(result.max_error_iv),
            'mean_error_iv': float(result.mean_error_iv),
            'n_iterations': int(result.n_iterations),
            'computation_time': float(result.computation_time),
            'success': bool(result.success),
            'message': str(result.message),
            'model_type': 'SABR Stochastic Volatility',
            'note': f'Beta fixed at {float(result.beta):.2f}. Calibrated to IV surface.',
            'errorMetrics': {
                'rmse_iv': float(result.rmse_iv),
                'max_error_iv': float(result.max_error_iv),
            },
        }

        return parameters, diagnostics

    @staticmethod
    def _calibrate_dupire_qmeasure(
        chain: OptionChain,
        dividend_yield: float = 0.0,
    ) -> Tuple[Dict[str, float], Dict]:
        """Extract Dupire local volatility surface from option chain (Q-measure)"""

        # Initialize calibrator
        calibrator = DupireCalibrator(
            smoothing=True,
            smoothing_sigma=1.0,
            interpolation='rbf',
        )

        # Run extraction
        result: DupireResult = calibrator.calibrate(
            chain=chain,
            spot=None,  # Will use chain.spot_price
            rate=None,  # Will use chain.risk_free_rate
            dividend_yield=dividend_yield,
            n_strikes=50,
            n_maturities=20,
        )

        # For Dupire, we return surface statistics rather than parameters
        # since it's a non-parametric model (convert to Python types)
        parameters = {
            'min_local_vol': float(np.min(result.local_vols)),
            'max_local_vol': float(np.max(result.local_vols)),
            'mean_local_vol': float(np.mean(result.local_vols)),
            'atm_local_vol': float(result.lv_function(chain.spot_price, result.maturities[len(result.maturities)//2])),
        }

        # Extract diagnostics (convert to Python types)
        diagnostics = {
            'n_options': int(result.n_options),
            'strike_range': [float(result.min_strike), float(result.max_strike)],
            'maturity_range': [float(result.min_maturity), float(result.max_maturity)],
            'grid_size': [int(len(result.maturities)), int(len(result.strikes))],
            'computation_time': float(result.computation_time),
            'smoothing_applied': bool(result.smoothing_applied),
            'model_type': 'Dupire Local Volatility',
            'note': 'Non-parametric model. Returns local vol surface σ_LV(K,T).',
            'surface_data': {
                'strikes': [float(x) for x in result.strikes.tolist()],
                'maturities': [float(x) for x in result.maturities.tolist()],
                'local_vols': [[float(x) for x in row] for row in result.local_vols.tolist()],
            },
            'errorMetrics': {},
        }

        return parameters, diagnostics


    @staticmethod
    def generate_vol_surface(
        ticker: str,
        model: str,
        parameters: Dict[str, float],
        spot_price: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
        n_strikes: int = 30,
        n_maturities: int = 20,
        strike_range: Tuple[float, float] = (0.7, 1.3),
        maturity_range: Tuple[float, float] = (0.05, 2.0),
    ) -> Dict:
        """
        Generate volatility surface from calibrated model parameters.

        Args:
            ticker: Ticker symbol
            model: Model name ('heston', 'sabr', 'dupire')
            parameters: Calibrated model parameters
            spot_price: Current spot price
            risk_free_rate: Risk-free rate
            dividend_yield: Dividend yield
            n_strikes: Number of strike points
            n_maturities: Number of maturity points
            strike_range: Strike range as (min_moneyness, max_moneyness)
            maturity_range: Maturity range in years (min, max)

        Returns:
            Dictionary with surface data (strikes, maturities, vols)
        """
        # Generate grid
        strikes = np.linspace(
            spot_price * strike_range[0],
            spot_price * strike_range[1],
            n_strikes
        )
        maturities = np.linspace(maturity_range[0], maturity_range[1], n_maturities)

        if model == 'heston':
            return CalibrationService._generate_heston_surface(
                parameters, strikes, maturities, spot_price, risk_free_rate, dividend_yield
            )
        elif model == 'sabr':
            return CalibrationService._generate_sabr_surface(
                parameters, strikes, maturities, spot_price, risk_free_rate, dividend_yield
            )
        elif model == 'dupire':
            # Dupire surface is already computed, just return it
            return {
                'strikes': parameters.get('strikes', strikes.tolist()),
                'maturities': parameters.get('maturities', maturities.tolist()),
                'vols': parameters.get('local_vols', []),
                'surface_type': 'local_vol',
            }
        else:
            raise ValueError(f"Unsupported model: {model}")

    @staticmethod
    def _generate_heston_surface(
        params: Dict[str, float],
        strikes: np.ndarray,
        maturities: np.ndarray,
        S0: float,
        r: float,
        q: float,
    ) -> Dict:
        """Generate implied volatility surface from Heston parameters."""
        from options_desk.pricer import COSPricer
        from options_desk.processes import Heston
        from options_desk.derivatives import EuropeanCall

        # Create Heston process
        heston = Heston(
            mu=r,
            kappa=params['kappa'],
            theta=params['theta'],
            sigma_v=params['xi'],
            rho=params['rho'],
            v0=params['v0'],
        )

        # Create pricer
        pricer = COSPricer(risk_free_rate=r, N=256, L=10.0)

        # Generate surface
        iv_surface = np.zeros((len(maturities), len(strikes)))

        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                # Price call option
                option = EuropeanCall(strike=K, maturity=T)
                try:
                    result = pricer.price(option, heston, X0=np.array([S0, params['v0']]))
                    call_price = max(result.price, 0.0)

                    # Back out implied volatility using Black-Scholes inversion
                    iv = CalibrationService._implied_vol_from_price(
                        call_price, S0, K, T, r, q, is_call=True
                    )
                    iv_surface[i, j] = iv
                except:
                    iv_surface[i, j] = 0.2  # Fallback

        return {
            'strikes': strikes.tolist(),
            'maturities': maturities.tolist(),
            'vols': iv_surface.tolist(),
            'surface_type': 'implied_vol',
        }

    @staticmethod
    def _generate_sabr_surface(
        params: Dict[str, float],
        strikes: np.ndarray,
        maturities: np.ndarray,
        S0: float,
        r: float,
        q: float,
    ) -> Dict:
        """Generate implied volatility surface from SABR parameters."""
        from options_desk.calibration.risk_neutral import SABRCalibrator

        calibrator = SABRCalibrator()

        # Generate surface
        iv_surface = np.zeros((len(maturities), len(strikes)))

        for i, T in enumerate(maturities):
            F = S0 * np.exp((r - q) * T)  # Forward price
            for j, K in enumerate(strikes):
                try:
                    iv = calibrator._sabr_hagan_iv(
                        F, K, T,
                        params['alpha'],
                        params['beta'],
                        params['rho'],
                        params['nu'],
                    )
                    iv_surface[i, j] = iv
                except:
                    iv_surface[i, j] = 0.2  # Fallback

        return {
            'strikes': strikes.tolist(),
            'maturities': maturities.tolist(),
            'vols': iv_surface.tolist(),
            'surface_type': 'implied_vol',
        }

    @staticmethod
    def _implied_vol_from_price(
        price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        is_call: bool = True,
    ) -> float:
        """
        Calculate implied volatility from option price using Newton-Raphson.

        Simple Black-Scholes IV inversion.
        """
        from scipy.stats import norm

        # Initial guess
        sigma = 0.2

        # Newton-Raphson iterations
        for _ in range(50):
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            if is_call:
                bs_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
                vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
            else:
                bs_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
                vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

            if vega < 1e-10:
                break

            diff = price - bs_price
            if abs(diff) < 1e-6:
                break

            sigma = sigma + diff / vega

            # Bounds
            sigma = max(0.01, min(sigma, 3.0))

        return sigma


# Singleton instance
calibration_service = CalibrationService()
