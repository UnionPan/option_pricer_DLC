"""
Multi-asset Heston QMLE calibration pipeline.

Pulls daily OHLC for a basket of equities via yfinance, runs
HestonQMLECalibrator on each asset in parallel via joblib, and persists
results as a parquet table for downstream use (stress scenario
generation, neural-hedging training, cross-asset risk analytics).

Outputs:
    - parquet table with columns
      [ticker, kappa, theta, sigma_v, rho, mu, v0,
       n_observations, log_likelihood, feller_ratio, variance_proxy_r2,
       converged, calibration_date]
    - calibration_report.json with summary stats (convergence rate,
      parameter distribution percentiles, failed tickers)

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .heston_qmle import HestonQMLECalibrator, HestonQMLEResult

logger = logging.getLogger(__name__)


# A canonical, diversified default basket (large-cap US equities across sectors).
# Used as the default when callers don't pass their own list.
DEFAULT_BASKET_50 = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "ORCL", "CRM",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "AXP", "SCHW", "USB",
    # Healthcare
    "JNJ", "PFE", "UNH", "MRK", "ABBV", "LLY", "TMO", "ABT", "DHR", "BMY",
    # Consumer
    "WMT", "PG", "KO", "PEP", "COST", "HD", "MCD", "NKE", "SBUX", "TGT",
    # Industrials / energy / materials
    "XOM", "CVX", "BA", "CAT", "GE", "HON", "RTX", "UNP", "LMT", "DE",
]


@dataclass
class AssetCalibration:
    """Per-asset calibration record."""
    ticker: str
    kappa: float
    theta: float
    sigma_v: float
    rho: float
    mu: float
    v0: float
    n_observations: int
    log_likelihood: float
    feller_ratio: float
    variance_proxy_r2: float
    converged: bool
    calibration_date: str
    error: str = ""

    @classmethod
    def from_qmle_result(cls, ticker: str, r: HestonQMLEResult, cal_date: str) -> "AssetCalibration":
        return cls(
            ticker=ticker, kappa=r.kappa, theta=r.theta, sigma_v=r.sigma_v,
            rho=r.rho, mu=r.mu, v0=r.v0,
            n_observations=r.n_observations, log_likelihood=r.log_likelihood,
            feller_ratio=r.feller_ratio, variance_proxy_r2=r.variance_proxy_r2,
            converged=r.converged, calibration_date=cal_date, error="",
        )

    @classmethod
    def failed(cls, ticker: str, msg: str, cal_date: str) -> "AssetCalibration":
        return cls(
            ticker=ticker, kappa=float("nan"), theta=float("nan"),
            sigma_v=float("nan"), rho=float("nan"), mu=float("nan"),
            v0=float("nan"), n_observations=0, log_likelihood=float("nan"),
            feller_ratio=float("nan"), variance_proxy_r2=float("nan"),
            converged=False, calibration_date=cal_date, error=msg,
        )


def _fetch_one_close(ticker: str, period: str = "5y") -> np.ndarray | None:
    """Pull daily adjusted-close for one ticker via yfinance. None on failure."""
    try:
        import yfinance as yf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        # In yfinance >=1.0 the OHLCV columns might be a MultiIndex
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        prices = close.dropna().to_numpy(dtype=np.float64).ravel()
        return prices if prices.size >= 60 else None
    except Exception as e:
        logger.warning("yfinance fetch failed for %s: %s", ticker, e)
        return None


def _calibrate_one(
    ticker: str,
    prices: np.ndarray,
    cal_date: str,
    smooth_window: int = 10,
    dt: float = 1.0 / 252.0,
) -> AssetCalibration:
    """Run Heston QMLE on one asset's prices. Catches any failure."""
    try:
        if prices is None or prices.size < 60:
            return AssetCalibration.failed(
                ticker, "insufficient data (<60 days)", cal_date,
            )
        result = HestonQMLECalibrator(smooth_window=smooth_window).fit(prices, dt=dt)
        return AssetCalibration.from_qmle_result(ticker, result, cal_date)
    except Exception as e:
        return AssetCalibration.failed(
            ticker, f"{type(e).__name__}: {e}", cal_date,
        )


def calibrate_universe(
    tickers: list[str],
    period: str = "5y",
    smooth_window: int = 10,
    dt: float = 1.0 / 252.0,
    n_jobs: int = -1,
    prefetched_prices: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Calibrate Heston QMLE for a list of tickers in parallel.

    Args:
        tickers: list of ticker symbols (e.g. ['AAPL', 'MSFT', ...]).
        period: yfinance period string ('1y', '2y', '5y', 'max').
        smooth_window: realized-variance proxy smoothing window (days).
        dt: time step in years (1/252 for daily trading days).
        n_jobs: joblib parallelism. -1 = all cores.
        prefetched_prices: optional pre-fetched ``{ticker: prices_1d}`` to
            skip the yfinance download step (useful for testing or when
            data lives in a local cache).

    Returns:
        pd.DataFrame with one row per ticker, columns as in
        :class:`AssetCalibration`.
    """
    cal_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if prefetched_prices is None:
        logger.info("Downloading %d tickers (period=%s) via yfinance...",
                    len(tickers), period)
        # Fetch sequentially (yfinance is rate-limited; parallel downloads
        # often get throttled and produce empty frames)
        prices_by_ticker = {t: _fetch_one_close(t, period=period) for t in tickers}
        n_ok = sum(1 for p in prices_by_ticker.values() if p is not None)
        logger.info("Fetched %d/%d tickers successfully", n_ok, len(tickers))
    else:
        prices_by_ticker = prefetched_prices

    logger.info("Running parallel Heston QMLE calibration (n_jobs=%s)...", n_jobs)
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_calibrate_one)(
            t, prices_by_ticker.get(t), cal_date,
            smooth_window=smooth_window, dt=dt,
        )
        for t in tickers
    )

    df = pd.DataFrame([asdict(r) for r in results])
    n_conv = int(df["converged"].sum())
    logger.info(
        "Calibration done: %d/%d converged (%.1f%%)",
        n_conv, len(df), 100.0 * n_conv / max(len(df), 1),
    )
    return df


def calibration_report(df: pd.DataFrame) -> dict:
    """Build summary stats over a calibration DataFrame.

    Returns a dict suitable for json.dumps with:
        - convergence rate
        - per-parameter median/quantile distribution (over converged assets)
        - failed tickers + reasons
    """
    converged = df[df["converged"]]
    failed = df[~df["converged"]][["ticker", "error"]].to_dict(orient="records")

    report: dict = {
        "n_total": int(len(df)),
        "n_converged": int(len(converged)),
        "convergence_rate": float(len(converged) / max(len(df), 1)),
        "failed": failed,
        "param_distribution": {},
    }

    if not converged.empty:
        for col in ("kappa", "theta", "sigma_v", "rho", "mu", "v0",
                    "feller_ratio", "variance_proxy_r2"):
            s = converged[col].dropna()
            if s.empty:
                continue
            report["param_distribution"][col] = {
                "p05": float(s.quantile(0.05)),
                "p25": float(s.quantile(0.25)),
                "p50": float(s.quantile(0.50)),
                "p75": float(s.quantile(0.75)),
                "p95": float(s.quantile(0.95)),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            }
        # Cross-asset average Feller violation rate
        report["feller_violation_rate"] = float(
            (converged["feller_ratio"] <= 1.0).mean()
        )

    return report


def save_calibration(
    df: pd.DataFrame,
    out_dir: str | Path,
    report: Optional[dict] = None,
) -> tuple[Path, Path]:
    """Persist calibration to parquet + report to json under out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "calibrations.parquet"
    df.to_parquet(parquet_path, index=False)

    report_path = out_dir / "calibration_report.json"
    if report is None:
        report = calibration_report(df)
    with report_path.open("w") as fp:
        json.dump(report, fp, indent=2, default=float)
    logger.info("Wrote %s and %s", parquet_path, report_path)
    return parquet_path, report_path
