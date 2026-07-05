"""
Local parquet price lake for universe-scale calibration.

Layout under ``root``:
    prices/<TICKER>.parquet   one file per ticker, DatetimeIndex,
                              columns = PRICE_COLUMNS
    coverage.json             ticker -> widest {start, end} range ever
                              requested (so dead tickers aren't refetched)

The fetcher is injectable (``FetcherFn``) so tests never touch the network;
the default is the chunked yfinance fetcher (``fetch_yfinance``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)

PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]

# (tickers, start_iso, end_iso) -> {ticker: OHLCV frame with DatetimeIndex}
FetcherFn = Callable[[list[str], str, str], dict[str, pd.DataFrame]]


@dataclass
class EnsureReport:
    fetched: list[str]
    cached: list[str]
    failed: dict[str, str]


class PriceStore:
    def __init__(self, root: str | Path, fetcher: FetcherFn | None = None):
        self.root = Path(root)
        self.prices_dir = self.root / "prices"
        self.prices_dir.mkdir(parents=True, exist_ok=True)
        self._coverage_path = self.root / "coverage.json"
        if fetcher is None:
            fetcher = fetch_yfinance
        self._fetcher = fetcher

    # -- storage ----------------------------------------------------------

    def _path(self, ticker: str) -> Path:
        return self.prices_dir / f"{ticker.upper()}.parquet"

    def get_prices(self, ticker: str) -> pd.DataFrame | None:
        p = self._path(ticker)
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index)
        return df[PRICE_COLUMNS]

    def put_prices(self, ticker: str, df: pd.DataFrame) -> None:
        """Upsert: merge with existing rows; newer write wins on overlap."""
        new = df.copy()
        new.index = pd.to_datetime(new.index)
        new = new[PRICE_COLUMNS]
        old = self.get_prices(ticker)
        if old is not None:
            new = pd.concat([old, new])
        new = new[~new.index.duplicated(keep="last")].sort_index()
        new.to_parquet(self._path(ticker))

    # -- coverage ledger ---------------------------------------------------

    def _load_coverage(self) -> dict:
        if self._coverage_path.exists():
            return json.loads(self._coverage_path.read_text())
        return {}

    def _save_coverage(self, cov: dict) -> None:
        self._coverage_path.write_text(json.dumps(cov, indent=0, sort_keys=True))

    @staticmethod
    def _covered(rec: dict | None, start: str, end: str) -> bool:
        return (rec is not None
                and rec["start"] <= start
                and rec["end"] >= end)

    # -- ensure ------------------------------------------------------------

    def ensure(self, tickers: list[str], start, end) -> EnsureReport:
        """Fetch price history for any ticker whose ledger doesn't already
        cover [start, end]. Records the attempt (success or not) in the
        ledger so dead tickers aren't refetched every run."""
        start, end = str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date())
        cov = self._load_coverage()
        tickers = [t.upper() for t in tickers]
        need = [t for t in tickers if not self._covered(cov.get(t), start, end)]
        cached = [t for t in tickers if t not in set(need)]

        fetched: list[str] = []
        failed: dict[str, str] = {}
        if need:
            logger.info("PriceStore.ensure: fetching %d/%d tickers (%s..%s)",
                        len(need), len(tickers), start, end)
            got = self._fetcher(need, start, end)
            got = {k.upper(): v for k, v in got.items()}
            for t in need:
                df = got.get(t)
                if df is not None and not df.empty:
                    self.put_prices(t, df)
                    fetched.append(t)
                else:
                    failed[t] = "no data returned"
                old = cov.get(t)
                cov[t] = {
                    "start": min(start, old["start"]) if old else start,
                    "end": max(end, old["end"]) if old else end,
                }
            self._save_coverage(cov)
        return EnsureReport(fetched=fetched, cached=cached, failed=failed)


_YF_RENAME = {"Open": "open", "High": "high", "Low": "low",
              "Close": "close", "Adj Close": "adj_close", "Volume": "volume"}


def _split_multi_ticker_frame(
    raw: pd.DataFrame, tickers: list[str],
) -> dict[str, pd.DataFrame]:
    """Split a (possibly MultiIndex-column) yf.download frame into
    per-ticker OHLCV frames with our canonical column names. Tickers with
    no data are simply absent from the result."""
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            if t not in raw.columns.get_level_values(0):
                continue
            df_t = raw[t].copy()
        else:
            df_t = raw.copy()
        df_t = df_t.rename(columns=_YF_RENAME)
        missing = [c for c in PRICE_COLUMNS if c not in df_t.columns]
        if missing:
            continue
        df_t = df_t[PRICE_COLUMNS].dropna(how="all")
        if not df_t.empty:
            out[t] = df_t
    return out


def fetch_yfinance(
    tickers: list[str],
    start: str,
    end: str,
    chunk_size: int = 100,
    max_retries: int = 3,
    pause: float = 1.0,
) -> dict[str, pd.DataFrame]:
    """Chunked multi-ticker yfinance download with exponential-backoff
    retries. Returns only tickers that came back with data."""
    import time

    import yfinance as yf

    out: dict[str, pd.DataFrame] = {}
    n_chunks = (len(tickers) + chunk_size - 1) // chunk_size
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        raw = None
        for attempt in range(max_retries):
            try:
                raw = yf.download(
                    chunk, start=start, end=end, group_by="ticker",
                    auto_adjust=False, actions=False,
                    progress=False, threads=True,
                )
                break
            except Exception as e:                       # noqa: BLE001
                logger.warning("yfinance chunk %d/%d attempt %d failed: %s",
                               i // chunk_size + 1, n_chunks, attempt + 1, e)
                time.sleep(pause * (2 ** attempt))
        if raw is None or raw.empty:
            continue
        chunk_result = _split_multi_ticker_frame(raw, chunk)
        out.update(chunk_result)
        logger.info("fetched chunk %d/%d: %d/%d tickers",
                    i // chunk_size + 1, n_chunks,
                    len(chunk_result), len(chunk))
        time.sleep(pause)
    return out
