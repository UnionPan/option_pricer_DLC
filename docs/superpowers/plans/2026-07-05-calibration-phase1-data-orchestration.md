# Calibration Phase 1: Data Layer + Orchestration Skeleton — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A parquet price lake (`PriceStore`), named universe manifests, and a model-registry/runner/results-store orchestration layer that calibrates 1000+ US equities with the *existing scipy calibrators* (GBM, GARCH, Heston QMLE), with checkpoint/resume and no repeated downloads.

**Architecture:** Four new units: `calibration/data/universe.py` (universe CSVs + sectors), `calibration/data/price_store.py` (per-ticker parquet files + coverage ledger + chunked yfinance fetcher, fetcher injectable for tests), `calibration/pipeline/` (registry mapping model name → fit function over 1D prices; results store with per-model parquet + `.done` markers + manifest; runner that wires universe → ensure → joblib → persist). The existing `multi_asset_pipeline.py` stays untouched; `scripts/calibrate_universe.py` is rewritten on top of the runner.

**Tech Stack:** pandas 2.x, pyarrow 24 (present), numpy, scipy, joblib, yfinance. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-05-universe-scale-p-measure-calibration-design.md`

## Global Constraints

- Python env: ALWAYS `source /home/union/miniconda3/etc/profile.d/conda.sh && conda activate options-desk` before running python/pytest. Bare `python` is the wrong env.
- Repo root: `/home/union/quant/option_pricer_DLC`. Run pytest from repo root.
- Tests must not touch the network. yfinance is only exercised manually in Task 10.
- Price lake lives at `data/price_lake/` and is gitignored. Universe CSVs live at `data/universes/` and ARE committed.
- `dt = 1/252` everywhere (daily bars).
- Do not modify `multi_asset_pipeline.py`, existing calibrators, or anything under `deep_hedging/` (in-flight work there — do not `git add -A`; stage files explicitly).
- Commit after every task with the exact files listed in that task.

---

### Task 1: Universe loader

**Files:**
- Create: `src/options_desk/calibration/data/universe.py`
- Test: `tests/test_universe.py`

**Interfaces:**
- Produces: `Universe` (frozen dataclass: `name: str`, `tickers: list[str]`, `sectors: dict[str, str]`), `load_universe(name_or_path, universe_dir=None) -> Universe`, `available_universes(universe_dir=None) -> list[str]`. Task 8's runner calls `load_universe` and reads `.tickers` / `.sectors`; Task 9's CLI calls `available_universes`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_universe.py
from pathlib import Path

import pytest

from options_desk.calibration.data.universe import (
    Universe,
    available_universes,
    load_universe,
)


def _write_csv(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / f"{name}.csv"
    p.write_text(text)
    return p


def test_load_universe_by_path(tmp_path):
    p = _write_csv(tmp_path, "mini", "ticker,sector\naapl,Tech\nMSFT,Tech\nXOM,Energy\n")
    u = load_universe(p)
    assert u.name == "mini"
    assert u.tickers == ["AAPL", "MSFT", "XOM"]      # upper-cased, order kept
    assert u.sectors["AAPL"] == "Tech"
    assert u.sectors["XOM"] == "Energy"


def test_load_universe_by_name_uses_universe_dir(tmp_path):
    _write_csv(tmp_path, "sp3", "ticker,sector\nAAA,X\nBBB,Y\nCCC,Z\n")
    u = load_universe("sp3", universe_dir=tmp_path)
    assert u.name == "sp3"
    assert len(u.tickers) == 3


def test_load_universe_dedupes_and_defaults_sector(tmp_path):
    p = _write_csv(tmp_path, "dupes", "ticker\nAAPL\nAAPL\nMSFT\n")
    u = load_universe(p)
    assert u.tickers == ["AAPL", "MSFT"]
    assert u.sectors["AAPL"] == "UNKNOWN"


def test_load_universe_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_universe("nope", universe_dir=tmp_path)


def test_load_universe_requires_ticker_column(tmp_path):
    p = _write_csv(tmp_path, "bad", "symbol\nAAPL\n")
    with pytest.raises(ValueError, match="ticker"):
        load_universe(p)


def test_available_universes(tmp_path):
    _write_csv(tmp_path, "b_uni", "ticker\nAAA\n")
    _write_csv(tmp_path, "a_uni", "ticker\nAAA\n")
    assert available_universes(universe_dir=tmp_path) == ["a_uni", "b_uni"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source /home/union/miniconda3/etc/profile.d/conda.sh && conda activate options-desk && pytest tests/test_universe.py -v`
Expected: FAIL / collection error — `ModuleNotFoundError` or `ImportError` on `universe`.

- [ ] **Step 3: Write the implementation**

```python
# src/options_desk/calibration/data/universe.py
"""
Named equity universes for large-scale calibration.

A universe is a CSV under ``data/universes/`` (repo root) with at least a
``ticker`` column and optionally a ``sector`` column (GICS sector, used by
the cross-asset hierarchical pooling stage).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# repo_root/src/options_desk/calibration/data/universe.py -> repo_root
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _default_universe_dir() -> Path:
    return _REPO_ROOT / "data" / "universes"


@dataclass(frozen=True)
class Universe:
    """An ordered list of tickers plus per-ticker sector labels."""

    name: str
    tickers: list[str]
    sectors: dict[str, str]

    def __len__(self) -> int:
        return len(self.tickers)


def load_universe(
    name_or_path: str | Path,
    universe_dir: str | Path | None = None,
) -> Universe:
    """Load a universe by CSV path or by name from the universe directory."""
    p = Path(name_or_path)
    if p.suffix.lower() != ".csv":
        base = Path(universe_dir) if universe_dir else _default_universe_dir()
        p = base / f"{name_or_path}.csv"
    if not p.exists():
        raise FileNotFoundError(f"universe file not found: {p}")

    df = pd.read_csv(p)
    if "ticker" not in df.columns:
        raise ValueError(f"universe CSV {p} has no 'ticker' column")

    raw = df["ticker"].astype(str).str.strip().str.upper()
    if "sector" in df.columns:
        sector_raw = df["sector"].astype(str).str.strip()
    else:
        sector_raw = pd.Series(["UNKNOWN"] * len(df))

    tickers: list[str] = []
    sectors: dict[str, str] = {}
    seen: set[str] = set()
    for t, s in zip(raw, sector_raw):
        if not t or t in seen:
            continue
        seen.add(t)
        tickers.append(t)
        sectors[t] = s if s and s.lower() != "nan" else "UNKNOWN"

    return Universe(name=p.stem, tickers=tickers, sectors=sectors)


def available_universes(universe_dir: str | Path | None = None) -> list[str]:
    base = Path(universe_dir) if universe_dir else _default_universe_dir()
    if not base.exists():
        return []
    return sorted(f.stem for f in base.glob("*.csv"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_universe.py -v` (env activated)
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/options_desk/calibration/data/universe.py tests/test_universe.py
git commit -m "feat(calibration): universe manifests (load_universe, sectors)"
```

---

### Task 2: Universe builder script (S&P 500/400/600/1500 from Wikipedia)

**Files:**
- Create: `scripts/build_universes.py`
- Test: `tests/test_build_universes.py`

**Interfaces:**
- Produces: `normalize_constituents(df, ticker_col, sector_col) -> pd.DataFrame[ticker, sector]` (importable from the script), and — when run manually with network — `data/universes/{sp500,sp400,sp600,sp1500}.csv`. Task 10 runs it manually.

Only the pure normalization function is unit-tested; the Wikipedia fetch is manual (Task 10).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_universes.py
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_universes import normalize_constituents  # noqa: E402


def test_normalize_constituents():
    raw = pd.DataFrame({
        "Symbol": [" aapl ", "BRK.B", "MSFT", "MSFT", ""],
        "GICS Sector": ["Information Technology", "Financials",
                        "Information Technology", "Information Technology", "X"],
    })
    out = normalize_constituents(raw, "Symbol", "GICS Sector")
    assert list(out.columns) == ["ticker", "sector"]
    # BRK.B -> BRK-B (yfinance convention), upper-cased, deduped, no empties
    assert out["ticker"].tolist() == ["AAPL", "BRK-B", "MSFT"]
    assert out.loc[out["ticker"] == "BRK-B", "sector"].item() == "Financials"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_build_universes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'build_universes'`.

- [ ] **Step 3: Write the script**

```python
# scripts/build_universes.py
"""
Build universe CSVs (ticker, sector) from Wikipedia S&P constituent lists.

Requires network. Run manually, then commit the CSVs:

    python scripts/build_universes.py            # writes data/universes/*.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "universes"

SOURCES = {
    "sp500": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
              "Symbol", "GICS Sector"),
    "sp400": ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
              "Symbol", "GICS Sector"),
    "sp600": ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
              "Symbol", "GICS Sector"),
}


def normalize_constituents(
    df: pd.DataFrame, ticker_col: str, sector_col: str,
) -> pd.DataFrame:
    """Standardize a raw constituents table to [ticker, sector].

    Upper-cases, trims, maps '.' share-class separators to '-' (yfinance
    convention, e.g. BRK.B -> BRK-B), drops blanks and duplicates.
    """
    out = pd.DataFrame({
        "ticker": (df[ticker_col].astype(str).str.strip().str.upper()
                   .str.replace(".", "-", regex=False)),
        "sector": df[sector_col].astype(str).str.strip(),
    })
    out = out[out["ticker"].str.len() > 0]
    out = out.drop_duplicates("ticker").sort_values("ticker")
    return out.reset_index(drop=True)


def _find_constituents_table(tables: list[pd.DataFrame], ticker_col: str) -> pd.DataFrame:
    for t in tables:
        if ticker_col in t.columns and len(t) > 100:
            return t
    raise RuntimeError(f"no table with column '{ticker_col}' and >100 rows found")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {}
    for name, (url, ticker_col, sector_col) in SOURCES.items():
        print(f"fetching {name}: {url}")
        tables = pd.read_html(url)
        raw = _find_constituents_table(tables, ticker_col)
        norm = normalize_constituents(raw, ticker_col, sector_col)
        norm.to_csv(OUT_DIR / f"{name}.csv", index=False)
        frames[name] = norm
        print(f"  wrote {name}.csv ({len(norm)} names)")

    sp1500 = (pd.concat(frames.values())
              .drop_duplicates("ticker").sort_values("ticker")
              .reset_index(drop=True))
    sp1500.to_csv(OUT_DIR / "sp1500.csv", index=False)
    print(f"  wrote sp1500.csv ({len(sp1500)} names)")


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_build_universes.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_universes.py tests/test_build_universes.py
git commit -m "feat(calibration): universe builder script for S&P 500/400/600/1500"
```

---

### Task 3: PriceStore — parquet lake with coverage ledger and injectable fetcher

**Files:**
- Create: `src/options_desk/calibration/data/price_store.py`
- Test: `tests/test_price_store.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `PriceStore(root, fetcher=fetch_yfinance)` with `get_prices(ticker) -> pd.DataFrame | None` (DatetimeIndex, columns `open, high, low, close, adj_close, volume`), `put_prices(ticker, df)` (upsert), `ensure(tickers, start, end) -> EnsureReport(fetched, cached, failed)`. `FetcherFn = Callable[[list[str], str, str], dict[str, pd.DataFrame]]`. `PRICE_COLUMNS` constant. Tasks 5, 8, 9 consume `PriceStore`; Task 4 adds the real yfinance fetcher into this same file.

Storage layout: one parquet per ticker at `<root>/prices/<TICKER>.parquet`; a JSON coverage ledger at `<root>/coverage.json` mapping ticker → the widest `{start, end}` date range ever *requested and attempted*. `ensure` refetches only when the requested range is not covered by the ledger — so permanently-dead tickers are not refetched every run.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_price_store.py
import numpy as np
import pandas as pd
import pytest

from options_desk.calibration.data.price_store import (
    PRICE_COLUMNS,
    EnsureReport,
    PriceStore,
)


def _frame(start="2024-01-01", n=10, base=100.0) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n)
    close = base + np.arange(n, dtype=float)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1,
         "close": close, "adj_close": close, "volume": 1e6},
        index=idx,
    )


def test_put_get_roundtrip(tmp_path):
    store = PriceStore(tmp_path)
    df = _frame()
    store.put_prices("aapl", df)
    got = store.get_prices("AAPL")           # case-insensitive
    assert got is not None
    assert list(got.columns) == PRICE_COLUMNS
    assert len(got) == 10
    assert got.index.is_monotonic_increasing


def test_get_missing_returns_none(tmp_path):
    assert PriceStore(tmp_path).get_prices("NOPE") is None


def test_put_upserts_and_dedupes(tmp_path):
    store = PriceStore(tmp_path)
    store.put_prices("MSFT", _frame("2024-01-01", n=10, base=100.0))
    # overlapping window with different values: last write wins on overlap
    store.put_prices("MSFT", _frame("2024-01-08", n=10, base=200.0))
    got = store.get_prices("MSFT")
    assert len(got) == len(pd.bdate_range("2024-01-01", periods=10).union(
        pd.bdate_range("2024-01-08", periods=10)))
    assert got["close"].iloc[-1] == pytest.approx(209.0)
    # overlap date takes the newer value
    overlap = pd.bdate_range("2024-01-08", periods=1)[0]
    assert got.loc[overlap, "close"] == pytest.approx(200.0)


def test_ensure_fetches_missing_then_caches(tmp_path):
    calls = []

    def fake_fetcher(tickers, start, end):
        calls.append(list(tickers))
        return {t: _frame() for t in tickers if t != "DEAD"}

    store = PriceStore(tmp_path, fetcher=fake_fetcher)
    rep = store.ensure(["AAPL", "MSFT", "DEAD"], "2024-01-01", "2024-06-01")
    assert isinstance(rep, EnsureReport)
    assert sorted(rep.fetched) == ["AAPL", "MSFT"]
    assert rep.failed == {"DEAD": "no data returned"}
    assert calls == [["AAPL", "MSFT", "DEAD"]]

    # Second ensure over the same range: everything (incl. DEAD) is covered
    rep2 = store.ensure(["AAPL", "MSFT", "DEAD"], "2024-01-01", "2024-06-01")
    assert sorted(rep2.cached) == ["AAPL", "DEAD", "MSFT"]
    assert rep2.fetched == [] and rep2.failed == {}
    assert len(calls) == 1                     # no second fetch


def test_ensure_refetches_when_range_extends(tmp_path):
    calls = []

    def fake_fetcher(tickers, start, end):
        calls.append((list(tickers), start, end))
        return {t: _frame() for t in tickers}

    store = PriceStore(tmp_path, fetcher=fake_fetcher)
    store.ensure(["AAPL"], "2024-01-01", "2024-06-01")
    store.ensure(["AAPL"], "2024-01-01", "2024-09-01")   # extends end
    assert len(calls) == 2
    # ledger now covers the union; a sub-range is cached
    rep = store.ensure(["AAPL"], "2024-02-01", "2024-08-01")
    assert rep.cached == ["AAPL"] and len(calls) == 2


def test_coverage_ledger_persists_across_instances(tmp_path):
    def fake_fetcher(tickers, start, end):
        return {t: _frame() for t in tickers}

    PriceStore(tmp_path, fetcher=fake_fetcher).ensure(
        ["AAPL"], "2024-01-01", "2024-06-01")

    def exploding(tickers, start, end):
        raise AssertionError("should not fetch")

    rep = PriceStore(tmp_path, fetcher=exploding).ensure(
        ["AAPL"], "2024-01-01", "2024-06-01")
    assert rep.cached == ["AAPL"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_price_store.py -v`
Expected: collection error — module does not exist.

- [ ] **Step 3: Write the implementation**

```python
# src/options_desk/calibration/data/price_store.py
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
            from .price_store import fetch_yfinance as fetcher  # type: ignore
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
```

(`fetch_yfinance` is added to this file in Task 4; the self-import default in
`__init__` resolves then. Until Task 4, always pass an explicit fetcher —
tests do.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_price_store.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/options_desk/calibration/data/price_store.py tests/test_price_store.py
git commit -m "feat(calibration): PriceStore parquet lake with coverage ledger"
```

---

### Task 4: Chunked yfinance fetcher

**Files:**
- Modify: `src/options_desk/calibration/data/price_store.py` (append; also simplify the `__init__` default)
- Test: `tests/test_price_store.py` (append)

**Interfaces:**
- Produces: `fetch_yfinance(tickers, start, end, chunk_size=100, max_retries=3, pause=1.0) -> dict[str, pd.DataFrame]` and the pure helper `_split_multi_ticker_frame(raw, tickers) -> dict[str, pd.DataFrame]`. Only the helper is unit-tested (no network).

- [ ] **Step 1: Write the failing test (append to tests/test_price_store.py)**

```python
def test_split_multi_ticker_frame():
    from options_desk.calibration.data.price_store import _split_multi_ticker_frame

    idx = pd.bdate_range("2024-01-01", periods=5)
    cols = pd.MultiIndex.from_product(
        [["AAPL", "MSFT"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]])
    raw = pd.DataFrame(1.0, index=idx, columns=cols)
    raw.loc[:, ("MSFT", "Close")] = float("nan")   # partially-nan column survives
    raw.loc[:, ("MSFT", "Open")] = float("nan")

    out = _split_multi_ticker_frame(raw, ["AAPL", "MSFT", "GONE"])
    assert set(out) == {"AAPL", "MSFT"}            # GONE absent, not an error
    assert list(out["AAPL"].columns) == PRICE_COLUMNS
    assert len(out["AAPL"]) == 5


def test_split_single_ticker_frame():
    from options_desk.calibration.data.price_store import _split_multi_ticker_frame

    idx = pd.bdate_range("2024-01-01", periods=5)
    raw = pd.DataFrame(
        {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0,
         "Adj Close": 1.0, "Volume": 1.0}, index=idx)
    out = _split_multi_ticker_frame(raw, ["AAPL"])
    assert set(out) == {"AAPL"}
    assert list(out["AAPL"].columns) == PRICE_COLUMNS
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_price_store.py -v -k split`
Expected: FAIL — `ImportError: cannot import name '_split_multi_ticker_frame'`.

- [ ] **Step 3: Implement (append to price_store.py; fix the default fetcher)**

First, replace the `__init__` fetcher-default lines:

```python
        if fetcher is None:
            fetcher = fetch_yfinance
        self._fetcher = fetcher
```

(Module-level name resolves at call time; defining `fetch_yfinance` below the
class is fine.) Then append:

```python
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
        out.update(_split_multi_ticker_frame(raw, chunk))
        logger.info("fetched chunk %d/%d: %d/%d tickers",
                    i // chunk_size + 1, n_chunks,
                    len(_split_multi_ticker_frame(raw, chunk)), len(chunk))
        time.sleep(pause)
    return out
```

- [ ] **Step 4: Run the full file's tests**

Run: `pytest tests/test_price_store.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/options_desk/calibration/data/price_store.py tests/test_price_store.py
git commit -m "feat(calibration): chunked yfinance fetcher with retries"
```

---

### Task 5: Aligned returns matrix with NaN policy

**Files:**
- Modify: `src/options_desk/calibration/data/price_store.py` (add method + dataclass)
- Test: `tests/test_price_store.py` (append)

**Interfaces:**
- Produces: `ReturnsMatrix` dataclass (`returns: np.ndarray (T, N) float32` log-returns, `dates: pd.DatetimeIndex` (T), `tickers: list[str]` (N), `excluded: dict[str, str]`), and `PriceStore.returns_matrix(tickers, start, end, min_obs=504, max_ffill=5, edge_tolerance_days=5) -> ReturnsMatrix`. Phase 3 (cross-asset) is the main consumer; Phase 1 only builds and tests it.

NaN policy (spec): a name needs ≥ `min_obs` observations in-window; interior gaps ≤ `max_ffill` business days are forward-filled; a name must start within `edge_tolerance_days` business days of the window start and end within the same tolerance of the window end, else it is excluded (`"partial window coverage"`). Every exclusion carries a reason.

- [ ] **Step 1: Write the failing tests (append to tests/test_price_store.py)**

```python
def _store_with(tmp_path, series: dict[str, pd.Series]) -> PriceStore:
    store = PriceStore(tmp_path, fetcher=lambda t, s, e: {})
    for tkr, s in series.items():
        close = s.astype(float)
        df = pd.DataFrame({c: close for c in ["open", "high", "low", "close",
                                              "adj_close"]})
        df["volume"] = 1e6
        store.put_prices(tkr, df)
    return store


def test_returns_matrix_alignment_and_shape(tmp_path):
    idx = pd.bdate_range("2023-01-02", periods=300)
    store = _store_with(tmp_path, {
        "AAA": pd.Series(range(100, 400), index=idx),
        "BBB": pd.Series(range(200, 500), index=idx),
    })
    rm = store.returns_matrix(["AAA", "BBB"], idx[0], idx[-1], min_obs=250)
    assert rm.tickers == ["AAA", "BBB"]
    assert rm.returns.shape == (299, 2)              # T-1 log returns
    assert rm.returns.dtype == np.float32
    assert rm.excluded == {}
    assert len(rm.dates) == 299


def test_returns_matrix_excludes_short_history(tmp_path):
    idx = pd.bdate_range("2023-01-02", periods=300)
    short_idx = idx[-100:]
    store = _store_with(tmp_path, {
        "AAA": pd.Series(range(100, 400), index=idx),
        "NEW": pd.Series(range(100, 200), index=short_idx),
    })
    rm = store.returns_matrix(["AAA", "NEW"], idx[0], idx[-1], min_obs=250)
    assert rm.tickers == ["AAA"]
    assert "NEW" in rm.excluded
    assert "insufficient" in rm.excluded["NEW"]


def test_returns_matrix_ffills_small_gaps_only(tmp_path):
    idx = pd.bdate_range("2023-01-02", periods=300)
    gappy = pd.Series(np.arange(100.0, 400.0), index=idx)
    gappy = gappy.drop(idx[50:53])                    # 3-day gap: OK
    holey = pd.Series(np.arange(100.0, 400.0), index=idx)
    holey = holey.drop(idx[100:110])                  # 10-day gap: excluded
    store = _store_with(tmp_path, {
        "AAA": pd.Series(np.arange(100.0, 400.0), index=idx),
        "GAP": gappy, "HOLE": holey,
    })
    rm = store.returns_matrix(["AAA", "GAP", "HOLE"], idx[0], idx[-1],
                              min_obs=250, max_ffill=5)
    assert rm.tickers == ["AAA", "GAP"]
    assert rm.excluded["HOLE"] == "gap exceeds max_ffill"
    assert not np.isnan(rm.returns).any()


def test_returns_matrix_excludes_partial_window(tmp_path):
    idx = pd.bdate_range("2023-01-02", periods=300)
    late_idx = idx[30:]                               # starts 30 bd late
    store = _store_with(tmp_path, {
        "AAA": pd.Series(np.arange(100.0, 400.0), index=idx),
        "LATE": pd.Series(np.arange(100.0, 370.0), index=late_idx),
    })
    rm = store.returns_matrix(["AAA", "LATE"], idx[0], idx[-1], min_obs=250)
    assert rm.tickers == ["AAA"]
    assert rm.excluded["LATE"] == "partial window coverage"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_price_store.py -v -k returns_matrix`
Expected: FAIL — `AttributeError: 'PriceStore' object has no attribute 'returns_matrix'`.

- [ ] **Step 3: Implement (add to price_store.py)**

Add near `EnsureReport`:

```python
import numpy as np    # add to imports at top of file


@dataclass
class ReturnsMatrix:
    """Aligned (T, N) float32 log-return matrix + provenance."""
    returns: "np.ndarray"          # (T, N)
    dates: pd.DatetimeIndex        # (T,) — date of each return row
    tickers: list[str]             # (N,)
    excluded: dict[str, str]       # ticker -> reason
```

Add the method to `PriceStore`:

```python
    def returns_matrix(
        self,
        tickers: list[str],
        start,
        end,
        min_obs: int = 504,
        max_ffill: int = 5,
        edge_tolerance_days: int = 5,
        price_col: str = "adj_close",
    ) -> ReturnsMatrix:
        """Aligned log-return matrix over [start, end] with an explicit
        NaN policy; every dropped name gets a reason in ``excluded``."""
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        excluded: dict[str, str] = {}
        series: dict[str, pd.Series] = {}
        for t in [t.upper() for t in tickers]:
            df = self.get_prices(t)
            if df is None:
                excluded[t] = "no data in store"
                continue
            s = df.loc[(df.index >= start) & (df.index <= end), price_col].dropna()
            s = s[s > 0]
            if len(s) < min_obs:
                excluded[t] = f"insufficient history ({len(s)} < {min_obs})"
                continue
            series[t] = s

        if not series:
            return ReturnsMatrix(
                returns=np.zeros((0, 0), dtype=np.float32),
                dates=pd.DatetimeIndex([]), tickers=[], excluded=excluded)

        panel = pd.DataFrame(series)          # union of all dates, NaN-padded
        grid = panel.index
        tol = pd.tseries.offsets.BDay(edge_tolerance_days)
        keep: list[str] = []
        for t in panel.columns:
            col = panel[t].dropna()
            if col.index[0] > grid[0] + tol or col.index[-1] < grid[-1] - tol:
                excluded[t] = "partial window coverage"
            elif panel[t].ffill(limit=max_ffill).loc[col.index[0]:].isna().any():
                excluded[t] = "gap exceeds max_ffill"
            else:
                keep.append(t)

        filled = panel[keep].ffill(limit=max_ffill).dropna(axis=0, how="any")
        log_prices = np.log(filled.to_numpy(dtype=np.float64))
        returns = np.diff(log_prices, axis=0).astype(np.float32)
        return ReturnsMatrix(
            returns=returns, dates=filled.index[1:],
            tickers=list(keep), excluded=excluded)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_price_store.py -v`
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add src/options_desk/calibration/data/price_store.py tests/test_price_store.py
git commit -m "feat(calibration): aligned returns matrix with explicit NaN policy"
```

---

### Task 6: Model registry wrapping existing scipy calibrators

**Files:**
- Create: `src/options_desk/calibration/pipeline/__init__.py`
- Create: `src/options_desk/calibration/pipeline/registry.py`
- Test: `tests/test_model_registry.py`

**Interfaces:**
- Consumes: existing `GBMCalibrator.fit(prices, dt=...)`, `GARCHCalibrator.fit(prices, dt=...)`, `HestonQMLECalibrator(smooth_window=10).fit(prices, dt=...)` (all under `options_desk.calibration.physical`).
- Produces: `ModelSpec` (frozen dataclass: `name: str`, `fit: FitFn`, `min_obs: int`), `FitFn = Callable[[np.ndarray, float], dict]` (1D prices + dt → flat dict of scalars), `register_model(spec)`, `get_model(name) -> ModelSpec`, `list_models() -> list[str]`. Registered at import: `"gbm"` (min_obs=60), `"garch"` (min_obs=250), `"heston_qmle"` (min_obs=60). Task 8's runner calls `get_model`; Task 9's CLI calls `list_models`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_model_registry.py
import numpy as np
import pytest

from options_desk.calibration.pipeline.registry import (
    ModelSpec,
    get_model,
    list_models,
    register_model,
)


def _gbm_prices(n=1000, mu=0.08, sigma=0.2, seed=0):
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    r = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)
    return 100.0 * np.exp(np.cumsum(r))


def test_builtin_models_registered():
    assert set(list_models()) >= {"gbm", "garch", "heston_qmle"}


def test_get_model_unknown_raises():
    with pytest.raises(KeyError, match="unknown model"):
        get_model("not_a_model")


def test_register_duplicate_raises():
    with pytest.raises(ValueError, match="already registered"):
        register_model(ModelSpec(name="gbm", fit=lambda p, dt: {}, min_obs=1))


def test_gbm_fit_returns_flat_scalar_dict():
    out = get_model("gbm").fit(_gbm_prices(), 1.0 / 252.0)
    assert isinstance(out, dict)
    assert out["sigma"] == pytest.approx(0.2, abs=0.03)
    assert all(isinstance(v, (int, float, bool, str)) for v in out.values())


def test_heston_qmle_fit_returns_flat_scalar_dict():
    out = get_model("heston_qmle").fit(_gbm_prices(n=1500), 1.0 / 252.0)
    assert {"kappa", "theta", "sigma_v", "rho", "mu", "v0"} <= set(out)
    assert all(isinstance(v, (int, float, bool, str)) for v in out.values())


def test_garch_fit_returns_flat_scalar_dict():
    out = get_model("garch").fit(_gbm_prices(n=1000), 1.0 / 252.0)
    assert {"omega", "alpha", "beta"} <= set(out)
    assert all(isinstance(v, (int, float, bool, str)) for v in out.values())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model_registry.py -v`
Expected: collection error — module does not exist.

- [ ] **Step 3: Write the implementation**

```python
# src/options_desk/calibration/pipeline/__init__.py
"""Universe-scale calibration orchestration: registry, results store, runner."""

from .registry import ModelSpec, get_model, list_models, register_model

__all__ = ["ModelSpec", "get_model", "list_models", "register_model"]
```

```python
# src/options_desk/calibration/pipeline/registry.py
"""
Model registry: model name -> fit function over one asset's 1-D price array.

Adding a model to the universe pipeline is one ``register_model`` call.
Fit functions return a FLAT dict of scalars (params + diagnostics); the
runner adds bookkeeping columns (ticker, sector, error, calibration_date).

Phase 2 will register JAX ``fit_batch`` implementations under the same
names; the per-asset ``fit`` path here is the scipy reference route.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Callable

import numpy as np

# (prices_1d, dt) -> flat dict of scalar params/diagnostics
FitFn = Callable[[np.ndarray, float], dict]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    fit: FitFn
    min_obs: int = 60


_REGISTRY: dict[str, ModelSpec] = {}


def register_model(spec: ModelSpec) -> None:
    if spec.name in _REGISTRY:
        raise ValueError(f"model '{spec.name}' already registered")
    _REGISTRY[spec.name] = spec


def get_model(name: str) -> ModelSpec:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown model '{name}'; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_models() -> list[str]:
    return sorted(_REGISTRY)


def _scalars(result) -> dict:
    """Flatten a calibration result dataclass to its scalar fields."""
    d = asdict(result) if is_dataclass(result) else dict(result)
    return {k: v for k, v in d.items()
            if isinstance(v, (int, float, bool, str, np.floating, np.integer))}


def _fit_gbm(prices: np.ndarray, dt: float) -> dict:
    from ..physical.gbm_calibrator import GBMCalibrator
    return _scalars(GBMCalibrator().fit(prices, dt=dt))


def _fit_garch(prices: np.ndarray, dt: float) -> dict:
    from ..physical.garch_calibrator import GARCHCalibrator
    return _scalars(GARCHCalibrator().fit(prices, dt=dt))


def _fit_heston_qmle(prices: np.ndarray, dt: float) -> dict:
    from ..physical.heston_qmle import HestonQMLECalibrator
    return _scalars(HestonQMLECalibrator(smooth_window=10).fit(prices, dt=dt))


register_model(ModelSpec(name="gbm", fit=_fit_gbm, min_obs=60))
register_model(ModelSpec(name="garch", fit=_fit_garch, min_obs=250))
register_model(ModelSpec(name="heston_qmle", fit=_fit_heston_qmle, min_obs=60))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model_registry.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/options_desk/calibration/pipeline/ tests/test_model_registry.py
git commit -m "feat(calibration): model registry over existing scipy calibrators"
```

---

### Task 7: Results store (run dirs, manifest, per-model parquet, done markers)

**Files:**
- Create: `src/options_desk/calibration/pipeline/results_store.py`
- Modify: `src/options_desk/calibration/pipeline/__init__.py`
- Test: `tests/test_results_store.py`

**Interfaces:**
- Produces: `new_run_dir(out_root, run_id=None) -> Path` (creates `out_root/<run_id or UTC stamp>`; reuses the dir if it exists — that's resume), `write_manifest(run_dir, updates: dict)` (merge-updates `manifest.json`), `read_manifest(run_dir) -> dict`, `save_model_results(run_dir, model, df)` (writes `<model>.parquet` + `<model>.done` marker), `load_model_results(run_dir, model) -> pd.DataFrame`, `is_model_done(run_dir, model) -> bool`. Task 8's runner uses all of these.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_results_store.py
import pandas as pd

from options_desk.calibration.pipeline.results_store import (
    is_model_done,
    load_model_results,
    new_run_dir,
    read_manifest,
    save_model_results,
    write_manifest,
)


def test_new_run_dir_creates_and_reuses(tmp_path):
    d1 = new_run_dir(tmp_path, run_id="myrun")
    assert d1.is_dir() and d1.name == "myrun"
    assert new_run_dir(tmp_path, run_id="myrun") == d1     # reuse = resume
    auto = new_run_dir(tmp_path)                           # UTC stamp name
    assert auto.is_dir() and auto != d1


def test_manifest_merge_roundtrip(tmp_path):
    d = new_run_dir(tmp_path, run_id="r")
    write_manifest(d, {"universe": "sp500", "models": ["gbm"]})
    write_manifest(d, {"status": "done"})
    m = read_manifest(d)
    assert m["universe"] == "sp500" and m["status"] == "done"


def test_read_manifest_empty(tmp_path):
    assert read_manifest(new_run_dir(tmp_path, run_id="r")) == {}


def test_model_results_and_done_marker(tmp_path):
    d = new_run_dir(tmp_path, run_id="r")
    assert not is_model_done(d, "gbm")
    df = pd.DataFrame({"ticker": ["AAPL"], "mu": [0.1], "converged": [True]})
    save_model_results(d, "gbm", df)
    assert is_model_done(d, "gbm")
    got = load_model_results(d, "gbm")
    assert got["ticker"].tolist() == ["AAPL"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_results_store.py -v`
Expected: collection error — module does not exist.

- [ ] **Step 3: Write the implementation**

```python
# src/options_desk/calibration/pipeline/results_store.py
"""
On-disk layout for calibration runs:

    <out_root>/<run_id>/
        manifest.json      run metadata, merge-updated as the run progresses
        <model>.parquet    one row per ticker
        <model>.done       completion marker (UTC timestamp) -> resume skips
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def new_run_dir(out_root: str | Path, run_id: str | None = None) -> Path:
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    d = Path(out_root) / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_manifest(run_dir: Path, updates: dict) -> None:
    m = read_manifest(run_dir)
    m.update(updates)
    (Path(run_dir) / "manifest.json").write_text(
        json.dumps(m, indent=2, default=str))


def read_manifest(run_dir: Path) -> dict:
    p = Path(run_dir) / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _model_paths(run_dir: Path, model: str) -> tuple[Path, Path]:
    d = Path(run_dir)
    return d / f"{model}.parquet", d / f"{model}.done"


def save_model_results(run_dir: Path, model: str, df: pd.DataFrame) -> Path:
    pq, done = _model_paths(run_dir, model)
    df.to_parquet(pq, index=False)
    done.write_text(datetime.now(timezone.utc).isoformat())
    return pq


def load_model_results(run_dir: Path, model: str) -> pd.DataFrame:
    pq, _ = _model_paths(run_dir, model)
    return pd.read_parquet(pq)


def is_model_done(run_dir: Path, model: str) -> bool:
    _, done = _model_paths(run_dir, model)
    return done.exists()
```

Update `pipeline/__init__.py`:

```python
# src/options_desk/calibration/pipeline/__init__.py
"""Universe-scale calibration orchestration: registry, results store, runner."""

from .registry import ModelSpec, get_model, list_models, register_model
from .results_store import (
    is_model_done,
    load_model_results,
    new_run_dir,
    read_manifest,
    save_model_results,
    write_manifest,
)

__all__ = [
    "ModelSpec", "get_model", "list_models", "register_model",
    "new_run_dir", "write_manifest", "read_manifest",
    "save_model_results", "load_model_results", "is_model_done",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_results_store.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/options_desk/calibration/pipeline/results_store.py \
        src/options_desk/calibration/pipeline/__init__.py tests/test_results_store.py
git commit -m "feat(calibration): results store with manifests and done markers"
```

---

### Task 8: Runner (universe → ensure → joblib calibrate → persist, with resume)

**Files:**
- Create: `src/options_desk/calibration/pipeline/runner.py`
- Modify: `src/options_desk/calibration/pipeline/__init__.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `Universe`/`load_universe` (Task 1), `PriceStore` (Task 3), `get_model` (Task 6), all of results_store (Task 7).
- Produces: `RunConfig` dataclass (`universe: str`, `models: list[str]`, `years: float = 5.0`, `dt: float = 1/252`, `n_jobs: int = -1`, `out_root: str | Path = "runs/calibration"`, `run_id: str | None = None`, `end: str | None = None`) and `run_calibration(cfg, store, universe=None) -> Path` (returns the run dir). Task 9's CLI builds a `RunConfig` and calls this.

Behavior contract:
1. Resolve window: `end` (default: today UTC, normalized) minus `years * 365.25` days.
2. `store.ensure(tickers, start, end)`; write manifest (universe name/size, window, models, ensure counts, created_at).
3. Per model, **skip if `is_model_done`** (resume); else joblib `Parallel(n_jobs, prefer="processes")` over tickers calling module-level `_calibrate_one`; every row has at least `ticker, converged, error`; add `sector` (from universe) and `calibration_date` columns; `save_model_results`.
4. A per-ticker exception becomes `converged=False, error="<Type>: <msg>"` — never a crash.
5. Finish: manifest update with per-model `{n, n_converged}` and `status: "complete"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_runner.py
import numpy as np
import pandas as pd
import pytest

from options_desk.calibration.data.price_store import PriceStore
from options_desk.calibration.data.universe import Universe
from options_desk.calibration.pipeline.registry import ModelSpec, register_model
from options_desk.calibration.pipeline.results_store import (
    is_model_done, load_model_results, read_manifest,
)
from options_desk.calibration.pipeline.runner import RunConfig, run_calibration


def _gbm_frame(n=800, seed=0):
    rng = np.random.default_rng(seed)
    r = 0.0003 + 0.012 * rng.standard_normal(n)
    close = pd.Series(100.0 * np.exp(np.cumsum(r)),
                      index=pd.bdate_range("2022-01-03", periods=n))
    df = pd.DataFrame({c: close for c in ["open", "high", "low", "close",
                                          "adj_close"]})
    df["volume"] = 1e6
    return df


@pytest.fixture()
def store(tmp_path):
    frames = {t: _gbm_frame(seed=i) for i, t in enumerate(["AAA", "BBB", "CCC"])}

    def fake_fetcher(tickers, start, end):
        return {t: frames[t] for t in tickers if t in frames}

    return PriceStore(tmp_path / "lake", fetcher=fake_fetcher)


UNI = Universe(name="test3", tickers=["AAA", "BBB", "CCC", "DEAD"],
               sectors={"AAA": "Tech", "BBB": "Energy", "CCC": "Tech",
                        "DEAD": "UNKNOWN"})


def test_run_calibration_end_to_end(tmp_path, store):
    cfg = RunConfig(universe="ignored", models=["gbm"], years=3.0,
                    n_jobs=1, out_root=tmp_path / "runs", run_id="r1",
                    end="2025-01-31")
    run_dir = run_calibration(cfg, store, universe=UNI)

    df = load_model_results(run_dir, "gbm")
    assert set(df["ticker"]) == {"AAA", "BBB", "CCC", "DEAD"}
    ok = df[df["ticker"] != "DEAD"]
    assert ok["converged"].all()
    assert (ok["error"] == "").all()
    assert "sigma" in df.columns and "sector" in df.columns
    dead = df[df["ticker"] == "DEAD"].iloc[0]
    assert not dead["converged"] and dead["error"] != ""

    m = read_manifest(run_dir)
    assert m["universe"] == "test3" and m["status"] == "complete"
    assert m["models"]["gbm"]["n_converged"] == 3


def test_run_calibration_resume_skips_done_models(tmp_path, store):
    cfg = RunConfig(universe="ignored", models=["gbm"], years=3.0,
                    n_jobs=1, out_root=tmp_path / "runs", run_id="r2",
                    end="2025-01-31")
    d = run_calibration(cfg, store, universe=UNI)
    assert is_model_done(d, "gbm")
    first = load_model_results(d, "gbm")

    # poison the registry entry: if resume re-runs the model, rows change
    d2 = run_calibration(cfg, store, universe=UNI)
    assert d2 == d
    pd.testing.assert_frame_equal(load_model_results(d, "gbm"), first)


def test_run_calibration_isolates_fit_exceptions(tmp_path, store):
    def exploding_fit(prices, dt):
        raise RuntimeError("boom")

    register_model(ModelSpec(name="exploder", fit=exploding_fit, min_obs=1))
    cfg = RunConfig(universe="ignored", models=["exploder"], years=3.0,
                    n_jobs=1, out_root=tmp_path / "runs", run_id="r3",
                    end="2025-01-31")
    d = run_calibration(cfg, store, universe=UNI)
    df = load_model_results(d, "exploder")
    assert (~df["converged"]).all()
    assert df["error"].str.contains("RuntimeError: boom").all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_runner.py -v`
Expected: collection error — `runner` module does not exist.

- [ ] **Step 3: Write the implementation**

```python
# src/options_desk/calibration/pipeline/runner.py
"""
Universe calibration runner: universe -> ensure prices -> per-model joblib
calibration -> parquet results, with checkpoint/resume at model granularity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..data.price_store import PriceStore
from ..data.universe import Universe, load_universe
from .registry import get_model
from .results_store import (
    is_model_done,
    new_run_dir,
    read_manifest,
    save_model_results,
    write_manifest,
)

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    universe: str                      # name or CSV path (ignored if a
                                       # Universe object is passed directly)
    models: list[str] = field(default_factory=lambda: ["heston_qmle"])
    years: float = 5.0
    dt: float = 1.0 / 252.0
    n_jobs: int = -1
    out_root: str | Path = "runs/calibration"
    run_id: str | None = None          # pass an existing id to resume
    end: str | None = None             # default: today (UTC)


def _calibrate_one(model_name: str, ticker: str,
                   prices: np.ndarray | None, dt: float) -> dict:
    """One (model, ticker) fit. Never raises."""
    spec = get_model(model_name)
    try:
        if prices is None or len(prices) < spec.min_obs:
            n = 0 if prices is None else len(prices)
            return {"ticker": ticker, "converged": False,
                    "error": f"insufficient data ({n} < {spec.min_obs} obs)"}
        out = spec.fit(np.asarray(prices, dtype=np.float64), dt)
        out.setdefault("converged", True)
        return {"ticker": ticker, "error": "", **out}
    except Exception as e:                                  # noqa: BLE001
        return {"ticker": ticker, "converged": False,
                "error": f"{type(e).__name__}: {e}"}


def run_calibration(
    cfg: RunConfig,
    store: PriceStore,
    universe: Universe | None = None,
) -> Path:
    """Run all requested models over the universe. Returns the run dir.

    Re-invoking with the same ``run_id`` resumes: models with a ``.done``
    marker are skipped.
    """
    if universe is None:
        universe = load_universe(cfg.universe)

    end = (pd.Timestamp(cfg.end) if cfg.end
           else pd.Timestamp(datetime.now(timezone.utc).date()))
    start = end - pd.Timedelta(days=round(cfg.years * 365.25))
    run_dir = new_run_dir(cfg.out_root, cfg.run_id)
    logger.info("run %s: universe=%s (%d names), window %s..%s, models=%s",
                run_dir.name, universe.name, len(universe),
                start.date(), end.date(), cfg.models)

    # validate models before any expensive work
    for m in cfg.models:
        get_model(m)

    rep = store.ensure(universe.tickers, start, end)
    write_manifest(run_dir, {
        "universe": universe.name,
        "n_tickers": len(universe),
        "start": str(start.date()), "end": str(end.date()),
        "requested_models": cfg.models,
        "ensure": {"fetched": len(rep.fetched), "cached": len(rep.cached),
                   "failed": rep.failed},
        "created_at": read_manifest(run_dir).get(
            "created_at", datetime.now(timezone.utc).isoformat()),
        "status": "running",
    })

    prices_by_ticker: dict[str, np.ndarray | None] = {}
    for t in universe.tickers:
        df = store.get_prices(t)
        if df is None:
            prices_by_ticker[t] = None
            continue
        s = df.loc[(df.index >= start) & (df.index <= end), "adj_close"].dropna()
        prices_by_ticker[t] = s.to_numpy(dtype=np.float64) if len(s) else None

    cal_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    model_stats = read_manifest(run_dir).get("models", {})
    for model in cfg.models:
        if is_model_done(run_dir, model):
            logger.info("model %s already done — skipping (resume)", model)
            continue
        logger.info("calibrating %s over %d names (n_jobs=%s)...",
                    model, len(universe), cfg.n_jobs)
        rows = Parallel(n_jobs=cfg.n_jobs, prefer="processes")(
            delayed(_calibrate_one)(model, t, prices_by_ticker[t], cfg.dt)
            for t in universe.tickers
        )
        df = pd.DataFrame(rows)
        df["sector"] = df["ticker"].map(universe.sectors).fillna("UNKNOWN")
        df["calibration_date"] = cal_date
        save_model_results(run_dir, model, df)
        n_conv = int(df["converged"].sum())
        model_stats[model] = {"n": int(len(df)), "n_converged": n_conv}
        write_manifest(run_dir, {"models": model_stats})
        logger.info("model %s: %d/%d converged", model, n_conv, len(df))

    write_manifest(run_dir, {"models": model_stats, "status": "complete",
                             "finished_at": datetime.now(timezone.utc).isoformat()})
    return run_dir
```

Add to `pipeline/__init__.py` imports/`__all__`:

```python
from .runner import RunConfig, run_calibration
```

and append `"RunConfig", "run_calibration"` to `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_runner.py -v`
Expected: 3 passed. Also run the whole new suite: `pytest tests/test_universe.py tests/test_price_store.py tests/test_model_registry.py tests/test_results_store.py tests/test_runner.py -q` — all green.

- [ ] **Step 5: Commit**

```bash
git add src/options_desk/calibration/pipeline/runner.py \
        src/options_desk/calibration/pipeline/__init__.py tests/test_runner.py
git commit -m "feat(calibration): universe runner with joblib + model-level resume"
```

---

### Task 9: CLI rewrite + gitignore

**Files:**
- Modify: `scripts/calibrate_universe.py` (full rewrite, shown below)
- Modify: `.gitignore` (add `data/price_lake/`)
- Test: `tests/test_calibrate_universe_cli.py`

**Interfaces:**
- Consumes: `RunConfig`/`run_calibration` (Task 8), `PriceStore` (Task 3), `load_universe`/`available_universes`/`Universe` (Task 1), `list_models` (Task 6), `read_manifest` (Task 7), `DEFAULT_BASKET_50` (existing).
- Produces: `build_parser()` and `main(argv=None)` importable from the script (tested); the CLI itself.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_calibrate_universe_cli.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import calibrate_universe as cli  # noqa: E402

from options_desk.calibration.pipeline.results_store import (  # noqa: E402
    load_model_results,
)


def test_parser_defaults():
    args = cli.build_parser().parse_args([])
    assert args.universe is None
    assert args.models == ["heston_qmle"]
    assert args.years == 5.0
    assert args.run_id is None


def test_main_end_to_end_with_fake_fetcher(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)

    def fake_fetcher(tickers, start, end):
        out = {}
        for t in tickers:
            r = 0.0003 + 0.012 * rng.standard_normal(700)
            close = pd.Series(100.0 * np.exp(np.cumsum(r)),
                              index=pd.bdate_range("2022-06-01", periods=700))
            df = pd.DataFrame({c: close for c in
                               ["open", "high", "low", "close", "adj_close"]})
            df["volume"] = 1e6
            out[t] = df
        return out

    monkeypatch.setattr(cli, "_default_fetcher", lambda: fake_fetcher)
    uni_csv = tmp_path / "mini.csv"
    uni_csv.write_text("ticker,sector\nAAA,Tech\nBBB,Energy\n")

    rc = cli.main([
        "--universe", str(uni_csv),
        "--models", "gbm",
        "--years", "2",
        "--jobs", "1",
        "--price-lake", str(tmp_path / "lake"),
        "--out-root", str(tmp_path / "runs"),
        "--run-id", "clitest",
        "--end", "2025-01-31",
    ])
    assert rc == 0
    df = load_model_results(tmp_path / "runs" / "clitest", "gbm")
    assert set(df["ticker"]) == {"AAA", "BBB"}
    assert df["converged"].all()


def test_list_models_flag(capsys):
    rc = cli.main(["--list-models"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "gbm" in out and "heston_qmle" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_calibrate_universe_cli.py -v`
Expected: FAIL — `AttributeError` (`build_parser` / `_default_fetcher` don't exist in the old script).

- [ ] **Step 3: Rewrite the script**

```python
# scripts/calibrate_universe.py
"""
Universe-scale P-measure calibration CLI.

Examples:
    # 50-name default basket, Heston QMLE (back-compatible default)
    python scripts/calibrate_universe.py

    # S&P 500, three models, resumable
    python scripts/calibrate_universe.py --universe sp500 \
        --models gbm garch heston_qmle --run-id sp500-2026-07-05

    # resume after interruption (completed models are skipped)
    python scripts/calibrate_universe.py --universe sp500 \
        --models gbm garch heston_qmle --run-id sp500-2026-07-05
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from options_desk.calibration.data.price_store import (  # noqa: E402
    PriceStore, fetch_yfinance,
)
from options_desk.calibration.data.universe import (  # noqa: E402
    Universe, available_universes, load_universe,
)
from options_desk.calibration.physical import DEFAULT_BASKET_50  # noqa: E402
from options_desk.calibration.pipeline import (  # noqa: E402
    RunConfig, list_models, read_manifest, run_calibration,
)


def _default_fetcher():
    return fetch_yfinance


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--universe", default=None,
                   help="universe name (data/universes/<name>.csv) or CSV path")
    p.add_argument("--tickers", nargs="*", default=None,
                   help="explicit tickers (overrides --universe)")
    p.add_argument("--models", nargs="*", default=["heston_qmle"],
                   help="registered models to run (see --list-models)")
    p.add_argument("--years", type=float, default=5.0,
                   help="lookback window in years")
    p.add_argument("--end", default=None,
                   help="window end date YYYY-MM-DD (default: today UTC)")
    p.add_argument("--jobs", type=int, default=-1,
                   help="joblib parallel jobs (-1 = all cores)")
    p.add_argument("--price-lake", default=str(ROOT / "data" / "price_lake"),
                   help="price lake root")
    p.add_argument("--out-root", default=str(ROOT / "runs" / "calibration"),
                   help="run output root")
    p.add_argument("--run-id", default=None,
                   help="run id; reuse an existing id to resume")
    p.add_argument("--list-models", action="store_true",
                   help="print registered models and exit")
    p.add_argument("--list-universes", action="store_true",
                   help="print available universes and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("calibrate_universe")

    if args.list_models:
        print("\n".join(list_models()))
        return 0
    if args.list_universes:
        print("\n".join(available_universes()) or "(none — run scripts/build_universes.py)")
        return 0

    if args.tickers:
        universe = Universe(name="adhoc",
                            tickers=[t.upper() for t in args.tickers],
                            sectors={t.upper(): "UNKNOWN" for t in args.tickers})
    elif args.universe:
        universe = load_universe(args.universe)
    else:
        universe = Universe(name="basket50", tickers=list(DEFAULT_BASKET_50),
                            sectors={t: "UNKNOWN" for t in DEFAULT_BASKET_50})

    store = PriceStore(args.price_lake, fetcher=_default_fetcher())
    cfg = RunConfig(universe=universe.name, models=args.models,
                    years=args.years, n_jobs=args.jobs,
                    out_root=args.out_root, run_id=args.run_id, end=args.end)

    run_dir = run_calibration(cfg, store, universe=universe)

    m = read_manifest(run_dir)
    log.info("=" * 60)
    log.info("run %s complete — universe=%s (%d names)",
             run_dir.name, m.get("universe"), m.get("n_tickers", 0))
    for model, s in m.get("models", {}).items():
        log.info("  %-14s %d/%d converged (%.1f%%)", model,
                 s["n_converged"], s["n"], 100.0 * s["n_converged"] / max(s["n"], 1))
    ens = m.get("ensure", {})
    log.info("  data: %d fetched, %d cached, %d failed",
             ens.get("fetched", 0), ens.get("cached", 0),
             len(ens.get("failed", {})))
    log.info("  output: %s", run_dir)
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Append to `.gitignore`:

```
data/price_lake/
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_calibrate_universe_cli.py -v`, then the full Phase 1 suite:
`pytest tests/test_universe.py tests/test_build_universes.py tests/test_price_store.py tests/test_model_registry.py tests/test_results_store.py tests/test_runner.py tests/test_calibrate_universe_cli.py -q`
Expected: all passed. Also confirm no regression: `pytest tests/test_calibration_service.py -q`.

- [ ] **Step 5: Commit**

```bash
git add scripts/calibrate_universe.py tests/test_calibrate_universe_cli.py .gitignore
git commit -m "feat(calibration): universe CLI on registry/runner with resume"
```

---

### Task 10: Manual verification (network) — build universes, smoke-run, scale-run

This task is manual/interactive (network + wall-clock); no new code.

- [ ] **Step 1: Build universe CSVs**

Run: `source /home/union/miniconda3/etc/profile.d/conda.sh && conda activate options-desk && python scripts/build_universes.py`
Expected: `data/universes/{sp500,sp400,sp600,sp1500}.csv` written; sp500 ≈ 500 rows, sp1500 ≈ 1500 rows. Spot-check `head data/universes/sp500.csv` shows `ticker,sector`.

- [ ] **Step 2: Commit the universe CSVs**

```bash
git add data/universes/*.csv
git commit -m "data: S&P 500/400/600/1500 universe manifests"
```

- [ ] **Step 3: Smoke run (default 50-name basket, all three models)**

Run: `python scripts/calibrate_universe.py --models gbm garch heston_qmle --run-id smoke50`
Expected: completes in a few minutes; log shows ≥ 45/50 converged per model; `runs/calibration/smoke50/` contains `manifest.json`, `{gbm,garch,heston_qmle}.parquet` + `.done`.

- [ ] **Step 4: Re-run to verify cache + resume**

Run the same command again.
Expected: log shows `cached: 50, fetched: 0` and `model ... already done — skipping (resume)` for all three; finishes in seconds.

- [ ] **Step 5: Scale run — S&P 500**

Run: `python scripts/calibrate_universe.py --universe sp500 --models gbm heston_qmle --run-id sp500-first`
Expected: first fill downloads ~500 names in chunks (~5-10 min); calibration converges for the large majority; results parquet has ~500 rows. Note wall-clock in the task report — this is the Phase 2 baseline.

- [ ] **Step 6: Report** — summarize convergence rates, wall-clock (fetch vs calibrate), and any failed tickers; no commit (run outputs are not committed).

---

## Self-Review Notes

- **Spec coverage:** Phase 1 spec items — PriceStore + chunked fetch + incremental (`ensure`/ledger) ✅ T3/T4; returns matrix + NaN policy ✅ T5; universes + sectors ✅ T1/T2; registry ✅ T6; results store/manifests/markers ✅ T7; runner + resume + failure isolation ✅ T8; CLI ✅ T9; real-data verification ✅ T10. Per-asset-chunk checkpoints are particle-filter-specific → Phase 4 by design.
- **Types:** `FetcherFn` signature consistent across T3 tests/impl/T4/T9 (`monkeypatch.setattr(cli, "_default_fetcher", ...)` patches the indirection added for exactly this purpose). `RunConfig` fields in T8 tests match the dataclass. `ModelSpec(name, fit, min_obs)` consistent T6/T8.
- **Known judgment calls:** coverage ledger trusts "requested range" not "rows present" — deliberate, so delisted/dead names aren't refetched every run; `put_prices` upsert keeps newest on overlap; matrix policy excludes partial-window names (fine for Phase 3 cross-asset use; per-asset calibration uses each name's own series, not the matrix).
