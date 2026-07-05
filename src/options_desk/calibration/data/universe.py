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
