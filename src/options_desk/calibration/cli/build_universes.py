"""
Build universe CSVs (ticker, sector) from Wikipedia S&P constituent lists.

Requires network. Run manually, then commit the CSVs:

    build-universes            # writes ./data/universes/*.csv

The output directory is resolved relative to the current working
directory; run from the repo root.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT_DIR = Path("data") / "universes"

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
    raise SystemExit(main())
