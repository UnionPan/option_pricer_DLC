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
