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
