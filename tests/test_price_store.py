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
