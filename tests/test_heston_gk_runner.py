"""
Tests for Garman-Klass OHLC Heston calibration in the runner.

Sets JAX_PLATFORMS=cpu at module top before any JAX imports.
"""
import os

# CRITICAL: Set JAX to CPU-only mode BEFORE any JAX imports
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import pandas as pd
import pytest

from options_desk.calibration.data.price_store import PriceStore
from options_desk.calibration.data.universe import Universe
from options_desk.calibration.pipeline.results_store import load_model_results
from options_desk.calibration.pipeline.runner import RunConfig, run_calibration


def _simulate_heston_euler(kappa, theta, sigma_v, rho, mu, v0, dt, n, seed, substeps=1):
    """
    Simulate Heston process with Euler-Maruyama discretization.

    dS/S = mu*dt + sqrt(v)*dW1
    dv = kappa*(theta - v)*dt + sigma_v*sqrt(v)*dW2
    corr(dW1, dW2) = rho

    Args:
        kappa, theta, sigma_v, rho, mu, v0: Heston parameters
        dt: daily time step (e.g., 1/252)
        n: number of days
        seed: random seed
        substeps: number of Euler substeps per day

    Returns:
        ohlc: dict with keys 'open', 'high', 'low', 'close' of shape (n,) for daily bars
    """
    rng = np.random.default_rng(seed)
    S0 = 100.0
    dt_sub = dt / substeps

    # Initialize paths
    S = np.zeros(n * substeps + 1)
    v = np.zeros(n * substeps + 1)
    S[0] = S0
    v[0] = v0

    # Cholesky decomposition for correlated Brownian motions
    sqrt_dt = np.sqrt(dt_sub)

    for i in range(n * substeps):
        Z1 = rng.standard_normal()
        Z2 = rng.standard_normal()

        # Correlated Brownian increments
        dW1 = sqrt_dt * Z1
        dW2 = sqrt_dt * (rho * Z1 + np.sqrt(1 - rho**2) * Z2)

        # Full truncation scheme: v >= 0
        v_current = max(v[i], 0.0)

        # Variance evolution
        v[i+1] = v[i] + kappa * (theta - v_current) * dt_sub + sigma_v * np.sqrt(v_current) * dW2

        # Spot evolution
        S[i+1] = S[i] * np.exp((mu - 0.5 * v_current) * dt_sub + np.sqrt(v_current) * dW1)

    # Build OHLC from sub-day prices
    ohlc = {'open': [], 'high': [], 'low': [], 'close': []}
    for day in range(n):
        idx_start = day * substeps
        idx_end = (day + 1) * substeps + 1
        day_prices = S[idx_start:idx_end]
        ohlc['open'].append(day_prices[0])
        ohlc['high'].append(np.max(day_prices))
        ohlc['low'].append(np.min(day_prices))
        ohlc['close'].append(day_prices[-1])

    for k in ohlc:
        ohlc[k] = np.array(ohlc[k])

    return ohlc


def _heston_ohlc_frame(n=800, seed=0, substeps=32):
    """Generate synthetic Heston OHLC price data."""
    # Heston parameters
    kappa = 3.0
    theta = 0.04
    sigma_v = 0.4
    rho = -0.6
    mu = 0.05
    v0 = theta
    dt = 1.0 / 252.0

    ohlc = _simulate_heston_euler(kappa, theta, sigma_v, rho, mu, v0, dt, n, seed, substeps)

    # Create DataFrame
    df = pd.DataFrame({
        'open': ohlc['open'],
        'high': ohlc['high'],
        'low': ohlc['low'],
        'close': ohlc['close'],
        'adj_close': ohlc['close'],  # No adjustment initially
        'volume': np.full(n, 1e6),
    }, index=pd.bdate_range("2022-01-03", periods=n))

    return df


@pytest.fixture()
def ohlc_store(tmp_path):
    """Synthetic OHLC price store with 6 tickers + 1 insufficient-data ticker."""
    frames = {
        t: _heston_ohlc_frame(n=700, seed=i, substeps=32)
        for i, t in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"])
    }
    # SHORT has only 40 observations: insufficient for heston_qmle_gk (min_obs=60)
    frames["SHORT"] = _heston_ohlc_frame(n=40, seed=10, substeps=32)

    def fake_fetcher(tickers, start, end):
        return {t: frames[t] for t in tickers if t in frames}

    return PriceStore(tmp_path / "lake", fetcher=fake_fetcher)


UNI = Universe(
    name="test_heston_gk",
    tickers=["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "SHORT"],
    sectors={
        "AAA": "Tech", "BBB": "Energy", "CCC": "Tech",
        "DDD": "Finance", "EEE": "Tech", "FFF": "Energy",
        "SHORT": "UNKNOWN",
    },
)


def test_heston_gk_runner_produces_converged_rows(tmp_path, ohlc_store):
    """
    Test (a): Runner with models=["heston_qmle_gk"] on a synthetic OHLC store
    produces converged rows with kappa/theta/sigma_v columns.
    """
    cfg = RunConfig(
        universe="ignored",
        models=["heston_qmle_gk"],
        years=2.5,
        n_jobs=1,
        out_root=tmp_path / "runs",
        run_id="gk_test",
        end="2025-01-31",
    )
    run_dir = run_calibration(cfg, ohlc_store, universe=UNI)
    df = load_model_results(run_dir, "heston_qmle_gk")

    # Check all tickers present
    assert set(df["ticker"]) == set(UNI.tickers)

    # Good tickers should converge
    good_tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    ok = df[df["ticker"].isin(good_tickers)]
    assert ok["converged"].all()
    assert (ok["error"] == "").all()

    # Check expected columns exist
    required_cols = ["kappa", "theta", "sigma_v", "rho", "mu", "v0"]
    for col in required_cols:
        assert col in df.columns
        # All converged rows should have valid values
        assert ok[col].notna().all()
        assert np.isfinite(ok[col]).all()

    # SHORT should have insufficient data
    short_row = df[df["ticker"] == "SHORT"].iloc[0]
    assert not short_row["converged"]
    assert "insufficient data" in short_row["error"]


def test_heston_gk_adjustment_invariance(tmp_path, ohlc_store):
    """
    Test (b): Adjustment invariance. Scale one ticker's O/H/L/C by 2x while
    keeping adj_close fixed -> GK params unchanged rel 1e-6 vs unscaled run.
    """
    # Run baseline calibration
    cfg_baseline = RunConfig(
        universe="ignored",
        models=["heston_qmle_gk"],
        years=2.5,
        n_jobs=1,
        out_root=tmp_path / "runs_baseline",
        run_id="baseline",
        end="2025-01-31",
    )
    run_dir_baseline = run_calibration(cfg_baseline, ohlc_store, universe=UNI)
    df_baseline = load_model_results(run_dir_baseline, "heston_qmle_gk")

    # Create modified store: scale AAA's OHLC by 2x but keep adj_close fixed
    # This simulates a 2:1 stock split
    def make_modified_store(tmp_path):
        frames_modified = {}
        for ticker in UNI.tickers:
            df = ohlc_store.get_prices(ticker)
            if df is not None:
                df_copy = df.copy()
                if ticker == "AAA":
                    # Scale OHLC by 2x (simulating split)
                    for col in ["open", "high", "low", "close"]:
                        df_copy[col] = df_copy[col] * 2.0
                    # adj_close stays the same (already adjusted for split)
                frames_modified[ticker] = df_copy

        def fake_fetcher(tickers, start, end):
            return {t: frames_modified[t] for t in tickers if t in frames_modified}

        return PriceStore(tmp_path / "lake_modified", fetcher=fake_fetcher)

    store_modified = make_modified_store(tmp_path)

    # Run modified calibration
    cfg_modified = RunConfig(
        universe="ignored",
        models=["heston_qmle_gk"],
        years=2.5,
        n_jobs=1,
        out_root=tmp_path / "runs_modified",
        run_id="modified",
        end="2025-01-31",
    )
    run_dir_modified = run_calibration(cfg_modified, store_modified, universe=UNI)
    df_modified = load_model_results(run_dir_modified, "heston_qmle_gk")

    # Check AAA parameters are unchanged
    aaa_baseline = df_baseline[df_baseline["ticker"] == "AAA"].iloc[0]
    aaa_modified = df_modified[df_modified["ticker"] == "AAA"].iloc[0]

    params = ["kappa", "theta", "sigma_v", "rho", "mu", "v0"]
    for param in params:
        val_baseline = aaa_baseline[param]
        val_modified = aaa_modified[param]
        assert np.isclose(val_baseline, val_modified, rtol=1e-6), (
            f"AAA {param} changed after adjustment scaling: "
            f"baseline={val_baseline}, modified={val_modified}"
        )


def test_heston_gk_missing_ohlc_produces_unconverged_row(tmp_path):
    """
    Test (c): A ticker with NaN high/low produces converged=False row, run
    completes, and its presence does NOT change other tickers' params
    (mask isolation).
    """
    # BAD: ALL high/low NaN -> every bar invalid -> dropped -> insufficient data
    def make_all_nan_ohlc_frame():
        df = _heston_ohlc_frame(n=700, seed=0, substeps=32)
        df["high"] = np.nan
        df["low"] = np.nan
        return df

    # PARTIAL: every other bar has NaN high/low -> those bars are dropped,
    # remaining ~350 valid bars still calibrate (participates in the batch
    # with a shorter, NaN-free series -> exercises padding/mask isolation)
    def make_partial_nan_ohlc_frame():
        df = _heston_ohlc_frame(n=700, seed=2, substeps=32)
        df.loc[df.index[::2], "high"] = np.nan
        df.loc[df.index[::2], "low"] = np.nan
        return df

    good_frame = _heston_ohlc_frame(n=700, seed=1, substeps=32)
    frames = {
        "GOOD": good_frame,
        "PARTIAL": make_partial_nan_ohlc_frame(),
        "BAD": make_all_nan_ohlc_frame(),
    }

    def fake_fetcher(tickers, start, end):
        return {t: frames[t] for t in tickers if t in frames}

    store = PriceStore(tmp_path / "lake_nan", fetcher=fake_fetcher)

    uni = Universe(
        name="test_nan",
        tickers=["GOOD", "PARTIAL", "BAD"],
        sectors={"GOOD": "Tech", "PARTIAL": "Tech", "BAD": "Tech"},
    )

    cfg = RunConfig(
        universe="ignored",
        models=["heston_qmle_gk"],
        years=2.5,
        n_jobs=1,
        out_root=tmp_path / "runs_nan",
        run_id="nan_test",
        end="2025-01-31",
    )
    run_dir = run_calibration(cfg, store, universe=uni)
    df = load_model_results(run_dir, "heston_qmle_gk")

    # GOOD should converge
    good_row = df[df["ticker"] == "GOOD"].iloc[0]
    assert good_row["converged"]
    assert good_row["error"] == ""

    # PARTIAL should converge on its valid (NaN-free) bars only
    partial_row = df[df["ticker"] == "PARTIAL"].iloc[0]
    assert partial_row["converged"]
    assert partial_row["error"] == ""
    for col in ["kappa", "theta", "sigma_v"]:
        assert np.isfinite(partial_row[col])

    # BAD should not converge (all bars invalid -> insufficient data)
    bad_row = df[df["ticker"] == "BAD"].iloc[0]
    assert not bad_row["converged"]
    # Error message should indicate the problem
    assert bad_row["error"] != ""

    # Mask isolation: run GOOD alone -> params must be unchanged by the
    # presence of the NaN tickers in the batch.
    def fetch_good_only(tickers, start, end):
        return {t: frames[t] for t in tickers if t == "GOOD"}

    store_solo = PriceStore(tmp_path / "lake_solo", fetcher=fetch_good_only)
    uni_solo = Universe(name="test_solo", tickers=["GOOD"],
                        sectors={"GOOD": "Tech"})
    cfg_solo = RunConfig(
        universe="ignored",
        models=["heston_qmle_gk"],
        years=2.5,
        n_jobs=1,
        out_root=tmp_path / "runs_solo",
        run_id="solo",
        end="2025-01-31",
    )
    run_dir_solo = run_calibration(cfg_solo, store_solo, universe=uni_solo)
    df_solo = load_model_results(run_dir_solo, "heston_qmle_gk")
    good_solo = df_solo[df_solo["ticker"] == "GOOD"].iloc[0]

    for param in ["kappa", "theta", "sigma_v", "rho", "mu", "v0"]:
        assert np.isclose(good_row[param], good_solo[param], rtol=1e-6), (
            f"GOOD {param} changed due to NaN tickers in batch: "
            f"batch={good_row[param]}, solo={good_solo[param]}"
        )


def test_heston_gk_no_batch_env_var_ignored(tmp_path, ohlc_store, monkeypatch):
    """
    OPTIONS_DESK_NO_BATCH=1 must be ignored for needs_ohlc models: the batch
    path is mandatory (no per-asset fallback), so the run still produces
    converged GK rows instead of joblib NotImplementedError rows.
    """
    monkeypatch.setenv("OPTIONS_DESK_NO_BATCH", "1")

    cfg = RunConfig(
        universe="ignored",
        models=["heston_qmle_gk"],
        years=2.5,
        n_jobs=1,
        out_root=tmp_path / "runs_no_batch",
        run_id="no_batch",
        end="2025-01-31",
    )
    run_dir = run_calibration(cfg, ohlc_store, universe=UNI)
    df = load_model_results(run_dir, "heston_qmle_gk")

    good_tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    ok = df[df["ticker"].isin(good_tickers)]
    assert ok["converged"].all()
    assert (ok["error"] == "").all()
    for col in ["kappa", "theta", "sigma_v"]:
        assert col in df.columns
        assert np.isfinite(ok[col]).all()
