"""
Tests for batch calibration dispatch in the runner.

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
from options_desk.calibration.pipeline.registry import ModelSpec, register_model
from options_desk.calibration.pipeline.results_store import load_model_results
from options_desk.calibration.pipeline.runner import RunConfig, run_calibration


def _gbm_frame(n=800, seed=0):
    """Generate synthetic GBM price data."""
    rng = np.random.default_rng(seed)
    r = 0.0003 + 0.012 * rng.standard_normal(n)
    close = pd.Series(
        100.0 * np.exp(np.cumsum(r)), index=pd.bdate_range("2022-01-03", periods=n)
    )
    df = pd.DataFrame(
        {c: close for c in ["open", "high", "low", "close", "adj_close"]}
    )
    df["volume"] = 1e6
    return df


@pytest.fixture()
def store(tmp_path):
    """Synthetic price store with 3 good tickers + 1 dead ticker."""
    frames = {t: _gbm_frame(seed=i) for i, t in enumerate(["AAA", "BBB", "CCC"])}
    # DEAD has only 30 observations: insufficient for gbm (min_obs=60)
    frames["DEAD"] = _gbm_frame(n=30, seed=3)

    def fake_fetcher(tickers, start, end):
        return {t: frames[t] for t in tickers if t in frames}

    return PriceStore(tmp_path / "lake", fetcher=fake_fetcher)


UNI = Universe(
    name="test_batch",
    tickers=["AAA", "BBB", "CCC", "DEAD"],
    sectors={"AAA": "Tech", "BBB": "Energy", "CCC": "Tech", "DEAD": "UNKNOWN"},
)


def test_batch_vs_per_asset_produces_same_results(tmp_path, store, monkeypatch):
    """
    Test (a): Runner with model gbm on a 3-good+1-dead universe produces the
    same converged set and mu/sigma within rel 1e-3 with and without
    OPTIONS_DESK_NO_BATCH=1.
    """
    # Run with batch path
    cfg_batch = RunConfig(
        universe="ignored",
        models=["gbm"],
        years=3.0,
        n_jobs=1,
        out_root=tmp_path / "runs_batch",
        run_id="batch",
        end="2025-01-31",
    )
    run_dir_batch = run_calibration(cfg_batch, store, universe=UNI)
    df_batch = load_model_results(run_dir_batch, "gbm")

    # Run with per-asset path (NO_BATCH=1)
    monkeypatch.setenv("OPTIONS_DESK_NO_BATCH", "1")
    cfg_no_batch = RunConfig(
        universe="ignored",
        models=["gbm"],
        years=3.0,
        n_jobs=1,
        out_root=tmp_path / "runs_no_batch",
        run_id="no_batch",
        end="2025-01-31",
    )
    run_dir_no_batch = run_calibration(cfg_no_batch, store, universe=UNI)
    df_no_batch = load_model_results(run_dir_no_batch, "gbm")

    # Check same converged set
    assert set(df_batch[df_batch["converged"]]["ticker"]) == set(
        df_no_batch[df_no_batch["converged"]]["ticker"]
    )
    assert set(df_batch[df_batch["converged"]]["ticker"]) == {"AAA", "BBB", "CCC"}

    # Check DEAD is not converged in both
    assert not df_batch[df_batch["ticker"] == "DEAD"].iloc[0]["converged"]
    assert not df_no_batch[df_no_batch["ticker"] == "DEAD"].iloc[0]["converged"]

    # Check mu and sigma match within rel tol 1e-3
    for ticker in ["AAA", "BBB", "CCC"]:
        row_batch = df_batch[df_batch["ticker"] == ticker].iloc[0]
        row_no_batch = df_no_batch[df_no_batch["ticker"] == ticker].iloc[0]

        mu_batch = row_batch["mu"]
        mu_no_batch = row_no_batch["mu"]
        sigma_batch = row_batch["sigma"]
        sigma_no_batch = row_no_batch["sigma"]

        # Relative tolerance 1e-3
        assert np.isclose(mu_batch, mu_no_batch, rtol=1e-3), (
            f"{ticker}: mu mismatch batch={mu_batch}, no_batch={mu_no_batch}"
        )
        assert np.isclose(sigma_batch, sigma_no_batch, rtol=1e-3), (
            f"{ticker}: sigma mismatch batch={sigma_batch}, no_batch={sigma_no_batch}"
        )


def _ou_frame(n=800, seed=0, kappa=15.0, theta=None, sigma=0.4):
    """Generate synthetic mean-reverting prices: OU process in log-price."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    theta = np.log(100.0) if theta is None else theta
    b = np.exp(-kappa * dt)
    noise_sd = sigma * np.sqrt((1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa))
    x = np.empty(n)
    x[0] = theta
    z = rng.standard_normal(n - 1)
    for i in range(1, n):
        x[i] = theta + (x[i - 1] - theta) * b + noise_sd * z[i - 1]
    close = pd.Series(np.exp(x), index=pd.bdate_range("2022-01-03", periods=n))
    df = pd.DataFrame(
        {c: close for c in ["open", "high", "low", "close", "adj_close"]}
    )
    df["volume"] = 1e6
    return df


@pytest.fixture()
def ou_store(tmp_path):
    """Synthetic mean-reverting price store: 3 good tickers + 1 dead ticker."""
    frames = {t: _ou_frame(seed=i) for i, t in enumerate(["AAA", "BBB", "CCC"])}
    frames["DEAD"] = _ou_frame(n=30, seed=3)

    def fake_fetcher(tickers, start, end):
        return {t: frames[t] for t in tickers if t in frames}

    return PriceStore(tmp_path / "lake", fetcher=fake_fetcher)


def test_ou_batch_vs_per_asset(tmp_path, ou_store, monkeypatch):
    """
    OU: batch path (log-price levels, JAX AR(1)) and per-asset scipy fallback
    must agree on kappa/theta/sigma within rel 1e-3.

    Regression test: the scipy fallback previously received RAW prices while
    the batch adapter used log-prices as the OU level series, so the two
    paths silently produced different parameters.
    """
    cfg_batch = RunConfig(
        universe="ignored",
        models=["ou"],
        years=3.0,
        n_jobs=1,
        out_root=tmp_path / "runs_ou_batch",
        run_id="ou_batch",
        end="2025-01-31",
    )
    run_dir_batch = run_calibration(cfg_batch, ou_store, universe=UNI)
    df_batch = load_model_results(run_dir_batch, "ou")

    monkeypatch.setenv("OPTIONS_DESK_NO_BATCH", "1")
    cfg_no_batch = RunConfig(
        universe="ignored",
        models=["ou"],
        years=3.0,
        n_jobs=1,
        out_root=tmp_path / "runs_ou_no_batch",
        run_id="ou_no_batch",
        end="2025-01-31",
    )
    run_dir_no_batch = run_calibration(cfg_no_batch, ou_store, universe=UNI)
    df_no_batch = load_model_results(run_dir_no_batch, "ou")

    # Same converged set
    assert set(df_batch[df_batch["converged"]]["ticker"]) == set(
        df_no_batch[df_no_batch["converged"]]["ticker"]
    )
    assert set(df_batch[df_batch["converged"]]["ticker"]) == {"AAA", "BBB", "CCC"}

    # kappa/theta/sigma agree within rel 1e-3
    for ticker in ["AAA", "BBB", "CCC"]:
        row_b = df_batch[df_batch["ticker"] == ticker].iloc[0]
        row_s = df_no_batch[df_no_batch["ticker"] == ticker].iloc[0]
        for param in ["kappa", "theta", "sigma"]:
            assert np.isclose(row_b[param], row_s[param], rtol=1e-3), (
                f"{ticker}: {param} mismatch batch={row_b[param]}, "
                f"per-asset={row_s[param]}"
            )


def test_batch_invalid_tickers_get_insufficient_data_rows(tmp_path, store):
    """
    Test (b): Invalid tickers get insufficient-data rows in batch mode.
    """
    cfg = RunConfig(
        universe="ignored",
        models=["gbm"],
        years=3.0,
        n_jobs=1,
        out_root=tmp_path / "runs",
        run_id="invalid_test",
        end="2025-01-31",
    )
    run_dir = run_calibration(cfg, store, universe=UNI)
    df = load_model_results(run_dir, "gbm")

    # DEAD should have insufficient data
    dead_row = df[df["ticker"] == "DEAD"].iloc[0]
    assert not dead_row["converged"]
    assert "insufficient data" in dead_row["error"]


def test_batch_calibration_fallback_on_exception(tmp_path, store, monkeypatch):
    """
    Test (c): Register a model whose fit_batch raises -> runner falls back to
    per-asset path and still produces complete results.
    """

    def _fit_exploder(prices: np.ndarray, dt: float) -> dict:
        """Per-asset fit that works."""
        returns = np.diff(np.log(prices))
        mu = np.mean(returns) / dt
        sigma = np.std(returns, ddof=1) / np.sqrt(dt)
        return {"mu": mu, "sigma": sigma, "converged": True}

    def _batch_exploder(price_arrays: list[np.ndarray]) -> dict:
        """Batch fit that always raises."""
        raise RuntimeError("batch exploder boom")

    # Register the exploder model (name distinct from test_runner.py's
    # "exploder": the registry is process-global and rejects duplicates)
    register_model(
        ModelSpec(
            name="batch_exploder", fit=_fit_exploder, min_obs=60,
            fit_batch=_batch_exploder,
        )
    )

    # Run calibration (should fall back to per-asset path)
    cfg = RunConfig(
        universe="ignored",
        models=["batch_exploder"],
        years=3.0,
        n_jobs=1,
        out_root=tmp_path / "runs_fallback",
        run_id="fallback",
        end="2025-01-31",
    )
    run_dir = run_calibration(cfg, store, universe=UNI)
    df = load_model_results(run_dir, "batch_exploder")

    # Should have results for all tickers
    assert set(df["ticker"]) == {"AAA", "BBB", "CCC", "DEAD"}

    # Good tickers should converge (via per-asset fallback)
    ok = df[df["ticker"].isin(["AAA", "BBB", "CCC"])]
    assert ok["converged"].all()
    assert (ok["error"] == "").all()

    # DEAD should still have insufficient data error
    dead = df[df["ticker"] == "DEAD"].iloc[0]
    assert not dead["converged"]
    assert "insufficient data" in dead["error"]
