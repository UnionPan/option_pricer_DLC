"""Tests for cross-asset stage integration (factor/dcc/pooling) in runner."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Set JAX to CPU before importing anything JAX-touching
os.environ["JAX_PLATFORMS"] = "cpu"

from options_desk.calibration.data.price_store import PriceStore
from options_desk.calibration.data.universe import Universe
from options_desk.calibration.pipeline import RunConfig, run_calibration


@pytest.fixture
def synthetic_store(tmp_path):
    """Create a synthetic price store with ~40 names × 700 bdays, 2-factor structure.

    The synthetic data has:
    - Market factor (affects all names)
    - Sector factor (affects names within each sector)
    - Idiosyncratic noise
    """
    # Parameters
    n_names = 40
    n_days = 700
    n_sectors = 4

    # Generate tickers and sectors
    tickers = [f"TICK{i:02d}" for i in range(n_names)]
    sectors = {t: f"SECTOR{i % n_sectors}" for i, t in enumerate(tickers)}

    # Generate synthetic returns with 2-factor structure
    np.random.seed(42)

    # Market factor (affects all)
    market_factor = np.random.randn(n_days) * 0.02

    # Sector factors
    sector_factors = {
        f"SECTOR{i}": np.random.randn(n_days) * 0.015
        for i in range(n_sectors)
    }

    # Generate returns: market + sector + idiosyncratic
    returns = {}
    for ticker in tickers:
        sector = sectors[ticker]
        idio = np.random.randn(n_days) * 0.01
        returns[ticker] = market_factor + sector_factors[sector] + idio

    # Convert returns to prices (cumulative product)
    dates = pd.bdate_range(start="2020-01-01", periods=n_days)
    price_store_root = tmp_path / "price_lake"
    price_store_root.mkdir()

    # PriceStore expects files in prices/ subdirectory
    prices_dir = price_store_root / "prices"
    prices_dir.mkdir()

    for ticker in tickers:
        # Start at price 100, apply returns
        log_prices = np.cumsum(returns[ticker])
        prices = 100.0 * np.exp(log_prices)

        df = pd.DataFrame({
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "adj_close": prices,
            "volume": np.random.randint(1e6, 1e8, n_days),
        }, index=dates)

        # Save to store (PriceStore uses prices/<TICKER>.parquet)
        df.to_parquet(prices_dir / f"{ticker}.parquet")

    # Create PriceStore with no-op fetcher
    def noop_fetcher(tickers, start, end):
        return {}

    store = PriceStore(price_store_root, fetcher=noop_fetcher)

    # Set up coverage ledger to indicate all data is already present
    coverage = {}
    for ticker in tickers:
        coverage[ticker] = {
            "start": "2019-01-01",
            "end": "2025-12-31",
        }
    import json
    coverage_path = price_store_root / "coverage.json"
    coverage_path.write_text(json.dumps(coverage, indent=0, sort_keys=True))

    # Create Universe
    universe = Universe(name="synthetic", tickers=tickers, sectors=sectors)

    return store, universe


def test_full_cross_asset_stage(synthetic_store, tmp_path):
    """Test full cross-asset stage: factor, dcc, pooling with gbm and heston_qmle."""
    store, universe = synthetic_store
    out_root = tmp_path / "runs"

    cfg = RunConfig(
        universe="synthetic",
        models=["gbm", "heston_qmle"],
        years=2.0,
        n_jobs=1,
        out_root=str(out_root),
        end="2022-09-01",  # Within synthetic data range (2020-01-01 to 2022-09-06)
        cross_asset=["factor", "dcc", "pooling"],
        pooling_model="heston_qmle",
    )

    run_dir = run_calibration(cfg, store, universe=universe)

    # Check model calibration outputs
    assert (run_dir / "gbm.parquet").exists()
    assert (run_dir / "gbm.done").exists()
    assert (run_dir / "heston_qmle.parquet").exists()
    assert (run_dir / "heston_qmle.done").exists()

    # Check factor model outputs
    assert (run_dir / "factor_model.npz").exists()
    assert (run_dir / "factor_summary.json").exists()
    assert (run_dir / "factor.done").exists()

    # Check DCC outputs
    assert (run_dir / "dcc.json").exists()
    assert (run_dir / "dcc.done").exists()

    # Check pooling outputs
    assert (run_dir / "heston_qmle_pooled.parquet").exists()
    assert (run_dir / "pooling.done").exists()

    # Check manifest
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert "cross_asset" in manifest
    assert set(manifest["cross_asset"]["completed"]) == {"factor", "dcc", "pooling"}
    assert "excluded" in manifest["cross_asset"]
    assert manifest["cross_asset"]["excluded"] >= 0

    # Verify factor model data
    factor_data = np.load(run_dir / "factor_model.npz")
    assert "loadings" in factor_data
    assert "factor_cov" in factor_data
    assert "resid_var" in factor_data
    assert "factors" in factor_data
    assert "tickers" in factor_data

    # Verify factor summary
    factor_summary = json.loads((run_dir / "factor_summary.json").read_text())
    assert "k" in factor_summary
    assert factor_summary["k"] >= 1
    assert "mp_edge" in factor_summary
    assert "n_names" in factor_summary
    assert "T" in factor_summary
    assert "min_eig_lower_bound" in factor_summary
    assert "top5_eigenvalue_shares" in factor_summary

    # Verify DCC result
    dcc_result = json.loads((run_dir / "dcc.json").read_text())
    assert "a" in dcc_result
    assert "b" in dcc_result
    assert "log_likelihood" in dcc_result
    assert "converged" in dcc_result
    assert "garch_params" in dcc_result
    assert "last_corr" in dcc_result
    assert isinstance(dcc_result["last_corr"], list)

    # Verify pooled parameters
    df_pooled = pd.read_parquet(run_dir / "heston_qmle_pooled.parquet")
    expected_params = ["kappa", "theta", "sigma_v", "rho", "mu", "v0"]
    for param in expected_params:
        assert f"{param}_pooled" in df_pooled.columns
        assert f"{param}_shrinkage" in df_pooled.columns


def test_cross_asset_resume(synthetic_store, tmp_path):
    """Test that cross-asset stages are skipped on resume (via .done markers)."""
    store, universe = synthetic_store
    out_root = tmp_path / "runs"

    cfg = RunConfig(
        universe="synthetic",
        models=["heston_qmle"],
        years=2.0,
        n_jobs=1,
        out_root=str(out_root),
        run_id="resume_test",
        end="2022-09-01",
        cross_asset=["factor", "dcc", "pooling"],
        pooling_model="heston_qmle",
    )

    # First run
    run_dir = run_calibration(cfg, store, universe=universe)
    assert (run_dir / "factor.done").exists()
    assert (run_dir / "dcc.done").exists()
    assert (run_dir / "pooling.done").exists()

    # Record timestamps
    factor_done_ts = (run_dir / "factor.done").read_text()
    dcc_done_ts = (run_dir / "dcc.done").read_text()
    pooling_done_ts = (run_dir / "pooling.done").read_text()

    # Second run (resume)
    run_dir2 = run_calibration(cfg, store, universe=universe)
    assert run_dir2 == run_dir

    # Verify .done markers unchanged (stages were skipped)
    assert (run_dir / "factor.done").read_text() == factor_done_ts
    assert (run_dir / "dcc.done").read_text() == dcc_done_ts
    assert (run_dir / "pooling.done").read_text() == pooling_done_ts


def test_cross_asset_pooling_missing_model(synthetic_store, tmp_path):
    """Test that pooling stage error is isolated when model parquet is missing."""
    store, universe = synthetic_store
    out_root = tmp_path / "runs"

    cfg = RunConfig(
        universe="synthetic",
        models=["gbm"],  # Only run gbm, not heston_qmle
        years=2.0,
        n_jobs=1,
        out_root=str(out_root),
        end="2022-09-01",
        cross_asset=["factor", "pooling"],  # Skip dcc to simplify
        pooling_model="heston_qmle",  # But ask to pool heston_qmle (missing!)
    )

    run_dir = run_calibration(cfg, store, universe=universe)

    # Factor should succeed
    assert (run_dir / "factor.done").exists()

    # Pooling should fail (recorded as error)
    assert not (run_dir / "pooling.done").exists()

    # Check manifest for error
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert "cross_asset" in manifest
    assert "errors" in manifest["cross_asset"]
    assert "pooling" in manifest["cross_asset"]["errors"]
    assert "FileNotFoundError" in manifest["cross_asset"]["errors"]["pooling"]

    # Factor should still be in completed
    assert "factor" in manifest["cross_asset"]["completed"]
    assert "pooling" not in manifest["cross_asset"]["completed"]

    # Run should still be marked complete
    assert manifest["status"] == "complete"


def test_factor_summary_min_eig_positive(synthetic_store, tmp_path):
    """Test that factor_summary.json min_eig_lower_bound > 0."""
    store, universe = synthetic_store
    out_root = tmp_path / "runs"

    cfg = RunConfig(
        universe="synthetic",
        models=["gbm"],
        years=2.0,
        n_jobs=1,
        out_root=str(out_root),
        end="2022-09-01",
        cross_asset=["factor"],
    )

    run_dir = run_calibration(cfg, store, universe=universe)

    factor_summary = json.loads((run_dir / "factor_summary.json").read_text())
    assert factor_summary["min_eig_lower_bound"] > 0, (
        f"Expected min_eig_lower_bound > 0, got {factor_summary['min_eig_lower_bound']}"
    )
