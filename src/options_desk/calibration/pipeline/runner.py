"""
Universe calibration runner: universe -> ensure prices -> per-model joblib
calibration -> parquet results, with checkpoint/resume at model granularity.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..cross_asset import fit_dcc, fit_factor_model, pool_parameters
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
    cross_asset: list[str] = field(default_factory=list)
                                       # allowed: "factor", "dcc", "pooling"
    pooling_model: str = "heston_qmle" # model to pool (must be in models list)


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


def _build_ohlc_data(
    store: PriceStore,
    tickers: list[str],
    start,
    end,
) -> dict[str, dict[str, np.ndarray] | None]:
    """
    Build OHLC data for all tickers with adjustment scaling.

    For each ticker, construct a dict with keys 'open', 'high', 'low', 'close'
    where each OHLC price is scaled by the adjustment factor (adj_close / close)
    to preserve GK variance ratios while making close-to-close returns adjustment-aware.

    Args:
        store: Price store
        tickers: List of tickers
        start: Start date
        end: End date

    Returns:
        Map of ticker -> OHLC dict (or None if no data)
    """
    ohlc_by_ticker = {}
    for t in tickers:
        df = store.get_prices(t)
        if df is None:
            ohlc_by_ticker[t] = None
            continue

        # Filter to window
        df = df.loc[(df.index >= start) & (df.index <= end)]

        if len(df) == 0:
            ohlc_by_ticker[t] = None
            continue

        # Check required columns exist
        required = ["open", "high", "low", "close", "adj_close"]
        if not all(col in df.columns for col in required):
            ohlc_by_ticker[t] = None
            continue

        open_vals = df["open"].to_numpy(dtype=np.float64)
        high_vals = df["high"].to_numpy(dtype=np.float64)
        low_vals = df["low"].to_numpy(dtype=np.float64)
        close_vals = df["close"].to_numpy(dtype=np.float64)
        adj_close_vals = df["adj_close"].to_numpy(dtype=np.float64)

        # A bar is valid only if ALL OHLC values and adjustment-factor inputs
        # are finite AND close > 0 (division guard). Invalid bars are DROPPED
        # from all four series, so the arrays handed to the batch adapter are
        # equal-length and NaN-free — pad_returns then yields identical masks
        # for open/high/low/close. Bars with close <= 0 or NaN inputs are never
        # silently given factor = 1.
        valid_bars = (
            np.isfinite(open_vals)
            & np.isfinite(high_vals)
            & np.isfinite(low_vals)
            & np.isfinite(close_vals)
            & np.isfinite(adj_close_vals)
            & (close_vals > 0)
        )

        # Adjustment factor on valid bars only: adj_close / close
        factor = adj_close_vals[valid_bars] / close_vals[valid_bars]

        # Scale OHLC by adjustment factor (GK ratios unchanged; close-to-close
        # returns become dividend/split-adjusted)
        ohlc_dict = {
            "open": open_vals[valid_bars] * factor,
            "high": high_vals[valid_bars] * factor,
            "low": low_vals[valid_bars] * factor,
            "close": close_vals[valid_bars] * factor,
        }

        ohlc_by_ticker[t] = ohlc_dict

    return ohlc_by_ticker


def _calibrate_batch(
    model_name: str,
    tickers: list[str],
    prices_by_ticker: dict[str, np.ndarray | None],
    dt: float,
) -> list[dict]:
    """
    Batch calibration path: split into valid/invalid, call fit_batch once,
    assemble rows. Falls back to per-asset joblib path if fit_batch raises.

    Args:
        model_name: Model to calibrate
        tickers: Ordered list of tickers (universe order)
        prices_by_ticker: Map of ticker -> prices (or None)
        dt: Time increment

    Returns:
        List of dicts (one per ticker), same schema as _calibrate_one
    """
    spec = get_model(model_name)

    # Split into valid (enough data) and invalid (insufficient data)
    valid_tickers = []
    valid_prices = []
    invalid_rows = []

    for ticker in tickers:
        prices = prices_by_ticker[ticker]
        if prices is None or len(prices) < spec.min_obs:
            n = 0 if prices is None else len(prices)
            invalid_rows.append({
                "ticker": ticker,
                "converged": False,
                "error": f"insufficient data ({n} < {spec.min_obs} obs)",
            })
        else:
            valid_tickers.append(ticker)
            valid_prices.append(np.asarray(prices, dtype=np.float64))

    # If no valid tickers, return only invalid rows
    if not valid_tickers:
        return invalid_rows

    # Try batch calibration
    try:
        result_dict = spec.fit_batch(valid_prices, dt)
        # result_dict has keys -> (N,)-arrays where N = len(valid_tickers)

        # Assemble rows
        valid_rows = []
        for i, ticker in enumerate(valid_tickers):
            row = {"ticker": ticker, "error": ""}
            for key, arr in result_dict.items():
                # Extract scalar value for this ticker
                val = arr[i]
                # Convert numpy types to native Python types for dict
                if isinstance(val, (np.integer, np.floating, np.bool_)):
                    val = val.item()
                row[key] = val
            valid_rows.append(row)

        # Combine valid and invalid rows in original ticker order
        ticker_to_row = {r["ticker"]: r for r in valid_rows + invalid_rows}
        return [ticker_to_row[t] for t in tickers]

    except Exception as e:  # noqa: BLE001
        # Batch calibration failed -> fall back to per-asset path
        logger.warning(
            "Batch calibration for model %s raised %s: %s; "
            "falling back to per-asset joblib path",
            model_name, type(e).__name__, e
        )
        # Return None to signal fallback to joblib
        return None


def _calibrate_batch_ohlc(
    model_name: str,
    tickers: list[str],
    ohlc_by_ticker: dict[str, dict[str, np.ndarray] | None],
    dt: float,
) -> list[dict] | None:
    """
    Batch calibration for OHLC models.

    Args:
        model_name: Model to calibrate
        tickers: Ordered list of tickers (universe order)
        ohlc_by_ticker: Map of ticker -> OHLC dict (or None)
        dt: Time increment

    Returns:
        List of dicts (one per ticker), or None if batch calibration fails
    """
    spec = get_model(model_name)

    # Split into valid (enough data and valid OHLC) and invalid
    valid_tickers = []
    valid_ohlc = []
    invalid_rows = []

    for ticker in tickers:
        ohlc = ohlc_by_ticker[ticker]

        # Check if we have OHLC data
        if ohlc is None:
            invalid_rows.append({
                "ticker": ticker,
                "converged": False,
                "error": "insufficient data (0 < {} obs)".format(spec.min_obs),
            })
            continue

        # Check length
        n = len(ohlc["close"])
        if n < spec.min_obs:
            invalid_rows.append({
                "ticker": ticker,
                "converged": False,
                "error": f"insufficient data ({n} < {spec.min_obs} obs)",
            })
            continue

        # Defensive check for NaN/inf: _build_ohlc_data drops invalid bars,
        # so this should never trigger for data assembled by the runner.
        has_invalid = False
        for col in ["open", "high", "low", "close"]:
            if not np.all(np.isfinite(ohlc[col])):
                has_invalid = True
                break

        if has_invalid:
            invalid_rows.append({
                "ticker": ticker,
                "converged": False,
                "error": "invalid OHLC data (NaN or inf values)",
            })
            continue

        valid_tickers.append(ticker)
        valid_ohlc.append(ohlc)

    # If no valid tickers, return only invalid rows
    if not valid_tickers:
        return invalid_rows

    # Try batch calibration
    try:
        result_dict = spec.fit_batch(valid_ohlc, dt)
        # result_dict has keys -> (N,)-arrays where N = len(valid_tickers)

        # Assemble rows
        valid_rows = []
        for i, ticker in enumerate(valid_tickers):
            row = {"ticker": ticker, "error": ""}
            for key, arr in result_dict.items():
                # Extract scalar value for this ticker
                val = arr[i]
                # Convert numpy types to native Python types for dict
                if isinstance(val, (np.integer, np.floating, np.bool_)):
                    val = val.item()
                row[key] = val
            valid_rows.append(row)

        # Combine valid and invalid rows in original ticker order
        ticker_to_row = {r["ticker"]: r for r in valid_rows + invalid_rows}
        return [ticker_to_row[t] for t in tickers]

    except Exception as e:  # noqa: BLE001
        # Batch calibration failed -> NO fallback for OHLC models
        logger.error(
            "Batch calibration for OHLC model %s raised %s: %s; "
            "no fallback available",
            model_name, type(e).__name__, e
        )
        # Return None to signal error
        return None


def _is_stage_done(run_dir: Path, stage: str) -> bool:
    """Check if a cross-asset stage has been completed."""
    return (run_dir / f"{stage}.done").exists()


def _mark_stage_done(run_dir: Path, stage: str) -> None:
    """Mark a cross-asset stage as completed."""
    (run_dir / f"{stage}.done").write_text(datetime.now(timezone.utc).isoformat())


def _run_cross_asset_stage(
    cfg: RunConfig,
    store: PriceStore,
    universe: Universe,
    start,
    end,
    run_dir: Path,
) -> None:
    """Run cross-asset analysis stages (factor/dcc/pooling) with isolation."""
    if not cfg.cross_asset:
        return

    logger.info("Running cross-asset stages: %s", cfg.cross_asset)

    # Initialize cross-asset manifest section
    ca_manifest = read_manifest(run_dir).get("cross_asset", {})
    if "errors" not in ca_manifest:
        ca_manifest["errors"] = {}
    if "completed" not in ca_manifest:
        ca_manifest["completed"] = []

    # Stage 1: Build returns matrix (needed for factor and for tracking excluded)
    rm = None
    T_window = round(cfg.years * 252)
    if "factor" in cfg.cross_asset or "dcc" in cfg.cross_asset:
        min_obs = min(504, int(0.8 * T_window))
        rm = store.returns_matrix(universe.tickers, start, end, min_obs=min_obs)
        ca_manifest["excluded"] = len(rm.excluded)
        write_manifest(run_dir, {"cross_asset": ca_manifest})
        logger.info("Returns matrix: %d names × %d days; excluded %d",
                    len(rm.tickers), len(rm.returns), len(rm.excluded))

    # Stage 2: Factor model
    if "factor" in cfg.cross_asset:
        if _is_stage_done(run_dir, "factor"):
            logger.info("Stage 'factor' already done — skipping (resume)")
        else:
            try:
                if rm is None or len(rm.tickers) == 0:
                    raise ValueError("No valid tickers for factor model")

                logger.info("Fitting factor model on %d names...", len(rm.tickers))
                fm = fit_factor_model(
                    rm.returns.astype(np.float64),
                    rm.tickers,
                )

                # Save factor_model.npz
                np.savez(
                    run_dir / "factor_model.npz",
                    loadings=fm.loadings,
                    factor_cov=fm.factor_cov,
                    resid_var=fm.resid_var,
                    factors=fm.factors,
                    tickers=np.array(fm.tickers),
                )

                # Save factor_summary.json
                T, N = rm.returns.shape
                # Compute eigenvalue shares
                corr = (rm.returns.T @ rm.returns) / (T - 1)
                eigvals = np.linalg.eigvalsh(corr)
                eigvals = np.sort(eigvals)[::-1]
                total_var = eigvals.sum()
                top5_shares = [float(eigvals[i] / total_var)
                               for i in range(min(5, len(eigvals)))]

                factor_summary = {
                    "k": int(fm.k),
                    "mp_edge": float(fm.mp_edge),
                    "n_names": N,
                    "T": T,
                    "min_eig_lower_bound": float(fm.cov().min_eig_lower_bound()),
                    "top5_eigenvalue_shares": top5_shares,
                }
                (run_dir / "factor_summary.json").write_text(
                    json.dumps(factor_summary, indent=2)
                )

                _mark_stage_done(run_dir, "factor")
                ca_manifest["completed"].append("factor")
                write_manifest(run_dir, {"cross_asset": ca_manifest})
                logger.info("Factor model: k=%d, mp_edge=%.4f", fm.k, fm.mp_edge)

            except Exception as e:  # noqa: BLE001
                err_msg = f"{type(e).__name__}: {e}"
                logger.error("Stage 'factor' failed: %s", err_msg)
                ca_manifest["errors"]["factor"] = err_msg
                write_manifest(run_dir, {"cross_asset": ca_manifest})

    # Stage 3: DCC
    if "dcc" in cfg.cross_asset:
        if _is_stage_done(run_dir, "dcc"):
            logger.info("Stage 'dcc' already done — skipping (resume)")
        else:
            try:
                # Load factors from this run
                factor_npz_path = run_dir / "factor_model.npz"
                if not factor_npz_path.exists():
                    raise FileNotFoundError(
                        "factor_model.npz not found; run 'factor' stage first"
                    )

                factor_data = np.load(factor_npz_path)
                factors = factor_data["factors"]

                logger.info("Fitting DCC on %d factors...", factors.shape[1])
                dcc_result = fit_dcc(factors)

                # Save dcc.json
                dcc_dict = {
                    "a": float(dcc_result.a),
                    "b": float(dcc_result.b),
                    "log_likelihood": float(dcc_result.log_likelihood),
                    "converged": bool(dcc_result.converged),
                    "garch_params": dcc_result.garch_params.to_dict(orient="records"),
                    "last_corr": dcc_result.last_corr.tolist(),
                    "qbar": dcc_result.qbar.tolist(),
                    "valid_factor_indices": dcc_result.valid_factor_indices.tolist(),
                    "n_factors_original": factors.shape[1],
                    "n_factors_used": len(dcc_result.garch_params),
                }
                (run_dir / "dcc.json").write_text(
                    json.dumps(dcc_dict, indent=2)
                )

                _mark_stage_done(run_dir, "dcc")
                ca_manifest["completed"].append("dcc")
                write_manifest(run_dir, {"cross_asset": ca_manifest})
                logger.info("DCC: a=%.4f, b=%.4f, converged=%s",
                            dcc_result.a, dcc_result.b, dcc_result.converged)

            except Exception as e:  # noqa: BLE001
                err_msg = f"{type(e).__name__}: {e}"
                logger.error("Stage 'dcc' failed: %s", err_msg)
                ca_manifest["errors"]["dcc"] = err_msg
                write_manifest(run_dir, {"cross_asset": ca_manifest})

    # Stage 4: Pooling
    if "pooling" in cfg.cross_asset:
        if _is_stage_done(run_dir, "pooling"):
            logger.info("Stage 'pooling' already done — skipping (resume)")
        else:
            try:
                model = cfg.pooling_model
                model_parquet = run_dir / f"{model}.parquet"

                if not model_parquet.exists():
                    raise FileNotFoundError(
                        f"{model}.parquet not found; run model calibration first"
                    )

                df = pd.read_parquet(model_parquet)

                # Determine parameters to pool based on model
                if model == "heston_qmle":
                    params = ["kappa", "theta", "sigma_v", "rho", "mu", "v0"]
                elif model == "garch":
                    params = ["omega", "alpha", "beta", "mu"]
                elif model == "gbm":
                    params = ["mu", "sigma"]
                else:
                    # Generic: find numeric columns excluding standard columns
                    exclude_cols = {"ticker", "sector", "calibration_date",
                                    "converged", "error"}
                    params = [c for c in df.columns
                              if c not in exclude_cols and df[c].dtype in
                              [np.float64, np.float32, np.int64, np.int32]]

                # Filter to only existing columns
                params = [p for p in params if p in df.columns]

                if not params:
                    raise ValueError(f"No parameters to pool for model {model}")

                logger.info("Pooling %d parameters for model %s...",
                            len(params), model)
                df_pooled = pool_parameters(df, params)

                # Save pooled results
                pooled_path = run_dir / f"{model}_pooled.parquet"
                df_pooled.to_parquet(pooled_path, index=False)

                _mark_stage_done(run_dir, "pooling")
                ca_manifest["completed"].append("pooling")
                write_manifest(run_dir, {"cross_asset": ca_manifest})
                logger.info("Pooling complete: %d parameters, output=%s",
                            len(params), pooled_path.name)

            except Exception as e:  # noqa: BLE001
                err_msg = f"{type(e).__name__}: {e}"
                logger.error("Stage 'pooling' failed: %s", err_msg)
                ca_manifest["errors"]["pooling"] = err_msg
                write_manifest(run_dir, {"cross_asset": ca_manifest})


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
    models_errors = read_manifest(run_dir).get("models_errors", {})

    for model in cfg.models:
        if is_model_done(run_dir, model):
            logger.info("model %s already done — skipping (resume)", model)
            continue
        logger.info("calibrating %s over %d names (n_jobs=%s)...",
                    model, len(universe), cfg.n_jobs)

        # Decide whether to use batch path or per-asset path.
        # For needs_ohlc models the batch path is MANDATORY: there is no
        # per-asset scipy fallback, so OPTIONS_DESK_NO_BATCH is ignored.
        spec = get_model(model)
        use_batch = (
            spec.fit_batch is not None
            and (spec.needs_ohlc
                 or os.environ.get("OPTIONS_DESK_NO_BATCH") != "1")
        )

        if use_batch:
            logger.info("Using batch calibration path for model %s", model)

            if spec.needs_ohlc:
                if os.environ.get("OPTIONS_DESK_NO_BATCH") == "1":
                    logger.info(
                        "OPTIONS_DESK_NO_BATCH=1 ignored for OHLC model %s: "
                        "batch path is mandatory (no per-asset fallback)",
                        model,
                    )
                # Build OHLC data for needs_ohlc models
                ohlc_by_ticker = _build_ohlc_data(store, universe.tickers, start, end)
                rows = _calibrate_batch_ohlc(
                    model, universe.tickers, ohlc_by_ticker, cfg.dt
                )
                # OHLC models have no fallback: if rows is None, record error and skip
                if rows is None:
                    err_msg = "Batch calibration failed (see logs)"
                    logger.error("Model %s (OHLC) batch calibration failed; no fallback available", model)
                    models_errors[model] = err_msg
                    write_manifest(run_dir, {"models_errors": models_errors})
                    continue
            else:
                rows = _calibrate_batch(model, universe.tickers, prices_by_ticker, cfg.dt)
                # If batch calibration failed (returns None), fall back to joblib
                if rows is None:
                    use_batch = False

        if not use_batch:
            logger.info("Using per-asset joblib path for model %s", model)
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

    # Cross-asset stages (factor, dcc, pooling)
    _run_cross_asset_stage(cfg, store, universe, start, end, run_dir)

    write_manifest(run_dir, {"models": model_stats, "status": "complete",
                             "finished_at": datetime.now(timezone.utc).isoformat()})
    return run_dir
