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
