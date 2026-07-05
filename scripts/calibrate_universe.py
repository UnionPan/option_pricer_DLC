"""
Universe-scale P-measure calibration CLI.

Examples:
    # 50-name default basket, Heston QMLE (back-compatible default)
    python scripts/calibrate_universe.py

    # S&P 500, three models, resumable
    python scripts/calibrate_universe.py --universe sp500 \
        --models gbm garch heston_qmle --run-id sp500-2026-07-05

    # resume after interruption (completed models are skipped)
    python scripts/calibrate_universe.py --universe sp500 \
        --models gbm garch heston_qmle --run-id sp500-2026-07-05
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from options_desk.calibration.data.price_store import (  # noqa: E402
    PriceStore, fetch_yfinance,
)
from options_desk.calibration.data.universe import (  # noqa: E402
    Universe, available_universes, load_universe,
)
from options_desk.calibration.physical import DEFAULT_BASKET_50  # noqa: E402
from options_desk.calibration.pipeline import (  # noqa: E402
    RunConfig, list_models, read_manifest, run_calibration,
)


def _default_fetcher():
    return fetch_yfinance


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--universe", default=None,
                   help="universe name (data/universes/<name>.csv) or CSV path")
    p.add_argument("--tickers", nargs="*", default=None,
                   help="explicit tickers (overrides --universe)")
    p.add_argument("--models", nargs="*", default=["heston_qmle"],
                   help="registered models to run (see --list-models)")
    p.add_argument("--years", type=float, default=5.0,
                   help="lookback window in years")
    p.add_argument("--end", default=None,
                   help="window end date YYYY-MM-DD (default: today UTC)")
    p.add_argument("--jobs", type=int, default=-1,
                   help="joblib parallel jobs (-1 = all cores)")
    p.add_argument("--price-lake", default=str(ROOT / "data" / "price_lake"),
                   help="price lake root")
    p.add_argument("--out-root", default=str(ROOT / "runs" / "calibration"),
                   help="run output root")
    p.add_argument("--run-id", default=None,
                   help="run id; reuse an existing id to resume")
    p.add_argument("--list-models", action="store_true",
                   help="print registered models and exit")
    p.add_argument("--list-universes", action="store_true",
                   help="print available universes and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("calibrate_universe")

    if args.list_models:
        print("\n".join(list_models()))
        return 0
    if args.list_universes:
        print("\n".join(available_universes()) or "(none — run scripts/build_universes.py)")
        return 0

    if args.tickers:
        universe = Universe(name="adhoc",
                            tickers=[t.upper() for t in args.tickers],
                            sectors={t.upper(): "UNKNOWN" for t in args.tickers})
    elif args.universe:
        universe = load_universe(args.universe)
    else:
        universe = Universe(name="basket50", tickers=list(DEFAULT_BASKET_50),
                            sectors={t: "UNKNOWN" for t in DEFAULT_BASKET_50})

    store = PriceStore(args.price_lake, fetcher=_default_fetcher())
    cfg = RunConfig(universe=universe.name, models=args.models,
                    years=args.years, n_jobs=args.jobs,
                    out_root=args.out_root, run_id=args.run_id, end=args.end)

    run_dir = run_calibration(cfg, store, universe=universe)

    m = read_manifest(run_dir)
    log.info("=" * 60)
    log.info("run %s complete — universe=%s (%d names)",
             run_dir.name, m.get("universe"), m.get("n_tickers", 0))
    for model, s in m.get("models", {}).items():
        log.info("  %-14s %d/%d converged (%.1f%%)", model,
                 s["n_converged"], s["n"], 100.0 * s["n_converged"] / max(s["n"], 1))
    ens = m.get("ensure", {})
    log.info("  data: %d fetched, %d cached, %d failed",
             ens.get("fetched", 0), ens.get("cached", 0),
             len(ens.get("failed", {})))
    log.info("  output: %s", run_dir)
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
