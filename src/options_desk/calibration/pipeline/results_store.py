"""
On-disk layout for calibration runs:

    <out_root>/<run_id>/
        manifest.json      run metadata, merge-updated as the run progresses
        <model>.parquet    one row per ticker
        <model>.done       completion marker (UTC timestamp) -> resume skips
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def new_run_dir(out_root: str | Path, run_id: str | None = None) -> Path:
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    d = Path(out_root) / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_manifest(run_dir: Path, updates: dict) -> None:
    m = read_manifest(run_dir)
    m.update(updates)
    (Path(run_dir) / "manifest.json").write_text(
        json.dumps(m, indent=2, default=str))


def read_manifest(run_dir: Path) -> dict:
    p = Path(run_dir) / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _model_paths(run_dir: Path, model: str) -> tuple[Path, Path]:
    d = Path(run_dir)
    return d / f"{model}.parquet", d / f"{model}.done"


def save_model_results(run_dir: Path, model: str, df: pd.DataFrame) -> Path:
    pq, done = _model_paths(run_dir, model)
    df.to_parquet(pq, index=False)
    done.write_text(datetime.now(timezone.utc).isoformat())
    return pq


def load_model_results(run_dir: Path, model: str) -> pd.DataFrame:
    pq, _ = _model_paths(run_dir, model)
    return pd.read_parquet(pq)


def is_model_done(run_dir: Path, model: str) -> bool:
    _, done = _model_paths(run_dir, model)
    return done.exists()
