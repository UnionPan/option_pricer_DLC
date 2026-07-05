import pandas as pd

from options_desk.calibration.pipeline.results_store import (
    is_model_done,
    load_model_results,
    new_run_dir,
    read_manifest,
    save_model_results,
    write_manifest,
)


def test_new_run_dir_creates_and_reuses(tmp_path):
    d1 = new_run_dir(tmp_path, run_id="myrun")
    assert d1.is_dir() and d1.name == "myrun"
    assert new_run_dir(tmp_path, run_id="myrun") == d1     # reuse = resume
    auto = new_run_dir(tmp_path)                           # UTC stamp name
    assert auto.is_dir() and auto != d1


def test_manifest_merge_roundtrip(tmp_path):
    d = new_run_dir(tmp_path, run_id="r")
    write_manifest(d, {"universe": "sp500", "models": ["gbm"]})
    write_manifest(d, {"status": "done"})
    m = read_manifest(d)
    assert m["universe"] == "sp500" and m["status"] == "done"


def test_read_manifest_empty(tmp_path):
    assert read_manifest(new_run_dir(tmp_path, run_id="r")) == {}


def test_model_results_and_done_marker(tmp_path):
    d = new_run_dir(tmp_path, run_id="r")
    assert not is_model_done(d, "gbm")
    df = pd.DataFrame({"ticker": ["AAPL"], "mu": [0.1], "converged": [True]})
    save_model_results(d, "gbm", df)
    assert is_model_done(d, "gbm")
    got = load_model_results(d, "gbm")
    assert got["ticker"].tolist() == ["AAPL"]
