import numpy as np
import pandas as pd
import pytest

from options_desk.calibration.data.price_store import PriceStore
from options_desk.calibration.data.universe import Universe
from options_desk.calibration.pipeline.registry import ModelSpec, register_model
from options_desk.calibration.pipeline.results_store import (
    is_model_done, load_model_results, read_manifest,
)
from options_desk.calibration.pipeline.runner import RunConfig, run_calibration


def _gbm_frame(n=800, seed=0):
    rng = np.random.default_rng(seed)
    r = 0.0003 + 0.012 * rng.standard_normal(n)
    close = pd.Series(100.0 * np.exp(np.cumsum(r)),
                      index=pd.bdate_range("2022-01-03", periods=n))
    df = pd.DataFrame({c: close for c in ["open", "high", "low", "close",
                                          "adj_close"]})
    df["volume"] = 1e6
    return df


@pytest.fixture()
def store(tmp_path):
    frames = {t: _gbm_frame(seed=i) for i, t in enumerate(["AAA", "BBB", "CCC"])}
    # DEAD has only 30 observations: sufficient for exploder (min_obs=1) but not for gbm (min_obs=60)
    frames["DEAD"] = _gbm_frame(n=30, seed=3)

    def fake_fetcher(tickers, start, end):
        return {t: frames[t] for t in tickers if t in frames}

    return PriceStore(tmp_path / "lake", fetcher=fake_fetcher)


UNI = Universe(name="test3", tickers=["AAA", "BBB", "CCC", "DEAD"],
               sectors={"AAA": "Tech", "BBB": "Energy", "CCC": "Tech",
                        "DEAD": "UNKNOWN"})


def test_run_calibration_end_to_end(tmp_path, store):
    cfg = RunConfig(universe="ignored", models=["gbm"], years=3.0,
                    n_jobs=1, out_root=tmp_path / "runs", run_id="r1",
                    end="2025-01-31")
    run_dir = run_calibration(cfg, store, universe=UNI)

    df = load_model_results(run_dir, "gbm")
    assert set(df["ticker"]) == {"AAA", "BBB", "CCC", "DEAD"}
    ok = df[df["ticker"] != "DEAD"]
    assert ok["converged"].all()
    assert (ok["error"] == "").all()
    assert "sigma" in df.columns and "sector" in df.columns
    dead = df[df["ticker"] == "DEAD"].iloc[0]
    assert not dead["converged"] and dead["error"] != ""

    m = read_manifest(run_dir)
    assert m["universe"] == "test3" and m["status"] == "complete"
    assert m["models"]["gbm"]["n_converged"] == 3


def test_run_calibration_resume_skips_done_models(tmp_path, store):
    cfg = RunConfig(universe="ignored", models=["gbm"], years=3.0,
                    n_jobs=1, out_root=tmp_path / "runs", run_id="r2",
                    end="2025-01-31")
    d = run_calibration(cfg, store, universe=UNI)
    assert is_model_done(d, "gbm")
    first = load_model_results(d, "gbm")

    # poison the registry entry: if resume re-runs the model, rows change
    d2 = run_calibration(cfg, store, universe=UNI)
    assert d2 == d
    pd.testing.assert_frame_equal(load_model_results(d, "gbm"), first)


def test_run_calibration_isolates_fit_exceptions(tmp_path, store):
    def exploding_fit(prices, dt):
        raise RuntimeError("boom")

    register_model(ModelSpec(name="exploder", fit=exploding_fit, min_obs=1))
    cfg = RunConfig(universe="ignored", models=["exploder"], years=3.0,
                    n_jobs=1, out_root=tmp_path / "runs", run_id="r3",
                    end="2025-01-31")
    d = run_calibration(cfg, store, universe=UNI)
    df = load_model_results(d, "exploder")
    assert (~df["converged"]).all()
    assert df["error"].str.contains("RuntimeError: boom").all()
