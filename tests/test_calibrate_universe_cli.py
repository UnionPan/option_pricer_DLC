import numpy as np
import pandas as pd

from options_desk.calibration.cli import calibrate as cli

from options_desk.calibration.pipeline.results_store import (
    load_model_results,
)


def test_parser_defaults():
    args = cli.build_parser().parse_args([])
    assert args.universe is None
    assert args.models == ["heston_qmle"]
    assert args.years == 5.0
    assert args.run_id is None


def test_main_end_to_end_with_fake_fetcher(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)

    def fake_fetcher(tickers, start, end):
        out = {}
        for t in tickers:
            r = 0.0003 + 0.012 * rng.standard_normal(700)
            close = pd.Series(100.0 * np.exp(np.cumsum(r)),
                              index=pd.bdate_range("2022-06-01", periods=700))
            df = pd.DataFrame({c: close for c in
                               ["open", "high", "low", "close", "adj_close"]})
            df["volume"] = 1e6
            out[t] = df
        return out

    monkeypatch.setattr(cli, "_default_fetcher", lambda: fake_fetcher)
    uni_csv = tmp_path / "mini.csv"
    uni_csv.write_text("ticker,sector\nAAA,Tech\nBBB,Energy\n")

    rc = cli.main([
        "--universe", str(uni_csv),
        "--models", "gbm",
        "--years", "2",
        "--jobs", "1",
        "--price-lake", str(tmp_path / "lake"),
        "--out-root", str(tmp_path / "runs"),
        "--run-id", "clitest",
        "--end", "2025-01-31",
    ])
    assert rc == 0
    df = load_model_results(tmp_path / "runs" / "clitest", "gbm")
    assert set(df["ticker"]) == {"AAA", "BBB"}
    assert df["converged"].all()


def test_list_models_flag(capsys):
    rc = cli.main(["--list-models"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "gbm" in out and "heston_qmle" in out
