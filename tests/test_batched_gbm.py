import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import pytest

from options_desk.calibration.physical.batched.common import pad_returns
from options_desk.calibration.physical.batched import gbm as bgbm
from options_desk.calibration.physical.gbm_calibrator import GBMCalibrator


def _paths(n_assets=8, n=1200, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_assets):
        mu, sig = 0.02 + 0.02 * i, 0.10 + 0.03 * i
        dt = 1 / 252
        r = (mu - 0.5 * sig**2) * dt + sig * np.sqrt(dt) * rng.standard_normal(n)
        out.append(100 * np.exp(np.cumsum(r)))
    return out


def test_parity_with_scipy_gbm():
    prices_list = _paths()
    rets = [np.diff(np.log(p)) for p in prices_list]
    R, M = pad_returns(rets)
    out = bgbm.fit_batch(R, M, 1 / 252)
    for i, p in enumerate(prices_list):
        ref = GBMCalibrator().fit(p, dt=1 / 252)
        assert out["mu"][i] == pytest.approx(ref.mu, rel=1e-4, abs=1e-6)
        assert out["sigma"][i] == pytest.approx(ref.sigma, rel=1e-4)
        assert out["log_likelihood"][i] == pytest.approx(ref.log_likelihood, rel=1e-3)


def test_padding_does_not_leak():
    rets = [np.full(500, 0.001), np.full(900, -0.0005)]
    R, M = pad_returns(rets)
    out = bgbm.fit_batch(R, M, 1 / 252)
    R2, M2 = pad_returns([rets[0]])
    out2 = bgbm.fit_batch(R2, M2, 1 / 252)
    assert out["mu"][0] == pytest.approx(out2["mu"][0], rel=1e-6)


def test_recovery():
    prices_list = _paths(n_assets=4, n=100_000, seed=1)
    rets = [np.diff(np.log(p)) for p in prices_list]
    R, M = pad_returns(rets)
    out = bgbm.fit_batch(R, M, 1 / 252)
    for i in range(4):
        assert out["sigma"][i] == pytest.approx(0.10 + 0.03 * i, rel=0.02)
