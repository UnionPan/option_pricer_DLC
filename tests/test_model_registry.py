import numpy as np
import pytest

from options_desk.calibration.pipeline.registry import (
    ModelSpec,
    get_model,
    list_models,
    register_model,
)


def _gbm_prices(n=1000, mu=0.08, sigma=0.2, seed=0):
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    r = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)
    return 100.0 * np.exp(np.cumsum(r))


def test_builtin_models_registered():
    assert set(list_models()) >= {"gbm", "garch", "heston_qmle"}


def test_get_model_unknown_raises():
    with pytest.raises(KeyError, match="unknown model"):
        get_model("not_a_model")


def test_register_duplicate_raises():
    with pytest.raises(ValueError, match="already registered"):
        register_model(ModelSpec(name="gbm", fit=lambda p, dt: {}, min_obs=1))


def test_gbm_fit_returns_flat_scalar_dict():
    out = get_model("gbm").fit(_gbm_prices(), 1.0 / 252.0)
    assert isinstance(out, dict)
    assert out["sigma"] == pytest.approx(0.2, abs=0.03)
    assert all(isinstance(v, (int, float, bool, str)) for v in out.values())


def test_heston_qmle_fit_returns_flat_scalar_dict():
    out = get_model("heston_qmle").fit(_gbm_prices(n=1500), 1.0 / 252.0)
    assert {"kappa", "theta", "sigma_v", "rho", "mu", "v0"} <= set(out)
    assert all(isinstance(v, (int, float, bool, str)) for v in out.values())


def test_garch_fit_returns_flat_scalar_dict():
    out = get_model("garch").fit(_gbm_prices(n=1000), 1.0 / 252.0)
    assert {"omega", "alpha", "beta"} <= set(out)
    assert all(isinstance(v, (int, float, bool, str)) for v in out.values())
