"""
Smoke tests for RL environments.
"""

import numpy as np
import pytest

from options_desk.simulations import (
    HestonEnv,
    MertonEnv,
    RoughBergomiEnv,
    CachedHestonEnv,
    build_heston_cache,
)


@pytest.mark.parametrize("env_cls,kwargs", [
    (HestonEnv, {"include_options": False, "task": "trading"}),
    (MertonEnv, {"include_options": False, "task": "trading"}),
    (RoughBergomiEnv, {}),
])
def test_env_step_smoke(env_cls, kwargs):
    env = env_cls(**kwargs)
    obs, info = env.reset(seed=123)
    assert obs is not None
    assert info is not None

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float)
    assert not (terminated and truncated)
    env.close()


@pytest.mark.slow
def test_cached_heston_env(tmp_path):
    cache_path = tmp_path / "heston_cache.zarr"
    build_heston_cache(
        store_path=str(cache_path),
        n_paths=10,
        max_steps=10,
        dt=1 / 252,
        seed=42,
    )

    env = CachedHestonEnv(str(cache_path), task="hedging")
    obs, info = env.reset(seed=1)
    assert "spot_price" in obs
    assert "option_features" in obs

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float)
    env.close()
