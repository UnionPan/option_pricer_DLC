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


def test_heston_env_exposes_floating_grid_mask():
    env = HestonEnv(
        include_options=True,
        task="trading",
        max_steps=5,
        option_grid={3: [1.0], 7: [1.0]},
    )
    obs, info = env.reset(seed=123)

    assert "action_mask" in obs
    assert obs["action_mask"].shape == (env.n_instruments,)
    assert obs["action_mask"].dtype == np.bool_
    assert np.array_equal(
        obs["action_mask"],
        np.array([True, True, True, False, False], dtype=bool),
    )
    assert np.allclose(obs["option_features"][-4:], 0.0)
    assert np.array_equal(info["action_mask"], obs["action_mask"])

    env.close()


def test_heston_env_forces_unavailable_option_positions_flat():
    env = HestonEnv(
        include_options=True,
        task="trading",
        max_steps=6,
        option_grid={6: [1.0]},
    )
    obs, info = env.reset(seed=123)

    assert np.array_equal(
        obs["action_mask"],
        np.array([True, True, True], dtype=bool),
    )

    action = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert not terminated
    assert not truncated
    assert np.array_equal(
        obs["action_mask"],
        np.array([True, False, False], dtype=bool),
    )
    assert np.allclose(info["positions"][1:], 0.0)
    assert np.allclose(obs["option_features"], 0.0)

    env.close()


def test_merton_env_exposes_floating_grid_mask():
    env = MertonEnv(
        include_options=True,
        task="trading",
        max_steps=5,
        option_maturities=[3, 7],
        option_moneyness=[1.0],
    )
    obs, info = env.reset(seed=123)

    assert "action_mask" in obs
    assert obs["action_mask"].shape == (env.n_instruments,)
    assert obs["action_mask"].dtype == np.bool_
    assert np.array_equal(
        obs["action_mask"],
        np.array([True, True, True, False, False], dtype=bool),
    )
    assert np.allclose(obs["option_features"][-4:], 0.0)
    assert np.array_equal(info["action_mask"], obs["action_mask"])

    env.close()


def test_merton_env_forces_unavailable_option_positions_flat():
    env = MertonEnv(
        include_options=True,
        task="trading",
        max_steps=6,
        option_maturities=[6],
        option_moneyness=[1.0],
    )
    obs, info = env.reset(seed=123)

    assert np.array_equal(
        obs["action_mask"],
        np.array([True, True, True], dtype=bool),
    )

    action = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert not terminated
    assert not truncated
    assert np.array_equal(
        obs["action_mask"],
        np.array([True, False, False], dtype=bool),
    )
    assert np.allclose(info["positions"][1:], 0.0)
    assert np.allclose(obs["option_features"], 0.0)

    env.close()
