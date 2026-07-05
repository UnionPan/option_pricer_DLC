import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import pytest

from options_desk.calibration.physical.batched.common import pad_returns as pad_levels
from options_desk.calibration.physical.batched import ou as bou
from options_desk.calibration.physical.ou_calibrator import OUCalibrator


def _simulate_ou_exact(kappa, theta, sigma, x0, dt, n, seed):
    """
    Simulate OU process with exact discretization.

    X_{t+1} = θ + (X_t - θ)*exp(-κΔt) + σ*sqrt((1 - exp(-2κΔt))/(2κ))*z
    """
    rng = np.random.default_rng(seed)
    exp_term = np.exp(-kappa * dt)
    noise_std = sigma * np.sqrt((1 - np.exp(-2 * kappa * dt)) / (2 * kappa))

    path = np.zeros(n + 1)
    path[0] = x0
    for i in range(n):
        path[i + 1] = theta + (path[i] - theta) * exp_term + noise_std * rng.standard_normal()
    return path


def _paths(n_assets=6, n=1200, seed=0):
    """Generate OU paths with varying parameters."""
    out = []
    for i in range(n_assets):
        kappa = 0.5 + 0.3 * i
        theta = 0.05 + 0.01 * i
        sigma = 0.10 + 0.02 * i
        x0 = theta + sigma * 0.5
        path = _simulate_ou_exact(kappa, theta, sigma, x0, 1/252, n, seed + i)
        out.append(path)
    return out


def test_parity_with_scipy_ou():
    """Test parity with OUCalibrator.fit(method='exact_mle') on simulated OU paths."""
    paths_list = _paths()
    levels, mask = pad_levels(paths_list)
    out = bou.fit_batch(levels, mask, 1 / 252)

    for i, path in enumerate(paths_list):
        ref = OUCalibrator(method='exact_mle').fit(path, dt=1 / 252)
        assert out["kappa"][i] == pytest.approx(ref.kappa, rel=1e-3, abs=1e-6)
        assert out["theta"][i] == pytest.approx(ref.theta, rel=1e-3, abs=1e-6)
        assert out["sigma"][i] == pytest.approx(ref.sigma, rel=1e-3, abs=1e-6)
        assert out["log_likelihood"][i] == pytest.approx(ref.log_likelihood, rel=1e-3)
        assert out["half_life"][i] == pytest.approx(ref.half_life, rel=1e-3)
        assert out["n_observations"][i] == ref.n_observations


def test_padding_does_not_leak():
    """Ensure padding does not affect calibration results."""
    # Generate two paths of different lengths
    path1 = _simulate_ou_exact(0.8, 0.05, 0.12, 0.07, 1/252, 500, seed=100)
    path2 = _simulate_ou_exact(1.2, 0.08, 0.15, 0.10, 1/252, 900, seed=101)

    # Calibrate batch with both paths
    levels, mask = pad_levels([path1, path2])
    out = bou.fit_batch(levels, mask, 1 / 252)

    # Calibrate path1 alone
    levels2, mask2 = pad_levels([path1])
    out2 = bou.fit_batch(levels2, mask2, 1 / 252)

    # Results for path1 should be identical (allowing for numerical precision)
    assert out["kappa"][0] == pytest.approx(out2["kappa"][0], rel=1e-4)
    assert out["theta"][0] == pytest.approx(out2["theta"][0], rel=1e-4)
    assert out["sigma"][0] == pytest.approx(out2["sigma"][0], rel=1e-4)
    assert out["log_likelihood"][0] == pytest.approx(out2["log_likelihood"][0], rel=1e-4)


def test_recovery():
    """Test parameter recovery on long simulated paths."""
    # Use longer paths for better recovery (matching GBM test length)
    n_assets = 4
    n = 100_000
    paths = []
    true_sigmas = []

    for i in range(n_assets):
        kappa = 0.8 + 0.4 * i
        theta = 0.05
        sigma = 0.10 + 0.02 * i  # Varying sigma
        x0 = theta
        path = _simulate_ou_exact(kappa, theta, sigma, x0, 1/252, n, seed=200 + i)
        paths.append(path)
        true_sigmas.append(sigma)

    levels, mask = pad_levels(paths)
    out = bou.fit_batch(levels, mask, 1 / 252)

    for i in range(n_assets):
        # Check sigma recovery within 2% (matching GBM test tolerance)
        assert out["sigma"][i] == pytest.approx(true_sigmas[i], rel=0.02)
