import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import pytest

from options_desk.calibration.physical.batched.common import pad_returns
from options_desk.calibration.physical.batched import rbergomi as brbergomi
from options_desk.calibration.physical.rough_bergomi_calibrator import RoughBergomiCalibrator


def _simulate_rbergomi_path(H, eta, xi0, mu, rho, dt, n, seed):
    """
    Simulate rough Bergomi-style path where log-volatility follows fBm.

    The forward variance follows:
        log(v_t) = log(xi0) + eta * W^H_t
    where W^H is fractional Brownian motion with Hurst parameter H.

    Returns are generated as:
        r_t = sqrt(v_t * dt) * z_t
    where z_t ~ N(0,1).

    Args:
        H: Hurst parameter (0 < H < 0.5)
        eta: volatility of volatility
        xi0: initial variance level
        mu: drift (not used in log-vol fBm model, but kept for interface compatibility)
        rho: correlation (not used in this simple generator, but kept for interface)
        dt: time step
        n: number of time steps
        seed: random seed

    Returns:
        returns: (n,) array of log returns
    """
    rng = np.random.default_rng(seed)

    # Generate fBm via Cholesky decomposition
    # Covariance: C(s,t) = 0.5 * (s^(2H) + t^(2H) - |t-s|^(2H))
    times = np.arange(1, n + 1) * dt
    T_len = len(times)

    # Build covariance matrix
    cov_matrix = np.zeros((T_len, T_len))
    for i in range(T_len):
        for j in range(T_len):
            s = times[i]
            t = times[j]
            cov_matrix[i, j] = 0.5 * (s**(2*H) + t**(2*H) - abs(t - s)**(2*H))

    # Cholesky decomposition
    L = np.linalg.cholesky(cov_matrix + 1e-10 * np.eye(T_len))

    # Generate fBm path
    Z = rng.standard_normal(T_len)
    W_H = L @ Z

    # Log-variance path
    log_v = np.log(xi0) + eta * W_H

    # Variance path
    v = np.exp(log_v)

    # Generate returns: r_t = sqrt(v_t * dt) * z_t
    Z_returns = rng.standard_normal(T_len)
    returns = np.sqrt(v * dt) * Z_returns

    return returns


def test_parity_vs_scipy():
    """
    Test parity with RoughBergomiCalibrator on 5 simulated fBm log-vol paths.

    Per brief: deterministic pipeline, should match tightly (rel=1e-3).
    """
    n_assets = 5
    n_days = 1500
    dt = 1 / 252
    window = 20
    max_lag = 10

    # True parameters
    H = 0.3
    eta = 1.2
    xi0 = 0.04
    mu = 0.05
    rho = -0.5

    prices_list = []
    for i in range(n_assets):
        # Simulate returns
        returns = _simulate_rbergomi_path(
            H, eta, xi0, mu, rho, dt, n_days, seed=500 + i
        )
        # Convert to prices (cumsum of returns, starting from 100)
        log_prices = np.cumsum(np.concatenate([[0], returns]))
        prices = 100.0 * np.exp(log_prices)
        prices_list.append(prices)

    # Compute returns for batched calibration
    returns_list = [np.diff(np.log(p)) for p in prices_list]
    returns, mask = pad_returns(returns_list)

    # Batched calibration
    out = brbergomi.fit_batch(returns, mask, dt, window=window, max_lag=max_lag)

    # Reference calibration
    for i, prices in enumerate(prices_list):
        ref = RoughBergomiCalibrator(window=window, max_lag=max_lag).fit(prices, dt=dt)

        # Check parameter parity (deterministic pipeline should match tightly)
        assert out["H"][i] == pytest.approx(ref.H, rel=1e-3), \
            f"Asset {i}: H mismatch {out['H'][i]} vs {ref.H}"
        assert out["eta"][i] == pytest.approx(ref.eta, rel=1e-3), \
            f"Asset {i}: eta mismatch {out['eta'][i]} vs {ref.eta}"
        assert out["xi0"][i] == pytest.approx(ref.xi0, rel=1e-3), \
            f"Asset {i}: xi0 mismatch {out['xi0'][i]} vs {ref.xi0}"
        assert out["converged"][i] == True, \
            f"Asset {i}: converged should be True"
        assert out["n_observations"][i] == ref.n_observations, \
            f"Asset {i}: n_observations mismatch"


def test_variogram_cannot_resolve_H_at_default_settings():
    """
    Document the reference estimator's H-resolution limit (faithful-port pin).

    At default settings the estimator's rolling window (20 days) exceeds the
    variogram lags (<= 10 days), so overlap-smoothing of the realized-variance
    proxy destroys the roughness signal: for genuinely rough paths (true
    H=0.1) BOTH the scipy reference and the batched port return H ~ 0.49,
    the clip ceiling. This test pins that behavior so the batched port stays
    a faithful replica of the normative scipy calibrator, and makes the
    limitation explicit instead of pretending recovery works. Universe-scale
    rough-vol estimation is deferred to the Phase 4 NPE estimator per the
    spec.
    """
    n_days = 1500
    dt = 1 / 252
    window = 20
    max_lag = 10

    true_H = 0.1
    eta = 1.0
    xi0 = 0.04

    for i in range(3):
        returns = _simulate_rbergomi_path(
            true_H, eta, xi0, 0.0, 0.0, dt, n_days, seed=610 + i
        )
        # Convert to prices
        log_prices = np.cumsum(np.concatenate([[0], returns]))
        prices = 100.0 * np.exp(log_prices)

        # Scipy reference: hits the 0.49 clip ceiling on rough paths
        ref = RoughBergomiCalibrator(window=window, max_lag=max_lag).fit(prices, dt=dt)
        assert ref.H == pytest.approx(0.49, abs=0.02), \
            f"Path {i}: scipy reference expected at clip ceiling, got H={ref.H}"

        # Batched port: must exhibit the same limitation (faithful port)
        returns_arr = np.diff(np.log(prices))
        returns_batch = returns_arr.reshape(1, -1)
        mask_batch = np.ones_like(returns_batch)

        out = brbergomi.fit_batch(returns_batch, mask_batch, dt, window=window, max_lag=max_lag)
        assert out["H"][0] == pytest.approx(0.49, abs=0.02), \
            f"Path {i}: batched port expected at clip ceiling, got H={out['H'][0]}"


def test_padding_does_not_leak():
    """
    Ensure padding does not affect calibration results (rel 1e-4).
    """
    dt = 1 / 252
    window = 20
    max_lag = 10

    H, eta, xi0 = 0.25, 1.0, 0.04

    # Simulate two paths of different lengths
    returns1 = _simulate_rbergomi_path(H, eta, xi0, 0.0, 0.0, dt, 500, seed=700)
    returns2 = _simulate_rbergomi_path(H, eta, xi0, 0.0, 0.0, dt, 900, seed=701)

    # Calibrate individually
    returns1_batch = returns1.reshape(1, -1)
    mask1 = np.ones_like(returns1_batch)
    out1_solo = brbergomi.fit_batch(returns1_batch, mask1, dt, window=window, max_lag=max_lag)

    returns2_batch = returns2.reshape(1, -1)
    mask2 = np.ones_like(returns2_batch)
    out2_solo = brbergomi.fit_batch(returns2_batch, mask2, dt, window=window, max_lag=max_lag)

    # Calibrate together (padded)
    returns_padded, mask_padded = pad_returns([returns1, returns2])
    out_combined = brbergomi.fit_batch(returns_padded, mask_padded, dt, window=window, max_lag=max_lag)

    # Check that results match (padding should not affect)
    for key in ["H", "eta", "xi0"]:
        assert out_combined[key][0] == pytest.approx(out1_solo[key][0], rel=1e-4), \
            f"Padding leaked for asset 0, key {key}"
        assert out_combined[key][1] == pytest.approx(out2_solo[key][0], rel=1e-4), \
            f"Padding leaked for asset 1, key {key}"
