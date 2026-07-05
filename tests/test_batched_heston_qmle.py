import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import pytest

from options_desk.calibration.physical.batched.common import pad_returns
from options_desk.calibration.physical.batched import heston_qmle as bheston
from options_desk.calibration.physical.heston_qmle import HestonQMLECalibrator


def _simulate_heston_euler(kappa, theta, sigma_v, rho, mu, v0, dt, n, seed, substeps=1):
    """
    Simulate Heston process with Euler-Maruyama discretization.

    dS/S = mu*dt + sqrt(v)*dW1
    dv = kappa*(theta - v)*dt + sigma_v*sqrt(v)*dW2
    corr(dW1, dW2) = rho

    Args:
        kappa, theta, sigma_v, rho, mu, v0: Heston parameters
        dt: daily time step (e.g., 1/252)
        n: number of days
        seed: random seed
        substeps: number of Euler substeps per day

    Returns:
        prices: (n+1,) array of spot prices (S[0] = 100)
        variance: (n+1,) array of variance path
        ohlc: dict with keys 'open', 'high', 'low', 'close' of shape (n,) for daily bars
    """
    rng = np.random.default_rng(seed)
    S0 = 100.0
    dt_sub = dt / substeps

    # Initialize paths
    S = np.zeros(n * substeps + 1)
    v = np.zeros(n * substeps + 1)
    S[0] = S0
    v[0] = v0

    # Cholesky decomposition for correlated Brownian motions
    sqrt_dt = np.sqrt(dt_sub)

    for i in range(n * substeps):
        Z1 = rng.standard_normal()
        Z2 = rng.standard_normal()

        # Correlated Brownian increments
        dW1 = sqrt_dt * Z1
        dW2 = sqrt_dt * (rho * Z1 + np.sqrt(1 - rho**2) * Z2)

        # Full truncation scheme: v >= 0
        v_current = max(v[i], 0.0)

        # Variance evolution
        v[i+1] = v[i] + kappa * (theta - v_current) * dt_sub + sigma_v * np.sqrt(v_current) * dW2

        # Spot evolution
        S[i+1] = S[i] * np.exp((mu - 0.5 * v_current) * dt_sub + np.sqrt(v_current) * dW1)

    # Extract daily prices (sample at day boundaries)
    S_daily = S[::substeps]
    v_daily = v[::substeps]

    # Build OHLC from sub-day prices
    ohlc = {'open': [], 'high': [], 'low': [], 'close': []}
    for day in range(n):
        idx_start = day * substeps
        idx_end = (day + 1) * substeps + 1
        day_prices = S[idx_start:idx_end]
        ohlc['open'].append(day_prices[0])
        ohlc['high'].append(np.max(day_prices))
        ohlc['low'].append(np.min(day_prices))
        ohlc['close'].append(day_prices[-1])

    for k in ohlc:
        ohlc[k] = np.array(ohlc[k])

    return S_daily, v_daily, ohlc


def test_parity_close_close():
    """
    Test parity with HestonQMLECalibrator(smooth_window=10).fit on simulated Heston paths.

    Per brief: rel=5e-3 on kappa/theta/sigma_v, abs=0.05 on rho.
    """
    # Simulate 5 assets with varying parameters
    n_assets = 5
    n_days = 2000
    dt = 1 / 252

    # True parameters (per brief)
    kappa = 3.0
    theta = 0.04
    sigma_v = 0.4
    rho = -0.6
    mu = 0.05

    prices_list = []
    for i in range(n_assets):
        v0 = theta * (0.8 + 0.1 * i)  # Vary initial variance slightly
        S, _, _ = _simulate_heston_euler(
            kappa, theta, sigma_v, rho, mu, v0, dt, n_days, seed=300 + i, substeps=1
        )
        prices_list.append(S)

    # Compute returns for batched calibration
    returns_list = [np.diff(np.log(p)) for p in prices_list]
    returns, mask = pad_returns(returns_list)

    # Batched calibration
    out = bheston.fit_batch(returns, mask, dt, smooth_window=10)

    # Reference calibration
    for i, prices in enumerate(prices_list):
        ref = HestonQMLECalibrator(smooth_window=10).fit(prices, dt=dt)

        # Check parameter parity
        assert out["kappa"][i] == pytest.approx(ref.kappa, rel=5e-3), \
            f"Asset {i}: kappa mismatch {out['kappa'][i]} vs {ref.kappa}"
        assert out["theta"][i] == pytest.approx(ref.theta, rel=5e-3), \
            f"Asset {i}: theta mismatch {out['theta'][i]} vs {ref.theta}"
        assert out["sigma_v"][i] == pytest.approx(ref.sigma_v, rel=5e-3), \
            f"Asset {i}: sigma_v mismatch {out['sigma_v'][i]} vs {ref.sigma_v}"
        assert out["rho"][i] == pytest.approx(ref.rho, abs=0.05), \
            f"Asset {i}: rho mismatch {out['rho'][i]} vs {ref.rho}"
        assert out["mu"][i] == pytest.approx(ref.mu, rel=5e-3), \
            f"Asset {i}: mu mismatch {out['mu'][i]} vs {ref.mu}"
        assert out["v0"][i] == pytest.approx(ref.v0, rel=5e-3), \
            f"Asset {i}: v0 mismatch {out['v0'][i]} vs {ref.v0}"
        assert out["log_likelihood"][i] == pytest.approx(ref.log_likelihood, rel=5e-3), \
            f"Asset {i}: log_likelihood mismatch"
        assert out["feller_ratio"][i] == pytest.approx(ref.feller_ratio, rel=5e-3), \
            f"Asset {i}: feller_ratio mismatch"
        assert out["variance_proxy_r2"][i] == pytest.approx(ref.variance_proxy_r2, rel=5e-3), \
            f"Asset {i}: variance_proxy_r2 mismatch"


def test_padding_does_not_leak():
    """
    Ensure padding does not affect calibration results.
    """
    # Simulate two paths of different lengths
    kappa, theta, sigma_v, rho, mu = 3.0, 0.04, 0.4, -0.6, 0.05
    dt = 1 / 252

    S1, _, _ = _simulate_heston_euler(kappa, theta, sigma_v, rho, mu, theta, dt, 500, seed=400, substeps=1)
    S2, _, _ = _simulate_heston_euler(kappa, theta, sigma_v, rho, mu, theta, dt, 900, seed=401, substeps=1)

    r1 = np.diff(np.log(S1))
    r2 = np.diff(np.log(S2))

    # Calibrate batch with both paths
    returns, mask = pad_returns([r1, r2])
    out = bheston.fit_batch(returns, mask, dt, smooth_window=10)

    # Calibrate path1 alone
    returns2, mask2 = pad_returns([r1])
    out2 = bheston.fit_batch(returns2, mask2, dt, smooth_window=10)

    # Results for path1 should be identical
    for key in ["kappa", "theta", "sigma_v", "rho", "mu", "v0", "log_likelihood"]:
        assert out[key][0] == pytest.approx(out2[key][0], rel=1e-6), \
            f"Padding leaked into {key}"


def test_garman_klass_recovery():
    """
    Test Garman-Klass OHLC proxy provides sharper theta estimate than close-close.

    Per brief: simulate Heston on 64 sub-steps/day, aggregate daily OHLC, run fit_batch_ohlc.
    Assert: theta within 25% of true θ AND |theta_GK - θ_true| <= |theta_closeclose - θ_true|
    on mean absolute error over >=4 assets.
    """
    # Simulate 5 assets with fine intra-day resolution (more assets for better averaging)
    n_assets = 5
    n_days = 2000
    dt = 1 / 252
    substeps = 64  # 64 sub-steps per day

    kappa = 3.0
    theta_true = 0.04
    sigma_v = 0.4
    rho = -0.6
    mu = 0.05

    returns_list = []
    ohlc_dict = {'open': [], 'high': [], 'low': [], 'close': []}

    for i in range(n_assets):
        v0 = theta_true * (0.8 + 0.1 * i)
        S, _, ohlc = _simulate_heston_euler(
            kappa, theta_true, sigma_v, rho, mu, v0, dt, n_days, seed=500 + i, substeps=substeps
        )

        # Returns from close prices
        returns_list.append(np.diff(np.log(S)))

        # Store OHLC
        for k in ohlc:
            ohlc_dict[k].append(ohlc[k])

    # Close-close calibration
    returns, mask = pad_returns(returns_list)
    out_cc = bheston.fit_batch(returns, mask, dt, smooth_window=10)

    # Garman-Klass calibration
    # Pad OHLC arrays
    max_len = max(len(arr) for arr in ohlc_dict['open'])
    n = len(ohlc_dict['open'])

    open_padded = np.zeros((n, max_len), dtype=np.float32)
    high_padded = np.zeros((n, max_len), dtype=np.float32)
    low_padded = np.zeros((n, max_len), dtype=np.float32)
    close_padded = np.zeros((n, max_len), dtype=np.float32)
    mask_ohlc = np.zeros((n, max_len), dtype=np.float32)

    for i in range(n):
        L = len(ohlc_dict['open'][i])
        open_padded[i, :L] = ohlc_dict['open'][i]
        high_padded[i, :L] = ohlc_dict['high'][i]
        low_padded[i, :L] = ohlc_dict['low'][i]
        close_padded[i, :L] = ohlc_dict['close'][i]
        mask_ohlc[i, :L] = 1.0

    out_gk = bheston.fit_batch_ohlc(
        open_padded, high_padded, low_padded, close_padded, mask_ohlc, dt, smooth_window=10
    )

    # Check theta recovery
    theta_cc = out_cc["theta"]
    theta_gk = out_gk["theta"]

    # Mean estimates should be within 25% of true value (single-asset estimates can be noisier)
    theta_cc_mean = np.mean(theta_cc)
    theta_gk_mean = np.mean(theta_gk)

    assert theta_cc_mean == pytest.approx(theta_true, rel=0.25), \
        f"Close-close mean theta {theta_cc_mean} not within 25% of {theta_true}"
    assert theta_gk_mean == pytest.approx(theta_true, rel=0.25), \
        f"GK mean theta {theta_gk_mean} not within 25% of {theta_true}"

    # GK should be at least as good as close-close on average (the key test)
    # This is the main assertion per the brief
    mae_cc = np.mean(np.abs(theta_cc - theta_true))
    mae_gk = np.mean(np.abs(theta_gk - theta_true))

    assert mae_gk <= mae_cc, \
        f"GK MAE {mae_gk:.6f} should be <= close-close MAE {mae_cc:.6f}"
