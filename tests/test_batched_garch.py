import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import pytest

from options_desk.calibration.physical.batched.common import pad_returns
from options_desk.calibration.physical.batched import garch as bgarch
from options_desk.calibration.physical.garch_calibrator import GARCHCalibrator


def _simulate_garch(omega, alpha, beta, n, mu=0.0, dt=1/252, seed=0):
    """
    Simulate GARCH(1,1) returns.

    Note: GARCH parameters omega, alpha, beta are in per-period units (not annualized).
    The model is:
        r_t = mu + eps_t
        eps_t = sigma_t * z_t
        sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2

    Args:
        omega: constant term (per-period variance units)
        alpha: ARCH coefficient
        beta: GARCH coefficient
        n: number of returns
        mu: drift per period (not annualized)
        dt: time increment (not used in simulation, only for reference)
        seed: random seed

    Returns:
        Array of log-returns
    """
    rng = np.random.default_rng(seed)

    # Initialize
    returns = np.zeros(n)
    sigma2 = np.zeros(n)

    # Start from unconditional variance
    sigma2[0] = omega / (1 - alpha - beta)

    for t in range(n):
        z = rng.standard_normal()
        # NOTE: No dt scaling! Returns are per-period.
        returns[t] = mu + np.sqrt(sigma2[t]) * z

        if t < n - 1:
            # Recursion uses demeaned squared returns
            sigma2[t + 1] = omega + alpha * (returns[t] - mu) ** 2 + beta * sigma2[t]

    return returns


def test_garch_recovery():
    """
    Test (a): simulate 6 GARCH(1,1) paths and verify parameter recovery.

    Recovery criteria:
    - alpha within ±0.04
    - beta within ±0.05
    - log-likelihood >= scipy GARCHCalibrator's logL - 0.5 per asset
    """
    # True parameters
    omega_true = 2e-6
    alpha_true = 0.08
    beta_true = 0.90
    n = 4000
    dt = 1 / 252

    # Simulate 6 assets with different seeds
    returns_list = []
    for i in range(6):
        returns = _simulate_garch(omega_true, alpha_true, beta_true, n, mu=0.0, dt=dt, seed=i)
        returns_list.append(returns)

    # Pad and fit with batched calibrator
    R, M = pad_returns(returns_list)
    result = bgarch.fit_batch(R, M, dt)

    # Compare with scipy reference for each asset
    for i, rets in enumerate(returns_list):
        # Fit with scipy reference
        prices = 100 * np.exp(np.insert(np.cumsum(rets), 0, 0))
        scipy_result = GARCHCalibrator().fit(prices, dt=dt)

        # Check parameter recovery
        assert result["alpha"][i] == pytest.approx(alpha_true, abs=0.04), \
            f"Asset {i}: alpha={result['alpha'][i]:.4f}, expected {alpha_true:.4f} ± 0.04"

        assert result["beta"][i] == pytest.approx(beta_true, abs=0.05), \
            f"Asset {i}: beta={result['beta'][i]:.4f}, expected {beta_true:.4f} ± 0.05"

        # Check likelihood parity (this is the GATE)
        assert result["log_likelihood"][i] >= scipy_result.log_likelihood - 0.5, \
            f"Asset {i}: JAX logL={result['log_likelihood'][i]:.2f}, " \
            f"scipy logL={scipy_result.log_likelihood:.2f}, diff={result['log_likelihood'][i] - scipy_result.log_likelihood:.2f}"

        # Verify convergence
        assert result["converged"][i], f"Asset {i} did not converge"


def test_fixed_params_nll_isolation():
    """
    Test (b'): FIXED-PARAMS NLL isolation check (no optimizer).

    Evaluates the masked NLL function directly -- the same code path the
    optimizer differentiates -- for the same asset padded-in-batch vs alone,
    at the same fixed (omega, alpha, beta, mu) and the same var_init.

    This cleanly proves the mask handling is exact: padded steps contribute
    zero NLL and carry sigma^2 forward unchanged. Any difference beyond
    float32 reduction jitter (rel=1e-5) would indicate a mask leak, and
    legitimizes the wider endpoint tolerances in test_padding_does_not_leak
    as optimizer-trajectory effects rather than mask bugs.
    """
    import jax.numpy as jnp
    from options_desk.calibration.physical.batched.garch import (
        _garch_log_likelihood,
        _unconstrain_params,
    )

    rets = _simulate_garch(2e-6, 0.08, 0.90, 500, seed=42)

    # Alone: T=500, mask all ones
    R_alone, M_alone = pad_returns([rets])
    # Padded: same asset right-padded to T=900 alongside a longer asset
    rets_long = _simulate_garch(2e-6, 0.10, 0.85, 900, seed=43)
    R_batch, M_batch = pad_returns([rets, rets_long])

    # Fixed parameters (arbitrary valid GARCH point) and shared var_init
    omega_fixed, alpha_fixed, beta_fixed = 2e-6, 0.08, 0.90
    mu_fixed = float(np.mean(rets))
    var_init = np.float32(np.var(rets, ddof=1))

    a, b, c = _unconstrain_params(omega_fixed, alpha_fixed, beta_fixed)
    params = jnp.array([a, b, c, mu_fixed])

    nll_alone = float(_garch_log_likelihood(
        params, jnp.asarray(R_alone[0]), jnp.asarray(M_alone[0]), var_init
    ))
    nll_padded = float(_garch_log_likelihood(
        params, jnp.asarray(R_batch[0]), jnp.asarray(M_batch[0]), var_init
    ))

    rel_diff = abs(nll_padded - nll_alone) / max(abs(nll_alone), 1e-30)
    assert rel_diff <= 1e-5, (
        f"Mask leak detected: NLL alone={nll_alone:.8f}, "
        f"NLL padded={nll_padded:.8f}, rel diff={rel_diff:.2e} > 1e-5"
    )


def test_padding_does_not_leak():
    """
    Test (b): verify that padding does not affect parameter estimates.

    Endpoint tolerances are wide (see below) because the multi-start Adam
    optimizer can follow different trajectories under vmap; the exactness of
    the mask handling itself is proven by test_fixed_params_nll_isolation.
    """
    # Create two assets with different lengths
    rets1 = _simulate_garch(2e-6, 0.08, 0.90, 500, seed=42)
    rets2 = _simulate_garch(2e-6, 0.10, 0.85, 900, seed=43)

    # Fit together (with padding)
    R, M = pad_returns([rets1, rets2])
    result_together = bgarch.fit_batch(R, M, 1/252)

    # Fit first asset alone
    R_alone, M_alone = pad_returns([rets1])
    result_alone = bgarch.fit_batch(R_alone, M_alone, 1/252)

    # Results should match within tolerance
    # Note: vmapping can cause subtle differences in optimization trajectory
    # due to JIT/vmap interactions with optax state, so we allow wider tolerance
    assert result_together["omega"][0] == pytest.approx(result_alone["omega"][0], rel=0.1)
    assert result_together["alpha"][0] == pytest.approx(result_alone["alpha"][0], rel=0.1)
    assert result_together["beta"][0] == pytest.approx(result_alone["beta"][0], rel=0.1)
    # Likelihood should be close even if parameters differ slightly
    assert result_together["log_likelihood"][0] == pytest.approx(result_alone["log_likelihood"][0], rel=1e-2)


def test_speed_sanity():
    """
    Test (c): verify that 6-asset fit completes in <60s on CPU.
    """
    import time

    # Simulate 6 assets
    returns_list = []
    for i in range(6):
        returns = _simulate_garch(2e-6, 0.08, 0.90, 4000, seed=i)
        returns_list.append(returns)

    R, M = pad_returns(returns_list)

    # Time the fit
    start = time.time()
    result = bgarch.fit_batch(R, M, 1/252)
    elapsed = time.time() - start

    assert elapsed < 60.0, f"Fit took {elapsed:.1f}s, expected <60s"

    # Verify all converged
    assert np.all(result["converged"]), "Not all assets converged"
