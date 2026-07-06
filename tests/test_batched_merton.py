import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import pytest

from options_desk.calibration.physical.batched.common import pad_returns
from options_desk.calibration.physical.batched import merton as bmerton
from options_desk.calibration.physical.merton_calibrator import MertonJumpCalibrator


def _simulate_merton(mu, sigma, lambda_, mu_j, sigma_j, n, dt=1/252, seed=0):
    """
    Simulate Merton jump-diffusion log-returns.

    Model:
        r_t = (mu - 0.5*sigma^2 - lambda*kappa)*dt + sigma*sqrt(dt)*z_t + sum_{i=1}^{N_t} J_i

    where:
        N_t ~ Poisson(lambda * dt)
        J_i ~ Normal(mu_j, sigma_j^2)
        kappa = E[exp(J) - 1] = exp(mu_j + 0.5*sigma_j^2) - 1

    Args:
        mu: drift (annualized)
        sigma: diffusion volatility (annualized)
        lambda_: jump intensity (per year)
        mu_j: jump mean
        sigma_j: jump std dev
        n: number of returns
        dt: time increment in years
        seed: random seed

    Returns:
        Array of log-returns
    """
    rng = np.random.default_rng(seed)

    # Expected jump size compensation
    kappa = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0

    returns = np.zeros(n)
    for t in range(n):
        # Diffusion component
        z = rng.standard_normal()
        drift = (mu - 0.5 * sigma**2 - lambda_ * kappa) * dt
        diffusion = sigma * np.sqrt(dt) * z

        # Jump component
        n_jumps = rng.poisson(lambda_ * dt)
        jump_sum = 0.0
        if n_jumps > 0:
            jumps = rng.normal(mu_j, sigma_j, size=n_jumps)
            jump_sum = np.sum(jumps)

        returns[t] = drift + diffusion + jump_sum

    return returns


def test_merton_recovery():
    """
    Test (a): simulate 4 Merton jump-diffusion paths and verify parameter recovery.

    Recovery criteria:
    - sigma within 15%
    - lambda within factor 2
    - log-likelihood >= scipy MertonJumpCalibrator's logL - 0.5 per asset
    """
    # True parameters
    mu_true = 0.05
    sigma_true = 0.15
    lambda_true = 10.0  # per year
    mu_j_true = -0.02
    sigma_j_true = 0.04
    n = 6000
    dt = 1 / 252

    # Simulate 4 assets with different seeds
    returns_list = []
    for i in range(4):
        returns = _simulate_merton(
            mu_true, sigma_true, lambda_true, mu_j_true, sigma_j_true, n, dt=dt, seed=i
        )
        returns_list.append(returns)

    # Pad and fit with batched calibrator
    R, M = pad_returns(returns_list)
    result = bmerton.fit_batch(R, M, dt)

    # Compare with scipy reference for each asset
    for i, rets in enumerate(returns_list):
        # Fit with scipy reference
        prices = 100 * np.exp(np.insert(np.cumsum(rets), 0, 0))
        scipy_result = MertonJumpCalibrator(k_max=5).fit(prices, dt=dt)

        # Check parameter recovery
        assert result["sigma"][i] == pytest.approx(sigma_true, rel=0.15), \
            f"Asset {i}: sigma={result['sigma'][i]:.4f}, expected {sigma_true:.4f} ± 15%"

        # Lambda within factor 2
        lambda_ratio = result["lam"][i] / lambda_true
        assert 0.5 <= lambda_ratio <= 2.0, \
            f"Asset {i}: lam={result['lam'][i]:.2f}, expected {lambda_true:.2f} (ratio={lambda_ratio:.2f})"

        # Check likelihood parity (this is the GATE)
        assert result["log_likelihood"][i] >= scipy_result.log_likelihood - 0.5, \
            f"Asset {i}: JAX logL={result['log_likelihood'][i]:.2f}, " \
            f"scipy logL={scipy_result.log_likelihood:.2f}, diff={result['log_likelihood'][i] - scipy_result.log_likelihood:.2f}"

        # Verify convergence
        assert result["converged"][i], f"Asset {i} did not converge"


def test_fixed_params_nll_isolation():
    """
    Test (b'): FIXED-PARAMS NLL isolation check (no optimizer).

    Evaluates the masked NLL function directly at fixed parameters
    for the same asset padded-in-batch vs alone.
    """
    import jax.numpy as jnp
    from options_desk.calibration.physical.batched.merton import _make_merton_nll_fn

    rets = _simulate_merton(0.05, 0.15, 10.0, -0.02, 0.04, 500, seed=42)

    # Alone: T=500, mask all ones
    R_alone, M_alone = pad_returns([rets])
    # Padded: same asset right-padded to T=900 alongside a longer asset
    rets_long = _simulate_merton(0.05, 0.15, 10.0, -0.02, 0.04, 900, seed=43)
    R_batch, M_batch = pad_returns([rets, rets_long])

    # Fixed parameters (arbitrary valid Merton point)
    dt = 1 / 252
    mu_fixed = 0.05
    sigma_fixed = 0.15
    lam_fixed = 10.0
    mu_j_fixed = -0.02
    sigma_j_fixed = 0.04

    # Unconstrained transforms: mu (free), sigma=softplus, lam=softplus, mu_j (free), sigma_j=softplus
    # sigma_fixed = softplus(a) => a = log(exp(sigma) - 1)
    a_sigma = np.log(np.exp(sigma_fixed) - 1.0)
    a_lam = np.log(np.exp(lam_fixed) - 1.0)
    a_sigma_j = np.log(np.exp(sigma_j_fixed) - 1.0)

    params = jnp.array([mu_fixed, a_sigma, a_lam, mu_j_fixed, a_sigma_j], dtype=jnp.float32)

    # Create NLL function with k_max=5
    nll_fn = _make_merton_nll_fn(k_max=5)

    nll_alone = float(nll_fn(
        params, jnp.asarray(R_alone[0], dtype=jnp.float32),
        jnp.asarray(M_alone[0], dtype=jnp.float32), dt
    ))
    nll_padded = float(nll_fn(
        params, jnp.asarray(R_batch[0], dtype=jnp.float32),
        jnp.asarray(M_batch[0], dtype=jnp.float32), dt
    ))

    rel_diff = abs(nll_padded - nll_alone) / max(abs(nll_alone), 1e-30)
    assert rel_diff <= 1e-5, (
        f"Mask leak detected: NLL alone={nll_alone:.8f}, "
        f"NLL padded={nll_padded:.8f}, rel diff={rel_diff:.2e} > 1e-5"
    )


def test_padding_does_not_leak():
    """
    Test (b): verify that padding does not affect parameter estimates.

    Endpoint tolerances are wide because the multi-start Adam optimizer
    can follow different trajectories under vmap.
    """
    # Create two assets with different lengths
    rets1 = _simulate_merton(0.05, 0.15, 10.0, -0.02, 0.04, 500, seed=42)
    rets2 = _simulate_merton(0.05, 0.15, 10.0, -0.02, 0.04, 900, seed=43)

    dt = 1 / 252

    # Fit together (with padding)
    R, M = pad_returns([rets1, rets2])
    result_together = bmerton.fit_batch(R, M, dt)

    # Fit first asset alone
    R_alone, M_alone = pad_returns([rets1])
    result_alone = bmerton.fit_batch(R_alone, M_alone, dt)

    # Results should match within tolerance
    assert result_together["mu"][0] == pytest.approx(result_alone["mu"][0], rel=0.1)
    assert result_together["sigma"][0] == pytest.approx(result_alone["sigma"][0], rel=0.1)
    assert result_together["lam"][0] == pytest.approx(result_alone["lam"][0], rel=0.1)
    assert result_together["mu_j"][0] == pytest.approx(result_alone["mu_j"][0], rel=0.1)
    assert result_together["sigma_j"][0] == pytest.approx(result_alone["sigma_j"][0], rel=0.1)
    # Likelihood should be close even if parameters differ slightly
    assert result_together["log_likelihood"][0] == pytest.approx(result_alone["log_likelihood"][0], rel=1e-2)
