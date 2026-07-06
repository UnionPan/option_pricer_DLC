"""
Test suite for DCC(1,1) correlation model.

Test families per brief:
1. Recovery: simulate DCC(1,1) with GARCH margins → recover a, b within tolerance
2. Correlation matrix properties: R_t symmetric, unit diagonal, positive definite
3. Constant-correlation data → fitted a near zero
4. Objective sanity: NLL at true params <= NLL at perturbed params
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import pytest
from options_desk.calibration.cross_asset.dcc import (
    fit_dcc,
    dcc_corr_path,
    DCCResult,
    _evaluate_dcc_nll,
)


def simulate_dcc_garch(k: int, T: int, a: float, b: float, seed: int = 42) -> np.ndarray:
    """
    Simulate DCC(1,1) with GARCH(1,1) margins.

    Args:
        k: Number of factors
        T: Number of time steps
        a: DCC parameter a
        b: DCC parameter b
        seed: Random seed

    Returns:
        factor_returns: (T, k) array of simulated returns
    """
    np.random.seed(seed)

    # GARCH(1,1) parameters for each factor
    omega = np.full(k, 0.05)
    alpha = np.full(k, 0.08)
    beta = np.full(k, 0.90)

    # Unconditional correlation matrix
    qbar = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            rho = 0.3 + 0.1 * np.sin(i + j)  # Some variation
            qbar[i, j] = qbar[j, i] = rho

    # Initialize
    sigma2 = np.zeros((k, T))
    eps = np.zeros((k, T))
    Q = np.zeros((k, k, T))
    R = np.zeros((k, k, T))

    # Initial variance (unconditional)
    for i in range(k):
        sigma2[i, 0] = omega[i] / (1 - alpha[i] - beta[i])

    Q[:, :, 0] = qbar.copy()
    R[:, :, 0] = qbar.copy()

    # Generate standardized innovations from multivariate normal
    z = np.random.multivariate_normal(np.zeros(k), R[:, :, 0])
    eps[:, 0] = z

    # Simulate forward
    for t in range(1, T):
        # Update GARCH variance
        for i in range(k):
            r_prev = eps[i, t - 1] * np.sqrt(sigma2[i, t - 1])
            sigma2[i, t] = omega[i] + alpha[i] * r_prev ** 2 + beta[i] * sigma2[i, t - 1]

        # Update DCC correlation
        eps_outer = np.outer(eps[:, t - 1], eps[:, t - 1])
        Q[:, :, t] = (1 - a - b) * qbar + a * eps_outer + b * Q[:, :, t - 1]

        # Normalize to get correlation
        q_diag = np.sqrt(np.diag(Q[:, :, t]))
        R[:, :, t] = Q[:, :, t] / np.outer(q_diag, q_diag)

        # Generate standardized innovations
        z = np.random.multivariate_normal(np.zeros(k), R[:, :, t])
        eps[:, t] = z

    # Construct returns
    returns = np.zeros((T, k))
    for t in range(T):
        returns[t, :] = eps[:, t] * np.sqrt(sigma2[:, t])

    return returns


def test_dcc_recovery():
    """Test 1: Recovery of DCC parameters from simulated data."""
    k = 5
    T = 3000
    true_a = 0.06
    true_b = 0.90

    # Simulate DCC-GARCH data
    factor_returns = simulate_dcc_garch(k, T, true_a, true_b, seed=123)

    # Fit DCC model
    result = fit_dcc(factor_returns)

    # Check recovery tolerances
    assert abs(result.a - true_a) < 0.04, f"a={result.a:.4f}, true={true_a}"
    assert abs(result.b - true_b) < 0.06, f"b={result.b:.4f}, true={true_b}"
    assert abs((result.a + result.b) - (true_a + true_b)) < 0.05, \
        f"a+b={result.a + result.b:.4f}, true={true_a + true_b}"

    # Check convergence
    assert result.converged, "Optimization should converge"

    # Check GARCH parameters are sensible
    assert result.garch_params.shape == (k, 4), "Should have k rows, 4 columns"
    assert np.all(result.garch_params['omega'] > 0), "omega should be positive"
    assert np.all(result.garch_params['alpha'] > 0), "alpha should be positive"
    assert np.all(result.garch_params['beta'] > 0), "beta should be positive"
    assert np.all(result.garch_params['alpha'] + result.garch_params['beta'] < 1), \
        "alpha + beta should be < 1"


def test_correlation_matrix_properties():
    """Test 2: All R_t from dcc_corr_path are valid correlation matrices."""
    k = 5
    T = 1000
    a = 0.05
    b = 0.90

    # Simulate data
    factor_returns = simulate_dcc_garch(k, T, a, b, seed=456)

    # Fit and get correlation path
    result = fit_dcc(factor_returns)
    R_path = dcc_corr_path(result, factor_returns)

    assert R_path.shape == (T, k, k), f"Shape should be (T, k, k), got {R_path.shape}"

    # Check properties at each time step
    for t in range(T):
        R_t = R_path[t]

        # Symmetric
        assert np.allclose(R_t, R_t.T, atol=1e-6), f"R_t[{t}] not symmetric"

        # Unit diagonal
        assert np.allclose(np.diag(R_t), 1.0, atol=1e-6), f"R_t[{t}] diagonal not 1"

        # Positive definite (all eigenvalues > 0)
        eigvals = np.linalg.eigvalsh(R_t)
        assert np.all(eigvals > -1e-8), f"R_t[{t}] has negative eigenvalues: {eigvals.min()}"


def test_constant_correlation():
    """Test 3: Constant-correlation data should yield a near zero."""
    k = 5
    T = 2000

    # Generate data with constant correlation (independent GARCH, no DCC dynamics)
    np.random.seed(789)

    # Fixed correlation matrix
    rho = 0.4
    corr = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            corr[i, j] = corr[j, i] = rho

    # GARCH parameters
    omega = np.full(k, 0.05)
    alpha = np.full(k, 0.08)
    beta = np.full(k, 0.90)

    # Initialize
    sigma2 = np.zeros((k, T))
    for i in range(k):
        sigma2[i, 0] = omega[i] / (1 - alpha[i] - beta[i])

    returns = np.zeros((T, k))

    # Simulate with constant correlation
    for t in range(T):
        if t > 0:
            for i in range(k):
                sigma2[i, t] = omega[i] + alpha[i] * returns[t - 1, i] ** 2 + beta[i] * sigma2[i, t - 1]

        # Generate correlated normal innovations
        z = np.random.multivariate_normal(np.zeros(k), corr)
        returns[t, :] = z * np.sqrt(sigma2[:, t])

    # Fit DCC model
    result = fit_dcc(returns)

    # a should be near zero for constant correlation
    assert result.a < 0.03, f"a={result.a:.4f} should be near zero for constant correlation"


def test_objective_sanity():
    """Test 4: NLL at true params <= NLL at perturbed params."""
    k = 5
    T = 2000
    true_a = 0.06
    true_b = 0.90

    # Simulate data
    factor_returns = simulate_dcc_garch(k, T, true_a, true_b, seed=101112)

    # Fit model
    result = fit_dcc(factor_returns)

    # Evaluate NLL at various parameter settings
    nll_fitted = _evaluate_dcc_nll(result.a, result.b, result.qbar,
                                    result.garch_params, factor_returns)

    # Perturbed parameters (should have worse NLL)
    nll_perturbed1 = _evaluate_dcc_nll(result.a + 0.1, result.b - 0.2, result.qbar,
                                        result.garch_params, factor_returns)

    nll_perturbed2 = _evaluate_dcc_nll(0.3, 0.5, result.qbar,
                                        result.garch_params, factor_returns)

    # Fitted NLL should be better (lower) than perturbed
    assert nll_fitted <= nll_perturbed1, \
        f"Fitted NLL {nll_fitted:.2f} should be <= perturbed1 {nll_perturbed1:.2f}"
    assert nll_fitted <= nll_perturbed2, \
        f"Fitted NLL {nll_fitted:.2f} should be <= perturbed2 {nll_perturbed2:.2f}"


def test_result_structure():
    """Test that DCCResult has the expected structure."""
    k = 3
    T = 500
    factor_returns = simulate_dcc_garch(k, T, 0.05, 0.90, seed=999)

    result = fit_dcc(factor_returns)

    # Check types
    assert isinstance(result, DCCResult)
    assert isinstance(result.a, float)
    assert isinstance(result.b, float)
    assert isinstance(result.qbar, np.ndarray)
    assert isinstance(result.last_corr, np.ndarray)
    assert isinstance(result.log_likelihood, float)
    assert isinstance(result.converged, bool)

    # Check shapes
    assert result.qbar.shape == (k, k)
    assert result.last_corr.shape == (k, k)

    # Check parameter constraints
    assert result.a > 0
    assert result.b > 0
    assert result.a + result.b < 1.0
