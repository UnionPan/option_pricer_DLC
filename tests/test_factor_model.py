"""Tests for POET factor covariance model with Marchenko-Pastur k selection."""

import numpy as np
import pytest

from options_desk.calibration.cross_asset.factor_model import (
    FactorCov,
    FactorModel,
    fit_factor_model,
)


class TestFactorModelRecovery:
    """Test 1: Recovery of true factor model."""

    def test_recovery_from_simulated_data(self):
        """Simulate data from true 3-factor model and verify recovery."""
        rng = np.random.RandomState(42)
        T = 400  # Smaller T increases sampling noise in sample cov
        N = 200  # Smaller N for better T/N ratio
        k_true = 3

        # Generate true factor model with strong factor structure
        # Larger loadings, smaller idio variance
        B_true = rng.randn(N, k_true)  # Loadings ~ N(0, 1)
        B_true[:, 0] *= 0.08
        B_true[:, 1] *= 0.05
        B_true[:, 2] *= 0.03

        factor_vols = np.array([0.015, 0.01, 0.008])  # Factor volatilities
        Omega_true = np.diag(factor_vols**2)
        D_true = np.ones(N) * 0.001**2  # Very small idio variance

        # True covariance
        Sigma_true = B_true @ Omega_true @ B_true.T + np.diag(D_true)

        # Generate returns
        factors = rng.randn(T, k_true) @ np.diag(factor_vols)
        idio = rng.randn(T, N) * 0.001
        returns = factors @ B_true.T + idio

        tickers = [f"ASSET_{i:03d}" for i in range(N)]

        # Fit model
        model = fit_factor_model(returns, tickers, k=None, threshold="auto")

        # Check k selection via MP edge
        assert model.k == k_true, f"Expected k={k_true}, got k={model.k}"

        # Get estimated covariance
        factor_cov = model.cov()
        Sigma_hat = factor_cov.to_dense()

        # Sample covariance (for comparison)
        returns_demeaned = returns - returns.mean(axis=0)
        Sigma_sample = (returns_demeaned.T @ returns_demeaned) / (T - 1)

        # Relative Frobenius error
        error_factor = np.linalg.norm(Sigma_hat - Sigma_true, "fro") / np.linalg.norm(
            Sigma_true, "fro"
        )
        error_sample = np.linalg.norm(Sigma_sample - Sigma_true, "fro") / np.linalg.norm(
            Sigma_true, "fro"
        )

        # Factor model should be better than sample covariance
        # With strong factor structure and k selection via MP edge,
        # the factor model provides dimensionality reduction benefit
        improvement_ratio = error_sample / error_factor
        assert improvement_ratio > 1.04, (
            f"Factor model error {error_factor:.4f} not sufficiently better than "
            f"sample cov error {error_sample:.4f} (improvement ratio {improvement_ratio:.3f} ≤ 1.04)"
        )


class TestNGreaterThanT:
    """Test 2: N > T regime."""

    def test_n_greater_than_t(self):
        """Test with N=400, T=250."""
        rng = np.random.RandomState(43)
        T = 250
        N = 400

        # Generate random returns (low-rank structure)
        k_sim = 5
        factors = rng.randn(T, k_sim)
        loadings = rng.randn(N, k_sim) * 0.05
        idio = rng.randn(T, N) * 0.01
        returns = factors @ loadings.T + idio

        tickers = [f"ASSET_{i:03d}" for i in range(N)]

        # Fit model
        model = fit_factor_model(returns, tickers, k=None, threshold="auto")

        # Check k is reasonable
        assert model.k <= 10, f"Expected k ≤ 10, got k={model.k}"

        # Check PD lower bound
        factor_cov = model.cov()
        min_eig_bound = factor_cov.min_eig_lower_bound()
        assert min_eig_bound > 0, f"Expected min_eig_lower_bound > 0, got {min_eig_bound}"

        # Check dense matrix is symmetric PD
        Sigma = factor_cov.to_dense()
        assert np.allclose(Sigma, Sigma.T), "Covariance not symmetric"

        eigvals = np.linalg.eigvalsh(Sigma)
        assert np.all(eigvals > 0), f"Negative eigenvalues found: min={eigvals.min()}"

        # Condition number comparison
        returns_demeaned = returns - returns.mean(axis=0)
        Sigma_sample = (returns_demeaned.T @ returns_demeaned) / (T - 1)

        cond_factor = np.linalg.cond(Sigma)
        cond_sample = np.linalg.cond(Sigma_sample)
        assert cond_factor < cond_sample, (
            f"Factor model condition number {cond_factor:.2e} not < "
            f"sample cov condition number {cond_sample:.2e}"
        )


class TestQuadForm:
    """Test 3: Quad form matches dense computation."""

    def test_quad_form_accuracy(self):
        """Test quad_form gives same result as dense computation."""
        rng = np.random.RandomState(44)
        T = 500
        N = 200

        # Generate returns
        k_sim = 4
        factors = rng.randn(T, k_sim) * 0.02
        loadings = rng.randn(N, k_sim) * 0.1
        idio = rng.randn(T, N) * 0.01
        returns = factors @ loadings.T + idio

        tickers = [f"ASSET_{i:03d}" for i in range(N)]

        # Fit model
        model = fit_factor_model(returns, tickers, k=None, threshold="auto")
        factor_cov = model.cov()

        # Test with random weight vector
        w = rng.randn(N)
        w /= np.abs(w).sum()  # Normalize

        # Quad form via efficient method
        quad_efficient = factor_cov.quad_form(w)

        # Quad form via dense
        Sigma = factor_cov.to_dense()
        quad_dense = w @ Sigma @ w

        rel_error = abs(quad_efficient - quad_dense) / abs(quad_dense)
        assert rel_error < 1e-10, f"Relative error {rel_error:.2e} >= 1e-10"


class TestDeterminism:
    """Test 4: Deterministic given seed."""

    def test_deterministic_with_seed(self):
        """Verify same results with same seed."""
        T = 300
        N = 150

        def generate_and_fit(seed):
            rng = np.random.RandomState(seed)
            k_sim = 3
            factors = rng.randn(T, k_sim) * 0.02
            loadings = rng.randn(N, k_sim) * 0.08
            idio = rng.randn(T, N) * 0.01
            returns = factors @ loadings.T + idio
            tickers = [f"ASSET_{i:03d}" for i in range(N)]
            return fit_factor_model(returns, tickers, k=None, threshold="auto")

        # Same seed should give identical results
        model1 = generate_and_fit(45)
        model2 = generate_and_fit(45)

        assert model1.k == model2.k
        assert np.allclose(model1.loadings, model2.loadings)
        assert np.allclose(model1.factor_cov, model2.factor_cov)
        assert np.allclose(model1.resid_var, model2.resid_var)
        assert np.allclose(model1.factors, model2.factors)
        assert model1.mp_edge == model2.mp_edge

        # Different seed should give different results
        model3 = generate_and_fit(99)
        assert not np.allclose(model1.loadings, model3.loadings)
