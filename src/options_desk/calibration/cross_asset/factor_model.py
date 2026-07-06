"""POET factor covariance model with Marchenko-Pastur k selection.

Implements factor model estimation with:
- Automatic k selection via Marchenko-Pastur eigenvalue edge
- N > T handling via smaller-side eigendecomposition
- POET soft-thresholding of residual covariances
- Factored representation for efficiency
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class FactorCov:
    """Factored covariance representation Σ = B Ω Bᵀ + D (+ optional sparse residual cov).

    Avoids densifying the full N×N matrix for efficient operations.
    """

    loadings: np.ndarray  # (N, k)  B
    factor_cov: np.ndarray  # (k, k)  Ω
    resid_var: np.ndarray  # (N,)    D diagonal
    resid_cov_sparse: np.ndarray | None  # (N, N) optional thresholded residual cov

    def to_dense(self) -> np.ndarray:
        """Densify to full N×N covariance matrix."""
        N = self.loadings.shape[0]
        Sigma = self.loadings @ self.factor_cov @ self.loadings.T
        if self.resid_cov_sparse is not None:
            Sigma += self.resid_cov_sparse
        else:
            Sigma += np.diag(self.resid_var)
        return Sigma

    def variance(self) -> np.ndarray:
        """Return diagonal variances without densifying."""
        # diag(B Ω Bᵀ) = sum_k B[:, k]^2 * Ω[k, k]
        factor_var = np.sum(
            self.loadings**2 * np.diag(self.factor_cov)[None, :], axis=1
        )
        if self.resid_cov_sparse is not None:
            return factor_var + np.diag(self.resid_cov_sparse)
        else:
            return factor_var + self.resid_var

    def quad_form(self, w: np.ndarray) -> float:
        """Compute wᵀ Σ w without densifying.

        wᵀ (B Ω Bᵀ + D) w = (Bᵀw)ᵀ Ω (Bᵀw) + wᵀ D w
        """
        Btw = self.loadings.T @ w  # (k,)
        factor_contrib = Btw @ self.factor_cov @ Btw
        if self.resid_cov_sparse is not None:
            resid_contrib = w @ self.resid_cov_sparse @ w
        else:
            resid_contrib = w @ self.resid_var * w  # element-wise for diagonal
        return float(factor_contrib + resid_contrib)

    def min_eig_lower_bound(self) -> float:
        """Lower bound on minimum eigenvalue.

        Since D is diagonal and positive, min(D) > 0 guarantees PD.
        For sparse residual cov, we use the diagonal floor.
        """
        return float(np.min(self.resid_var))


@dataclass
class FactorModel:
    """Factor model with loadings, factor covariances, and residual structure."""

    loadings: np.ndarray  # (N, k)  B — on RETURN scale
    factor_cov: np.ndarray  # (k, k)  Ω (diagonal if orthogonalized)
    resid_var: np.ndarray  # (N,)    D diagonal (floored residual variances)
    resid_cov_sparse: np.ndarray | None  # (N, N) thresholded residual cov
    factors: np.ndarray  # (T, k)  estimated factor returns
    k: int
    mp_edge: float  # Marchenko-Pastur eigenvalue edge
    tickers: list[str]

    def cov(self) -> FactorCov:
        """Return factored covariance wrapper."""
        return FactorCov(
            loadings=self.loadings,
            factor_cov=self.factor_cov,
            resid_var=self.resid_var,
            resid_cov_sparse=self.resid_cov_sparse,
        )


def fit_factor_model(
    returns: np.ndarray,
    tickers: list[str],
    k: int | None = None,
    threshold: str | float = "auto",
) -> FactorModel:
    """Fit POET factor model with Marchenko-Pastur k selection.

    Args:
        returns: (T, N) return matrix
        tickers: List of N ticker names
        k: Number of factors (None = auto-select via MP edge)
        threshold: POET soft-threshold ("auto" = sqrt(log(N)/T) or float)

    Returns:
        FactorModel with loadings on return scale, factored covariance
    """
    T, N = returns.shape
    assert len(tickers) == N

    # Step 1: Standardize (demean, unit variance)
    mean = returns.mean(axis=0)
    returns_demeaned = returns - mean
    std = returns_demeaned.std(axis=0, ddof=1)
    std = np.where(std < 1e-12, 1.0, std)  # Avoid division by zero
    returns_std = returns_demeaned / std

    # Step 2: Sample correlation matrix via eigendecomposition of smaller side
    # For N > T: use T×T trick: if C = (1/(T-1)) Xᵀ X, eigenvectors of (1/(T-1)) X Xᵀ
    # give us the principal components, and we can recover loadings
    if N > T:
        # Eigendecompose (T, T) matrix: (1/(T-1)) X Xᵀ where X is (T, N)
        gram = (returns_std @ returns_std.T) / (T - 1)
        eigvals, eigvecs_T = np.linalg.eigh(gram)
        # Sort descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs_T = eigvecs_T[:, idx]

        # The eigenvalues are the same as the correlation matrix eigenvalues
        # (for the non-zero ones)
        corr_eigvals = eigvals
    else:
        # Eigendecompose (N, N) correlation matrix
        corr = (returns_std.T @ returns_std) / (T - 1)
        eigvals, eigvecs = np.linalg.eigh(corr)
        # Sort descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        corr_eigvals = eigvals

    # Step 3: Determine k via Marchenko-Pastur edge if needed
    gamma = N / T
    mp_edge = (1 + np.sqrt(gamma)) ** 2

    if k is None:
        # Count eigenvalues above MP edge
        k = int(np.sum(corr_eigvals > mp_edge))
        k = max(1, k)  # At least 1 factor

    # Step 4: PCA on standardized returns
    # Key: X_std (T, N) = F_std (T, k) @ L_std (k, N) + E_std
    # where F_std has cov I_k and L_std is loadings in standardized space
    if N > T:
        # Eigendecompose Gram matrix X X^T / (T-1)
        # Eigenvectors U give factor scores: F_std = U_k
        # Loadings: L_std = Lambda_k^{1/2} V_k^T where V_k = X^T U_k / sqrt((T-1) lambda_k)
        factors_std = eigvecs_T[:, :k]  # (T, k) - orthonormal
        loadings_std = (returns_std.T @ factors_std) / np.sqrt((T - 1) * eigvals[:k])[None, :]  # (N, k)
    else:
        # Eigendecompose correlation matrix X^T X / (T-1)
        # Loadings in correlation space: V_k
        # Factors: F_std = X V_k (T, k)
        loadings_std = eigvecs[:, :k]  # (N, k)
        factors_std = returns_std @ loadings_std  # (T, k)

    # Verify: factors_std should have sample cov close to diag(eigvals[:k])
    # Actually, for normalized factors: cov(F_std) should be I

    # Step 5: Map to RETURN scale
    # X_demeaned = diag(std) @ X_std = diag(std) @ F_std @ L_std^T + diag(std) @ E_std
    #             = F_std @ (diag(std) @ L_std)^T + E_demeaned
    # So loadings on return scale: B = diag(std) @ L_std (N, k)
    loadings = std[:, None] * loadings_std  # (N, k)

    # Factor covariance: cov(F_std)
    # In PCA, factors are principal components with cov = diag(eigvals)
    # But we want orthonormal factors, so we need to scale
    # Actually: cov(F_std) = cov(X @ V) = V^T @ cov(X) @ V = V^T @ I @ V = I (for correlation)
    # But sample cov will be diag(eigvals)
    factor_cov_std = (factors_std.T @ factors_std) / (T - 1)  # Should be ~ diag(eigvals[:k])

    # For the factored representation to work, we need:
    # cov(X_demeaned) = B @ Omega @ B^T + D
    # where B = diag(std) @ L_std
    # and Omega = cov(F_std)
    factor_cov = factor_cov_std

    # Step 6: Compute residuals
    fitted_std = factors_std @ loadings_std.T  # (T, N) in standardized space
    fitted = fitted_std * std[None, :]  # (T, N) on return scale
    residuals = returns_demeaned - fitted
    resid_var = np.var(residuals, axis=0, ddof=1)
    resid_var = np.maximum(resid_var, 1e-8)  # Floor at 1e-8

    # Step 7: Optional POET soft-thresholding
    resid_cov_sparse = None
    if threshold != 0:
        # Compute residual covariance matrix
        resid_cov = (residuals.T @ residuals) / (T - 1)

        # Compute correlation matrix for thresholding
        resid_std_for_corr = np.sqrt(np.diag(resid_cov))
        resid_std_for_corr = np.where(resid_std_for_corr < 1e-12, 1.0, resid_std_for_corr)
        resid_corr = resid_cov / (resid_std_for_corr[:, None] * resid_std_for_corr[None, :])

        # Determine threshold
        if threshold == "auto":
            tau = np.sqrt(np.log(N) / T)
        else:
            tau = float(threshold)

        # Soft-threshold: keep |c_ij| > tau
        mask = np.abs(resid_corr) > tau
        np.fill_diagonal(mask, True)  # Always keep diagonal

        # Thresholded correlation
        resid_corr_thresh = np.where(mask, resid_corr, 0.0)

        # Convert back to covariance scale
        resid_cov_thresh = resid_corr_thresh * resid_std_for_corr[:, None] * resid_std_for_corr[None, :]

        # Ensure diagonal is at least resid_var (floored)
        # Thresholding may have reduced the diagonal estimate
        current_diag = np.diag(resid_cov_thresh)
        np.fill_diagonal(resid_cov_thresh, np.maximum(current_diag, resid_var))

        # PD repair: eigenvalue clipping
        eigvals_resid, eigvecs_resid = np.linalg.eigh(resid_cov_thresh)
        if np.any(eigvals_resid < 0):
            # Clip negative eigenvalues
            eigvals_resid = np.maximum(eigvals_resid, 1e-10)
            resid_cov_thresh = eigvecs_resid @ np.diag(eigvals_resid) @ eigvecs_resid.T

        resid_cov_sparse = resid_cov_thresh

    # Step 8: Factors on return scale (for interpretation)
    # These are the factor time series, computed as projections
    factors = factors_std @ np.diag(np.sqrt(eigvals[:k]))

    return FactorModel(
        loadings=loadings,
        factor_cov=factor_cov,
        resid_var=resid_var,
        resid_cov_sparse=resid_cov_sparse,
        factors=factors,
        k=k,
        mp_edge=mp_edge,
        tickers=tickers,
    )
