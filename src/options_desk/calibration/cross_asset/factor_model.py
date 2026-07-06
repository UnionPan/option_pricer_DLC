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
    resid_min_eig: float | None = None  # min eigenvalue of resid_cov_sparse (post-PD-repair)

    def to_dense(self) -> np.ndarray:
        """Densify to full N×N covariance matrix."""
        Sigma = self.loadings @ self.factor_cov @ self.loadings.T
        if self.resid_cov_sparse is not None:
            Sigma += self.resid_cov_sparse
        else:
            Sigma += np.diag(self.resid_var)
        return Sigma

    def variance(self) -> np.ndarray:
        """Return diagonal variances without densifying.

        diag(B Ω Bᵀ)_i = B_i· Ω B_i·ᵀ = sum_j B_ij (B Ω)_ij — valid for
        arbitrary (not just diagonal) Ω.
        """
        factor_var = np.sum(self.loadings * (self.loadings @ self.factor_cov), axis=1)
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
            resid_contrib = float(w @ self.resid_cov_sparse @ w)
        else:
            # Diagonal D: wᵀ D w = sum_i w_i² D_ii
            resid_contrib = float(np.sum(w**2 * self.resid_var))
        return float(factor_contrib + resid_contrib)

    def min_eig_lower_bound(self) -> float:
        """Lower bound on minimum eigenvalue of Σ = B Ω Bᵀ + Θ.

        B Ω Bᵀ is PSD, so min_eig(Σ) ≥ min_eig(Θ).
        - Diagonal Θ = D: bound is min(D) > 0 (floored at 1e-8).
        - Sparse Θ: bound is the actual minimum eigenvalue of the
          PD-repaired residual covariance (computed once at fit time).
        """
        if self.resid_cov_sparse is not None:
            if self.resid_min_eig is not None:
                return float(self.resid_min_eig)
            return float(np.linalg.eigvalsh(self.resid_cov_sparse)[0])
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
    resid_min_eig: float | None = None  # min eigenvalue of resid_cov_sparse (post-PD-repair)

    def cov(self) -> FactorCov:
        """Return factored covariance wrapper."""
        return FactorCov(
            loadings=self.loadings,
            factor_cov=self.factor_cov,
            resid_var=self.resid_var,
            resid_cov_sparse=self.resid_cov_sparse,
            resid_min_eig=self.resid_min_eig,
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

    # Step 7: Optional POET SOFT thresholding (Fan-Liao-Mincheva 2013)
    # s(c) = sign(c) * max(|c| - tau, 0) applied to off-diagonal residual
    # correlations, diagonal kept.
    resid_cov_sparse = None
    resid_min_eig = None
    if threshold != 0:
        # Residual covariance and correlation
        resid_cov = (residuals.T @ residuals) / (T - 1)
        resid_std_for_corr = np.sqrt(np.diag(resid_cov))
        resid_std_for_corr = np.where(resid_std_for_corr < 1e-12, 1.0, resid_std_for_corr)
        resid_corr = resid_cov / (resid_std_for_corr[:, None] * resid_std_for_corr[None, :])

        # Threshold on correlation scale
        if threshold == "auto":
            tau = np.sqrt(np.log(N) / T)
        else:
            tau = float(threshold)

        # SOFT threshold off-diagonals: sign(c) * max(|c| - tau, 0)
        resid_corr_soft = np.sign(resid_corr) * np.maximum(np.abs(resid_corr) - tau, 0.0)
        np.fill_diagonal(resid_corr_soft, 1.0)  # Diagonal untouched (correlation = 1)

        # Back to covariance scale
        resid_cov_thresh = (
            resid_corr_soft * resid_std_for_corr[:, None] * resid_std_for_corr[None, :]
        )

        # PD repair — documented order:
        #  (1) set diagonal to the floored resid_var,
        #  (2) eigenvalue-clip at 1e-10 and reconstruct,
        #  (3) restore the floored diagonal exactly,
        #  (4) if step (3) reintroduced a negative eigenvalue, clip ONCE more
        #      (the diagonal may then deviate slightly from resid_var; PD wins).
        np.fill_diagonal(resid_cov_thresh, resid_var)  # (1)

        eigvals_resid, eigvecs_resid = np.linalg.eigh(resid_cov_thresh)
        if eigvals_resid[0] < 0:  # (2)
            eigvals_resid = np.maximum(eigvals_resid, 1e-10)
            resid_cov_thresh = eigvecs_resid @ np.diag(eigvals_resid) @ eigvecs_resid.T
            np.fill_diagonal(resid_cov_thresh, resid_var)  # (3)
            min_eig_check = np.linalg.eigvalsh(resid_cov_thresh)[0]
            if min_eig_check < 0:  # (4)
                ev, evec = np.linalg.eigh(resid_cov_thresh)
                ev = np.maximum(ev, 1e-10)
                resid_cov_thresh = evec @ np.diag(ev) @ evec.T

        # Enforce exact symmetry (eigh reconstruction can drift at ~1e-17)
        resid_cov_thresh = 0.5 * (resid_cov_thresh + resid_cov_thresh.T)

        # Store the ACTUAL minimum eigenvalue post-repair — this is what
        # min_eig_lower_bound() reports for the sparse-residual path.
        resid_min_eig = float(np.linalg.eigvalsh(resid_cov_thresh)[0])

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
        resid_min_eig=resid_min_eig,
    )
