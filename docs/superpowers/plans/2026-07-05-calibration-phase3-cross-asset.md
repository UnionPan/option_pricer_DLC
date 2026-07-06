# Calibration Phase 3: Cross-Asset Suite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The spec's Layer 3: factor covariance that is well-conditioned at N≈3000>T (PCA + Marchenko-Pastur k selection + POET-style residual thresholding), DCC(1,1) on the factor returns, and empirical-Bayes hierarchical pooling of per-name parameters toward sector means — all wired into the runner/CLI as a `--cross-asset` stage consuming `PriceStore.returns_matrix` (its first real consumer).

**Architecture:** New package `src/options_desk/calibration/cross_asset/` with `factor_model.py`, `dcc.py`, `pooling.py`; runner gains a cross-asset stage after the per-model loop; results land in the run dir (`factor_model.npz` + `factor_summary.json`, `dcc.json`, `<model>_pooled.parquet`). Existing `correlation.py` (Ledoit-Wolf/RMT) stays untouched; the MP edge logic is reimplemented locally where cleaner but with a parity check against `correlation.py`'s Marchenko-Pastur bounds if reusable.

**Tech Stack:** numpy/scipy for factor model + pooling (N=3000 eigendecomp is fine on CPU); jax+optax for the DCC optimizer (reusing `batched/garch.py` for step 1). pandas/pyarrow for outputs.

**Spec:** `docs/superpowers/specs/2026-07-05-universe-scale-p-measure-calibration-design.md` (Layer 3)

## Global Constraints

- Env: ALWAYS `source /home/union/miniconda3/etc/profile.d/conda.sh && conda activate options-desk`; pytest from repo root; tests network-free; tests set `os.environ.setdefault("JAX_PLATFORMS","cpu")` before importing jax (directly or transitively via the pipeline).
- Numerics in float64 numpy for the factor model and pooling (no JAX needed); DCC step-2 optimizer in JAX float32 with float64 outputs.
- Every produced covariance/correlation must be symmetric PD (assert smallest eigenvalue > 0 in tests); Σ is returned in FACTORED form (B, Ω, D) plus helpers — never materialize a dense 3000×3000 unless asked (`to_dense()` helper fine for N ≤ ~1500 tests).
- Stage files explicitly per task; never `git add -A`; no `__pycache__`. Commit per task with trailer:
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
- TDD per task. Do not modify `correlation.py`, existing calibrators, or Phase 1/2 modules except where a task names them.

---

### Task 1: Factor covariance (`factor_model.py`)

**Files:** Create `src/options_desk/calibration/cross_asset/__init__.py`, `src/options_desk/calibration/cross_asset/factor_model.py`; Test `tests/test_factor_model.py`.

**Interfaces (produced):**
```python
@dataclass
class FactorModel:
    loadings: np.ndarray        # (N, k)  B — on the RETURN scale (not standardized)
    factor_cov: np.ndarray      # (k, k)  Ω (diagonal if factors orthogonalized)
    resid_var: np.ndarray       # (N,)    D diagonal (POET residual variances)
    resid_cov_sparse: np.ndarray | None  # (N, N) thresholded residual cov (optional)
    factors: np.ndarray         # (T, k)  estimated factor returns
    k: int
    mp_edge: float              # Marchenko-Pastur eigenvalue edge used for k
    tickers: list[str]
    def cov(self) -> "FactorCov": ...      # factored Σ = B Ω Bᵀ + D wrapper
@dataclass
class FactorCov:
    # factored covariance with helpers:
    def to_dense(self) -> np.ndarray
    def variance(self) -> np.ndarray                  # diag(Σ)
    def quad_form(self, w: np.ndarray) -> float       # wᵀ Σ w without densifying
    def min_eig_lower_bound(self) -> float            # min(D) > 0 ⇒ PD
def fit_factor_model(returns: np.ndarray, tickers: list[str],
                     k: int | None = None,
                     threshold: str | float = "auto") -> FactorModel
```
**Method:** standardize columns (demean, unit variance); eigendecompose the T×T or N×N sample correlation (use the smaller side; N>T ⇒ T×T trick); `k` = number of eigenvalues above the Marchenko-Pastur edge `(1+sqrt(N/T))²` (of the correlation matrix) when `k=None`; principal-component factors; loadings mapped back to return scale via the per-name std; residual variances floored at `1e-8`; optional POET soft-threshold on residual covariance: keep entries `|c_ij| > threshold·sqrt(c_ii·c_jj)` (default `"auto"` = `sqrt(log(N)/T)` rate), then PD-repair the residual block by eigenvalue clipping at 0 + restore floored diagonal.

**Tests:** (1) recovery: simulate T=800, N=300 from a true 3-factor model (random B, factor vols {3%, 2%, 1.5%} daily, idio vol 1%) → `fit_factor_model` picks k=3 via MP edge; relative Frobenius error of Σ̂ vs true Σ < half that of the raw sample covariance; (2) N>T regime: N=400, T=250 → runs, k reasonable (≤10), `FactorCov.min_eig_lower_bound() > 0`, `to_dense()` symmetric PD (eigvalsh > 0), condition number < sample-cov condition number; (3) quad_form matches dense computation rel 1e-10; (4) deterministic given seed.

Commit: `feat(calibration): POET factor covariance with MP-edge k selection`.

---

### Task 2: Hierarchical pooling (`pooling.py`)

**Files:** Create `src/options_desk/calibration/cross_asset/pooling.py`; Test `tests/test_pooling.py`.

**Interface:**
```python
def pool_parameters(df: pd.DataFrame, params: list[str],
                    sector_col: str = "sector",
                    min_sector_size: int = 5) -> pd.DataFrame
# returns a copy with added columns: f"{p}_pooled", f"{p}_shrinkage" per p
```
**Method (positive-part James-Stein per (param, sector)):** for each param p and sector s with n_s ≥ min_sector_size names: sector mean m_s; per-name noise variance estimated as σ̂² = (robust within-sector residual variance via MAD²·1.4826²); shrinkage weight `w = max(0, 1 − (n_s − 3)·σ̂² / Σ_i (x_i − m_s)²)` … use the standard JS positive-part estimator: `x_pooled_i = m_s + (1 − w)·(x_i − m_s)` where `w = min(1, (n_s − 3)·σ̂² / SS)` with SS = Σ(x_i − m_s)². Sectors smaller than min_sector_size (incl. UNKNOWN) shrink toward the GLOBAL mean with the same rule. Non-finite param values pass through untouched with shrinkage 0. Clip pooled values of positivity-constrained params (kappa, theta, sigma_v, omega, alpha, beta, lam, sigma_j) at a small positive floor when the raw value was positive.

**Tests:** (1) heavy-noise sector: simulate one sector of 40 names whose true param is constant μ + large noise → pooled values have ≥ 60% lower MSE vs truth than raw; shrinkage column in (0,1]; (2) tight sector (noise ≈ 0) → shrinkage ≈ 0, pooled ≈ raw; (3) small sector (n=3) shrinks toward global mean; (4) NaN param rows untouched; (5) sector means preserved (mean of pooled == sector mean of raw within fp tolerance).

Commit: `feat(calibration): empirical-Bayes sector pooling of calibrated parameters`.

---

### Task 3: DCC(1,1) on factor returns (`dcc.py`)

**Files:** Create `src/options_desk/calibration/cross_asset/dcc.py`; Test `tests/test_dcc.py`.

**Interface:**
```python
@dataclass
class DCCResult:
    a: float; b: float
    qbar: np.ndarray            # (k, k) unconditional corr of std residuals
    garch_params: pd.DataFrame  # per-factor omega/alpha/beta
    last_corr: np.ndarray       # (k, k) R_T
    log_likelihood: float
    converged: bool
def fit_dcc(factor_returns: np.ndarray) -> DCCResult      # (T, k), k ≤ ~50
def dcc_corr_path(result: DCCResult, factor_returns: np.ndarray) -> np.ndarray  # (T, k, k)
```
**Method:** step 1 — univariate GARCH(1,1) per factor via `batched.garch.fit_batch` (factors as "assets"); standardized residuals ε_t = r_t/σ_t. Step 2 — correlation targeting: Q̄ = corr(ε); recursion `Q_t = (1−a−b)·Q̄ + a·ε_{t−1}ε_{t−1}ᵀ + b·Q_{t−1}`, `R_t = diag(Q_t)^{−1/2} Q_t diag(Q_t)^{−1/2}`; Gaussian copula NLL `0.5·Σ_t [log|R_t| + ε_tᵀ R_t^{−1} ε_t − ε_tᵀε_t]` via `lax.scan` (k ≤ 50 ⇒ dense solves fine); optimize (a,b) on the constrained surface a,b>0, a+b<0.999 via the same sigmoid-pair transform as batched GARCH; optax.adam(5e-3), 500 steps, 2 starts ((a,b) ∈ {(0.05,0.90),(0.02,0.95)}).

**Tests:** (1) recovery: simulate DCC(1,1) with k=5, T=3000, a=0.06, b=0.90, unit-variance GARCH margins → recovered a within ±0.04, b within ±0.06, and `a+b` within ±0.05 (persistence is the well-identified quantity); (2) all R_t from `dcc_corr_path` symmetric, unit diagonal, eigvalsh > 0; (3) constant-correlation data (a=b=0 world) → fitted a ≈ 0 (< 0.03); (4) fixed-params NLL padding-free (no padding here — instead assert NLL at true params ≤ NLL at perturbed params on simulated data, a sanity of the objective).

Commit: `feat(calibration): DCC(1,1) on factor returns via JAX two-step`.

---

### Task 4: Runner + CLI cross-asset stage

**Files:** Modify `src/options_desk/calibration/pipeline/runner.py` (RunConfig + stage), `scripts/calibrate_universe.py` (`--cross-asset` flag), `src/options_desk/calibration/cross_asset/__init__.py` (exports); Test `tests/test_cross_asset_stage.py`.

**Contract:** `RunConfig` gains `cross_asset: list[str] = field(default_factory=list)` (allowed: "factor", "dcc", "pooling") and `pooling_model: str = "heston_qmle"`. In `run_calibration`, after the model loop and before the final manifest write, if `cfg.cross_asset`:
1. Build `rm = store.returns_matrix(universe.tickers, start, end, min_obs=min(504, int(0.8·T_window)))`; record `rm.excluded` count in the manifest under `cross_asset.excluded`.
2. `"factor"`: `fit_factor_model(rm.returns.astype(np.float64), rm.tickers)`; save `np.savez(run_dir/"factor_model.npz", loadings=..., factor_cov=..., resid_var=..., factors=..., tickers=...)` + `factor_summary.json` (k, mp_edge, n_names, T, min-eig lower bound, top-5 eigenvalue shares); write `factor.done` marker (resume-skippable like models).
3. `"dcc"` (requires factor; if factor not requested this run, load factors from an existing `factor_model.npz` in the run dir or error the stage — record error in manifest, don't crash): `fit_dcc(factors)`; save `dcc.json` (a, b, log_likelihood, converged, garch_params records, last_corr as nested list) + `dcc.done`.
4. `"pooling"`: load `run_dir/<pooling_model>.parquet` (must exist from this run — else record stage error), `pool_parameters(df, params=<numeric param columns for that model: for heston_qmle ["kappa","theta","sigma_v","rho","mu","v0"]>)`, save `<model>_pooled.parquet` + `pooling.done`.
5. Stage failures are isolated: record `cross_asset.errors[stage]` in the manifest, continue; `status` stays "complete" with stages listed under `cross_asset.completed`.

CLI: `--cross-asset` (nargs="*", choices factor/dcc/pooling, default []), `--pooling-model` (default heston_qmle) → RunConfig passthrough.

**Tests (synthetic store fixture, ~40 names × 700 bdays with a 2-factor structure planted):** (a) full stage `["factor","dcc","pooling"]` with models=["gbm","heston_qmle"] → run dir contains factor_model.npz, factor_summary.json, dcc.json, heston_qmle_pooled.parquet, and manifest lists all three completed; (b) resume: second invocation skips all three (`.done` markers); (c) pooling with a missing model parquet records a stage error and the other stages still complete; (d) factor_summary.json min-eig bound > 0.

Commit: `feat(calibration): cross-asset stage (factor/dcc/pooling) in runner + CLI`.

---

### Task 5: Real-data verification (manual, network OK)

No new code. (1) `python scripts/calibrate_universe.py --universe sp500 --models gbm heston_qmle garch --cross-asset factor dcc pooling --run-id sp500-cross` (lake warm). (2) Report: chosen k and top eigenvalue shares (expect market factor ≈ 30–50% of variance), min-eig bound, DCC (a, b, a+b — expect persistence 0.95–0.999), pooling shrinkage medians per Heston param (expect meaningful shrinkage on kappa/rho, little on theta). (3) Sanity: factor-1 loadings should be predominantly one-signed (market factor) — report the sign fraction. (4) Append `## Phase 3 verification` to `.superpowers/sdd/progress.md`. No commits.

---

## Self-Review Notes
- Spec Layer-3 coverage: factor covariance (MP-edge k, POET residual, PD, snapshots) ✅ T1/T4; DCC on factors ✅ T3/T4; hierarchical pooling ✅ T2/T4; persisted date-stamped outputs ✅ T4; N>T conditioning ✅ T1 test 2. returns_matrix finally consumed ✅ T4.
- Judgment calls: Σ kept factored (never densify at N=3000); DCC persistence (a+b) is the assertion target, not a and b separately, matching identifiability; pooling uses positive-part JS (classical, defensible) rather than full hierarchical Bayes — documented upgrade path.
