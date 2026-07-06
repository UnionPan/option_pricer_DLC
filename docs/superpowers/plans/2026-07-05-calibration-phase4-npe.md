# Calibration Phase 4: Full-Fidelity SV at Scale (NPE + OHLC wiring) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The amended spec's Phase 4: (a) wire the Phase 2 Garman-Klass OHLC proxy into the runner as a first-class model (`heston_qmle_gk`); (b) amortized neural posterior estimation for Heston under P — simulate (θ, path) pairs from a JAX Heston simulator, train a conditional mixture-density posterior over θ given path summary statistics, register `heston_npe` so the whole universe calibrates in one forward pass WITH posterior uncertainties (feeding Phase 3 pooling); (c) cross-validate NPE posteriors against the existing scipy particle filter on a small name sample.

**Architecture:** New package `src/options_desk/calibration/physical/batched/npe/` (`simulate.py` prior + JAX Euler full-truncation Heston simulator + summary features; `model.py` flax MDN q(θ|s); `train.py` training loop + checkpoint save/load; `estimator.py` fit_batch-compatible inference + registration). Runner gains OHLC assembly for models flagged `needs_ohlc` (per-bar adjustment scaling so GK ratios stay raw-consistent while returns are adjusted). `scripts/train_npe_heston.py` does the real GPU training run; tests train tiny throwaway models only.

**Tech Stack:** jax/flax/optax (present). MDN (K=8 Gaussian mixture over standardized θ) rather than a full flow — declared v1 in the spec's spirit ("conditional normalizing flow" upgradeable; MDN IS a conditional density estimator and keeps the dependency surface zero). Document the upgrade path in the module docstring.

## Global Constraints

- Env: `source /home/union/miniconda3/etc/profile.d/conda.sh && conda activate options-desk`; pytest from repo root; tests network-free, `os.environ.setdefault("JAX_PLATFORMS","cpu")` before jax imports; tiny fixtures only in tests (real training happens in Task 5 on GPU).
- θ = (kappa, theta, sigma_v, rho, mu, v0). Prior (annualized, document): kappa~U(0.5,15), theta~U(0.005,0.25), sigma_v~U(0.1,1.5), rho~U(−0.95,0.1), mu~U(−0.1,0.3), v0~U(0.005,0.25). Train/infer in a standardized unconstrained space (log for positive params, atanh(rho/0.99), affine for mu); posterior moments reported back on the natural scale via sampling (4096 samples) from the MDN.
- Summary features are the ONLY interface between paths and the network: a fixed, named 16-vector computed identically at train and inference time by ONE shared function (masked, works on (N,T) with mask): {std, skew, excess kurtosis of returns; acf of r² at lags 1,5,10,21; acf of |r| at lags 1,5; log mean RV(21), log std RV(21); corr(r_t, RV_{t+1}−RV_t); corr(r_t, r²_{t+1}); mean|r|; fraction |r|>2σ; log std of RV(63)}. Features must be finite for any non-degenerate path (guard divisions).
- Checkpoints: pickle dict {"params": flax params, "feature_mean/std", "theta_mean/std", "config"} at data/npe/heston_mdn.pkl (gitignored: add data/npe/ to .gitignore in Task 4).
- Stage files explicitly; no `__pycache__`; commit per task with trailer:
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

---

### Task 1: OHLC flow — `heston_qmle_gk` end to end

**Files:** Modify `pipeline/registry.py` (ModelSpec gains `needs_ohlc: bool = False`; new registration `heston_qmle_gk` whose BatchFitFn signature for OHLC models is `Callable[[list[dict[str,np.ndarray]], float], dict]` receiving per-ticker `{"open","high","low","close","adj_close"}` arrays), `pipeline/runner.py` (when `spec.needs_ohlc`, assemble per-ticker OHLC dicts from `store.get_prices` over the window; per-bar adjustment: factor = adj_close/close, scale O/H/L/C by it so GK ratios are unchanged and close-returns are adjusted; no per-asset joblib fallback for OHLC models — insufficient-data rows same as batch path, and a fit_batch exception marks the whole model errored in the manifest rather than falling back), and adapter calling `batched.heston_qmle.fit_batch_ohlc`; Test `tests/test_heston_gk_runner.py`.

**Tests:** synthetic store fixture with OHLC built from a fine-grid Heston sim (reuse the pattern from tests/test_batched_heston_qmle.py's GK test, smaller): (a) runner with models=["heston_qmle_gk"] produces converged rows with kappa/theta/sigma_v columns; (b) adjustment invariance: multiplying a ticker's O/H/L/C by a constant and its adj_close consistently leaves GK params unchanged (rel 1e-6); (c) a ticker with missing OHLC columns (NaN high/low) → converged=False row, not a crash. Commit: `feat(calibration): heston_qmle_gk model with OHLC flow in runner`.

---

### Task 2: NPE simulator + priors + summary features (`npe/simulate.py`)

**Files:** Create `batched/npe/__init__.py`, `batched/npe/simulate.py`; Test `tests/test_npe_simulate.py`.

**Interfaces:**
```python
PRIOR_LOW, PRIOR_HIGH: dict[str, float]                  # per Global Constraints
def sample_prior(key, n) -> jnp.ndarray                  # (n, 6) natural scale
def simulate_heston_paths(key, thetas (n,6), T, dt=1/252) -> jnp.ndarray  # (n, T) log-returns; Euler full truncation, v floored at 0 inside sqrt
def summary_features(returns (N,T), mask (N,T)) -> jnp.ndarray  # (N, 16), the shared train/infer featurizer
FEATURE_NAMES: list[str]  # len 16
def to_unconstrained(thetas (n,6)) -> (n,6); def to_natural(z (n,6)) -> (n,6)
```
**Tests:** (a) shapes + determinism per key; (b) prior samples within bounds, round-trip to_unconstrained∘to_natural = identity (rtol 1e-5); (c) features finite on 500 simulated paths AND on a constant-price path (guards); (d) feature sanity: higher sigma_v prior slice → higher mean r²-acf(1) feature than low sigma_v slice (vol clustering signal exists); (e) simulator moment check: for fixed θ (kappa=3, theta=0.04, ...) mean annualized realized variance over 2000 paths within 15% of theta. Commit: `feat(calibration): NPE prior, JAX Heston simulator, summary features`.

---

### Task 3: MDN + training loop (`npe/model.py`, `npe/train.py`)

**Files:** Create `batched/npe/model.py`, `batched/npe/train.py`; Test `tests/test_npe_train.py`.

**Interfaces:**
```python
class ConditionalMDN(nn.Module):   # hidden (128,128) relu, K=8 comps, diag covs
    # __call__(s (B,16)) -> (logits (B,K), means (B,K,6), log_scales (B,K,6))
def mdn_nll(params, apply_fn, s, z) -> scalar             # stable logsumexp
def train_mdn(key, features, thetas_unconstrained, *, epochs, batch_size=512,
              lr=1e-3, val_frac=0.1, hidden=(128,128), n_components=8) -> TrainedNPE
@dataclass TrainedNPE: params; feature_mean; feature_std; theta_mean; theta_std; config: dict
def save_npe(t: TrainedNPE, path); def load_npe(path) -> TrainedNPE
def sample_posterior(t: TrainedNPE, apply_fn_or_none, s_raw (N,16), key, n_samples=4096) -> (N, n_samples, 6) natural scale
```
Standardization: features and θ standardized inside train (stats stored). log_scales clipped to [−7, 2].

**Tests (tiny: 4000 sims of T=256 from Task 2, epochs=30, hidden=(32,32), K=4 — must run < ~120s CPU):** (a) val NLL decreases ≥ 20% from epoch 0; (b) save/load roundtrip → identical posterior samples given same key; (c) posterior samples within prior bounds' natural ranges (soft check: 99% inside 1.5× ranges); (d) informativeness: posterior std of theta (the well-identified param) < prior std of theta on average. Commit: `feat(calibration): conditional MDN posterior + training loop for NPE`.

---

### Task 4: Amortized estimator + registration + training script

**Files:** Create `batched/npe/estimator.py`, `scripts/train_npe_heston.py`; Modify `pipeline/registry.py` (register `heston_npe`), `.gitignore` (+`data/npe/`); Test `tests/test_npe_estimator.py`.

**Interfaces:** `estimator.fit_batch(returns (N,T), mask, dt, checkpoint_path=DEFAULT) -> dict` with keys kappa, theta, sigma_v, rho, mu, v0 (posterior MEANS), plus `{p}_std` posterior stds, `log_likelihood` = mean MDN log-density of the posterior mean (diagnostic), `converged` = features finite & checkpoint loaded, `n_observations`. Registry: `heston_npe` registered ONLY when the default checkpoint file exists at import... no — register unconditionally; the adapter raises a clear FileNotFoundError("train first: scripts/train_npe_heston.py") which the runner's error isolation converts to an errored model. `scripts/train_npe_heston.py`: args --n-sims 200000 --T 1260 --epochs 200 --out data/npe/heston_mdn.pkl; simulates in chunks on GPU, trains, saves, prints held-out recovery correlations per param.

**Tests (tiny in-test training, reuse Task 3 scale):** (a) end-to-end recovery: train tiny NPE, run estimator.fit_batch on 200 fresh simulated paths with known θ → Pearson r(posterior mean, truth) ≥ 0.8 for theta, ≥ 0.4 for kappa and sigma_v (tiny-scale bounds; real-scale bounds checked in Task 5); (b) coverage: fraction of truths within ±1 posterior std in [0.45, 0.90] for theta (68% nominal, wide tolerance at tiny scale); (c) runner integration: register a tmp checkpoint path via monkeypatched DEFAULT, run through run_calibration on the synthetic store fixture → converged rows with kappa/…/v0_std columns; (d) missing checkpoint → model errored in manifest, run completes. Commit: `feat(calibration): amortized heston_npe estimator + training script`.

---

### Task 5: GPU training + universe calibration + PF cross-validation (manual)

No new code except throwaway analysis in the report. (1) Train for real: `python scripts/train_npe_heston.py --n-sims 200000 --T 1260 --epochs 200` on GPU (~expect minutes to tens of minutes); record held-out recovery correlations (expect theta r>0.95, sigma_v r>0.8, kappa r>0.6, rho r>0.5; mu weakly identified — report whatever it is). (2) `python scripts/calibrate_universe.py --universe sp500 --models heston_npe --run-id sp500-npe` (lake warm; inference should be seconds). Report wall-clock split train vs infer, convergence count. (3) Compare per-name posterior means vs the sp500-cross heston_qmle parquet: rank correlations per param (expect theta/v0 high, kappa moderate). (4) PF cross-validation: pick 8 names spread across sectors; run the existing scipy HestonParticleFilter with the NPE posterior-mean params vs the QMLE params on each name's return series; report per-name PF log-likelihoods — NPE params should achieve ≥ QMLE params' PF logL on the majority of names (this is the fidelity claim; report honestly either way). (5) Append `## Phase 4 verification` to `.superpowers/sdd/progress.md`; commit nothing except (optionally) the report notes file.

---

## Self-Review Notes
- Amended-spec coverage: 4a OHLC wiring ✅ T1; 4b NPE (simulate→train→amortized universe calibration, posterior widths available for pooling) ✅ T2-T5; 4c PF cross-validation ✅ T5. rBergomi NPE explicitly deferred (documented stretch; same machinery applies — the simulator module is the only swap).
- Judgment calls: MDN over normalizing flow (v1, zero new deps, documented upgrade path); summary statistics over raw-path encoders (robustness + train/infer symmetry via one shared featurizer); posterior moments via sampling (exact for MDN would be closed-form for means — sampling keeps rho/positivity transforms honest).
