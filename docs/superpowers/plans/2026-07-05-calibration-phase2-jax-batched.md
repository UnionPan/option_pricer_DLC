# Calibration Phase 2: JAX Batched Calibrators — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** JAX `fit_batch` implementations of GBM, OU, Heston QMLE (+ Garman-Klass OHLC proxy), rBergomi variogram, GARCH(1,1) and Merton, vmapped over assets on the GPU; runner dispatches to the batch path when a model provides it. The whole S&P 500 calibrates in one compiled call per model.

**Architecture:** New package `src/options_desk/calibration/physical/batched/` — one module per model exposing `fit_batch(returns (N,T) or prices (N,T+1), mask, dt) -> dict[str, np.ndarray]`. A shared `common.py` holds padding utilities and the masked-statistics helpers. `ModelSpec` gains an optional `fit_batch` field; the runner uses it when present (padded arrays, one call), else falls back to the joblib per-asset path. Existing scipy calibrators are the normative references: every batched model ships a parity test against its scipy twin and a parameter-recovery test on simulated data.

**Tech Stack:** jax 0.9.2 (CUDA), optax 0.2.8 (Adam multi-start for GARCH/Merton; no jaxopt/optimistix in env), numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-07-05-universe-scale-p-measure-calibration-design.md` (Layer 2)

## Global Constraints

- Env: ALWAYS `source /home/union/miniconda3/etc/profile.d/conda.sh && conda activate options-desk`. Run pytest from repo root. Tests must not touch the network.
- Tests run JAX on CPU deterministically: set `JAX_PLATFORMS=cpu` inside test files via `os.environ.setdefault("JAX_PLATFORMS", "cpu")` BEFORE importing jax (keeps CI/GPU-free correctness; GPU is exercised in the Task 8 benchmark).
- All fit_batch functions: pure, `jax.jit`-compatible, float64 disabled (float32 default) EXCEPT where parity demands float64 — enable per-module with `jax.config.update("jax_enable_x64", True)` is FORBIDDEN (global); instead do the final statistics in numpy float64 where needed. Parity tolerances are chosen accordingly and stated per task.
- Scipy calibrators are normative. If a parity test fails, fix the JAX side to match the scipy convention (read the scipy file), never the reverse. Do not modify any existing calibrator.
- Contract (all models): inputs `returns: (N, T) float32` (log-returns, padded with 0.0), `mask: (N, T) float32` (1=valid), `dt: float`. Heston GK variant additionally takes OHLC arrays. Output: plain dict of numpy arrays, each shape (N,), including `converged (bool)` and `log_likelihood` where defined. Padding must NOT affect results (masked statistics only).
- Stage files explicitly per task; never `git add -A`; no `__pycache__`. Commit after each task with trailer:
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
- Follow the working pattern of Phase 1: TDD per task (write tests → fail → implement → pass → commit).

---

### Task 1: Batched package scaffolding + masked helpers + batched GBM

**Files:**
- Create: `src/options_desk/calibration/physical/batched/__init__.py`
- Create: `src/options_desk/calibration/physical/batched/common.py`
- Create: `src/options_desk/calibration/physical/batched/gbm.py`
- Test: `tests/test_batched_gbm.py`

**Interfaces (produced, used by all later tasks):**
```python
# common.py
def pad_returns(returns_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """list of 1-D float arrays -> (returns (N,T), mask (N,T)) float32,
    right-padded with zeros; T = max length."""
def masked_mean(x, mask, axis=-1): ...      # jnp; sum(x*mask)/sum(mask)
def masked_var(x, mask, axis=-1): ...       # MLE variance (ddof=0) under mask
# gbm.py
def fit_batch(returns: jnp.ndarray, mask: jnp.ndarray, dt: float) -> dict:
    """keys: mu, sigma, log_likelihood, n_observations, converged"""
```

**Math (must match `GBMCalibrator.fit` conventions — read `physical/gbm_calibrator.py` first and reconcile):** with r̄ = masked mean, s² = masked MLE variance of log-returns: `sigma = sqrt(s²/dt)`, `mu = r̄/dt + 0.5·sigma²`, `log_likelihood = -0.5·n·(log(2π) + log(s²)) - 0.5·n`, `converged = n >= 2`.

**Tests (write first):**
```python
# tests/test_batched_gbm.py
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import pytest

from options_desk.calibration.physical.batched.common import pad_returns
from options_desk.calibration.physical.batched import gbm as bgbm
from options_desk.calibration.physical.gbm_calibrator import GBMCalibrator


def _paths(n_assets=8, n=1200, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_assets):
        mu, sig = 0.02 + 0.02 * i, 0.10 + 0.03 * i
        dt = 1 / 252
        r = (mu - 0.5 * sig**2) * dt + sig * np.sqrt(dt) * rng.standard_normal(n)
        out.append(100 * np.exp(np.cumsum(r)))
    return out


def test_parity_with_scipy_gbm():
    prices_list = _paths()
    rets = [np.diff(np.log(p)) for p in prices_list]
    R, M = pad_returns(rets)
    out = bgbm.fit_batch(R, M, 1 / 252)
    for i, p in enumerate(prices_list):
        ref = GBMCalibrator().fit(p, dt=1 / 252)
        assert out["mu"][i] == pytest.approx(ref.mu, rel=1e-4, abs=1e-6)
        assert out["sigma"][i] == pytest.approx(ref.sigma, rel=1e-4)
        assert out["log_likelihood"][i] == pytest.approx(ref.log_likelihood, rel=1e-3)


def test_padding_does_not_leak():
    rets = [np.full(500, 0.001), np.full(900, -0.0005)]
    R, M = pad_returns(rets)
    out = bgbm.fit_batch(R, M, 1 / 252)
    R2, M2 = pad_returns([rets[0]])
    out2 = bgbm.fit_batch(R2, M2, 1 / 252)
    assert out["mu"][0] == pytest.approx(out2["mu"][0], rel=1e-6)


def test_recovery():
    prices_list = _paths(n_assets=4, n=100_000, seed=1)
    rets = [np.diff(np.log(p)) for p in prices_list]
    R, M = pad_returns(rets)
    out = bgbm.fit_batch(R, M, 1 / 252)
    for i in range(4):
        assert out["sigma"][i] == pytest.approx(0.10 + 0.03 * i, rel=0.02)
```

Implementation sketch (complete it, keep it ~this small):
```python
# batched/gbm.py
import jax, jax.numpy as jnp, numpy as np
from .common import masked_mean, masked_var

@jax.jit
def _fit(returns, mask, dt):
    n = mask.sum(-1)
    rbar = masked_mean(returns, mask)
    s2 = masked_var(returns, mask)
    sigma = jnp.sqrt(s2 / dt)
    mu = rbar / dt + 0.5 * sigma**2
    ll = -0.5 * n * (jnp.log(2 * jnp.pi) + jnp.log(s2)) - 0.5 * n
    return mu, sigma, ll, n

def fit_batch(returns, mask, dt):
    mu, sigma, ll, n = _fit(jnp.asarray(returns), jnp.asarray(mask), dt)
    return {"mu": np.asarray(mu, np.float64), "sigma": np.asarray(sigma, np.float64),
            "log_likelihood": np.asarray(ll, np.float64),
            "n_observations": np.asarray(n, np.int64),
            "converged": np.asarray(n >= 2)}
```
If parity fails on float32 precision, compute the masked stats in numpy float64 directly (this model is closed-form; jit is optional). Commit: `feat(calibration): batched JAX GBM + masked helpers`.

---

### Task 2: Batched OU (exact MLE via AR(1) closed form)

**Files:** Create `batched/ou.py`; Test `tests/test_batched_ou.py`.

Read `physical/ou_calibrator.py` `method='exact_mle'` first; replicate its estimator. Core: regress x_{t+1} on x_t (masked, closed-form OLS) → a, b, resid var s²; then `kappa = -log(b)/dt`, `theta = a/(1-b)`, `sigma = s·sqrt(2·kappa/(1-b²))` (reconcile exact formulas/log-lik with the scipy file — it is normative). fit_batch input here is the LEVEL series (N,T) + mask, not returns; add `pad_levels = pad_returns` alias usage. Output keys: `kappa, theta, sigma, log_likelihood, half_life, n_observations, converged` (match the scipy result fields that are scalars).

Tests mirror Task 1: (a) parity vs `OUCalibrator.fit(series, dt, method='exact_mle')` on 6 simulated OU paths (simulate with exact discretization: x_{t+1} = θ+(x_t−θ)e^{−κdt} + σ·sqrt((1−e^{−2κdt})/(2κ))·z), rel=1e-3; (b) padding-no-leak; (c) recovery κ within 15% on T=50_000. Commit: `feat(calibration): batched JAX OU exact MLE`.

---

### Task 3: Batched Heston QMLE + Garman-Klass OHLC proxy

**Files:** Create `batched/heston_qmle.py`; Test `tests/test_batched_heston_qmle.py`.

Read `physical/heston_qmle.py` FIRST — it is normative for the close-close mode. Replicate its exact pipeline in vectorized JAX/numpy: (1) RV proxy = rolling mean over `smooth_window=10` of squared returns / dt (masked rolling via convolution with a ones-kernel normalized by valid counts); (2) AR(1) step on v_t (masked OLS + its likelihood evaluation, matching the scipy optimizer's objective — if scipy optimizes, initialize from the closed-form OLS and refine with 200 Adam steps (optax, lr=1e-3) on the same negative log-likelihood, vmapped); (3) rho = masked corr(r_t, Δv_t); (4) mu = mean(r)/dt, v0 = last RV. Output keys: `kappa, theta, sigma_v, rho, mu, v0, log_likelihood, feller_ratio, variance_proxy_r2, converged, n_observations`.

**GK variant** (new capability, additive): `fit_batch_ohlc(open_, high, low, close, mask, dt, proxy="garman_klass")` — replaces step (1)'s squared-return proxy with the Garman-Klass estimator per bar: `σ²_GK = 0.5·ln(H/L)² − (2ln2−1)·ln(C/O)²`, divided by dt, then the same smoothing/AR(1)/rho/mu steps on that proxy.

Tests: (a) parity close-close vs `HestonQMLECalibrator(smooth_window=10).fit` on 5 simulated Heston paths (Euler, dt=1/252, T=2000; params κ=3, θ=0.04, ξ=0.4, ρ=−0.6, μ=0.05) — rel=5e-3 on kappa/theta/sigma_v, abs=0.05 on rho; (b) padding-no-leak; (c) GK recovery: simulate Heston on a fine grid (64 sub-steps/day), aggregate each day to OHLC, run fit_batch_ohlc, assert theta within 25% of true θ and the GK-based theta estimate is at least as close to truth as the close-close one on the same data (GK is the better proxy — that's the point). Commit: `feat(calibration): batched Heston QMLE with Garman-Klass OHLC proxy`.

---

### Task 4: Batched rBergomi variogram

**Files:** Create `batched/rbergomi.py`; Test `tests/test_batched_rbergomi.py`.

Read `physical/rough_bergomi_calibrator.py` first (window=20, max_lag=10 defaults); replicate: rolling RV (window) → log-RV variogram over lags 1..max_lag → OLS of log E[(log v_{t+Δ} − log v_t)²] on log Δ → slope = 2H, intercept → η; ξ₀ from mean RV. Vectorize lags with a fixed (max_lag,) axis; masked means throughout. Output keys: `hurst, eta, xi0, converged, n_observations` (reconcile names with the scipy result dataclass — normative).

Tests: (a) parity vs scipy on 5 simulated rBergomi-ish paths (simulate log-vol as fBm via Cholesky on T=1500 — a helper in the test file, ~15 lines: covariance C(s,t)=0.5·(s^{2H}+t^{2H}−|t−s|^{2H})); H recovery within 0.1 abs for H∈{0.1, 0.3}; (b) padding-no-leak. Commit: `feat(calibration): batched rBergomi variogram estimator`.

---

### Task 5: Batched GARCH(1,1) QMLE (optax Adam multi-start)

**Files:** Create `batched/garch.py`; Test `tests/test_batched_garch.py`.

Read `physical/garch_calibrator.py` first (its likelihood is normative). Parameterize unconstrained: `omega = softplus(a)`, `(alpha, beta) = sigmoid pair scaled so alpha+beta<0.999` (e.g. alpha = 0.999·sigmoid(b)·sigmoid(c), beta = 0.999·sigmoid(b)·(1−sigmoid(c))). Likelihood: `lax.scan` GARCH recursion σ²_t = ω + α·r²_{t−1} + β·σ²_{t−1}, Gaussian NLL, masked. Optimize with optax.adam(1e-2), 800 steps, from a 4-point start grid ((α,β) ∈ {(0.05,0.90),(0.10,0.85),(0.02,0.95),(0.08,0.80)}, ω matched to sample variance), vmapped over (starts × assets); take best final NLL per asset. `converged = isfinite(best_nll)`.

Tests: (a) simulate 6 GARCH paths (ω=2e-6, α=0.08, β=0.90, T=4000): recovery α within ±0.04, β within ±0.05, and **achieved log-likelihood ≥ scipy GARCHCalibrator's logL − 0.5** per asset (likelihood parity is the robust criterion; parameters may differ within tolerance); (b) padding-no-leak; (c) speed sanity: 6 assets fit in <60 s on CPU. Commit: `feat(calibration): batched GARCH(1,1) QMLE via optax multi-start`.

---

### Task 6: Batched Merton jump-diffusion MLE

**Files:** Create `batched/merton.py`; Test `tests/test_batched_merton.py`.

Read `physical/merton_calibrator.py` first (truncated Poisson-mixture likelihood, k_max=5 — normative). Params (unconstrained transforms): mu, sigma=softplus, lam=softplus (jump intensity), mu_j, sigma_j=softplus. NLL: `logsumexp` over k=0..k_max of Poisson(k;λdt) · N(r; (mu−0.5σ²)dt + k·mu_j, σ²dt + k·σ_j²) — fully vectorized over (assets, T, k). optax.adam(5e-3), 1000 steps, 3 starts (λ ∈ {5, 20, 60}/yr), best NLL per asset.

Tests: (a) simulate 4 Merton paths (T=6000, μ=0.05, σ=0.15, λ=10/yr, μ_j=−0.02, σ_j=0.04): recovery σ within 15%, λ within factor 2; achieved logL ≥ scipy's − 0.5; (b) padding-no-leak. Commit: `feat(calibration): batched Merton jump-diffusion MLE`.

---

### Task 7: Registry + runner batch path

**Files:** Modify `pipeline/registry.py` (add `fit_batch: BatchFitFn | None = None` to ModelSpec + register batched impls under the SAME names), `pipeline/runner.py` (batch dispatch), `physical/batched/__init__.py` (exports); Test `tests/test_runner_batch.py`.

Contract:
```python
BatchFitFn = Callable[[list[np.ndarray]], dict]   # per-model adapter:
# takes the ordered list of per-ticker price arrays (None already filtered),
# handles pad/returns internally, returns dict of (N,) numpy arrays
```
Registry: for gbm/garch/heston_qmle (and new registrations `ou`, `merton`, `rbergomi` with per-asset scipy fit fns as the fallback path + batched fit_batch), attach adapters that (a) compute log-returns (levels for OU) per ticker, (b) pad, (c) call the module fit_batch, (d) return column dict. Runner: if `spec.fit_batch is not None`, split tickers into (valid ≥ min_obs) and (invalid → insufficient-data rows); one fit_batch call for valid; assemble the same row schema as the joblib path (ticker, error="", converged, params..., sector, calibration_date). Env flag `OPTIONS_DESK_NO_BATCH=1` forces the joblib path (escape hatch + test lever).

Tests: (a) runner with model gbm on the Phase 1 synthetic-store fixture produces the SAME converged tickers and mu/sigma within rel 1e-3 whether OPTIONS_DESK_NO_BATCH is set or not; (b) invalid tickers still get insufficient-data rows in batch mode; (c) a fit_batch raising → runner falls back to joblib path for that model with a logged warning (wrap the batch call in try/except). Commit: `feat(calibration): runner batch dispatch for JAX calibrators`.

---

### Task 8: GPU benchmark + verification (manual, network OK)

No new code. (1) `python scripts/calibrate_universe.py --universe sp500 --models gbm ou heston_qmle rbergomi garch merton --run-id sp500-batched` on the GPU (default env). Price lake is already warm → fetch ≈ 0. (2) Record per-model wall-clock from the run log; compare gbm/heston_qmle medians (kappa/theta/sigma_v/mu/sigma) against the Phase 1 `sp500-first` run parquets — assert medians agree within 10% (report, don't gate hard). (3) Append a `## Phase 2 benchmark` section to `.superpowers/sdd/progress.md` with the table (model × wall-clock × convergence%). Target: all six models over 503 names in well under a minute of calibrate time total. If GARCH/Merton exceed ~2 min on GPU, note it — acceptable, they're iterative.

---

## Self-Review Notes
- Spec coverage: all six Layer-2 models ✅ (T1-T6), GK proxy (Phase-4a pulled forward per amendment) ✅ T3, runner GPU path ✅ T7, parity+recovery test families ✅ every task, benchmark vs Phase 1 baseline ✅ T8. NPE is Phase 4, not here.
- The scipy-is-normative rule + likelihood-parity criterion for optimizer-based models (GARCH/Merton) avoids brittle parameter-equality tests.
- Known judgment call: batch path falls back to joblib on any batch exception — keeps the CLI robust while the JAX path matures.
