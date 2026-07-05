# Universe-Scale P-Measure Calibration — Design

**Date:** 2026-07-05
**Status:** Approved
**Goal:** Scale the physical-measure (P) calibration module from a 50-name,
single-model pipeline to 1000–3000+ US equities, with both single-asset and
cross-asset calibration, on a single machine with a CUDA GPU.

## Context and motivation

The current stack (`src/options_desk/calibration/`) has a strong single-asset
calibrator ladder (GBM MLE, GARCH(1,1) QMLE, OU, Merton jump-diffusion,
Heston QMLE, rough Bergomi variogram, Heston/rBergomi particle filters,
regime switching) but does not scale:

1. `multi_asset_pipeline.py` is hardcoded to Heston QMLE over a 50-name
   basket; other models have no multi-asset path.
2. Price data is fetched per-ticker from yfinance with no local cache; a
   3000-name universe would refetch everything every run and hit rate limits.
3. All calibrators are per-asset scipy/numpy loops. The particle filters cost
   5–20 s/asset (hours for the universe on CPU).
4. Cross-asset support is limited to static correlation matrices
   (`correlation.py`: sample/EWMA/Ledoit-Wolf/RMT). At N=3000 with T≈1250
   daily observations, N > T: the sample estimator is rank-deficient and a
   factor structure is required.
5. No checkpoint/resume, no incremental runs, no unified results store.

## Decisions (from design discussion)

- **Data source:** yfinance + a local parquet cache. The fetcher sits behind
  a store interface so a paid bulk API can be swapped in later.
- **Model set at universe scale:** all of it — fast core (GBM, GARCH, Heston
  QMLE), moderate (Merton, OU, rBergomi variogram), and the particle filters
  (feasible only via GPU batching).
- **Cross-asset deliverables:** robust factor covariance at N=3000,
  statistical factor model (PCA/POET), DCC-GARCH on factor returns, and
  hierarchical (empirical-Bayes) pooling of per-name parameters.
- **Compute target:** single machine, JAX on the local CUDA GPU. No
  distributed infrastructure (Ray/Dask rejected as unnecessary at this scale).
- **Approach:** JAX batched core under a layered pipeline (Approach B),
  built in four phases. Incremental scipy-only extension (Approach A) was
  rejected because it cannot deliver particle filters at scale.

## Architecture

Four layers. New code lives beside the existing module; existing scipy
calibrators are untouched and become golden references.

```
src/options_desk/calibration/
  data/
    price_store.py          # parquet price lake + incremental fetcher
    universe.py             # named universes + sector metadata
  physical/batched/         # JAX batched calibrators (new)
    gbm.py, ou.py, heston_qmle.py, rbergomi.py, garch.py, merton.py
    particle/heston_pf.py, rbergomi_pf.py
  cross_asset/              # new package
    factor_model.py         # PCA/POET factor covariance
    dcc.py                  # DCC-GARCH on factor returns
    pooling.py              # empirical-Bayes hierarchical shrinkage
  pipeline/                 # new package
    registry.py, runner.py, results_store.py
scripts/calibrate_universe.py   # upgraded CLI
```

### Layer 1 — Data

**`PriceStore`** — a local parquet price lake under `data/price_lake/`
(columns: date, ticker, open, high, low, close, adj_close, volume;
partitioned by ticker prefix).

- `ensure(tickers, start, end)` — fetch only missing (ticker, date-range)
  slices. Fetching uses yfinance's native multi-ticker `yf.download` in
  chunks of ~100 tickers per call, with exponential-backoff retries and
  polite rate limiting. First full fill of 3000 names ≈ 30–60 min; daily
  incremental updates run in seconds.
- `returns_matrix(tickers, start, end, min_obs)` — aligned `(T, N)` float32
  log-return matrix plus a validity mask, built once and shared by every
  downstream consumer.

**`universe.py`** — named universes as CSVs in `data/universes/` (initially
S&P 500, S&P 1500, and a Russell-3000 approximation list), each with GICS
sector metadata fetched once via yfinance and cached. Sector labels feed the
hierarchical pooling stage.

**NaN policy (explicit and reported):** a name needs ≥ 2 years of history;
gaps ≤ 5 trading days are forward-filled; anything worse excludes the name,
and every exclusion is recorded in the run report with a reason.

### Layer 2 — Batched JAX calibrators

Common contract, one per model:

```python
def fit_batch(returns: (N, T), mask: (N, T), dt: float) -> BatchResult
# BatchResult: per-parameter arrays (N,), log_likelihood (N,),
#              converged (N,) bool, model-specific diagnostics
```

Pure JAX functions, `jit`-compiled, `vmap`-ed over the asset axis. (The
data layer emits `(T, N)`; the runner transposes once so calibrators vmap
over a leading asset axis.)

- **GBM, OU (exact MLE), Heston QMLE, rBergomi variogram:** closed-form /
  method-of-moments computations — direct vectorization; the whole universe
  calibrates in one GPU call per model.
- **GARCH(1,1) QMLE, Merton MLE:** the likelihood recursion is a `lax.scan`;
  optimization is a jit'd L-BFGS (`optimistix` or `jaxopt`) vmapped across
  assets, with multi-start from a small grid of initial values, taking the
  best likelihood per asset.
- **Particle filters (Heston, rBergomi):** `lax.scan` over time, `vmap` over
  assets, particles as an inner vectorized dimension; systematic resampling;
  float32 throughout. GPU memory scales with N × n_particles, so the runner
  chunks assets (~256 per chunk, tunable). Target: universe-scale particle
  filtering in minutes instead of CPU-hours.
- **Validation:** the existing scipy calibrators are the golden references.
  Two test families per model: (a) parity — JAX vs scipy on the same
  synthetic series within documented tolerances; (b) recovery — simulate
  from known parameters, assert the estimator recovers them within
  statistical error.

### Layer 3 — Cross-asset

- **Factor covariance (`factor_model.py`)** — PCA on standardized returns;
  the number of factors k is chosen by the Marchenko-Pastur edge (reusing
  the existing RMT machinery in `correlation.py`); residual covariance is
  POET-style sparse-thresholded. Output Σ = B Ω Bᵀ + D is positive definite
  and well-conditioned at N = 3000 > T ≈ 1250. Date-stamped covariance
  snapshots are persisted each run. This is the workhorse cross-asset
  deliverable.
- **DCC-GARCH (`dcc.py`)** — estimated on the k ≈ 10–50 factor returns, not
  the raw names. Standard two-step: univariate GARCH per factor (reusing the
  batched GARCH), then DCC(1,1) on standardized residuals via a jit'd
  optimizer. Output: DCC parameters plus the time series of factor
  correlation matrices.
- **Hierarchical pooling (`pooling.py`)** — empirical-Bayes shrinkage of
  per-name parameter estimates (Heston κ/θ/ρ, GARCH α/β, Merton jump
  intensity) toward sector-level means, with James-Stein-style weights from
  per-name estimation variance vs cross-sectional dispersion. Outputs both
  raw and shrunk parameter tables. This directly addresses the documented
  noisiness of per-name κ and ρ in the Heston QMLE.

### Layer 4 — Orchestration

- **Registry** — model name → batched fit function + output schema. Adding a
  model is one registration.
- **Runner** — load universe → `ensure` data → build returns matrix → run
  selected models on GPU → cross-asset stage → persist. Per-asset failures
  isolate to `converged=False` rows; a failing name never crashes the run.
- **Results store** — `runs/calibration/<UTC-date>/` parquet, partitioned by
  model, plus a run-manifest JSON (universe hash, data range, model list,
  timings, exclusions and reasons, package versions).
- **Checkpoint/resume** — per-model completion markers; per-asset-chunk
  checkpoints for the particle filters; re-invoking the CLI on an
  interrupted run resumes where it stopped.
- **CLI** — upgraded `scripts/calibrate_universe.py`:
  `--universe sp1500 --models gbm garch heston_qmle hpf
  --cross-asset factor dcc pooling [--resume RUN_ID]`.

## Error handling

- Data: outlier detection and cleaning reuse `data_utils.py`; QC failures
  are exclusions with reasons, not crashes.
- Calibration: non-convergence, Feller violations, and boundary-pinned
  parameters are flagged per name in the output schema.
- GPU: OOM is avoided by asset chunking with a tunable chunk size; chunk
  size failures fall back to smaller chunks.

## Testing

- Unit: parity + parameter-recovery tests per batched calibrator (as above).
- Cross-asset: PD and conditioning assertions on factor covariance;
  DCC on simulated data with known parameters; pooling shrinkage weights
  sanity checks (weights → 1 as per-name noise → ∞).
- Integration: a ~20-name end-to-end run against a committed price fixture
  (no network in tests).

## Phasing

Each phase is independently useful and gets its own implementation plan:

1. **Data layer + orchestration skeleton** — PriceStore, universes,
   registry/runner/results store wired to the *existing scipy* calibrators.
   Immediately unlocks 1000+ name runs for the fast models via joblib.
2. **JAX fast core** — batched GBM, OU, Heston QMLE, rBergomi variogram,
   GARCH, Merton + parity/recovery tests; runner switches these models to
   the GPU path.
3. **Cross-asset suite** — factor covariance, hierarchical pooling,
   DCC-GARCH on factors.
4. **Particle filters on GPU** — batched Heston and rBergomi particle
   filters with asset chunking and chunk checkpoints.

## Out of scope

- Distributed compute (Ray/Dask), databases (DuckDB/Timescale) — single
  machine + parquet is sufficient at this scale.
- Intraday data and intraday recalibration.
- Q-measure (option-surface) calibration changes; joint P/Q comes later and
  builds on this foundation.
- Portfolio construction / downstream consumers of the covariance.
