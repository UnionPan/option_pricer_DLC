# Phase 4 Merton Regression Diagnosis

## Symptom

`tests/test_batched_merton.py::test_merton_recovery` passed at merge commit
`16327ae` (~174 s for the full 3-test file) but later **failed** with an
`AssertionError` on the log-likelihood parity gate and took ~22 minutes for
the single recovery test.

## Root Cause

The original `_make_merton_nll_fn` used **nested `jax.vmap`** to compute the
Poisson-mixture NLL: an outer vmap over T=6000 returns, each containing an
inner vmap over K=6 mixture components.  Combined with `lax.scan` of 1000
Adam steps (each calling `value_and_grad` of that NLL), the XLA program was
enormous.

Under **JAX 0.9.2** the XLA compiler's handling of nested-vmap-inside-scan
regressed dramatically: compilation of this pattern went from ~2 minutes
(JAX 0.4-0.5 era) to **10-20 minutes**.  Since `fit_batch` additionally
vmapped the optimizer over 3 starting points and 4 assets, the total
compilation exploded to 22+ minutes.

The **assertion failure** was a secondary consequence: the nested-vmap NLL
and the equivalent broadcast-array NLL produce identical function values
(verified to exact float32 match), but their autodiff graphs reduce sums in
different orders, yielding slightly different gradients.  This caused the
Adam optimizer to follow a marginally different trajectory.  At 1000 steps
the broadcast path converged to logL=18 632.88 vs. the scipy reference's
18 633.55 (diff = −0.67, failing the −0.5 gate).  At 1500 steps it reaches
18 633.45 (diff = −0.10, comfortably passing).

## Fix (single file: `batched/merton.py`)

1. **Replaced nested `jax.vmap` NLL with array broadcasting** over `(T, K)`.
   XLA program shrinks from O(T·K) traced ops to a handful of broadcasts +
   one `logsumexp`.  Compilation: **< 1 s** (vs 10+ minutes).

2. **Replaced `jax.vmap` over assets and starts with a Python loop**.  The
   `@jax.jit`-compiled single-start function compiles once and is reused for
   all (asset, start) pairs with the same T dimension.

3. **Increased Adam steps from 1000 to 1500** to compensate for the slightly
   different gradient-reduction ordering in the broadcast NLL.  Verified
   that 1500 steps produce logL within 0.1 of scipy for all 4 test seeds.

4. **Module-level JIT cache** (`_FIT_SINGLE_START_CACHE`) so repeated
   `fit_batch` calls with the same `k_max` skip recompilation entirely.

## Timing

| Metric                  | Before (nested vmap) | After (broadcast)  |
|-------------------------|---------------------:|-------------------:|
| XLA compilation         | 10-20 min            | < 1 s              |
| `fit_batch` (4 assets)  | ~22 min              | **3.7 s**          |
| `test_merton_recovery`  | ~22 min (FAIL)       | ~20 min* (PASS)    |

\* The ~20 min remaining runtime is entirely the **scipy reference**
`MertonJumpCalibrator.fit()` which uses a pure-Python loop over 6000
returns.  The JAX portion is 3.7 s.

## Verification

- `test_fixed_params_nll_isolation`: **PASS** (10.6 s)
- `test_padding_does_not_leak`: **PASS** (8.9 s)
- `test_merton_recovery` JAX side: all 4 assets pass sigma (< 15%),
  lambda (factor 2), and logL (> scipy − 0.5) checks.  Full test
  runtime is dominated by scipy reference fits.
- NLL numerical equivalence: original nested-vmap and new broadcast NLL
  produce **identical float32 values** at the same parameter point.
