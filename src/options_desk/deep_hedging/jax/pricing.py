"""
JAX-native option grid pricing for the deep hedging environment.

Prices the entire FloatingOptionGrid at each time step using the COS method,
staying entirely in JAX arrays (no to_numpy() calls) so the result can feed
directly into a ``lax.scan`` rollout body.

The grid is pre-compiled once via ``compile_padded_grid()`` into fixed-shape
arrays (padded to max_strikes_per_maturity) so that ``vmap`` over maturities
and the enclosing ``lax.scan`` both see static shapes.

author: Yunian Pan
email: yp1170@nyu.edu
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from options_desk.pricer._jax_fourier_pricer import (
    HestonCFParams,
    _chi_k,
    _cos_truncation_range,
    _jax_cos_price_multi,
    _psi_k,
    jax_heston_cf,
)
from options_desk.processes._jax_backend import configure_jax_runtime

from .env import DeepHedgingEnvConfig, FloatingOptionGrid

configure_jax_runtime()


# ============================================================================
# Market parameter container
# ============================================================================

class HestonMarketParams(NamedTuple):
    """Risk-neutral Heston parameters for option pricing within the env."""
    kappa: float
    theta: float
    sigma_v: float
    rho: float
    r: float       # risk-free rate
    q: float       # dividend yield


# ============================================================================
# Padded grid for JIT-compatible fixed-shape arrays
# ============================================================================

class _PaddedGrid(NamedTuple):
    """Pre-compiled fixed-shape arrays for JIT-compatible grid pricing.

    All arrays have the maturity dimension as axis 0 (length = n_maturities)
    and the strike/moneyness dimension padded to ``max_strikes`` (axis 1).
    Invalid entries are masked with ``moneyness_mask``.

    Instrument layout in the env price vector:
        [underlying, call_m0_k0, put_m0_k0, call_m0_k1, put_m0_k1, ...]
    So maturity ``m`` with ``K`` strikes occupies 2*K slots starting at
    ``instrument_offset[m]``.
    """
    maturities_steps: jnp.ndarray     # (n_maturities,) int — maturity in env steps
    maturities_years: jnp.ndarray     # (n_maturities,) float — maturity in years
    moneyness: jnp.ndarray            # (n_maturities, max_strikes) — padded moneyness
    moneyness_mask: jnp.ndarray       # (n_maturities, max_strikes) bool — True = valid
    instrument_offset: jnp.ndarray    # (n_maturities,) int — start index in price vector
    n_instruments: int                # total instruments (1 + 2 * total_strikes)
    max_strikes: int                  # max strikes per maturity (padding width)
    n_maturities: int

    # Flat instrument table (scatter-free pricing) — see compile_padded_grid
    # NOTE: bool arrays are stored as float32 (1.0/0.0) because the MPS
    # backend corrupts bool dtype inside JIT closures.
    flat_maturity_idx: jnp.ndarray    # (n_options,) int — maturity index for each option
    flat_moneyness: jnp.ndarray       # (n_options,) — moneyness for each option
    flat_is_call: jnp.ndarray         # (n_options,) float32 — 1.0 for call, 0.0 for put
    flat_valid: jnp.ndarray           # (n_options,) float32 — 1.0 for real instrument

    # Pre-built action masks — depends only on (grid, horizon), not state
    # Shape: (horizon+1, n_instruments) bool.  Built once by compile_padded_grid.
    action_masks: np.ndarray | None   # None when horizon unknown at compile time


def compile_padded_grid(
    grid: FloatingOptionGrid,
    dt: float,
    horizon_steps: int | None = None,
) -> _PaddedGrid:
    """Convert a ``FloatingOptionGrid`` into padded arrays for JIT pricing.

    Called once at environment construction time — not inside a scan body.

    Also builds a flat instrument table for scatter-free pricing: each of
    the ``n_options`` option slots maps to (maturity_index, moneyness, is_call).

    If ``horizon_steps`` is provided, pre-computes action masks for the
    entire horizon so they don't need to be rebuilt every simulation call.
    """
    maturities = list(grid.maturities)
    n_maturities = len(maturities)
    max_strikes = max(len(grid.moneyness_by_maturity[m]) for m in maturities)

    mat_steps = np.array(maturities, dtype=np.int32)
    mat_years = mat_steps.astype(np.float64) * dt

    moneyness_arr = np.zeros((n_maturities, max_strikes), dtype=np.float64)
    mask_arr = np.zeros((n_maturities, max_strikes), dtype=bool)

    offset = 1  # instrument 0 is the underlying
    offsets = np.zeros(n_maturities, dtype=np.int32)

    # Build flat instrument table (excluding underlying at index 0)
    # Store booleans as float32 — MPS corrupts bool in JIT closures
    n_options = grid.n_options  # 2 * total_strikes
    flat_mat_idx = np.zeros(n_options, dtype=np.int32)
    flat_money = np.zeros(n_options, dtype=np.float64)
    flat_call = np.zeros(n_options, dtype=np.float32)
    flat_valid = np.zeros(n_options, dtype=np.float32)

    flat_pos = 0
    for i, m in enumerate(maturities):
        strikes_m = list(grid.moneyness_by_maturity[m])
        n_k = len(strikes_m)
        moneyness_arr[i, :n_k] = strikes_m
        mask_arr[i, :n_k] = True
        offsets[i] = offset
        for k_idx in range(n_k):
            # call
            flat_mat_idx[flat_pos] = i
            flat_money[flat_pos] = strikes_m[k_idx]
            flat_call[flat_pos] = 1.0
            flat_valid[flat_pos] = 1.0
            flat_pos += 1
            # put
            flat_mat_idx[flat_pos] = i
            flat_money[flat_pos] = strikes_m[k_idx]
            flat_call[flat_pos] = 0.0
            flat_valid[flat_pos] = 1.0
            flat_pos += 1
        offset += 2 * n_k

    # Pre-compute action masks if horizon is known
    masks = None
    if horizon_steps is not None:
        from .env import build_action_mask
        masks = np.stack([
            build_action_mask(grid, t, horizon_steps)
            for t in range(horizon_steps + 1)
        ], axis=0)

    return _PaddedGrid(
        maturities_steps=jnp.array(mat_steps),
        maturities_years=jnp.array(mat_years),
        moneyness=jnp.array(moneyness_arr),
        moneyness_mask=jnp.array(mask_arr),
        instrument_offset=jnp.array(offsets),
        n_instruments=grid.n_instruments,
        max_strikes=max_strikes,
        n_maturities=n_maturities,
        flat_maturity_idx=jnp.array(flat_mat_idx),
        flat_moneyness=jnp.array(flat_money),
        flat_is_call=jnp.array(flat_call),
        flat_valid=jnp.array(flat_valid),
        action_masks=masks,
    )


# ============================================================================
# Single-maturity pricing slice (vmap target)
# ============================================================================

def _price_maturity_slice(
    spot: jnp.ndarray,
    v0: jnp.ndarray,
    T: jnp.ndarray,
    moneyness_row: jnp.ndarray,
    mask_row: jnp.ndarray,
    market: HestonMarketParams,
    N_cos: int,
    L_cos: float,
) -> jnp.ndarray:
    """Price calls and puts for one maturity, returning interleaved prices.

    Args:
        spot: scalar JAX array — current spot price
        v0: scalar JAX array — current instantaneous variance
        T: scalar JAX array — time to maturity in years
        moneyness_row: (max_strikes,) — relative strikes (padded)
        mask_row: (max_strikes,) bool — True for valid strikes
        market: Heston risk-neutral parameters
        N_cos: number of COS expansion terms
        L_cos: COS truncation parameter

    Returns:
        (2 * max_strikes,) — interleaved [call_0, put_0, call_1, put_1, ...]
        Invalid entries (from padding) are zeroed out.
    """
    mu = market.r - market.q
    log_S0 = jnp.log(spot)
    sigma_approx = jnp.sqrt(jnp.maximum(v0, 1e-8))

    # Truncation range for COS (stays in JAX, no Python float conversion)
    c1 = log_S0 + (mu - 0.5 * v0) * T
    c2 = jnp.maximum(v0 * T, 1e-12)
    a = c1 - L_cos * jnp.sqrt(c2)
    b = c1 + L_cos * jnp.sqrt(c2)

    # CF on the k-grid (shared across all strikes in this maturity)
    k_grid = jnp.arange(N_cos)
    ba = b - a
    u_k = k_grid * jnp.pi / ba

    cf_params = HestonCFParams(
        v0=v0, kappa=market.kappa, theta=market.theta,
        sigma_v=market.sigma_v, rho=market.rho, mu=mu,
    )
    cf_vals = jax_heston_cf(cf_params, u_k, log_S0, v0, T)

    # Absolute strikes from moneyness
    strikes = spot * moneyness_row  # (max_strikes,)

    # Price calls
    is_call = jnp.ones(moneyness_row.shape[0], dtype=bool)
    call_prices = _jax_cos_price_multi(cf_vals, strikes, T, market.r, a, b, is_call)

    # Price puts
    is_put = jnp.zeros(moneyness_row.shape[0], dtype=bool)
    put_prices = _jax_cos_price_multi(cf_vals, strikes, T, market.r, a, b, is_put)

    # Mask invalid (padded) entries
    call_prices = jnp.where(mask_row, call_prices, 0.0)
    put_prices = jnp.where(mask_row, put_prices, 0.0)

    # Floor at zero (no negative prices)
    call_prices = jnp.maximum(call_prices, 0.0)
    put_prices = jnp.maximum(put_prices, 0.0)

    # Interleave: [call_0, put_0, call_1, put_1, ...]
    # Use stack+ravel instead of .at[::2].set() for MPS compatibility under vmap
    interleaved = jnp.stack([call_prices, put_prices], axis=-1).ravel()

    return interleaved


# ============================================================================
# Full grid pricing
# ============================================================================

def price_option_grid(
    spot: jnp.ndarray,
    v0: jnp.ndarray,
    padded_grid: _PaddedGrid,
    market: HestonMarketParams,
    time_index: int,
    horizon_steps: int,
    N_cos: int = 128,
    L_cos: float = 12.0,
) -> jnp.ndarray:
    """Price the entire option grid at the current state.

    Uses a flat-table approach: all options are priced via a single
    broadcast operation, with no ``vmap``, ``fori_loop``, or ``.at[].set()``
    scatter. This is MPS-safe and vmap-compatible.

    Each option instrument is mapped to its maturity's CF via
    ``padded_grid.flat_maturity_idx``, so the COS summation over k terms
    happens in one (n_options, N_cos) broadcast.

    Args:
        spot: scalar JAX array — current spot
        v0: scalar JAX array — current instantaneous variance
        padded_grid: pre-compiled padded grid from ``compile_padded_grid``
        market: risk-neutral Heston parameters
        time_index: current env step
        horizon_steps: total episode horizon
        N_cos: COS expansion terms (static for JIT)
        L_cos: COS truncation parameter

    Returns:
        (n_instruments,) price vector.
        Index 0 is the underlying (= spot).
        Remaining indices are interleaved call/put prices per maturity.
    """
    mu = market.r - market.q
    log_S0 = jnp.log(spot)
    remaining = horizon_steps - time_index
    n_mat = padded_grid.n_maturities
    n_options = padded_grid.flat_maturity_idx.shape[0]

    # --- Step 1: Compute per-maturity COS quantities ---
    # Truncation ranges: (n_mat,) each
    T_all = padded_grid.maturities_years                    # (n_mat,)
    c1 = log_S0 + (mu - 0.5 * v0) * T_all
    c2 = jnp.maximum(v0 * T_all, 1e-12)
    a_all = c1 - L_cos * jnp.sqrt(c2)                      # (n_mat,)
    b_all = c1 + L_cos * jnp.sqrt(c2)                      # (n_mat,)
    ba_all = b_all - a_all                                  # (n_mat,)

    # k-grid: (N_cos,)
    k_grid = jnp.arange(N_cos)

    # u_k per maturity: (n_mat, N_cos)
    u_k_all = k_grid[None, :] * jnp.pi / ba_all[:, None]

    # Heston CF per maturity: (n_mat, N_cos) complex
    cf_params = HestonCFParams(
        v0=v0, kappa=market.kappa, theta=market.theta,
        sigma_v=market.sigma_v, rho=market.rho, mu=mu,
    )
    # Evaluate CF for each maturity — broadcast over (n_mat, N_cos)
    d = jnp.sqrt(
        (cf_params.rho * cf_params.sigma_v * 1j * u_k_all - cf_params.kappa)**2
        + cf_params.sigma_v**2 * (1j * u_k_all + u_k_all**2)
    )
    g = (
        (cf_params.kappa - cf_params.rho * cf_params.sigma_v * 1j * u_k_all - d)
        / (cf_params.kappa - cf_params.rho * cf_params.sigma_v * 1j * u_k_all + d)
    )
    exp_dt = jnp.exp(-d * T_all[:, None])
    C = (
        cf_params.mu * 1j * u_k_all * T_all[:, None]
        + (cf_params.kappa * cf_params.theta / cf_params.sigma_v**2)
        * (
            (cf_params.kappa - cf_params.rho * cf_params.sigma_v * 1j * u_k_all - d)
            * T_all[:, None]
            - 2.0 * jnp.log((1.0 - g * exp_dt) / (1.0 - g))
        )
    )
    D = (
        (cf_params.kappa - cf_params.rho * cf_params.sigma_v * 1j * u_k_all - d)
        / cf_params.sigma_v**2
    ) * ((1.0 - exp_dt) / (1.0 - g * exp_dt))
    cf_all = jnp.exp(C + D * v0 + 1j * u_k_all * log_S0)  # (n_mat, N_cos)

    # Fourier density coefficients: F_k = Re{cf * exp(-i*u_k*a)}
    F_k_all = jnp.real(
        cf_all * jnp.exp(-1j * u_k_all * a_all[:, None])
    )  # (n_mat, N_cos)

    # --- Step 2: Gather per-option quantities from maturity tables ---
    mat_idx = padded_grid.flat_maturity_idx                 # (n_options,) int
    moneyness = padded_grid.flat_moneyness                  # (n_options,)
    is_call = padded_grid.flat_is_call                      # (n_options,) bool

    # Per-option COS parameters (gathered from maturity arrays)
    T_opt = T_all[mat_idx]                                  # (n_options,)
    a_opt = a_all[mat_idx]                                  # (n_options,)
    b_opt = b_all[mat_idx]                                  # (n_options,)
    ba_opt = ba_all[mat_idx]                                # (n_options,)
    F_k_opt = F_k_all[mat_idx]                              # (n_options, N_cos)

    # Availability: maturity must be <= remaining steps
    mat_steps_opt = padded_grid.maturities_steps[mat_idx]   # (n_options,)
    available = mat_steps_opt <= remaining                   # (n_options,) bool

    # Absolute strikes
    strikes_opt = spot * moneyness                          # (n_options,)
    log_K = jnp.log(strikes_opt)                            # (n_options,)

    # --- Step 3: COS payoff coefficients (n_options, N_cos) ---
    w = k_grid[None, :] * jnp.pi / ba_opt[:, None]         # (n_options, N_cos)

    def _F_antideriv(x_col):
        """Antiderivative of exp(x)*cos(w*(x-a)). x_col: (n_options,)."""
        x = x_col[:, None]                                 # (n_options, 1)
        a = a_opt[:, None]                                  # (n_options, 1)
        return jnp.exp(x) / (1.0 + w**2) * (
            jnp.cos(w * (x - a)) + w * jnp.sin(w * (x - a))
        )

    # Call payoff: chi(log_K, b), psi(log_K, b)
    chi_call = _F_antideriv(b_opt) - _F_antideriv(log_K)   # (n_options, N_cos)
    safe_k = jnp.where(k_grid == 0, 1.0, k_grid)           # (N_cos,)
    psi_call = (ba_opt[:, None] / (safe_k[None, :] * jnp.pi)) * (
        jnp.sin(w * (b_opt[:, None] - a_opt[:, None]))
        - jnp.sin(w * (log_K[:, None] - a_opt[:, None]))
    )
    psi_call = jnp.where(
        k_grid[None, :] == 0,
        (b_opt - log_K)[:, None],
        psi_call,
    )
    V_call = 2.0 / ba_opt[:, None] * (chi_call - strikes_opt[:, None] * psi_call)

    # Put payoff: chi(a, log_K), psi(a, log_K)
    chi_put = _F_antideriv(log_K) - _F_antideriv(a_opt)
    psi_put = (ba_opt[:, None] / (safe_k[None, :] * jnp.pi)) * (
        jnp.sin(w * (log_K[:, None] - a_opt[:, None]))
        - jnp.sin(w * (a_opt[:, None] - a_opt[:, None]))   # = 0
    )
    psi_put = jnp.where(
        k_grid[None, :] == 0,
        (log_K - a_opt)[:, None],
        psi_put,
    )
    V_put = 2.0 / ba_opt[:, None] * (-chi_put + strikes_opt[:, None] * psi_put)

    # Select call or put per instrument (float mask: 1.0=call, 0.0=put)
    V_k = is_call[:, None] * V_call + (1.0 - is_call[:, None]) * V_put  # (n_options, N_cos)

    # Half weight for k=0
    half_weight = jnp.where(k_grid == 0, 0.5, 1.0)         # (N_cos,)

    # --- Step 4: Price = exp(-rT) * sum(F_k * V_k * half_weight) ---
    option_prices = jnp.exp(-market.r * T_opt) * jnp.sum(
        F_k_opt * V_k * half_weight[None, :], axis=-1
    )  # (n_options,)

    # Floor at zero, mask unavailable and invalid (float masks)
    option_prices = jnp.maximum(option_prices, 0.0)
    available_f = jnp.where(available, 1.0, 0.0)
    option_prices = option_prices * available_f * padded_grid.flat_valid

    # Build price vector: [spot, option_0, option_1, ...]
    return jnp.concatenate([spot[jnp.newaxis], option_prices])
