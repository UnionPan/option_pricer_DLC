"""
Vectorized Black-Scholes implied volatility inversion via JAX.

Replaces per-option Newton-Raphson loops with a single JIT-compiled
batch operation over all options simultaneously.

For N options, the NumPy version runs N sequential Newton solves.
This version runs all N in parallel with fixed-iteration Newton,
compiled to a single XLA kernel.

author: Yunian Pan
email: yp1170@nyu.edu
"""

import jax
import jax.numpy as jnp
import numpy as np

from ..processes._jax_utils import to_numpy


# ============================================================================
# Vectorized BS price + vega (all options at once)
# ============================================================================
def _bs_price_and_vega(S, K, T, r, q, sigma, is_call):
    """
    Vectorized Black-Scholes price and vega.

    All inputs are arrays of shape (n,). Returns (prices, vegas) of shape (n,).
    """
    sqrt_T = jnp.sqrt(T)
    d1 = (jnp.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    # Normal CDF and PDF via JAX
    nd1 = jax.scipy.stats.norm.cdf(d1)
    nd2 = jax.scipy.stats.norm.cdf(d2)
    npd1 = jax.scipy.stats.norm.pdf(d1)

    # Call and put prices
    call_price = S * jnp.exp(-q * T) * nd1 - K * jnp.exp(-r * T) * nd2
    put_price = K * jnp.exp(-r * T) * (1.0 - nd2) - S * jnp.exp(-q * T) * (1.0 - nd1)

    prices = jnp.where(is_call, call_price, put_price)
    vega = S * jnp.exp(-q * T) * npd1 * sqrt_T

    return prices, vega


# ============================================================================
# Vectorized Newton-Raphson IV solver
# ============================================================================
@jax.jit
def implied_vol_batch(
    S: jnp.ndarray,
    K: jnp.ndarray,
    T: jnp.ndarray,
    r: jnp.ndarray,
    q: jnp.ndarray,
    market_prices: jnp.ndarray,
    is_call: jnp.ndarray,
    sigma_init: jnp.ndarray,
    sigma_min: float = 0.005,
    sigma_max: float = 5.0,
) -> jnp.ndarray:
    """
    Batch implied volatility via vectorized Newton-Raphson.

    All N options are solved simultaneously in each iteration.
    Fixed iteration count (no early exit) for JIT compatibility.

    Args:
        S: Spot prices, shape (n,)
        K: Strikes, shape (n,)
        T: Maturities, shape (n,)
        r: Risk-free rates, shape (n,)
        q: Dividend yields, shape (n,)
        market_prices: Market option prices, shape (n,)
        is_call: Boolean call/put flags, shape (n,)
        sigma_init: Initial IV guesses, shape (n,)
        sigma_min: Floor for IV
        sigma_max: Ceiling for IV

    Returns:
        Implied volatilities, shape (n,)
    """
    sigma = jnp.clip(sigma_init, sigma_min, sigma_max)

    def newton_step(sigma, _):
        prices, vega = _bs_price_and_vega(S, K, T, r, q, sigma, is_call)
        diff = prices - market_prices

        # Newton update with vega floor to avoid division by zero
        vega_safe = jnp.maximum(vega, 1e-10)
        update = diff / vega_safe

        # Damped Newton: clip step size
        update = jnp.clip(update, -0.5, 0.5)

        sigma_new = sigma - update
        sigma_new = jnp.clip(sigma_new, sigma_min, sigma_max)

        return sigma_new, None

    # Run 20 fixed iterations via lax.scan (no Python loop)
    sigma_final, _ = jax.lax.scan(newton_step, sigma, jnp.arange(20))

    return sigma_final


# ============================================================================
# Convenience: numpy in/out wrapper
# ============================================================================
def implied_vol_batch_np(
    S, K, T, r, q, market_prices, is_call,
    sigma_init=None,
    sigma_min=0.005, sigma_max=5.0,
):
    """
    Batch IV inversion with numpy arrays in and out.

    Drop-in replacement for per-option Newton loops.

    Args:
        S: float or array — spot price(s)
        K: array — strikes
        T: float or array — maturities
        r: float or array — risk-free rate
        q: float or array — dividend yield
        market_prices: array — option prices
        is_call: bool array or list — True for calls
        sigma_init: array — initial guesses (default: 0.2)
        sigma_min: IV floor
        sigma_max: IV ceiling

    Returns:
        numpy array of implied volatilities
    """
    n = len(K)

    S_j = jnp.broadcast_to(jnp.float64(S), (n,)) if np.isscalar(S) else jnp.array(S, dtype=jnp.float64)
    K_j = jnp.array(K, dtype=jnp.float64)
    T_j = jnp.broadcast_to(jnp.float64(T), (n,)) if np.isscalar(T) else jnp.array(T, dtype=jnp.float64)
    r_j = jnp.broadcast_to(jnp.float64(r), (n,)) if np.isscalar(r) else jnp.array(r, dtype=jnp.float64)
    q_j = jnp.broadcast_to(jnp.float64(q), (n,)) if np.isscalar(q) else jnp.array(q, dtype=jnp.float64)
    prices_j = jnp.array(market_prices, dtype=jnp.float64)
    is_call_j = jnp.array(is_call, dtype=bool) if not isinstance(is_call, jnp.ndarray) else is_call

    if sigma_init is None:
        sigma_init_j = jnp.full(n, 0.2)
    else:
        sigma_init_j = jnp.array(sigma_init, dtype=jnp.float64)

    result = implied_vol_batch(
        S_j, K_j, T_j, r_j, q_j, prices_j, is_call_j,
        sigma_init_j, sigma_min, sigma_max,
    )

    return to_numpy(result)
