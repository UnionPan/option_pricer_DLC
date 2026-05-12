"""
Andersen 2008 Quadratic-Exponential (QE) scheme for stochastic-vol simulation.

The QE scheme samples the CIR variance ``v_{t+dt}`` from an exact-conditional
distribution (matched moments under either a quadratic or exponential law,
depending on the dimensionless ratio ``ψ = Var(v)/E[v]²``). It is
**positivity-preserving by construction** — variance can never go negative,
even at borderline-Feller parameters where Euler-Maruyama produces ``v < 0``
and downstream pricers blow up.

The corresponding spot update uses the log-Euler form with a central
trapezoidal approximation of the integrated variance — the
"Andersen central discretization" (Andersen 2008, §3.4).

Models supported:
- ``qe_heston_step``         — Heston (CIR variance + correlated GBM spot)
- ``qe_bates_step``          — Heston + Merton jumps (CIR variance unchanged)
- ``qe_three_half_step``     — 3/2 model (1/v is CIR; reparameterize, apply QE, invert)
- ``qe_four_half_step``      — 4/2 model (CIR variance same as Heston, combined spot SDE)

Reference:
    Andersen, L. (2008). "Simple and efficient simulation of the Heston
    stochastic volatility model." Journal of Computational Finance 11(3).

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Andersen recommends ψ_crit ∈ [1.0, 2.0]; 1.5 is the standard mid-range choice.
DEFAULT_PSI_CRIT: float = 1.5

# Numerical guards — keep all in one place for auditability.
_EPS_M = 1e-12      # guard against m → 0 in conditional moments
_EPS_PSI = 1e-12    # guard against ψ → 0
_EPS_U = 1e-12      # guard against U → 1 in inverse-CDF


# ---------------------------------------------------------------------------
# Core: QE step for a CIR variance process
# ---------------------------------------------------------------------------


def _qe_variance_step_cir(
    v0: jnp.ndarray,
    kappa: float,
    theta: float,
    sigma_v: float,
    dt: float,
    key: jax.Array,
    psi_crit: float = DEFAULT_PSI_CRIT,
) -> tuple[jnp.ndarray, jax.Array]:
    """One QE step for a CIR variance process ``dv = κ(θ−v)dt + σ_v √v dW``.

    The conditional law of ``v_{t+dt} | v_t = v0`` is matched to either a
    quadratic distribution (low ψ, high variance regime) or an exponential
    distribution (high ψ, low variance regime). The switch threshold ψ_crit
    is a free parameter; Andersen recommends 1.5.

    Args:
        v0: scalar (or batched) current variance ≥ 0
        kappa, theta, sigma_v: CIR parameters
        dt: time step
        key: JAX PRNG key
        psi_crit: switching threshold (default 1.5)

    Returns:
        ``(v_next, new_key)`` — variance ≥ 0 by construction.

    Notes:
        Both branches are computed unconditionally and selected via
        ``jnp.where`` — keeps the code JIT-friendly with no Python-side
        branching. Each branch consumes exactly one normal sample plus
        one uniform sample so PRNG consumption is static (required for
        ``lax.scan`` body usage).
    """
    # Conditional moments of v_{t+dt} | v_t = v0
    e_kdt = jnp.exp(-kappa * dt)
    one_minus_e = 1.0 - e_kdt
    m = theta + (v0 - theta) * e_kdt
    s2 = (
        v0 * sigma_v ** 2 * e_kdt * one_minus_e / kappa
        + theta * sigma_v ** 2 * one_minus_e ** 2 / (2.0 * kappa)
    )

    m_safe = jnp.maximum(m, _EPS_M)
    psi = s2 / (m_safe * m_safe)
    psi_safe = jnp.maximum(psi, _EPS_PSI)

    # ---- Quadratic branch (ψ ≤ ψ_crit) — Andersen eq. 27 ----
    inv_psi = 1.0 / psi_safe
    two_inv_psi = 2.0 * inv_psi
    inner = jnp.maximum(two_inv_psi * (two_inv_psi - 1.0), 0.0)
    b2 = two_inv_psi - 1.0 + jnp.sqrt(inner)
    b = jnp.sqrt(jnp.maximum(b2, 0.0))
    a = m / (1.0 + b2)

    # ---- Exponential branch (ψ > ψ_crit) ----
    p = (psi - 1.0) / (psi + 1.0)
    p = jnp.clip(p, 0.0, 1.0 - _EPS_U)
    beta = jnp.maximum((1.0 - p) / m_safe, _EPS_M)

    # ---- Sample noise (always both, then select) ----
    key, k_z, k_u = jax.random.split(key, 3)
    Z = jax.random.normal(k_z)
    U = jax.random.uniform(k_u, minval=0.0, maxval=1.0)

    # Quadratic candidate: v = a (b + Z)²  (always ≥ 0)
    v_quad = a * (b + Z) ** 2

    # Exponential candidate: inverse CDF of Ψ(x) = p + (1-p)(1 - exp(-βx))
    # If U ≤ p: v = 0;  else v = -ln((1-U)/(1-p)) / β
    arg = jnp.maximum((1.0 - p) / jnp.maximum(1.0 - U, _EPS_U), _EPS_M)
    v_exp_pos = jnp.log(arg) / beta
    v_exp = jnp.where(U <= p, 0.0, v_exp_pos)

    v_next = jnp.where(psi <= psi_crit, v_quad, v_exp)
    v_next = jnp.maximum(v_next, 0.0)  # safety floor
    return v_next, key


# ---------------------------------------------------------------------------
# Spot updates that use a QE-stepped variance
# ---------------------------------------------------------------------------


def _log_spot_step_central(
    log_spot: jnp.ndarray,
    v0: jnp.ndarray,
    v_next: jnp.ndarray,
    mu: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    dt: float,
    key: jax.Array,
) -> tuple[jnp.ndarray, jax.Array]:
    """Log-Euler spot update with central trapezoidal variance integral.

    Conditional on the QE-sampled ``v_{t+dt}``, integrates the log-spot SDE::

        d(log S) = (μ − ½ v) dt + (ρ/σ_v)(dv − κ(θ − v) dt) + √((1−ρ²) v) dZ

    using the trapezoidal approximation
    ``∫_t^{t+dt} v_s ds ≈ ½(v_t + v_{t+dt}) dt``.

    Returns ``(log_spot_next, new_key)``.
    """
    I = 0.5 * (v0 + v_next) * dt  # trapezoidal integrated variance

    key, k_zs = jax.random.split(key)
    Z_S = jax.random.normal(k_zs)

    drift_term = mu * dt - 0.5 * I
    martingale_term = (rho / sigma_v) * (
        v_next - v0 - kappa * theta * dt + kappa * I
    )
    diffusion_term = jnp.sqrt((1.0 - rho ** 2) * jnp.maximum(I, 0.0)) * Z_S

    log_spot_next = log_spot + drift_term + martingale_term + diffusion_term
    return log_spot_next, key


# ---------------------------------------------------------------------------
# Heston QE step (variance + correlated spot)
# ---------------------------------------------------------------------------


def qe_heston_step(
    spot: jnp.ndarray,
    variance: jnp.ndarray,
    dt: float,
    mu: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    key: jax.Array,
    psi_crit: float = DEFAULT_PSI_CRIT,
) -> tuple[jnp.ndarray, jnp.ndarray, jax.Array]:
    """One Andersen-QE step for Heston ``(S, v)``.

    Args:
        spot: current spot price
        variance: current variance
        dt: time step
        mu: risk-neutral drift (r − q)
        kappa, theta, sigma_v: CIR variance parameters
        rho: spot–variance correlation
        key: JAX PRNG key
        psi_crit: QE switching threshold (default 1.5)

    Returns:
        ``(next_spot, next_variance, next_key)`` — variance guaranteed ≥ 0.
    """
    v_next, key = _qe_variance_step_cir(
        variance, kappa, theta, sigma_v, dt, key, psi_crit,
    )
    log_spot = jnp.log(jnp.maximum(spot, 1e-300))
    log_spot_next, key = _log_spot_step_central(
        log_spot, variance, v_next, mu, kappa, theta, sigma_v, rho, dt, key,
    )
    next_spot = jnp.exp(log_spot_next)
    return next_spot, v_next, key


# ---------------------------------------------------------------------------
# Bates QE step (Heston variance + Merton-style log-normal jumps)
# ---------------------------------------------------------------------------


def qe_bates_step(
    spot: jnp.ndarray,
    variance: jnp.ndarray,
    dt: float,
    mu: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    lambda_j: float,
    mu_j: float,
    sigma_j: float,
    key: jax.Array,
    psi_crit: float = DEFAULT_PSI_CRIT,
) -> tuple[jnp.ndarray, jnp.ndarray, jax.Array]:
    """One QE step for Bates (Heston + Merton jumps).

    Variance dynamics identical to Heston (CIR via QE).
    Spot adds a compensated compound-Poisson log-normal jump::

        log S_{t+dt} = log S_t + (μ − λ k − ½ v̄) dt + … + Σ_{i=1..N_t} J_i

    where ``k = exp(μ_j + ½ σ_j²) − 1`` is the jump compensator and jumps
    are log-normal: ``J_i ~ N(μ_j, σ_j²)``.

    Args:
        lambda_j: jump intensity (jumps per unit time)
        mu_j: mean of log jump size
        sigma_j: std of log jump size
    """
    v_next, key = _qe_variance_step_cir(
        variance, kappa, theta, sigma_v, dt, key, psi_crit,
    )

    # Continuous spot update with jump compensator absorbed into drift
    jump_compensator = jnp.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0
    mu_compensated = mu - lambda_j * jump_compensator
    log_spot = jnp.log(jnp.maximum(spot, 1e-300))
    log_spot_cont, key = _log_spot_step_central(
        log_spot, variance, v_next, mu_compensated, kappa, theta, sigma_v,
        rho, dt, key,
    )

    # Compound Poisson log-jump:
    #   N ~ Poisson(λ dt);  Σ J_i | N ~ N(N μ_j, N σ_j²)
    key, k_n, k_j = jax.random.split(key, 3)
    n_jumps = jax.random.poisson(k_n, lambda_j * dt)
    n_jumps_f = jnp.asarray(n_jumps, dtype=log_spot.dtype)
    Z_j = jax.random.normal(k_j)
    log_jump = n_jumps_f * mu_j + jnp.sqrt(n_jumps_f) * sigma_j * Z_j

    log_spot_next = log_spot_cont + log_jump
    next_spot = jnp.exp(log_spot_next)
    return next_spot, v_next, key


# ---------------------------------------------------------------------------
# 3/2 model: y = 1/v is CIR, so apply QE to y and invert
# ---------------------------------------------------------------------------


def qe_three_half_step(
    spot: jnp.ndarray,
    variance: jnp.ndarray,
    dt: float,
    mu: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    key: jax.Array,
    psi_crit: float = DEFAULT_PSI_CRIT,
) -> tuple[jnp.ndarray, jnp.ndarray, jax.Array]:
    """One QE step for the 3/2 model.

    The 3/2 variance SDE is::

        dv = κ v (θ − v) dt + σ_v v^{3/2} dW

    Under the substitution ``y = 1/v``, Itô gives a CIR-type SDE for ``y``
    with effective parameters (Baldeaux 2012 §3)::

        κ̃ = κ θ
        θ̃ = (κ θ + σ_v²) / (κ θ)
        σ̃_v = σ_v

    Apply QE to ``y`` and invert.

    NOTE: This implementation uses a simplified spot update that ignores
    the spot–y correlation sign flip. For research-grade accuracy on long
    horizons, implement the full Itô lemma adjustment to the spot SDE.
    """
    kappa_tilde = kappa * theta
    theta_tilde = (kappa * theta + sigma_v ** 2) / (kappa * theta)
    sigma_tilde = sigma_v

    y0 = 1.0 / jnp.maximum(variance, 1e-300)
    y_next, key = _qe_variance_step_cir(
        y0, kappa_tilde, theta_tilde, sigma_tilde, dt, key, psi_crit,
    )
    v_next = 1.0 / jnp.maximum(y_next, 1e-300)

    # Simplified log-Euler spot update (correlated, no martingale-term coupling)
    log_spot = jnp.log(jnp.maximum(spot, 1e-300))
    I = 0.5 * (variance + v_next) * dt

    key, k_zv, k_zorth = jax.random.split(key, 3)
    Z_v = jax.random.normal(k_zv)
    Z_orth = jax.random.normal(k_zorth)

    drift_term = mu * dt - 0.5 * I
    diffusion_term = jnp.sqrt(jnp.maximum(I, 0.0)) * (
        rho * Z_v + jnp.sqrt(1.0 - rho ** 2) * Z_orth
    )
    log_spot_next = log_spot + drift_term + diffusion_term
    next_spot = jnp.exp(log_spot_next)
    return next_spot, v_next, key


# ---------------------------------------------------------------------------
# 4/2 model: combined Heston + 3/2 spot dynamics with shared CIR variance
# ---------------------------------------------------------------------------


def qe_four_half_step(
    spot: jnp.ndarray,
    variance: jnp.ndarray,
    dt: float,
    mu: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    a_coef: float,   # Heston-side coefficient
    b_coef: float,   # 3/2-side coefficient
    key: jax.Array,
    psi_crit: float = DEFAULT_PSI_CRIT,
) -> tuple[jnp.ndarray, jnp.ndarray, jax.Array]:
    """One QE step for the 4/2 model (Grasselli 2017).

    Variance is plain CIR (same as Heston) — QE step is identical.
    Spot instantaneous-vol coefficient is ``a √v + b/√v``::

        d(log S) = (μ − ½ (a √v + b/√v)²) dt + (a √v + b/√v) dW_S

    ``(a, b) = (1, 0)`` reduces to Heston; ``(0, 1)`` reduces to a
    3/2-like vol coefficient.
    """
    v_next, key = _qe_variance_step_cir(
        variance, kappa, theta, sigma_v, dt, key, psi_crit,
    )
    log_spot = jnp.log(jnp.maximum(spot, 1e-300))

    # Trapezoidal integrals
    I_v = 0.5 * (variance + v_next) * dt
    I_inv_v = 0.5 * (
        1.0 / jnp.maximum(variance, 1e-300)
        + 1.0 / jnp.maximum(v_next, 1e-300)
    ) * dt

    # ∫ (a√v + b/√v)² ds = a² ∫v + 2ab dt + b² ∫(1/v)
    integrated_inst_var = (
        a_coef ** 2 * I_v
        + 2.0 * a_coef * b_coef * dt
        + b_coef ** 2 * I_inv_v
    )

    key, k_zv, k_zorth = jax.random.split(key, 3)
    Z_v = jax.random.normal(k_zv)
    Z_orth = jax.random.normal(k_zorth)

    diffusion_term = jnp.sqrt(jnp.maximum(integrated_inst_var, 0.0)) * (
        rho * Z_v + jnp.sqrt(1.0 - rho ** 2) * Z_orth
    )
    log_spot_next = (
        log_spot + mu * dt - 0.5 * integrated_inst_var + diffusion_term
    )
    next_spot = jnp.exp(log_spot_next)
    return next_spot, v_next, key
