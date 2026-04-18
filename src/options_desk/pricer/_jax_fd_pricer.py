"""
JAX-accelerated Finite Difference PDE solver for option pricing.

Solves the Black-Scholes PDE:
    dV/dt + (r-q)S dV/dS + 0.5*sigma^2*S^2 d2V/dS2 - rV = 0

using explicit, implicit, or Crank-Nicolson (theta-method) time stepping
with a Thomas algorithm O(N) tridiagonal solver (replacing scipy.sparse.linalg.spsolve).

Time loop is fully JIT-compiled via lax.scan.
American exercise via jnp.maximum inside the scan body.

author: Yunian Pan
email: yp1170@nyu.edu
"""

import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np

from ..processes._jax_utils import to_numpy


# ============================================================================
# Thomas algorithm — O(N) tridiagonal solver
# ============================================================================

def _thomas_solve(
    a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray, d: jnp.ndarray,
) -> jnp.ndarray:
    """Solve tridiagonal system Ax = d using the Thomas algorithm.

    Storage convention (length-N arrays, matching interior+boundary grid):
        a: sub-diagonal,   a[0] unused (set to 0)
        b: main diagonal
        c: super-diagonal, c[-1] unused (set to 0)
        d: right-hand side

    Returns x of shape (N,).
    """
    # Forward sweep: compute modified c' and d'
    c0 = c[0] / b[0]
    d0 = d[0] / b[0]

    def forward_step(carry, inputs):
        c_prev, d_prev = carry
        a_i, b_i, c_i, d_i = inputs
        denom = b_i - a_i * c_prev
        c_new = c_i / denom
        d_new = (d_i - a_i * d_prev) / denom
        return (c_new, d_new), (c_new, d_new)

    _, (c_rest, d_rest) = lax.scan(
        forward_step, (c0, d0), (a[1:], b[1:], c[1:], d[1:]),
    )
    c_prime = jnp.concatenate([jnp.array([c0]), c_rest])
    d_prime = jnp.concatenate([jnp.array([d0]), d_rest])

    # Back substitution
    def backward_step(x_next, inputs):
        c_i, d_i = inputs
        x_i = d_i - c_i * x_next
        return x_i, x_i

    x_last = d_prime[-1]
    _, x_rest = lax.scan(
        backward_step, x_last, (c_prime[:-1], d_prime[:-1]), reverse=True,
    )
    return jnp.concatenate([x_rest, jnp.array([x_last])])


def _tridiag_matvec(
    a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray, x: jnp.ndarray,
) -> jnp.ndarray:
    """Multiply tridiagonal matrix (a, b, c diagonals) by vector x."""
    y = b * x
    y = y.at[1:].add(a[1:] * x[:-1])
    y = y.at[:-1].add(c[:-1] * x[1:])
    return y


# ============================================================================
# FD coefficient construction (vectorized, no loops)
# ============================================================================

def _build_fd_coeffs(
    S_grid: jnp.ndarray, dS: float, dt: float,
    sigma: float, r: float, q: float, theta: float,
) -> tuple[jnp.ndarray, ...]:
    """Build LHS and RHS tridiagonal coefficients for the theta-method.

    theta=0: explicit,  theta=1: implicit,  theta=0.5: Crank-Nicolson

    PDE operator L(V)_i = p*(V_{i+1} - 2V_i + V_{i-1}) + qd*(V_{i+1} - V_{i-1}) - r*V_i
    where  p  = 0.5*sigma^2*S_i^2 / dS^2   (diffusion)
           qd = (r-q)*S_i / (2*dS)          (drift, central difference)

    LHS matrix A = I - theta*dt*L     (to be inverted)
    RHS matrix B = I + (1-theta)*dt*L (applied to current V)

    Returns (a_L, b_L, c_L, a_R, b_R, c_R) — each shape (N_S,).
    Boundary rows (0 and -1) are identity.
    """
    N = S_grid.shape[0]

    # Operator coefficients (full grid, but only interior matters)
    p = 0.5 * sigma**2 * S_grid**2 / dS**2
    qd = (r - q) * S_grid / (2.0 * dS)

    # LHS: A = I - theta*dt*L
    a_L = jnp.zeros(N).at[1:-1].set(-theta * dt * (p[1:-1] - qd[1:-1]))
    b_L = jnp.ones(N).at[1:-1].set(1.0 + theta * dt * (2.0 * p[1:-1] + r))
    c_L = jnp.zeros(N).at[1:-1].set(-theta * dt * (p[1:-1] + qd[1:-1]))

    # RHS: B = I + (1-theta)*dt*L
    ot = 1.0 - theta
    a_R = jnp.zeros(N).at[1:-1].set(ot * dt * (p[1:-1] - qd[1:-1]))
    b_R = jnp.ones(N).at[1:-1].set(1.0 - ot * dt * (2.0 * p[1:-1] + r))
    c_R = jnp.zeros(N).at[1:-1].set(ot * dt * (p[1:-1] + qd[1:-1]))

    return a_L, b_L, c_L, a_R, b_R, c_R


# ============================================================================
# Boundary conditions (call/put)
# ============================================================================

def _lower_bc(tau: float, K: float, r: float, is_call: bool) -> float:
    """BC at S = 0.  tau = remaining time to maturity."""
    return jnp.where(is_call, 0.0, K * jnp.exp(-r * tau))


def _upper_bc(
    S_max: float, tau: float, K: float, r: float, is_call: bool,
) -> float:
    """BC at S = S_max."""
    return jnp.where(is_call, S_max - K * jnp.exp(-r * tau), 0.0)


# ============================================================================
# FD time-stepping via lax.scan
# ============================================================================

@jax.jit
def _jax_fd_solve(
    V_terminal: jnp.ndarray,
    a_L: jnp.ndarray, b_L: jnp.ndarray, c_L: jnp.ndarray,
    a_R: jnp.ndarray, b_R: jnp.ndarray, c_R: jnp.ndarray,
    K: float, T: float, r: float, S_max: float,
    dt: float,
    is_call: bool, american: bool,
    exercise_vals: jnp.ndarray,
    step_indices: jnp.ndarray,
) -> jnp.ndarray:
    """Full backward-in-time FD solve.

    Iterates from n = N_t-1 (terminal) down to n = 1, producing V at t = 0.
    step_indices must be jnp.arange(N_t-1, 0, -1), passed from outside JIT.
    """
    def step(V, step_idx):
        # tau = time to maturity at the TARGET time level (step_idx - 1)
        tau = T - (step_idx - 1) * dt

        # RHS = B * V^n
        rhs = _tridiag_matvec(a_R, b_R, c_R, V)

        # Overwrite boundary rows with BCs at the target time
        bc_low = _lower_bc(tau, K, r, is_call)
        bc_up = _upper_bc(S_max, tau, K, r, is_call)
        rhs = rhs.at[0].set(bc_low)
        rhs = rhs.at[-1].set(bc_up)

        # Solve A * V^{n-1} = rhs
        V_new = _thomas_solve(a_L, b_L, c_L, rhs)

        # American constraint: V >= intrinsic at every step
        V_new = jnp.where(american, jnp.maximum(V_new, exercise_vals), V_new)

        return V_new, None

    V_final, _ = lax.scan(step, V_terminal, step_indices)
    return V_final


# ============================================================================
# Greeks from grid
# ============================================================================

def _fd_greeks(
    S0: float, S_grid: jnp.ndarray, V: jnp.ndarray, dS: float,
) -> dict:
    """Compute delta and gamma via centred differences on the FD grid."""
    idx = int(jnp.argmin(jnp.abs(S_grid - S0)))
    idx = max(1, min(idx, len(S_grid) - 2))

    delta = float((V[idx + 1] - V[idx - 1]) / (2.0 * dS))
    gamma = float((V[idx + 1] - 2.0 * V[idx] + V[idx - 1]) / dS**2)
    return {"delta": delta, "gamma": gamma}


# ============================================================================
# Public convenience wrapper
# ============================================================================

def jax_fd_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool = True,
    american: bool = False,
    scheme: str = "crank-nicolson",
    N_S: int = 200,
    N_t: int = 500,
    S_max: float | None = None,
    compute_greeks: bool = False,
) -> float | dict:
    """Price a European or American vanilla option via finite differences.

    Args:
        S0: Spot price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        q: Continuous dividend yield
        sigma: Volatility
        is_call: True for call, False for put
        american: True for American exercise
        scheme: 'explicit', 'implicit', or 'crank-nicolson'
        N_S: Number of spatial grid points
        N_t: Number of time steps
        S_max: Upper boundary for spot grid (auto-set if None)
        compute_greeks: Return dict with price, delta, gamma

    Returns:
        float price, or dict {'price', 'delta', 'gamma'} if compute_greeks
    """
    if T <= 0:
        intrinsic = max(S0 - K, 0.0) if is_call else max(K - S0, 0.0)
        if compute_greeks:
            return {"price": intrinsic, "delta": 0.0, "gamma": 0.0}
        return intrinsic

    theta_map = {"explicit": 0.0, "implicit": 1.0, "crank-nicolson": 0.5}
    scheme_lower = scheme.lower()
    if scheme_lower not in theta_map:
        raise ValueError(f"Unknown scheme: {scheme}")
    theta = theta_map[scheme_lower]

    # Grid setup
    if S_max is None:
        S_max = S0 * np.exp(3.0 * sigma * np.sqrt(T))

    S_grid = jnp.linspace(0.0, S_max, N_S)
    dS = float(S_grid[1] - S_grid[0])
    dt = T / (N_t - 1)

    # Stability check for explicit scheme
    if theta == 0.0:
        max_dt = 0.5 * dS**2 / (sigma**2 * S_max**2)
        if dt > max_dt:
            raise ValueError(
                f"Explicit scheme unstable: dt={dt:.2e} > max_dt={max_dt:.2e}. "
                f"Increase N_t or decrease N_S."
            )

    # Terminal condition (payoff at maturity)
    V_terminal = jnp.where(is_call, jnp.maximum(S_grid - K, 0.0),
                           jnp.maximum(K - S_grid, 0.0))
    exercise_vals = V_terminal

    # Build tridiagonal coefficients
    a_L, b_L, c_L, a_R, b_R, c_R = _build_fd_coeffs(
        S_grid, dS, dt, sigma, r, q, theta,
    )

    # Solve
    step_indices = jnp.arange(N_t - 1, 0, -1)
    V = _jax_fd_solve(
        V_terminal, a_L, b_L, c_L, a_R, b_R, c_R,
        K, T, r, S_max, dt, is_call, american, exercise_vals,
        step_indices,
    )

    # Interpolate to S0
    price = float(jnp.interp(S0, S_grid, V))

    if compute_greeks:
        greeks = _fd_greeks(S0, S_grid, V, dS)
        greeks["price"] = price
        return greeks

    return price
