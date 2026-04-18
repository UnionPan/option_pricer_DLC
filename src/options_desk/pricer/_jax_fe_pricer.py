"""
JAX-accelerated Finite Element (Galerkin, P1) solver for option pricing.

Solves the weak formulation of the Black-Scholes PDE with piecewise-linear
(hat) basis functions.  All element matrices are assembled vectorised over
elements into tridiagonal storage, then solved per time-step with the Thomas
algorithm from ``_jax_fd_pricer``.

Supports:
- Crank-Nicolson (default) and implicit Euler time stepping
- Uniform and adaptive (tanh-clustering) mesh
- European and American options

author: Yunian Pan
email: yp1170@nyu.edu
"""

import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np

from ..processes._jax_utils import to_numpy
from ._jax_fd_pricer import _thomas_solve, _tridiag_matvec


# ============================================================================
# Mesh construction
# ============================================================================

def _uniform_mesh(S_min: float, S_max: float, N_nodes: int) -> jnp.ndarray:
    """Uniform mesh on [S_min, S_max]."""
    return jnp.linspace(S_min, S_max, N_nodes)


def _adaptive_mesh(
    S_min: float, S_max: float, K: float, N_nodes: int,
    concentration: float = 3.0,
) -> jnp.ndarray:
    """Non-uniform mesh via tanh mapping, clustered near strike K.

    Fixed-size output — JIT compatible (no recursive bisection).
    """
    xi = np.linspace(0.0, 1.0, N_nodes)
    xi_K = (K - S_min) / (S_max - S_min)
    z = np.tanh(concentration * (xi - xi_K))
    z_min = np.tanh(concentration * (0.0 - xi_K))
    z_max = np.tanh(concentration * (1.0 - xi_K))
    nodes = S_min + (S_max - S_min) * (z - z_min) / (z_max - z_min)
    return jnp.asarray(nodes)


# ============================================================================
# Vectorised tridiagonal assembly (P1 elements, 1-D)
# ============================================================================

def _assemble_fe_tridiag(
    nodes: jnp.ndarray, sigma: float, r: float, q: float, dt: float,
    theta: float,
) -> tuple[jnp.ndarray, ...]:
    """Assemble LHS and RHS tridiagonal diagonals for the theta-method.

    Matrices involved (all tridiagonal for P1 on 1-D):
        M  — lumped mass   M_ii = sum of row of consistent mass matrix
        K  — stiffness     K_ij = integral 0.5 sigma^2 S^2 phi_i' phi_j' dx
        C  — convection    C_ij = integral S phi_i phi_j' dx
        R  — reaction      R = -M

    Uses lumped mass (diagonal M) for unconditional stability when the
    diffusion coefficient 0.5*sigma^2*S^2 is large relative to element size.

    Combined PDE operator: L = -K + (r-q-sigma^2)*C - r*M

    Time-stepping (theta-method):
        LHS = M - theta*dt*L
        RHS = M + (1-theta)*dt*L

    Returns (a_L, b_L, c_L, a_R, b_R, c_R) with boundary rows = identity/zero.
    """
    N = nodes.shape[0]

    # Element lengths and midpoints (N-1 elements)
    h = nodes[1:] - nodes[:-1]                        # (N-1,)
    S_mid = 0.5 * (nodes[:-1] + nodes[1:])            # (N-1,)
    coeff = 0.5 * sigma**2 * S_mid**2                 # diffusion coeff per element

    # --- Mass matrix M (lumped — diagonal only) ---
    # Lumped mass: row-sum of consistent mass (h_e/6)*[[2,1],[1,2]]
    # Row sum for node e: h_e/2.  Each interior node touches 2 elements.
    # M_main[i] = h[i-1]/2 + h[i]/2
    M_lower = jnp.zeros(N)
    M_main = jnp.zeros(N).at[:-1].add(h / 2.0).at[1:].add(h / 2.0)
    M_upper = jnp.zeros(N)

    # --- Stiffness matrix K ---
    # Element e local stiffness: (coeff_e/h_e)*[[1,-1],[-1,1]]
    ch = coeff / h
    K_lower = jnp.zeros(N).at[1:].add(-ch)
    K_main = jnp.zeros(N).at[:-1].add(ch).at[1:].add(ch)
    K_upper = jnp.zeros(N).at[:-1].add(-ch)

    # --- Convection matrix C ---
    # Element e local convection: S_mid_e * [[-0.5, 0.5], [-0.5, 0.5]]
    # C[e,e]   += -0.5*S_mid_e      (local [0,0])
    # C[e,e+1] +=  0.5*S_mid_e      (local [0,1])
    # C[e+1,e] += -0.5*S_mid_e      (local [1,0])
    # C[e+1,e+1] += 0.5*S_mid_e     (local [1,1])
    C_lower = jnp.zeros(N).at[1:].add(-0.5 * S_mid)
    C_main = jnp.zeros(N).at[:-1].add(-0.5 * S_mid).at[1:].add(0.5 * S_mid)
    C_upper = jnp.zeros(N).at[:-1].add(0.5 * S_mid)

    # --- Combined operator L = -K + (r-q-sigma^2)*C - r*M ---
    # After integration by parts of 0.5*sigma^2*S^2*V'', stiffness enters
    # with a negative sign and the convection coefficient picks up -sigma^2.
    conv_coeff = r - q - sigma**2
    L_lower = -K_lower + conv_coeff * C_lower - r * M_lower
    L_main = -K_main + conv_coeff * C_main - r * M_main
    L_upper = -K_upper + conv_coeff * C_upper - r * M_upper

    # --- LHS = M - theta*dt*L,  RHS = M + (1-theta)*dt*L ---
    td = theta * dt
    otd = (1.0 - theta) * dt

    a_L = M_lower - td * L_lower
    b_L = M_main - td * L_main
    c_L = M_upper - td * L_upper

    a_R = M_lower + otd * L_lower
    b_R = M_main + otd * L_main
    c_R = M_upper + otd * L_upper

    # Boundary rows: LHS → identity, RHS → zero (BCs overwritten each step)
    a_L = a_L.at[0].set(0.0).at[-1].set(0.0)
    b_L = b_L.at[0].set(1.0).at[-1].set(1.0)
    c_L = c_L.at[0].set(0.0).at[-1].set(0.0)

    a_R = a_R.at[0].set(0.0).at[-1].set(0.0)
    b_R = b_R.at[0].set(0.0).at[-1].set(0.0)
    c_R = c_R.at[0].set(0.0).at[-1].set(0.0)

    return a_L, b_L, c_L, a_R, b_R, c_R


# ============================================================================
# Boundary conditions
# ============================================================================

def _lower_bc(tau: float, K: float, r: float, is_call: bool) -> float:
    return jnp.where(is_call, 0.0, K * jnp.exp(-r * tau))


def _upper_bc(
    S_max: float, tau: float, K: float, r: float, is_call: bool,
) -> float:
    return jnp.where(is_call, S_max - K * jnp.exp(-r * tau), 0.0)


# ============================================================================
# FE time-stepping via lax.scan
# ============================================================================

@jax.jit
def _jax_fe_solve(
    V_terminal: jnp.ndarray,
    a_L_cn: jnp.ndarray, b_L_cn: jnp.ndarray, c_L_cn: jnp.ndarray,
    a_R_cn: jnp.ndarray, b_R_cn: jnp.ndarray, c_R_cn: jnp.ndarray,
    a_L_im: jnp.ndarray, b_L_im: jnp.ndarray, c_L_im: jnp.ndarray,
    a_R_im: jnp.ndarray, b_R_im: jnp.ndarray, c_R_im: jnp.ndarray,
    K: float, T: float, r: float, S_max: float,
    dt: float,
    is_call: bool, american: bool,
    exercise_vals: jnp.ndarray,
    step_indices: jnp.ndarray,
    rannacher_threshold: int = 0,
) -> jnp.ndarray:
    """Full backward-in-time FE solve with Rannacher smoothing.

    Uses fully implicit steps for the first few time steps (near terminal
    condition) to damp oscillations from the non-smooth payoff, then
    switches to Crank-Nicolson for the remaining steps.

    rannacher_threshold: step indices >= this value use implicit; below use CN.
    """
    def step(V, step_idx):
        tau = T - (step_idx - 1) * dt

        # Select implicit or CN coefficients based on proximity to terminal
        use_implicit = step_idx >= rannacher_threshold
        a_L = jnp.where(use_implicit, a_L_im, a_L_cn)
        b_L = jnp.where(use_implicit, b_L_im, b_L_cn)
        c_L = jnp.where(use_implicit, c_L_im, c_L_cn)
        a_R = jnp.where(use_implicit, a_R_im, a_R_cn)
        b_R = jnp.where(use_implicit, b_R_im, b_R_cn)
        c_R = jnp.where(use_implicit, c_R_im, c_R_cn)

        rhs = _tridiag_matvec(a_R, b_R, c_R, V)

        bc_low = _lower_bc(tau, K, r, is_call)
        bc_up = _upper_bc(S_max, tau, K, r, is_call)
        rhs = rhs.at[0].set(bc_low)
        rhs = rhs.at[-1].set(bc_up)

        V_new = _thomas_solve(a_L, b_L, c_L, rhs)

        V_new = jnp.where(american, jnp.maximum(V_new, exercise_vals), V_new)
        return V_new, None

    V_final, _ = lax.scan(step, V_terminal, step_indices)
    return V_final


# ============================================================================
# Greeks from FE solution
# ============================================================================

def _fe_greeks(
    S0: float, nodes: jnp.ndarray, V: jnp.ndarray,
) -> dict:
    """Compute delta and gamma from the FE nodal solution."""
    idx = int(jnp.searchsorted(nodes, S0)) - 1
    idx = max(0, min(idx, len(nodes) - 2))

    h = float(nodes[idx + 1] - nodes[idx])
    delta = float((V[idx + 1] - V[idx]) / h)

    if idx > 0:
        h_left = float(nodes[idx] - nodes[idx - 1])
        delta_left = float((V[idx] - V[idx - 1]) / h_left)
        gamma = (delta - delta_left) / (0.5 * (h + h_left))
    else:
        gamma = 0.0

    return {"delta": delta, "gamma": float(gamma)}


# ============================================================================
# Public convenience wrapper
# ============================================================================

def jax_fe_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool = True,
    american: bool = False,
    N_elements: int = 100,
    N_t: int = 500,
    S_max: float | None = None,
    adaptive: bool = True,
    concentration: float = 3.0,
    compute_greeks: bool = False,
) -> float | dict:
    """Price a European or American vanilla option via finite elements (P1).

    Args:
        S0: Spot price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        q: Continuous dividend yield
        sigma: Volatility
        is_call: True for call, False for put
        american: True for American exercise
        N_elements: Number of elements (N_nodes = N_elements + 1)
        N_t: Number of time steps
        S_max: Upper boundary (auto-set if None)
        adaptive: Use tanh-clustered mesh near strike
        concentration: Clustering strength for adaptive mesh
        compute_greeks: Return dict with price, delta, gamma

    Returns:
        float price, or dict {'price', 'delta', 'gamma'} if compute_greeks
    """
    if T <= 0:
        intrinsic = max(S0 - K, 0.0) if is_call else max(K - S0, 0.0)
        if compute_greeks:
            return {"price": intrinsic, "delta": 0.0, "gamma": 0.0}
        return intrinsic

    if S_max is None:
        S_max = S0 * np.exp(3.0 * sigma * np.sqrt(T))

    N_nodes = N_elements + 1

    if adaptive:
        nodes = _adaptive_mesh(0.0, S_max, K, N_nodes, concentration)
    else:
        nodes = _uniform_mesh(0.0, S_max, N_nodes)

    dt = T / (N_t - 1)

    V_terminal = jnp.where(is_call, jnp.maximum(nodes - K, 0.0),
                           jnp.maximum(K - nodes, 0.0))
    exercise_vals = V_terminal

    # CN coefficients (theta=0.5)
    a_Lcn, b_Lcn, c_Lcn, a_Rcn, b_Rcn, c_Rcn = _assemble_fe_tridiag(
        nodes, sigma, r, q, dt, 0.5,
    )
    # Implicit coefficients (theta=1.0) for Rannacher smoothing
    a_Lim, b_Lim, c_Lim, a_Rim, b_Rim, c_Rim = _assemble_fe_tridiag(
        nodes, sigma, r, q, dt, 1.0,
    )

    step_indices = jnp.arange(N_t - 1, 0, -1)
    # Rannacher: use implicit for the first 4 steps (highest step indices)
    rannacher_threshold = N_t - 1 - 4
    V = _jax_fe_solve(
        V_terminal,
        a_Lcn, b_Lcn, c_Lcn, a_Rcn, b_Rcn, c_Rcn,
        a_Lim, b_Lim, c_Lim, a_Rim, b_Rim, c_Rim,
        K, T, r, float(S_max), dt, is_call, american, exercise_vals,
        step_indices, rannacher_threshold,
    )

    price = float(jnp.interp(S0, nodes, V))

    if compute_greeks:
        greeks = _fe_greeks(S0, nodes, V)
        greeks["price"] = price
        return greeks

    return price
