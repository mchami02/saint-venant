"""ARZ model: pressure, Rusanov flux, and time stepping with boundary conditions."""

from typing import Tuple
import numpy as np
from .config import ARZConfig


def pressure(rho: np.ndarray, gamma: float) -> np.ndarray:
    """Pressure p(ρ) = ρ^γ."""
    return np.power(rho, gamma)


def dp_drho(rho: np.ndarray, gamma: float) -> np.ndarray:
    """Derivative p'(ρ)."""
    if gamma == 1.0:
        return np.full_like(rho, gamma)
    return gamma * np.power(rho, gamma - 1)


def _get_eigenvalues(rho: float, v: float, gamma: float) -> Tuple[float, float]:
    """Return eigenvalues (λ1, λ2) for ARZ at given state."""
    # λ1 = v, λ2 = v - ρ * p'(ρ)
    lam1 = v
    lam2 = v - rho * gamma * (rho ** (gamma - 1)) if rho > 1e-10 else v
    return lam1, lam2


def rusanov_flux(uL: np.ndarray, uR: np.ndarray, gamma: float, eps: float = 1e-10) -> np.ndarray:
    """
    Rusanov (local Lax-Friedrichs) numerical flux for ARZ.
    uL, uR are (2,) arrays [rho, rho*w].
    Most diffusive but most stable.
    """
    rhoL, rho_wL = uL[0], uL[1]
    rhoR, rho_wR = uR[0], uR[1]

    wL = rho_wL / rhoL if rhoL > eps else 0.0
    wR = rho_wR / rhoR if rhoR > eps else 0.0
    vL = wL - pressure(np.array([rhoL]), gamma)[0]
    vR = wR - pressure(np.array([rhoR]), gamma)[0]

    fL = np.array([rhoL * vL, rho_wL * vL])
    fR = np.array([rhoR * vR, rho_wR * vR])

    sL = max(abs(vL), abs(vL - rhoL * dp_drho(np.array([rhoL]), gamma)[0]))
    sR = max(abs(vR), abs(vR - rhoR * dp_drho(np.array([rhoR]), gamma)[0]))
    smax = max(sL, sR)

    return 0.5 * (fL + fR) - 0.5 * smax * (uR - uL)


def hll_flux(uL: np.ndarray, uR: np.ndarray, gamma: float, eps: float = 1e-10) -> np.ndarray:
    """
    HLL (Harten-Lax-van Leer) numerical flux for ARZ.
    Less diffusive than Rusanov, uses wave speed estimates.
    """
    rhoL, rho_wL = uL[0], uL[1]
    rhoR, rho_wR = uR[0], uR[1]

    wL = rho_wL / rhoL if rhoL > eps else 0.0
    wR = rho_wR / rhoR if rhoR > eps else 0.0
    vL = wL - pressure(np.array([rhoL]), gamma)[0]
    vR = wR - pressure(np.array([rhoR]), gamma)[0]

    # Physical fluxes
    fL = np.array([rhoL * vL, rho_wL * vL])
    fR = np.array([rhoR * vR, rho_wR * vR])

    # Wave speed estimates using eigenvalues
    lam1L, lam2L = _get_eigenvalues(rhoL, vL, gamma)
    lam1R, lam2R = _get_eigenvalues(rhoR, vR, gamma)

    # HLL wave speeds: min/max of all eigenvalues
    sL = min(lam1L, lam2L, lam1R, lam2R)
    sR = max(lam1L, lam2L, lam1R, lam2R)

    # HLL flux formula
    if sL >= 0:
        return fL
    elif sR <= 0:
        return fR
    else:
        return (sR * fL - sL * fR + sL * sR * (uR - uL)) / (sR - sL)


def _weno5_weights(b0, b1, b2, d0, d1, d2):
    """Compute WENO nonlinear weights from smoothness indicators."""
    eps = 1e-6
    w0 = d0 / (eps + b0)**2
    w1 = d1 / (eps + b1)**2
    w2 = d2 / (eps + b2)**2
    ws = w0 + w1 + w2
    return w0 / ws, w1 / ws, w2 / ws


def _weno5_reconstruct(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    WENO-5 reconstruction from cell averages (vectorized).

    Given cell averages v[0..N-1] with 4 ghost cells on each side,
    returns (v_minus, v_plus) at nx+1 cell interfaces.

    v_minus[j] = v^-_{(j+3)+1/2} (value from left of interface)
    v_plus[j]  = v^+_{(j+3)+1/2} (value from right of interface)
    """
    N = len(v)
    # Physical cells at indices 4..N-5, interfaces at i+1/2 for i=3..N-5
    # That gives N-8+1 = nx+1 interfaces
    ni = N - 7  # number of interfaces

    # Slices for v^- at interface i+1/2, i = 3..N-5
    # Stencil: {v_{i-2}, v_{i-1}, v_i, v_{i+1}, v_{i+2}}
    a = v[1:1+ni]   # v_{i-2}
    b = v[2:2+ni]   # v_{i-1}
    c = v[3:3+ni]   # v_i
    d = v[4:4+ni]   # v_{i+1}
    e = v[5:5+ni]   # v_{i+2}

    q0 = (1/3)*a - (7/6)*b + (11/6)*c
    q1 = -(1/6)*b + (5/6)*c + (1/3)*d
    q2 = (1/3)*c + (5/6)*d - (1/6)*e

    b0 = (13/12)*(a - 2*b + c)**2 + (1/4)*(a - 4*b + 3*c)**2
    b1 = (13/12)*(b - 2*c + d)**2 + (1/4)*(b - d)**2
    b2 = (13/12)*(c - 2*d + e)**2 + (1/4)*(3*c - 4*d + e)**2

    w0, w1, w2 = _weno5_weights(b0, b1, b2, 1/10, 6/10, 3/10)
    v_minus = w0*q0 + w1*q1 + w2*q2

    # Slices for v^+ at interface i+1/2, i = 3..N-5
    # Stencil centered on cell i+1: {v_{i-1}, v_i, v_{i+1}, v_{i+2}, v_{i+3}}
    p = v[2:2+ni]   # v_{i-1}
    q = v[3:3+ni]   # v_i
    r = v[4:4+ni]   # v_{i+1}
    s = v[5:5+ni]   # v_{i+2}
    t = v[6:6+ni]   # v_{i+3}

    q0r = (1/3)*t - (7/6)*s + (11/6)*r
    q1r = -(1/6)*s + (5/6)*r + (1/3)*q
    q2r = (1/3)*r + (5/6)*q - (1/6)*p

    b0r = (13/12)*(t - 2*s + r)**2 + (1/4)*(t - 4*s + 3*r)**2
    b1r = (13/12)*(s - 2*r + q)**2 + (1/4)*(s - q)**2
    b2r = (13/12)*(r - 2*q + p)**2 + (1/4)*(3*r - 4*q + p)**2

    w0r, w1r, w2r = _weno5_weights(b0r, b1r, b2r, 1/10, 6/10, 3/10)
    v_plus = w0r*q0r + w1r*q1r + w2r*q2r

    return v_minus, v_plus


def _hll_flux_vectorized(rhoL: np.ndarray, rho_wL: np.ndarray,
                         rhoR: np.ndarray, rho_wR: np.ndarray,
                         gamma: float, eps: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized HLL flux for arrays of left/right states."""
    wL = np.where(rhoL > eps, rho_wL / rhoL, 0.0)
    wR = np.where(rhoR > eps, rho_wR / rhoR, 0.0)
    vL = wL - np.power(rhoL, gamma)
    vR = wR - np.power(rhoR, gamma)

    fL_rho = rhoL * vL
    fL_rw = rho_wL * vL
    fR_rho = rhoR * vR
    fR_rw = rho_wR * vR

    dpL = gamma * np.power(np.maximum(rhoL, eps), gamma - 1)
    dpR = gamma * np.power(np.maximum(rhoR, eps), gamma - 1)

    lam1L, lam2L = vL, vL - rhoL * dpL
    lam1R, lam2R = vR, vR - rhoR * dpR

    sL = np.minimum(np.minimum(lam1L, lam2L), np.minimum(lam1R, lam2R))
    sR = np.maximum(np.maximum(lam1L, lam2L), np.maximum(lam1R, lam2R))

    denom = sR - sL
    denom = np.where(np.abs(denom) < 1e-14, 1e-14, denom)

    hll_rho = (sR * fL_rho - sL * fR_rho + sL * sR * (rhoR - rhoL)) / denom
    hll_rw = (sR * fL_rw - sL * fR_rw + sL * sR * (rho_wR - rho_wL)) / denom

    flux_rho = np.where(sL >= 0, fL_rho, np.where(sR <= 0, fR_rho, hll_rho))
    flux_rw = np.where(sL >= 0, fL_rw, np.where(sR <= 0, fR_rw, hll_rw))

    return flux_rho, flux_rw


def _ghost_cells_weno(
    rho: np.ndarray,
    rho_w: np.ndarray,
    bc_type: str,
    t: float,
    config: "ARZConfig",
) -> Tuple[np.ndarray, np.ndarray]:
    """Build extended state with 4 ghost cells on each side for WENO-5."""
    if bc_type == "periodic":
        rho_g = np.concatenate([rho[-4:], rho, rho[:4]])
        rho_w_g = np.concatenate([rho_w[-4:], rho_w, rho_w[:4]])
    else:
        rho_g = np.concatenate([[rho[0]]*4, rho, [rho[-1]]*4])
        rho_w_g = np.concatenate([[rho_w[0]]*4, rho_w, [rho_w[-1]]*4])

    return rho_g, rho_w_g


def _ghost_cells(
    rho: np.ndarray,
    rho_w: np.ndarray,
    bc_type: str,
    t: float,
    config: ARZConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build extended state with ghost cells for the chosen BC."""
    nx = rho.size
    eps = 1e-12

    if config.bc_left is not None:
        rho_left, v_left = config.bc_left[0], config.bc_left[1]
        w_left = v_left + pressure(np.array([rho_left]), config.gamma)[0]
        rho_w_left = rho_left * w_left
    else:
        rho_left, v_left = 0.5, 1.0
        w_left = v_left + pressure(np.array([rho_left]), config.gamma)[0]
        rho_w_left = rho_left * w_left

    if config.bc_right is not None:
        rho_right, v_right = config.bc_right[0], config.bc_right[1]
        w_right = v_right + pressure(np.array([rho_right]), config.gamma)[0]
        rho_w_right = rho_right * w_right
    else:
        rho_right, v_right = 0.3, 0.5
        w_right = v_right + pressure(np.array([rho_right]), config.gamma)[0]
        rho_w_right = rho_right * w_right

    if bc_type == "periodic":
        rho_g = np.concatenate([[rho[-1]], rho, [rho[0]]])
        rho_w_g = np.concatenate([[rho_w[-1]], rho_w, [rho_w[0]]])
    elif bc_type == "inflow_outflow":
        rho_g = np.concatenate([[rho_left], rho, [rho[-1]]])
        rho_w_g = np.concatenate([[rho_w_left], rho_w, [rho_w[-1]]])
    elif bc_type == "time_varying_inflow":
        if config.bc_left_time is not None:
            rho_left_t, v_left_t = config.bc_left_time(t)
        else:
            rho_left_t = rho_left + 2 * np.sin(2 * np.pi * t / 2.0)
            v_left_t = v_left + 0.1 * np.sin(2 * np.pi * t / 1.5)
        w_left_t = v_left_t + pressure(np.array([rho_left_t]), config.gamma)[0]
        rho_w_left_t = rho_left_t * w_left_t
        rho_g = np.concatenate([[rho_left_t], rho, [rho[-1]]])
        rho_w_g = np.concatenate([[rho_w_left_t], rho_w, [rho_w[-1]]])
    elif bc_type == "dirichlet":
        rho_g = np.concatenate([[rho_left], rho, [rho_right]])
        rho_w_g = np.concatenate([[rho_w_left], rho_w, [rho_w_right]])
    else:
        # zero_gradient
        rho_g = np.concatenate([[rho[0]], rho, [rho[-1]]])
        rho_w_g = np.concatenate([[rho_w[0]], rho_w, [rho_w[-1]]])

    return rho_g, rho_w_g


def run(
    rho0: np.ndarray,
    w0: np.ndarray,
    config: ARZConfig,
    bc_type: str = "zero_gradient",
    flux_type: str = "rusanov",
    reconstruction: str = "constant",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run ARZ solver from initial (rho0, w0) with given boundary condition.

    Parameters
    ----------
    rho0, w0 : 1D arrays (length nx)
        Initial density and effective velocity w = v + p(ρ).
    config : ARZConfig
        Grid, time, and model parameters.
    bc_type : str
        One of: "zero_gradient", "periodic", "inflow_outflow",
        "time_varying_inflow", "dirichlet".
    flux_type : str
        One of: "rusanov" (most diffusive), "hll" (less diffusive).
    reconstruction : str
        One of: "constant" (first-order Godunov), "weno5" (fifth-order WENO).

    Returns
    -------
    rho_history, w_history, v_history : (nt+1, nx) arrays
    """
    nx, nt = config.nx, config.nt
    dx, dt, gamma = config.dx, config.dt, config.gamma
    eps = 1e-12

    if flux_type == "hll":
        flux_fn = hll_flux
    else:
        flux_fn = rusanov_flux

    use_weno = reconstruction == "weno5"

    rho = rho0.copy()
    rho_w = rho * w0.copy()

    rho_history = np.zeros((nt + 1, nx))
    w_history = np.zeros((nt + 1, nx))
    v_history = np.zeros((nt + 1, nx))

    rho_history[0] = rho
    w_history[0] = rho_w / (rho + eps)
    v_history[0] = w_history[0] - pressure(rho, gamma)

    def _compute_rhs(rho_loc, rho_w_loc, t_loc):
        """Compute -1/dx * (F_{i+1/2} - F_{i-1/2}) for all cells."""
        if use_weno:
            rho_g, rho_w_g = _ghost_cells_weno(rho_loc, rho_w_loc, bc_type, t_loc, config)
            rho_L, rho_R = _weno5_reconstruct(rho_g)
            rho_w_L, rho_w_R = _weno5_reconstruct(rho_w_g)

            f_rho, f_rw = _hll_flux_vectorized(rho_L, rho_w_L, rho_R, rho_w_R, gamma)
            f_rho = f_rho[:nx+1]
            f_rw = f_rw[:nx+1]

            drho = -(1.0 / dx) * (f_rho[1:] - f_rho[:-1])
            drho_w = -(1.0 / dx) * (f_rw[1:] - f_rw[:-1])
            return drho, drho_w
        else:
            rho_g, rho_w_g = _ghost_cells(rho_loc, rho_w_loc, bc_type, t_loc, config)
            fluxes = np.zeros((nx + 1, 2))
            for i in range(nx + 1):
                uL = np.array([rho_g[i], rho_w_g[i]])
                uR = np.array([rho_g[i + 1], rho_w_g[i + 1]])
                fluxes[i] = flux_fn(uL, uR, gamma)

        drho = -(1.0 / dx) * (fluxes[1:, 0] - fluxes[:-1, 0])
        drho_w = -(1.0 / dx) * (fluxes[1:, 1] - fluxes[:-1, 1])
        return drho, drho_w

    for n in range(nt):
        t = n * dt

        if use_weno:
            # SSP-RK3 time integration (3rd-order, TVD)
            k1_rho, k1_rw = _compute_rhs(rho, rho_w, t)
            rho_1 = np.maximum(rho + dt * k1_rho, 0.0)
            rho_w_1 = rho_w + dt * k1_rw

            k2_rho, k2_rw = _compute_rhs(rho_1, rho_w_1, t + dt)
            rho_2 = np.maximum(0.75 * rho + 0.25 * (rho_1 + dt * k2_rho), 0.0)
            rho_w_2 = 0.75 * rho_w + 0.25 * (rho_w_1 + dt * k2_rw)

            k3_rho, k3_rw = _compute_rhs(rho_2, rho_w_2, t + 0.5 * dt)
            rho_new = np.maximum((1/3) * rho + (2/3) * (rho_2 + dt * k3_rho), 0.0)
            rho_w_new = (1/3) * rho_w + (2/3) * (rho_w_2 + dt * k3_rw)
        else:
            # Forward Euler
            k1_rho, k1_rw = _compute_rhs(rho, rho_w, t)
            rho_new = np.maximum(rho + dt * k1_rho, 0.0)
            rho_w_new = rho_w + dt * k1_rw

        rho = rho_new
        rho_w = rho_w_new

        w = rho_w / (rho + eps)
        v = w - pressure(rho, gamma)

        rho_history[n + 1] = rho
        w_history[n + 1] = w
        v_history[n + 1] = v

    return rho_history, w_history, v_history
