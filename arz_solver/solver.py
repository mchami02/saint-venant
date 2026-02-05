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


def rusanov_flux(uL: np.ndarray, uR: np.ndarray, gamma: float, eps: float = 1e-10) -> np.ndarray:
    """
    Rusanov (local Lax-Friedrichs) numerical flux for ARZ.
    uL, uR are (2,) arrays [rho, rho*w].
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

    Returns
    -------
    rho_history, w_history, v_history : (nt+1, nx) arrays
    """
    nx, nt = config.nx, config.nt
    dx, dt, gamma = config.dx, config.dt, config.gamma
    eps = 1e-12

    rho = rho0.copy()
    rho_w = rho * w0.copy()

    rho_history = np.zeros((nt + 1, nx))
    w_history = np.zeros((nt + 1, nx))
    v_history = np.zeros((nt + 1, nx))

    rho_history[0] = rho
    w_history[0] = rho_w / (rho + eps)
    v_history[0] = w_history[0] - pressure(rho, gamma)

    for n in range(nt):
        t = n * dt
        rho_g, rho_w_g = _ghost_cells(rho, rho_w, bc_type, t, config)

        fluxes = np.zeros((nx + 1, 2))
        for i in range(nx + 1):
            uL = np.array([rho_g[i], rho_w_g[i]])
            uR = np.array([rho_g[i + 1], rho_w_g[i + 1]])
            fluxes[i] = rusanov_flux(uL, uR, gamma)

        rho_new = rho - (dt / dx) * (fluxes[1:, 0] - fluxes[:-1, 0])
        rho_w_new = rho_w - (dt / dx) * (fluxes[1:, 1] - fluxes[:-1, 1])

        rho = rho_new
        rho_w = rho_w_new

        w = rho_w / (rho + eps)
        v = w - pressure(rho, gamma)

        rho_history[n + 1] = rho
        w_history[n + 1] = w
        v_history[n + 1] = v

    return rho_history, w_history, v_history
