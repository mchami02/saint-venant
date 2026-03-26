"""Time integration and main solve loop for the 1D Euler system (PyTorch)."""

import torch

from .boundary import apply_ghost_cells
from .flux import hll, hllc, rusanov
from .physics import conservative_to_primitive, primitive_to_conservative
from .weno import weno5_reconstruct


def solve(
    rho0: torch.Tensor,
    rho_u0: torch.Tensor,
    E0: torch.Tensor,
    *,
    nx: int,
    dx: float,
    dt: float,
    nt: int,
    gamma: float,
    bc_type: str = "extrap",
    flux_type: str = "hllc",
    reconstruction: str = "weno5",
    max_value: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Run the 1D Euler solver.

    Parameters
    ----------
    rho0, rho_u0, E0 : 1-D tensors of length *nx* — initial conservative state.
    nx, dx, dt, nt : grid / time parameters.
    gamma : ratio of specific heats.
    bc_type : "extrap", "periodic", or "wall".
    flux_type : "hllc", "hll", or "rusanov".
    reconstruction : "constant" (1st order) or "weno5" (5th order).
    max_value : if set, terminate early when any value exceeds this threshold.

    Returns
    -------
    rho_hist, u_hist, p_hist : tensors of shape (nt+1, nx).
    valid : True if the solution remained finite (and within max_value).
    """
    use_weno = reconstruction == "weno5"
    n_ghost = 4 if use_weno else 1

    flux_fns = {"hllc": hllc, "hll": hll, "rusanov": rusanov}
    flux_fn = flux_fns[flux_type]

    rho = rho0.clone()
    rho_u = rho_u0.clone()
    E = E0.clone()

    rho_hist = torch.zeros(nt + 1, nx, device=rho.device, dtype=rho.dtype)
    u_hist = torch.zeros_like(rho_hist)
    p_hist = torch.zeros_like(rho_hist)

    _, u0_prim, p0_prim = conservative_to_primitive(rho0, rho_u0, E0, gamma)
    rho_hist[0] = rho0
    u_hist[0] = u0_prim
    p_hist[0] = p0_prim

    # ------------------------------------------------------------------ RHS
    def _compute_rhs(
        rho_loc: torch.Tensor,
        rho_u_loc: torch.Tensor,
        E_loc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rho_g, rho_u_g, E_g = apply_ghost_cells(
            rho_loc, rho_u_loc, E_loc, bc_type, n_ghost=n_ghost
        )

        if use_weno:
            rhoL, rhoR = weno5_reconstruct(rho_g)
            rho_uL, rho_uR = weno5_reconstruct(rho_u_g)
            EL, ER = weno5_reconstruct(E_g)
            rhoL = rhoL[: nx + 1]
            rhoR = rhoR[: nx + 1]
            rho_uL = rho_uL[: nx + 1]
            rho_uR = rho_uR[: nx + 1]
            EL = EL[: nx + 1]
            ER = ER[: nx + 1]
        else:
            rhoL = rho_g[: nx + 1]
            rhoR = rho_g[1 : nx + 2]
            rho_uL = rho_u_g[: nx + 1]
            rho_uR = rho_u_g[1 : nx + 2]
            EL = E_g[: nx + 1]
            ER = E_g[1 : nx + 2]

        f_rho, f_rho_u, f_E = flux_fn(
            rhoL, rho_uL, EL, rhoR, rho_uR, ER, gamma
        )

        drho = -(1.0 / dx) * (f_rho[1:] - f_rho[:-1])
        drho_u = -(1.0 / dx) * (f_rho_u[1:] - f_rho_u[:-1])
        dE = -(1.0 / dx) * (f_E[1:] - f_E[:-1])
        return drho, drho_u, dE

    # --------------------------------------------------------------- march
    valid = True
    for n in range(nt):
        if use_weno:
            # SSP-RK3
            k1_rho, k1_rho_u, k1_E = _compute_rhs(rho, rho_u, E)
            rho_1 = (rho + dt * k1_rho).clamp(min=0.0)
            rho_u_1 = rho_u + dt * k1_rho_u
            E_1 = E + dt * k1_E

            k2_rho, k2_rho_u, k2_E = _compute_rhs(rho_1, rho_u_1, E_1)
            rho_2 = (0.75 * rho + 0.25 * (rho_1 + dt * k2_rho)).clamp(min=0.0)
            rho_u_2 = 0.75 * rho_u + 0.25 * (rho_u_1 + dt * k2_rho_u)
            E_2 = 0.75 * E + 0.25 * (E_1 + dt * k2_E)

            k3_rho, k3_rho_u, k3_E = _compute_rhs(rho_2, rho_u_2, E_2)
            rho = (
                (1 / 3) * rho + (2 / 3) * (rho_2 + dt * k3_rho)
            ).clamp(min=0.0)
            rho_u = (1 / 3) * rho_u + (2 / 3) * (rho_u_2 + dt * k3_rho_u)
            E = (1 / 3) * E + (2 / 3) * (E_2 + dt * k3_E)
        else:
            # Forward Euler
            k1_rho, k1_rho_u, k1_E = _compute_rhs(rho, rho_u, E)
            rho = (rho + dt * k1_rho).clamp(min=0.0)
            rho_u = rho_u + dt * k1_rho_u
            E = E + dt * k1_E

        _, u_prim, p_prim = conservative_to_primitive(rho, rho_u, E, gamma)

        rho_hist[n + 1] = rho
        u_hist[n + 1] = u_prim
        p_hist[n + 1] = p_prim

        # Check for NaN/Inf always; extreme values only if max_value is set
        is_bad = (
            not torch.isfinite(rho).all()
            or not torch.isfinite(u_prim).all()
            or not torch.isfinite(p_prim).all()
        )
        if not is_bad and max_value is not None:
            is_bad = (
                rho.abs().max() > max_value
                or u_prim.abs().max() > max_value
                or p_prim.abs().max() > max_value
            )
        if is_bad:
            rho_hist[n + 2 :] = float("nan")
            u_hist[n + 2 :] = float("nan")
            p_hist[n + 2 :] = float("nan")
            valid = False
            break

    return rho_hist, u_hist, p_hist, valid
