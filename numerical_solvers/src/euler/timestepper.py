"""Time integration and main solve loop for the 1D Euler system (PyTorch).

Uses CFL-based adaptive sub-stepping: within each output time interval dt,
the solver takes multiple smaller steps whose size is determined by the
maximum wave speed, ensuring the CFL condition is always satisfied.
"""

import torch

from .boundary import apply_ghost_cells
from .flux import hll, hllc, rusanov
from .physics import conservative_to_primitive, sound_speed
from .weno import weno5_reconstruct

_MAX_SUBSTEPS = 1000


def _max_wave_speed(
    rho: torch.Tensor,
    rho_u: torch.Tensor,
    E: torch.Tensor,
    gamma: float,
    eps: float = 1e-12,
) -> float:
    """Compute max(|u| + c) over all cells (and all batch samples).

    Works for both 1-D (nx,) and 2-D (B, nx) tensors.
    """
    _, u, p = conservative_to_primitive(rho, rho_u, E, gamma, eps=eps)
    c = sound_speed(rho, p, gamma, eps=eps)
    return (u.abs() + c).max().item()


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
    cfl: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Run the 1D Euler solver.

    Parameters
    ----------
    rho0, rho_u0, E0 : 1-D tensors of length *nx* — initial conservative state.
    nx, dx, dt, nt : grid / time parameters.  *dt* is the output time interval;
        the actual integration step is determined adaptively from the CFL condition.
    gamma : ratio of specific heats.
    bc_type : "extrap", "periodic", or "wall".
    flux_type : "hllc", "hll", or "rusanov".
    reconstruction : "constant" (1st order) or "weno5" (5th order).
    max_value : if set, terminate early when any value exceeds this threshold.
    cfl : CFL number for adaptive sub-stepping.  Defaults to 0.9 (weno5) or
        0.5 (constant).

    Returns
    -------
    rho_hist, u_hist, p_hist : tensors of shape (nt+1, nx).
    valid : True if the solution remained finite (and within max_value).
    """
    use_weno = reconstruction == "weno5"
    n_ghost = 4 if use_weno else 1
    cfl_eff = cfl if cfl is not None else (0.9 if use_weno else 0.5)

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
            rhoL = rhoL[..., : nx + 1]
            rhoR = rhoR[..., : nx + 1]
            rho_uL = rho_uL[..., : nx + 1]
            rho_uR = rho_uR[..., : nx + 1]
            EL = EL[..., : nx + 1]
            ER = ER[..., : nx + 1]
        else:
            rhoL = rho_g[..., : nx + 1]
            rhoR = rho_g[..., 1 : nx + 2]
            rho_uL = rho_u_g[..., : nx + 1]
            rho_uR = rho_u_g[..., 1 : nx + 2]
            EL = E_g[..., : nx + 1]
            ER = E_g[..., 1 : nx + 2]

        f_rho, f_rho_u, f_E = flux_fn(
            rhoL, rho_uL, EL, rhoR, rho_uR, ER, gamma
        )

        drho = -(1.0 / dx) * (f_rho[..., 1:] - f_rho[..., :-1])
        drho_u = -(1.0 / dx) * (f_rho_u[..., 1:] - f_rho_u[..., :-1])
        dE = -(1.0 / dx) * (f_E[..., 1:] - f_E[..., :-1])
        return drho, drho_u, dE

    # ----------------------------------------------------- single sub-step
    def _step(rho_s, rho_u_s, E_s, dt_sub):
        if use_weno:
            # SSP-RK3
            k1_rho, k1_rho_u, k1_E = _compute_rhs(rho_s, rho_u_s, E_s)
            rho_1 = (rho_s + dt_sub * k1_rho).clamp(min=0.0)
            rho_u_1 = rho_u_s + dt_sub * k1_rho_u
            E_1 = E_s + dt_sub * k1_E

            k2_rho, k2_rho_u, k2_E = _compute_rhs(rho_1, rho_u_1, E_1)
            rho_2 = (0.75 * rho_s + 0.25 * (rho_1 + dt_sub * k2_rho)).clamp(min=0.0)
            rho_u_2 = 0.75 * rho_u_s + 0.25 * (rho_u_1 + dt_sub * k2_rho_u)
            E_2 = 0.75 * E_s + 0.25 * (E_1 + dt_sub * k2_E)

            k3_rho, k3_rho_u, k3_E = _compute_rhs(rho_2, rho_u_2, E_2)
            rho_n = (
                (1 / 3) * rho_s + (2 / 3) * (rho_2 + dt_sub * k3_rho)
            ).clamp(min=0.0)
            rho_u_n = (1 / 3) * rho_u_s + (2 / 3) * (rho_u_2 + dt_sub * k3_rho_u)
            E_n = (1 / 3) * E_s + (2 / 3) * (E_2 + dt_sub * k3_E)
        else:
            # Forward Euler
            k1_rho, k1_rho_u, k1_E = _compute_rhs(rho_s, rho_u_s, E_s)
            rho_n = (rho_s + dt_sub * k1_rho).clamp(min=0.0)
            rho_u_n = rho_u_s + dt_sub * k1_rho_u
            E_n = E_s + dt_sub * k1_E
        return rho_n, rho_u_n, E_n

    # --------------------------------------------------------------- march
    valid = True
    for n in range(nt):
        # Adaptive sub-stepping within this output interval
        t_rem = dt
        n_subs = 0
        while t_rem > 1e-14:
            s_max = _max_wave_speed(rho, rho_u, E, gamma)
            dt_sub = min(cfl_eff * dx / max(s_max, 1e-12), t_rem)
            rho, rho_u, E = _step(rho, rho_u, E, dt_sub)
            # Enforce minimum internal energy (prevents negative pressure)
            ke = 0.5 * rho_u**2 / rho.clamp(min=1e-12)
            E = torch.max(E, ke + 1e-10)
            t_rem -= dt_sub
            n_subs += 1

            # Bail early on NaN
            if not torch.isfinite(rho).all():
                valid = False
                break

            if n_subs >= _MAX_SUBSTEPS:
                valid = False
                break

        if not valid:
            rho_hist[n + 1 :] = float("nan")
            u_hist[n + 1 :] = float("nan")
            p_hist[n + 1 :] = float("nan")
            break

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


def solve_batch(
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
    cfl: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch-parallel 1D Euler solver.

    Parameters
    ----------
    rho0, rho_u0, E0 : 2-D tensors of shape (B, nx) — initial conservative state.
    nx, dx, dt, nt : grid / time parameters.  *dt* is the output time interval;
        the actual integration step is determined adaptively from the CFL condition.
    gamma : ratio of specific heats.
    bc_type : "extrap", "periodic", or "wall".
    flux_type : "hllc", "hll", or "rusanov".
    reconstruction : "constant" (1st order) or "weno5" (5th order).
    max_value : if set, freeze samples whose values exceed this threshold.
    cfl : CFL number for adaptive sub-stepping.  Defaults to 0.9 (weno5) or
        0.5 (constant).  The sub-step dt is shared across all alive samples
        (determined by the most restrictive wave speed).

    Returns
    -------
    rho_hist, u_hist, p_hist : tensors of shape (B, nt+1, nx).
    valid : boolean tensor of shape (B,) — True for samples that stayed finite.
    """
    B = rho0.shape[0]
    use_weno = reconstruction == "weno5"
    n_ghost = 4 if use_weno else 1
    cfl_eff = cfl if cfl is not None else (0.9 if use_weno else 0.5)

    flux_fns = {"hllc": hllc, "hll": hll, "rusanov": rusanov}
    flux_fn = flux_fns[flux_type]

    rho = rho0.clone()
    rho_u = rho_u0.clone()
    E = E0.clone()

    rho_hist = torch.zeros(B, nt + 1, nx, device=rho.device, dtype=rho.dtype)
    u_hist = torch.zeros_like(rho_hist)
    p_hist = torch.zeros_like(rho_hist)

    _, u0_prim, p0_prim = conservative_to_primitive(rho0, rho_u0, E0, gamma)
    rho_hist[:, 0] = rho0
    u_hist[:, 0] = u0_prim
    p_hist[:, 0] = p0_prim

    alive = torch.ones(B, dtype=torch.bool, device=rho.device)
    first_bad_step = torch.full((B,), nt + 1, dtype=torch.long, device=rho.device)

    # ------------------------------------------------------------------ RHS
    def _compute_rhs_batch(
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
            rhoL = rhoL[..., : nx + 1]
            rhoR = rhoR[..., : nx + 1]
            rho_uL = rho_uL[..., : nx + 1]
            rho_uR = rho_uR[..., : nx + 1]
            EL = EL[..., : nx + 1]
            ER = ER[..., : nx + 1]
        else:
            rhoL = rho_g[..., : nx + 1]
            rhoR = rho_g[..., 1 : nx + 2]
            rho_uL = rho_u_g[..., : nx + 1]
            rho_uR = rho_u_g[..., 1 : nx + 2]
            EL = E_g[..., : nx + 1]
            ER = E_g[..., 1 : nx + 2]

        f_rho, f_rho_u, f_E = flux_fn(
            rhoL, rho_uL, EL, rhoR, rho_uR, ER, gamma
        )

        drho = -(1.0 / dx) * (f_rho[..., 1:] - f_rho[..., :-1])
        drho_u = -(1.0 / dx) * (f_rho_u[..., 1:] - f_rho_u[..., :-1])
        dE = -(1.0 / dx) * (f_E[..., 1:] - f_E[..., :-1])
        return drho, drho_u, dE

    # ----------------------------------------------------- single sub-step
    def _step_batch(rho_s, rho_u_s, E_s, dt_sub, alive_exp):
        if use_weno:
            # SSP-RK3
            k1_rho, k1_rho_u, k1_E = _compute_rhs_batch(rho_s, rho_u_s, E_s)
            rho_1 = (rho_s + dt_sub * k1_rho).clamp(min=0.0)
            rho_u_1 = rho_u_s + dt_sub * k1_rho_u
            E_1 = E_s + dt_sub * k1_E

            k2_rho, k2_rho_u, k2_E = _compute_rhs_batch(rho_1, rho_u_1, E_1)
            rho_2 = (0.75 * rho_s + 0.25 * (rho_1 + dt_sub * k2_rho)).clamp(min=0.0)
            rho_u_2 = 0.75 * rho_u_s + 0.25 * (rho_u_1 + dt_sub * k2_rho_u)
            E_2 = 0.75 * E_s + 0.25 * (E_1 + dt_sub * k2_E)

            k3_rho, k3_rho_u, k3_E = _compute_rhs_batch(rho_2, rho_u_2, E_2)
            new_rho = (
                (1 / 3) * rho_s + (2 / 3) * (rho_2 + dt_sub * k3_rho)
            ).clamp(min=0.0)
            new_rho_u = (1 / 3) * rho_u_s + (2 / 3) * (rho_u_2 + dt_sub * k3_rho_u)
            new_E = (1 / 3) * E_s + (2 / 3) * (E_2 + dt_sub * k3_E)
        else:
            # Forward Euler
            k1_rho, k1_rho_u, k1_E = _compute_rhs_batch(rho_s, rho_u_s, E_s)
            new_rho = (rho_s + dt_sub * k1_rho).clamp(min=0.0)
            new_rho_u = rho_u_s + dt_sub * k1_rho_u
            new_E = E_s + dt_sub * k1_E

        # Freeze dead samples at their last valid state
        rho_n = torch.where(alive_exp, new_rho, rho_s)
        rho_u_n = torch.where(alive_exp, new_rho_u, rho_u_s)
        E_n = torch.where(alive_exp, new_E, E_s)
        return rho_n, rho_u_n, E_n

    # --------------------------------------------------------------- march
    for step in range(nt):
        alive_exp = alive.unsqueeze(-1)  # (B, 1)

        # Adaptive sub-stepping within this output interval
        t_rem = dt
        n_subs = 0
        while t_rem > 1e-14:
            if not alive.any():
                break

            # Max wave speed across alive samples
            _, u_tmp, p_tmp = conservative_to_primitive(rho, rho_u, E, gamma)
            c_tmp = sound_speed(rho, p_tmp, gamma)
            ws = u_tmp.abs() + c_tmp  # (B, nx)
            ws_alive = torch.where(alive_exp, ws, torch.zeros_like(ws))
            s_max = ws_alive.max().item()

            dt_sub = min(cfl_eff * dx / max(s_max, 1e-12), t_rem)
            rho, rho_u, E = _step_batch(rho, rho_u, E, dt_sub, alive_exp)
            # Enforce minimum internal energy (prevents negative pressure)
            ke = 0.5 * rho_u**2 / rho.clamp(min=1e-12)
            E = torch.max(E, ke + 1e-10)
            t_rem -= dt_sub
            n_subs += 1

            # Bail early on NaN — mark affected samples as dead
            finite_ok = torch.isfinite(rho).all(dim=-1)
            newly_dead = alive & ~finite_ok
            if newly_dead.any():
                first_bad_step = torch.where(newly_dead, step + 1, first_bad_step)
                alive = alive & finite_ok
                alive_exp = alive.unsqueeze(-1)

            if n_subs >= _MAX_SUBSTEPS:
                # Mark remaining alive samples as dead
                still_alive = alive.clone()
                if still_alive.any():
                    first_bad_step = torch.where(
                        still_alive, step + 1, first_bad_step
                    )
                    alive = alive & False
                    alive_exp = alive.unsqueeze(-1)
                break

        if not alive.any():
            rho_hist[:, step + 2 :] = float("nan")
            u_hist[:, step + 2 :] = float("nan")
            p_hist[:, step + 2 :] = float("nan")
            break

        _, u_prim, p_prim = conservative_to_primitive(rho, rho_u, E, gamma)

        rho_hist[:, step + 1] = rho
        u_hist[:, step + 1] = u_prim
        p_hist[:, step + 1] = p_prim

        # Per-sample validity check
        step_ok = (
            torch.isfinite(rho).all(dim=-1)
            & torch.isfinite(u_prim).all(dim=-1)
            & torch.isfinite(p_prim).all(dim=-1)
        )
        if max_value is not None:
            step_ok = step_ok & (rho.abs().amax(dim=-1) <= max_value)
            step_ok = step_ok & (u_prim.abs().amax(dim=-1) <= max_value)
            step_ok = step_ok & (p_prim.abs().amax(dim=-1) <= max_value)

        newly_dead = alive & ~step_ok
        if newly_dead.any():
            first_bad_step = torch.where(newly_dead, step + 1, first_bad_step)
        alive = alive & step_ok

        if not alive.any():
            rho_hist[:, step + 2 :] = float("nan")
            u_hist[:, step + 2 :] = float("nan")
            p_hist[:, step + 2 :] = float("nan")
            break

    # Post-loop: fill NaN from each sample's failure step onward
    valid = first_bad_step > nt
    for b in range(B):
        s = first_bad_step[b].item()
        if s <= nt:
            rho_hist[b, s + 1 :] = float("nan")
            u_hist[b, s + 1 :] = float("nan")
            p_hist[b, s + 1 :] = float("nan")

    return rho_hist, u_hist, p_hist, valid
