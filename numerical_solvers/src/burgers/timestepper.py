"""Time integration and main solve loop for the inviscid Burgers equation.

Uses CFL-based adaptive sub-stepping: within each output time interval dt,
the solver takes multiple smaller steps whose size is determined by the
maximum wave speed |u|, ensuring the CFL condition is always satisfied.

Control flow mirrors ``src/euler/timestepper.py`` with a single conserved
scalar ``u`` instead of (rho, rho_u, E).
"""

import torch

from .boundary import apply_ghost_cells
from .flux import godunov, rusanov
from .weno import weno5_reconstruct

_MAX_SUBSTEPS = 1000


def _max_speed(u: torch.Tensor) -> float:
    """Max |u| across all cells (and all batch samples if ndim >= 2)."""
    return u.abs().max().item()


def solve(
    u0: torch.Tensor,
    *,
    nx: int,
    dx: float,
    dt: float,
    nt: int,
    bc_type: str = "extrap",
    flux_type: str = "godunov",
    reconstruction: str = "weno5",
    max_value: float | None = None,
    cfl: float | None = None,
) -> tuple[torch.Tensor, bool]:
    """Run the 1D Burgers solver.

    Parameters
    ----------
    u0 : 1-D tensor of length *nx* — initial conserved state.
    nx, dx, dt, nt : grid / time parameters. *dt* is the output interval;
        the actual integration step is determined adaptively from CFL.
    bc_type : "extrap", "periodic", or "wall".
    flux_type : "godunov" or "rusanov".
    reconstruction : "constant" (1st order FE) or "weno5" (SSP-RK3).
    max_value : if set, terminate early when |u| exceeds this threshold.
    cfl : CFL number.  Defaults to 0.9 (weno5) or 0.5 (constant).

    Returns
    -------
    u_hist : tensor of shape (nt+1, nx).
    valid : True if the solution remained finite.
    """
    use_weno = reconstruction == "weno5"
    n_ghost = 4 if use_weno else 1
    cfl_eff = cfl if cfl is not None else (0.9 if use_weno else 0.5)

    flux_fns = {"godunov": godunov, "rusanov": rusanov}
    flux_fn = flux_fns[flux_type]

    u = u0.clone()
    u_hist = torch.zeros(nt + 1, nx, device=u.device, dtype=u.dtype)
    u_hist[0] = u0

    def _compute_rhs(u_loc: torch.Tensor) -> torch.Tensor:
        u_g = apply_ghost_cells(u_loc, bc_type, n_ghost=n_ghost)

        if use_weno:
            uL_full, uR_full = weno5_reconstruct(u_g)
            uL = uL_full[..., : nx + 1]
            uR = uR_full[..., : nx + 1]
        else:
            uL = u_g[..., : nx + 1]
            uR = u_g[..., 1 : nx + 2]

        F = flux_fn(uL, uR)  # (..., nx+1)
        return -(1.0 / dx) * (F[..., 1:] - F[..., :-1])

    def _step(u_s: torch.Tensor, dt_sub: float) -> torch.Tensor:
        if use_weno:
            k1 = _compute_rhs(u_s)
            u1 = u_s + dt_sub * k1
            k2 = _compute_rhs(u1)
            u2 = 0.75 * u_s + 0.25 * (u1 + dt_sub * k2)
            k3 = _compute_rhs(u2)
            return (1.0 / 3.0) * u_s + (2.0 / 3.0) * (u2 + dt_sub * k3)
        k1 = _compute_rhs(u_s)
        return u_s + dt_sub * k1

    valid = True
    for n in range(nt):
        t_rem = dt
        n_subs = 0
        while t_rem > 1e-14:
            s_max = _max_speed(u)
            dt_sub = min(cfl_eff * dx / max(s_max, 1e-12), t_rem)
            u = _step(u, dt_sub)
            t_rem -= dt_sub
            n_subs += 1

            if not torch.isfinite(u).all():
                valid = False
                break
            if n_subs >= _MAX_SUBSTEPS:
                valid = False
                break

        if not valid:
            u_hist[n + 1 :] = float("nan")
            break

        u_hist[n + 1] = u

        is_bad = not torch.isfinite(u).all()
        if not is_bad and max_value is not None:
            is_bad = u.abs().max() > max_value
        if is_bad:
            u_hist[n + 2 :] = float("nan")
            valid = False
            break

    return u_hist, valid


def solve_batch(
    u0: torch.Tensor,
    *,
    nx: int,
    dx: float,
    dt: float,
    nt: int,
    bc_type: str = "extrap",
    flux_type: str = "godunov",
    reconstruction: str = "weno5",
    max_value: float | None = None,
    cfl: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch-parallel 1D Burgers solver.

    Parameters
    ----------
    u0 : 2-D tensor of shape (B, nx).

    Returns
    -------
    u_hist : tensor of shape (B, nt+1, nx).
    valid : boolean tensor of shape (B,).
    """
    B = u0.shape[0]
    use_weno = reconstruction == "weno5"
    n_ghost = 4 if use_weno else 1
    cfl_eff = cfl if cfl is not None else (0.9 if use_weno else 0.5)

    flux_fns = {"godunov": godunov, "rusanov": rusanov}
    flux_fn = flux_fns[flux_type]

    u = u0.clone()
    u_hist = torch.zeros(B, nt + 1, nx, device=u.device, dtype=u.dtype)
    u_hist[:, 0] = u0

    alive = torch.ones(B, dtype=torch.bool, device=u.device)
    first_bad_step = torch.full((B,), nt + 1, dtype=torch.long, device=u.device)

    def _compute_rhs_batch(u_loc: torch.Tensor) -> torch.Tensor:
        u_g = apply_ghost_cells(u_loc, bc_type, n_ghost=n_ghost)
        if use_weno:
            uL_full, uR_full = weno5_reconstruct(u_g)
            uL = uL_full[..., : nx + 1]
            uR = uR_full[..., : nx + 1]
        else:
            uL = u_g[..., : nx + 1]
            uR = u_g[..., 1 : nx + 2]
        F = flux_fn(uL, uR)
        return -(1.0 / dx) * (F[..., 1:] - F[..., :-1])

    def _step_batch(u_s: torch.Tensor, dt_sub: float, alive_exp: torch.Tensor) -> torch.Tensor:
        if use_weno:
            k1 = _compute_rhs_batch(u_s)
            u1 = u_s + dt_sub * k1
            k2 = _compute_rhs_batch(u1)
            u2 = 0.75 * u_s + 0.25 * (u1 + dt_sub * k2)
            k3 = _compute_rhs_batch(u2)
            new_u = (1.0 / 3.0) * u_s + (2.0 / 3.0) * (u2 + dt_sub * k3)
        else:
            k1 = _compute_rhs_batch(u_s)
            new_u = u_s + dt_sub * k1
        return torch.where(alive_exp, new_u, u_s)

    for step in range(nt):
        alive_exp = alive.unsqueeze(-1)

        t_rem = dt
        n_subs = 0
        while t_rem > 1e-14:
            if not alive.any():
                break
            ws = u.abs()
            ws_alive = torch.where(alive_exp, ws, torch.zeros_like(ws))
            s_max = ws_alive.max().item()
            dt_sub = min(cfl_eff * dx / max(s_max, 1e-12), t_rem)
            u = _step_batch(u, dt_sub, alive_exp)
            t_rem -= dt_sub
            n_subs += 1

            finite_ok = torch.isfinite(u).all(dim=-1)
            newly_dead = alive & ~finite_ok
            if newly_dead.any():
                first_bad_step = torch.where(newly_dead, step + 1, first_bad_step)
                alive = alive & finite_ok
                alive_exp = alive.unsqueeze(-1)

            if n_subs >= _MAX_SUBSTEPS:
                if alive.any():
                    first_bad_step = torch.where(alive, step + 1, first_bad_step)
                    alive = alive & False
                    alive_exp = alive.unsqueeze(-1)
                break

        if not alive.any():
            u_hist[:, step + 2 :] = float("nan")
            break

        u_hist[:, step + 1] = u

        step_ok = torch.isfinite(u).all(dim=-1)
        if max_value is not None:
            step_ok = step_ok & (u.abs().amax(dim=-1) <= max_value)

        newly_dead = alive & ~step_ok
        if newly_dead.any():
            first_bad_step = torch.where(newly_dead, step + 1, first_bad_step)
        alive = alive & step_ok

        if not alive.any():
            u_hist[:, step + 2 :] = float("nan")
            break

    valid = first_bad_step > nt
    for b in range(B):
        s = first_bad_step[b].item()
        if s <= nt:
            u_hist[b, s + 1 :] = float("nan")

    return u_hist, valid
