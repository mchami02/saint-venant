"""Time integration and main solve loop for the ARZ system (PyTorch)."""

from collections.abc import Callable

import torch

from .boundary import apply_ghost_cells
from .flux import hll, rusanov
from .physics import pressure
from .weno import weno5_reconstruct


def solve(
    rho0: torch.Tensor,
    rho_w0: torch.Tensor,
    *,
    nx: int,
    dx: float,
    dt: float,
    nt: int,
    gamma: float,
    bc_type: str = "zero_gradient",
    flux_type: str = "hll",
    reconstruction: str = "weno5",
    bc_left: tuple[float, float] | None = None,
    bc_right: tuple[float, float] | None = None,
    bc_left_time: Callable[[float], tuple[float, float]] | None = None,
    max_value: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Run the ARZ solver.

    Parameters
    ----------
    rho0, rho_w0 : 1-D tensors of length *nx* — initial conservative state.
    nx, dx, dt, nt : grid / time parameters.
    gamma : pressure exponent.
    bc_type : boundary condition type.
    flux_type : "rusanov" or "hll".
    reconstruction : "constant" (1st order) or "weno5" (5th order).
    bc_left, bc_right : static Dirichlet values (rho, v).
    bc_left_time : time-varying left BC callable.

    max_value : if set, terminate early when any value exceeds this threshold.

    Returns
    -------
    rho_hist, w_hist, v_hist : tensors of shape (nt+1, nx).
    valid : True if the solution remained finite (and within max_value).
    """
    eps = 1e-12
    use_weno = reconstruction == "weno5"
    n_ghost = 4 if use_weno else 1

    flux_fn = hll if flux_type == "hll" else rusanov

    rho = rho0.clone()
    rho_w = rho_w0.clone()

    rho_hist = torch.zeros(nt + 1, nx, device=rho.device, dtype=rho.dtype)
    w_hist = torch.zeros_like(rho_hist)
    v_hist = torch.zeros_like(rho_hist)

    rho_hist[0] = rho
    w_hist[0] = torch.where(rho > eps, rho_w / rho, torch.zeros_like(rho_w))
    v_hist[0] = w_hist[0] - pressure(rho, gamma)

    # ------------------------------------------------------------------ RHS
    def _compute_rhs(
        rho_loc: torch.Tensor, rho_w_loc: torch.Tensor, t_loc: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rho_g, rho_w_g = apply_ghost_cells(
            rho_loc,
            rho_w_loc,
            bc_type,
            t_loc,
            n_ghost=n_ghost,
            gamma=gamma,
            bc_left=bc_left,
            bc_right=bc_right,
            bc_left_time=bc_left_time,
        )

        if use_weno:
            rho_L, rho_R = weno5_reconstruct(rho_g)
            rho_w_L, rho_w_R = weno5_reconstruct(rho_w_g)
            rho_L = rho_L[..., : nx + 1]
            rho_R = rho_R[..., : nx + 1]
            rho_w_L = rho_w_L[..., : nx + 1]
            rho_w_R = rho_w_R[..., : nx + 1]
        else:
            # Constant reconstruction: left state = cell i, right = cell i+1
            rho_L = rho_g[..., : nx + 1]
            rho_R = rho_g[..., 1 : nx + 2]
            rho_w_L = rho_w_g[..., : nx + 1]
            rho_w_R = rho_w_g[..., 1 : nx + 2]

        f_rho, f_rw = flux_fn(rho_L, rho_w_L, rho_R, rho_w_R, gamma)

        drho = -(1.0 / dx) * (f_rho[..., 1:] - f_rho[..., :-1])
        drho_w = -(1.0 / dx) * (f_rw[..., 1:] - f_rw[..., :-1])
        return drho, drho_w

    # --------------------------------------------------------------- march
    valid = True
    for n in range(nt):
        t = n * dt

        if use_weno:
            # SSP-RK3
            k1_rho, k1_rw = _compute_rhs(rho, rho_w, t)
            rho_1 = (rho + dt * k1_rho).clamp(min=0.0)
            rho_w_1 = rho_w + dt * k1_rw
            rho_w_1 = torch.where(rho_1 > 0, rho_w_1, torch.zeros_like(rho_w_1))

            k2_rho, k2_rw = _compute_rhs(rho_1, rho_w_1, t + dt)
            rho_2 = (0.75 * rho + 0.25 * (rho_1 + dt * k2_rho)).clamp(min=0.0)
            rho_w_2 = 0.75 * rho_w + 0.25 * (rho_w_1 + dt * k2_rw)
            rho_w_2 = torch.where(rho_2 > 0, rho_w_2, torch.zeros_like(rho_w_2))

            k3_rho, k3_rw = _compute_rhs(rho_2, rho_w_2, t + 0.5 * dt)
            rho = ((1 / 3) * rho + (2 / 3) * (rho_2 + dt * k3_rho)).clamp(min=0.0)
            rho_w = (1 / 3) * rho_w + (2 / 3) * (rho_w_2 + dt * k3_rw)
            rho_w = torch.where(rho > 0, rho_w, torch.zeros_like(rho_w))
        else:
            # Forward Euler
            k1_rho, k1_rw = _compute_rhs(rho, rho_w, t)
            rho = (rho + dt * k1_rho).clamp(min=0.0)
            rho_w = rho_w + dt * k1_rw
            rho_w = torch.where(rho > 0, rho_w, torch.zeros_like(rho_w))

        w = torch.where(rho > eps, rho_w / rho, torch.zeros_like(rho_w))
        v = w - pressure(rho, gamma)

        rho_hist[n + 1] = rho
        w_hist[n + 1] = w
        v_hist[n + 1] = v

        # Check for NaN/Inf always; extreme values only if max_value is set
        is_bad = not torch.isfinite(rho).all() or not torch.isfinite(v).all()
        if not is_bad and max_value is not None:
            is_bad = rho.abs().max() > max_value or v.abs().max() > max_value
        if is_bad:
            rho_hist[n + 2 :] = float("nan")
            w_hist[n + 2 :] = float("nan")
            v_hist[n + 2 :] = float("nan")
            valid = False
            break

    return rho_hist, w_hist, v_hist, valid


def solve_batch(
    rho0: torch.Tensor,
    rho_w0: torch.Tensor,
    *,
    nx: int,
    dx: float,
    dt: float,
    nt: int,
    gamma: float,
    bc_type: str = "zero_gradient",
    flux_type: str = "hll",
    reconstruction: str = "weno5",
    bc_left: tuple[float, float] | None = None,
    bc_right: tuple[float, float] | None = None,
    max_value: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch-parallel ARZ solver.

    Parameters
    ----------
    rho0, rho_w0 : 2-D tensors of shape (B, nx) — initial conservative state.
    nx, dx, dt, nt : grid / time parameters.
    gamma : pressure exponent.
    bc_type : boundary condition type (``"time_varying_inflow"`` not supported).
    flux_type : "rusanov" or "hll".
    reconstruction : "constant" (1st order) or "weno5" (5th order).
    bc_left, bc_right : static Dirichlet values (rho, v).
    max_value : if set, freeze samples whose values exceed this threshold.

    Returns
    -------
    rho_hist, w_hist, v_hist : tensors of shape (B, nt+1, nx).
    valid : boolean tensor of shape (B,) — True for samples that stayed finite.
    """
    if bc_type == "time_varying_inflow":
        raise ValueError(
            "time_varying_inflow is not supported in batch mode. "
            "Use solve() for single-sample time-varying BCs."
        )

    eps = 1e-12
    B = rho0.shape[0]
    use_weno = reconstruction == "weno5"
    n_ghost = 4 if use_weno else 1
    flux_fn = hll if flux_type == "hll" else rusanov

    rho = rho0.clone()
    rho_w = rho_w0.clone()

    rho_hist = torch.zeros(B, nt + 1, nx, device=rho.device, dtype=rho.dtype)
    w_hist = torch.zeros_like(rho_hist)
    v_hist = torch.zeros_like(rho_hist)

    rho_hist[:, 0] = rho
    w0 = torch.where(rho > eps, rho_w / rho, torch.zeros_like(rho_w))
    w_hist[:, 0] = w0
    v_hist[:, 0] = w0 - pressure(rho, gamma)

    alive = torch.ones(B, dtype=torch.bool, device=rho.device)
    first_bad_step = torch.full((B,), nt + 1, dtype=torch.long, device=rho.device)

    # ------------------------------------------------------------------ RHS
    def _compute_rhs_batch(
        rho_loc: torch.Tensor, rho_w_loc: torch.Tensor, t_loc: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rho_g, rho_w_g = apply_ghost_cells(
            rho_loc,
            rho_w_loc,
            bc_type,
            t_loc,
            n_ghost=n_ghost,
            gamma=gamma,
            bc_left=bc_left,
            bc_right=bc_right,
        )

        if use_weno:
            rho_L, rho_R = weno5_reconstruct(rho_g)
            rho_w_L, rho_w_R = weno5_reconstruct(rho_w_g)
            rho_L = rho_L[..., : nx + 1]
            rho_R = rho_R[..., : nx + 1]
            rho_w_L = rho_w_L[..., : nx + 1]
            rho_w_R = rho_w_R[..., : nx + 1]
        else:
            rho_L = rho_g[..., : nx + 1]
            rho_R = rho_g[..., 1 : nx + 2]
            rho_w_L = rho_w_g[..., : nx + 1]
            rho_w_R = rho_w_g[..., 1 : nx + 2]

        f_rho, f_rw = flux_fn(rho_L, rho_w_L, rho_R, rho_w_R, gamma)

        drho = -(1.0 / dx) * (f_rho[..., 1:] - f_rho[..., :-1])
        drho_w = -(1.0 / dx) * (f_rw[..., 1:] - f_rw[..., :-1])
        return drho, drho_w

    # --------------------------------------------------------------- march
    for step in range(nt):
        t = step * dt
        alive_exp = alive.unsqueeze(-1)  # (B, 1)

        if use_weno:
            # SSP-RK3
            k1_rho, k1_rw = _compute_rhs_batch(rho, rho_w, t)
            rho_1 = (rho + dt * k1_rho).clamp(min=0.0)
            rho_w_1 = rho_w + dt * k1_rw
            rho_w_1 = torch.where(rho_1 > 0, rho_w_1, torch.zeros_like(rho_w_1))

            k2_rho, k2_rw = _compute_rhs_batch(rho_1, rho_w_1, t + dt)
            rho_2 = (0.75 * rho + 0.25 * (rho_1 + dt * k2_rho)).clamp(min=0.0)
            rho_w_2 = 0.75 * rho_w + 0.25 * (rho_w_1 + dt * k2_rw)
            rho_w_2 = torch.where(rho_2 > 0, rho_w_2, torch.zeros_like(rho_w_2))

            k3_rho, k3_rw = _compute_rhs_batch(rho_2, rho_w_2, t + 0.5 * dt)
            new_rho = (
                (1 / 3) * rho + (2 / 3) * (rho_2 + dt * k3_rho)
            ).clamp(min=0.0)
            new_rho_w = (1 / 3) * rho_w + (2 / 3) * (rho_w_2 + dt * k3_rw)
            new_rho_w = torch.where(
                new_rho > 0, new_rho_w, torch.zeros_like(new_rho_w)
            )
        else:
            # Forward Euler
            k1_rho, k1_rw = _compute_rhs_batch(rho, rho_w, t)
            new_rho = (rho + dt * k1_rho).clamp(min=0.0)
            new_rho_w = rho_w + dt * k1_rw
            new_rho_w = torch.where(
                new_rho > 0, new_rho_w, torch.zeros_like(new_rho_w)
            )

        # Freeze dead samples at their last valid state
        rho = torch.where(alive_exp, new_rho, rho)
        rho_w = torch.where(alive_exp, new_rho_w, rho_w)

        w = torch.where(rho > eps, rho_w / rho, torch.zeros_like(rho_w))
        v = w - pressure(rho, gamma)

        rho_hist[:, step + 1] = rho
        w_hist[:, step + 1] = w
        v_hist[:, step + 1] = v

        # Per-sample validity check
        step_ok = torch.isfinite(rho).all(dim=-1) & torch.isfinite(v).all(dim=-1)
        if max_value is not None:
            step_ok = step_ok & (rho.abs().amax(dim=-1) <= max_value)
            step_ok = step_ok & (v.abs().amax(dim=-1) <= max_value)

        newly_dead = alive & ~step_ok
        if newly_dead.any():
            first_bad_step = torch.where(newly_dead, step + 1, first_bad_step)
        alive = alive & step_ok

        if not alive.any():
            rho_hist[:, step + 2 :] = float("nan")
            w_hist[:, step + 2 :] = float("nan")
            v_hist[:, step + 2 :] = float("nan")
            break

    # Post-loop: fill NaN from each sample's failure step onward
    valid = first_bad_step > nt
    for b in range(B):
        s = first_bad_step[b].item()
        if s <= nt:
            rho_hist[b, s + 1 :] = float("nan")
            w_hist[b, s + 1 :] = float("nan")
            v_hist[b, s + 1 :] = float("nan")

    return rho_hist, w_hist, v_hist, valid
