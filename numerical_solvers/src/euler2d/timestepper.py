"""Time integration and main solve loop for the 2D compressible Euler system.

Unsplit finite-volume update: one pair of (x-flux, y-flux) computations per
sub-step, combined into a single conservative update. Time integration is
Forward Euler (for ``reconstruction="constant"``) or SSP-RK3 (for
``reconstruction="weno5"``). Sub-stepping is CFL-adaptive within each
output interval ``dt``.
"""

import torch

from .boundary import apply_ghost_cells
from .flux import hll, hllc, rusanov
from .physics import conservative_to_primitive, sound_speed
from .weno import weno5_x, weno5_y

_MAX_SUBSTEPS = 2000


def _max_wave_speeds(
    rho: torch.Tensor,
    rho_u: torch.Tensor,
    rho_v: torch.Tensor,
    E: torch.Tensor,
    gamma: float,
    eps: float = 1e-12,
) -> tuple[float, float]:
    """Return (max(|u|+c), max(|v|+c)) over the grid (and batch)."""
    _, u, v, p = conservative_to_primitive(rho, rho_u, rho_v, E, gamma, eps=eps)
    c = sound_speed(rho, p, gamma, eps=eps)
    return (u.abs() + c).max().item(), (v.abs() + c).max().item()


def _compute_divergence(
    rho, rho_u, rho_v, E,
    *,
    nx: int, ny: int,
    dx: float, dy: float,
    bc_type: str,
    n_ghost: int,
    flux_fn,
    gamma: float,
    use_weno: bool,
):
    """Return the conservative spatial divergence of (rho, rho_u, rho_v, E).

    Output shapes match the interior grid (..., ny, nx).
    """
    rho_g, rho_u_g, rho_v_g, E_g = apply_ghost_cells(
        rho, rho_u, rho_v, E, bc_type, n_ghost=n_ghost,
    )

    if use_weno:
        rhoL_x, rhoR_x = weno5_x(rho_g)
        rho_uL_x, rho_uR_x = weno5_x(rho_u_g)
        rho_vL_x, rho_vR_x = weno5_x(rho_v_g)
        EL_x, ER_x = weno5_x(E_g)
        # Restrict to interior rows (strip y ghost cells) and to nx+1 x-ifaces
        def _trim_x(q):
            return q[..., n_ghost:-n_ghost, : nx + 1]
        rhoL_x = _trim_x(rhoL_x); rhoR_x = _trim_x(rhoR_x)
        rho_uL_x = _trim_x(rho_uL_x); rho_uR_x = _trim_x(rho_uR_x)
        rho_vL_x = _trim_x(rho_vL_x); rho_vR_x = _trim_x(rho_vR_x)
        EL_x = _trim_x(EL_x); ER_x = _trim_x(ER_x)

        rhoL_y, rhoR_y = weno5_y(rho_g)
        rho_uL_y, rho_uR_y = weno5_y(rho_u_g)
        rho_vL_y, rho_vR_y = weno5_y(rho_v_g)
        EL_y, ER_y = weno5_y(E_g)
        def _trim_y(q):
            return q[..., : ny + 1, n_ghost:-n_ghost]
        rhoL_y = _trim_y(rhoL_y); rhoR_y = _trim_y(rhoR_y)
        rho_uL_y = _trim_y(rho_uL_y); rho_uR_y = _trim_y(rho_uR_y)
        rho_vL_y = _trim_y(rho_vL_y); rho_vR_y = _trim_y(rho_vR_y)
        EL_y = _trim_y(EL_y); ER_y = _trim_y(ER_y)
    else:
        # 1st order: L = interior-1, R = interior in each direction.
        # Interior rows / columns of the ghost-padded state.
        interior = rho_g[..., n_ghost:-n_ghost if n_ghost > 0 else None, :]
        # x-direction interfaces (on the rows corresponding to interior y)
        def _x_lr(q):
            qi = q[..., n_ghost:-n_ghost, :]
            return qi[..., : nx + 1], qi[..., 1 : nx + 2]

        rhoL_x, rhoR_x = _x_lr(rho_g)
        rho_uL_x, rho_uR_x = _x_lr(rho_u_g)
        rho_vL_x, rho_vR_x = _x_lr(rho_v_g)
        EL_x, ER_x = _x_lr(E_g)

        def _y_lr(q):
            qi = q[..., :, n_ghost:-n_ghost]
            return qi[..., : ny + 1, :], qi[..., 1 : ny + 2, :]

        rhoL_y, rhoR_y = _y_lr(rho_g)
        rho_uL_y, rho_uR_y = _y_lr(rho_u_g)
        rho_vL_y, rho_vR_y = _y_lr(rho_v_g)
        EL_y, ER_y = _y_lr(E_g)
        _ = interior  # silence unused warning when weno branch inactive

    # X-sweep: rho_u is normal momentum, rho_v is tangential (passive)
    Fx_rho, Fx_rho_u, Fx_rho_v, Fx_E = flux_fn(
        rhoL_x, rho_uL_x, rho_vL_x, EL_x,
        rhoR_x, rho_uR_x, rho_vR_x, ER_x,
        gamma,
    )
    # Y-sweep: rho_v is normal momentum, rho_u is tangential (passive)
    Gy_rho, Gy_rho_v, Gy_rho_u, Gy_E = flux_fn(
        rhoL_y, rho_vL_y, rho_uL_y, EL_y,
        rhoR_y, rho_vR_y, rho_uR_y, ER_y,
        gamma,
    )

    drho = -(1.0 / dx) * (Fx_rho[..., :, 1:] - Fx_rho[..., :, :-1]) \
           - (1.0 / dy) * (Gy_rho[..., 1:, :] - Gy_rho[..., :-1, :])
    drho_u = -(1.0 / dx) * (Fx_rho_u[..., :, 1:] - Fx_rho_u[..., :, :-1]) \
             - (1.0 / dy) * (Gy_rho_u[..., 1:, :] - Gy_rho_u[..., :-1, :])
    drho_v = -(1.0 / dx) * (Fx_rho_v[..., :, 1:] - Fx_rho_v[..., :, :-1]) \
             - (1.0 / dy) * (Gy_rho_v[..., 1:, :] - Gy_rho_v[..., :-1, :])
    dE = -(1.0 / dx) * (Fx_E[..., :, 1:] - Fx_E[..., :, :-1]) \
         - (1.0 / dy) * (Gy_E[..., 1:, :] - Gy_E[..., :-1, :])

    return drho, drho_u, drho_v, dE


def _step(
    rho, rho_u, rho_v, E, dt_sub, *,
    nx, ny, dx, dy, bc_type, n_ghost, flux_fn, gamma, use_weno,
):
    kwargs = dict(
        nx=nx, ny=ny, dx=dx, dy=dy, bc_type=bc_type,
        n_ghost=n_ghost, flux_fn=flux_fn, gamma=gamma, use_weno=use_weno,
    )
    if use_weno:
        k1_rho, k1_ru, k1_rv, k1_E = _compute_divergence(
            rho, rho_u, rho_v, E, **kwargs,
        )
        rho_1 = (rho + dt_sub * k1_rho).clamp(min=0.0)
        rho_u_1 = rho_u + dt_sub * k1_ru
        rho_v_1 = rho_v + dt_sub * k1_rv
        E_1 = E + dt_sub * k1_E

        k2_rho, k2_ru, k2_rv, k2_E = _compute_divergence(
            rho_1, rho_u_1, rho_v_1, E_1, **kwargs,
        )
        rho_2 = (0.75 * rho + 0.25 * (rho_1 + dt_sub * k2_rho)).clamp(min=0.0)
        rho_u_2 = 0.75 * rho_u + 0.25 * (rho_u_1 + dt_sub * k2_ru)
        rho_v_2 = 0.75 * rho_v + 0.25 * (rho_v_1 + dt_sub * k2_rv)
        E_2 = 0.75 * E + 0.25 * (E_1 + dt_sub * k2_E)

        k3_rho, k3_ru, k3_rv, k3_E = _compute_divergence(
            rho_2, rho_u_2, rho_v_2, E_2, **kwargs,
        )
        rho_n = ((1 / 3) * rho + (2 / 3) * (rho_2 + dt_sub * k3_rho)).clamp(min=0.0)
        rho_u_n = (1 / 3) * rho_u + (2 / 3) * (rho_u_2 + dt_sub * k3_ru)
        rho_v_n = (1 / 3) * rho_v + (2 / 3) * (rho_v_2 + dt_sub * k3_rv)
        E_n = (1 / 3) * E + (2 / 3) * (E_2 + dt_sub * k3_E)
    else:
        k1_rho, k1_ru, k1_rv, k1_E = _compute_divergence(
            rho, rho_u, rho_v, E, **kwargs,
        )
        rho_n = (rho + dt_sub * k1_rho).clamp(min=0.0)
        rho_u_n = rho_u + dt_sub * k1_ru
        rho_v_n = rho_v + dt_sub * k1_rv
        E_n = E + dt_sub * k1_E
    return rho_n, rho_u_n, rho_v_n, E_n


def solve(
    rho0: torch.Tensor,
    rho_u0: torch.Tensor,
    rho_v0: torch.Tensor,
    E0: torch.Tensor,
    *,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    dt: float,
    nt: int,
    gamma: float,
    bc_type: str = "extrap",
    flux_type: str = "hllc",
    reconstruction: str = "constant",
    max_value: float | None = None,
    cfl: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Run the 2D Euler solver (single sample).

    Parameters
    ----------
    rho0, rho_u0, rho_v0, E0 : tensors of shape (ny, nx).

    Returns
    -------
    rho_hist, u_hist, v_hist, p_hist : tensors of shape (nt+1, ny, nx).
    valid : True if the solution remained finite (and within max_value).
    """
    use_weno = reconstruction == "weno5"
    n_ghost = 4 if use_weno else 1
    cfl_eff = cfl if cfl is not None else (0.5 if use_weno else 0.45)

    flux_fns = {"hllc": hllc, "hll": hll, "rusanov": rusanov}
    flux_fn = flux_fns[flux_type]

    rho = rho0.clone()
    rho_u = rho_u0.clone()
    rho_v = rho_v0.clone()
    E = E0.clone()

    rho_hist = torch.zeros(nt + 1, ny, nx, device=rho.device, dtype=rho.dtype)
    u_hist = torch.zeros_like(rho_hist)
    v_hist = torch.zeros_like(rho_hist)
    p_hist = torch.zeros_like(rho_hist)

    _, u0, v0, p0 = conservative_to_primitive(rho0, rho_u0, rho_v0, E0, gamma)
    rho_hist[0] = rho0
    u_hist[0] = u0
    v_hist[0] = v0
    p_hist[0] = p0

    step_kwargs = dict(
        nx=nx, ny=ny, dx=dx, dy=dy,
        bc_type=bc_type, n_ghost=n_ghost,
        flux_fn=flux_fn, gamma=gamma, use_weno=use_weno,
    )

    valid = True
    for n in range(nt):
        t_rem = dt
        n_subs = 0
        while t_rem > 1e-14:
            sx, sy = _max_wave_speeds(rho, rho_u, rho_v, E, gamma)
            dt_sub = cfl_eff / max(sx / dx + sy / dy, 1e-12)
            dt_sub = min(dt_sub, t_rem)

            rho, rho_u, rho_v, E = _step(
                rho, rho_u, rho_v, E, dt_sub, **step_kwargs,
            )
            # Floor internal energy to prevent negative pressure
            ke = 0.5 * (rho_u * rho_u + rho_v * rho_v) / rho.clamp(min=1e-12)
            E = torch.max(E, ke + 1e-10)
            t_rem -= dt_sub
            n_subs += 1

            if not torch.isfinite(rho).all():
                valid = False
                break
            if n_subs >= _MAX_SUBSTEPS:
                valid = False
                break

        if not valid:
            rho_hist[n + 1 :] = float("nan")
            u_hist[n + 1 :] = float("nan")
            v_hist[n + 1 :] = float("nan")
            p_hist[n + 1 :] = float("nan")
            break

        _, u_p, v_p, p_p = conservative_to_primitive(rho, rho_u, rho_v, E, gamma)
        rho_hist[n + 1] = rho
        u_hist[n + 1] = u_p
        v_hist[n + 1] = v_p
        p_hist[n + 1] = p_p

        is_bad = (
            not torch.isfinite(rho).all()
            or not torch.isfinite(u_p).all()
            or not torch.isfinite(v_p).all()
            or not torch.isfinite(p_p).all()
        )
        if not is_bad and max_value is not None:
            is_bad = (
                rho.abs().max() > max_value
                or u_p.abs().max() > max_value
                or v_p.abs().max() > max_value
                or p_p.abs().max() > max_value
            )
        if is_bad:
            rho_hist[n + 2 :] = float("nan")
            u_hist[n + 2 :] = float("nan")
            v_hist[n + 2 :] = float("nan")
            p_hist[n + 2 :] = float("nan")
            valid = False
            break

    return rho_hist, u_hist, v_hist, p_hist, valid


def solve_batch(
    rho0: torch.Tensor,
    rho_u0: torch.Tensor,
    rho_v0: torch.Tensor,
    E0: torch.Tensor,
    *,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    dt: float,
    nt: int,
    gamma: float,
    bc_type: str = "extrap",
    flux_type: str = "hllc",
    reconstruction: str = "constant",
    max_value: float | None = None,
    cfl: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the 2D Euler solver on a batch of samples.

    Parameters
    ----------
    rho0, rho_u0, rho_v0, E0 : tensors of shape (B, ny, nx).

    Returns
    -------
    rho_hist, u_hist, v_hist, p_hist : tensors of shape (B, nt+1, ny, nx).
    valid : boolean tensor of shape (B,).
    """
    B = rho0.shape[0]
    use_weno = reconstruction == "weno5"
    n_ghost = 4 if use_weno else 1
    cfl_eff = cfl if cfl is not None else (0.5 if use_weno else 0.45)

    flux_fns = {"hllc": hllc, "hll": hll, "rusanov": rusanov}
    flux_fn = flux_fns[flux_type]

    rho = rho0.clone()
    rho_u = rho_u0.clone()
    rho_v = rho_v0.clone()
    E = E0.clone()

    rho_hist = torch.zeros(B, nt + 1, ny, nx, device=rho.device, dtype=rho.dtype)
    u_hist = torch.zeros_like(rho_hist)
    v_hist = torch.zeros_like(rho_hist)
    p_hist = torch.zeros_like(rho_hist)

    _, u0, v0, p0 = conservative_to_primitive(rho0, rho_u0, rho_v0, E0, gamma)
    rho_hist[:, 0] = rho0
    u_hist[:, 0] = u0
    v_hist[:, 0] = v0
    p_hist[:, 0] = p0

    alive = torch.ones(B, dtype=torch.bool, device=rho.device)
    first_bad_step = torch.full((B,), nt + 1, dtype=torch.long, device=rho.device)

    step_kwargs = dict(
        nx=nx, ny=ny, dx=dx, dy=dy,
        bc_type=bc_type, n_ghost=n_ghost,
        flux_fn=flux_fn, gamma=gamma, use_weno=use_weno,
    )

    for step in range(nt):
        alive_exp = alive.view(-1, 1, 1)

        t_rem = dt
        n_subs = 0
        while t_rem > 1e-14:
            if not alive.any():
                break
            _, u_tmp, v_tmp, p_tmp = conservative_to_primitive(
                rho, rho_u, rho_v, E, gamma,
            )
            c_tmp = sound_speed(rho, p_tmp, gamma)
            sx = (u_tmp.abs() + c_tmp)
            sy = (v_tmp.abs() + c_tmp)
            ws_alive_x = torch.where(alive_exp, sx, torch.zeros_like(sx))
            ws_alive_y = torch.where(alive_exp, sy, torch.zeros_like(sy))
            s_x = ws_alive_x.max().item()
            s_y = ws_alive_y.max().item()
            dt_sub = cfl_eff / max(s_x / dx + s_y / dy, 1e-12)
            dt_sub = min(dt_sub, t_rem)

            new_rho, new_ru, new_rv, new_E = _step(
                rho, rho_u, rho_v, E, dt_sub, **step_kwargs,
            )
            rho = torch.where(alive_exp, new_rho, rho)
            rho_u = torch.where(alive_exp, new_ru, rho_u)
            rho_v = torch.where(alive_exp, new_rv, rho_v)
            E = torch.where(alive_exp, new_E, E)
            ke = 0.5 * (rho_u * rho_u + rho_v * rho_v) / rho.clamp(min=1e-12)
            E = torch.max(E, ke + 1e-10)

            t_rem -= dt_sub
            n_subs += 1

            finite_ok = (
                torch.isfinite(rho).reshape(B, -1).all(dim=-1)
                & torch.isfinite(rho_u).reshape(B, -1).all(dim=-1)
                & torch.isfinite(rho_v).reshape(B, -1).all(dim=-1)
                & torch.isfinite(E).reshape(B, -1).all(dim=-1)
            )
            newly_dead = alive & ~finite_ok
            if newly_dead.any():
                first_bad_step = torch.where(newly_dead, step + 1, first_bad_step)
                alive = alive & finite_ok
                alive_exp = alive.view(-1, 1, 1)

            if n_subs >= _MAX_SUBSTEPS:
                if alive.any():
                    first_bad_step = torch.where(alive, step + 1, first_bad_step)
                    alive = alive & False
                break

        if not alive.any():
            rho_hist[:, step + 2 :] = float("nan")
            u_hist[:, step + 2 :] = float("nan")
            v_hist[:, step + 2 :] = float("nan")
            p_hist[:, step + 2 :] = float("nan")
            break

        _, u_p, v_p, p_p = conservative_to_primitive(rho, rho_u, rho_v, E, gamma)
        rho_hist[:, step + 1] = rho
        u_hist[:, step + 1] = u_p
        v_hist[:, step + 1] = v_p
        p_hist[:, step + 1] = p_p

        per_sample_finite = (
            torch.isfinite(rho).reshape(B, -1).all(dim=-1)
            & torch.isfinite(u_p).reshape(B, -1).all(dim=-1)
            & torch.isfinite(v_p).reshape(B, -1).all(dim=-1)
            & torch.isfinite(p_p).reshape(B, -1).all(dim=-1)
        )
        step_ok = per_sample_finite
        if max_value is not None:
            step_ok = step_ok & (rho.reshape(B, -1).abs().amax(dim=-1) <= max_value)
            step_ok = step_ok & (u_p.reshape(B, -1).abs().amax(dim=-1) <= max_value)
            step_ok = step_ok & (v_p.reshape(B, -1).abs().amax(dim=-1) <= max_value)
            step_ok = step_ok & (p_p.reshape(B, -1).abs().amax(dim=-1) <= max_value)

        newly_dead = alive & ~step_ok
        if newly_dead.any():
            first_bad_step = torch.where(newly_dead, step + 1, first_bad_step)
        alive = alive & step_ok

        if not alive.any():
            rho_hist[:, step + 2 :] = float("nan")
            u_hist[:, step + 2 :] = float("nan")
            v_hist[:, step + 2 :] = float("nan")
            p_hist[:, step + 2 :] = float("nan")
            break

    valid = first_bad_step > nt
    for b in range(B):
        s = first_bad_step[b].item()
        if s <= nt:
            rho_hist[b, s + 1 :] = float("nan")
            u_hist[b, s + 1 :] = float("nan")
            v_hist[b, s + 1 :] = float("nan")
            p_hist[b, s + 1 :] = float("nan")

    return rho_hist, u_hist, v_hist, p_hist, valid
