"""Ghost cell boundary conditions for the 1D Euler system (PyTorch)."""

import torch


def apply_ghost_cells(
    rho: torch.Tensor,
    rho_u: torch.Tensor,
    E: torch.Tensor,
    bc_type: str,
    *,
    n_ghost: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build extended state with *n_ghost* ghost cells on each side.

    Parameters
    ----------
    rho, rho_u, E : 1-D tensors of length nx (conservative variables).
    bc_type : one of "extrap", "periodic", "wall".
    n_ghost : number of ghost cells per side (1 for constant, 4 for WENO-5).
    """
    ng = n_ghost

    if bc_type == "periodic":
        rho_g = torch.cat([rho[..., -ng:], rho, rho[..., :ng]], dim=-1)
        rho_u_g = torch.cat([rho_u[..., -ng:], rho_u, rho_u[..., :ng]], dim=-1)
        E_g = torch.cat([E[..., -ng:], E, E[..., :ng]], dim=-1)

    elif bc_type == "wall":
        # Reflecting: density and energy are mirrored, momentum flips sign
        rho_g = torch.cat([rho[..., :ng].flip(-1), rho, rho[..., -ng:].flip(-1)], dim=-1)
        rho_u_g = torch.cat([-rho_u[..., :ng].flip(-1), rho_u, -rho_u[..., -ng:].flip(-1)], dim=-1)
        E_g = torch.cat([E[..., :ng].flip(-1), E, E[..., -ng:].flip(-1)], dim=-1)

    else:
        # extrap (zero-gradient, default)
        rho_g = torch.cat(
            [rho[..., :1].expand(*rho.shape[:-1], ng), rho, rho[..., -1:].expand(*rho.shape[:-1], ng)], dim=-1
        )
        rho_u_g = torch.cat(
            [rho_u[..., :1].expand(*rho_u.shape[:-1], ng), rho_u, rho_u[..., -1:].expand(*rho_u.shape[:-1], ng)],
            dim=-1,
        )
        E_g = torch.cat(
            [E[..., :1].expand(*E.shape[:-1], ng), E, E[..., -1:].expand(*E.shape[:-1], ng)], dim=-1
        )

    return rho_g, rho_u_g, E_g
