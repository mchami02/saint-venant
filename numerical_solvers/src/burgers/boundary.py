"""Ghost cell boundary conditions for the inviscid Burgers equation (PyTorch)."""

import torch


def apply_ghost_cells(
    u: torch.Tensor,
    bc_type: str,
    *,
    n_ghost: int,
) -> torch.Tensor:
    """Build an extended state with *n_ghost* ghost cells on each side.

    Parameters
    ----------
    u : tensor of shape (..., nx).
    bc_type : "extrap", "periodic", or "wall" (reflecting, sign-flip).
    n_ghost : number of ghost cells per side.
    """
    ng = n_ghost

    if bc_type == "periodic":
        return torch.cat([u[..., -ng:], u, u[..., :ng]], dim=-1)

    if bc_type == "wall":
        # Reflecting BC: the conserved variable flips sign across the wall
        # (since f(u) = u^2/2 is symmetric and a wall imposes u = 0).
        return torch.cat(
            [-u[..., :ng].flip(-1), u, -u[..., -ng:].flip(-1)],
            dim=-1,
        )

    # extrap (zero-gradient, default)
    left = u[..., :1].expand(*u.shape[:-1], ng)
    right = u[..., -1:].expand(*u.shape[:-1], ng)
    return torch.cat([left, u, right], dim=-1)
