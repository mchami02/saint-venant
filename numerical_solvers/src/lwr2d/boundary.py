"""Ghost cell boundary conditions for 2D scalar conservation laws (PyTorch)."""

import torch
import torch.nn.functional as F


def apply_ghost_cells_2d(
    rho: torch.Tensor,
    bc_type: str,
    *,
    n_ghost: int = 1,
    bc_value: float | None = None,
) -> torch.Tensor:
    """Pad *rho* (ny, nx) with ghost cells, returning (ny+2g, nx+2g).

    Parameters
    ----------
    rho : 2-D tensor of shape (ny, nx).
    bc_type : one of "zero_gradient", "periodic", "dirichlet".
    n_ghost : number of ghost layers per side.
    bc_value : constant value for Dirichlet BCs.
    """
    g = n_ghost

    if bc_type == "periodic":
        # Pad using wrap mode via F.pad (needs 4-D input)
        rho_4d = rho.unsqueeze(0).unsqueeze(0)  # (1,1,ny,nx)
        padded = F.pad(rho_4d, (g, g, g, g), mode="circular")
        return padded.squeeze(0).squeeze(0)

    if bc_type == "dirichlet":
        if bc_value is None:
            raise ValueError("bc_value is required for dirichlet BCs")
        padded = torch.full(
            (rho.shape[0] + 2 * g, rho.shape[1] + 2 * g),
            bc_value,
            device=rho.device,
            dtype=rho.dtype,
        )
        padded[g:-g, g:-g] = rho
        return padded

    # zero_gradient (default): replicate boundary values
    rho_4d = rho.unsqueeze(0).unsqueeze(0)
    padded = F.pad(rho_4d, (g, g, g, g), mode="replicate")
    return padded.squeeze(0).squeeze(0)
