"""Ghost cell boundary conditions for the 2D Euler system (PyTorch).

Handles all four edges symmetrically. Wall BCs use a true mirror (flip of
the n_ghost nearest interior cells) with normal momentum sign-flipped at
each wall; tangential momentum is preserved.
"""

import torch
import torch.nn.functional as F


def _pad_periodic(q: torch.Tensor, g: int) -> torch.Tensor:
    """Circular pad on the last two axes."""
    orig_shape = q.shape
    q4 = q.reshape(-1, 1, orig_shape[-2], orig_shape[-1])
    padded = F.pad(q4, (g, g, g, g), mode="circular")
    return padded.reshape(*orig_shape[:-2], orig_shape[-2] + 2 * g, orig_shape[-1] + 2 * g)


def _pad_extrap(q: torch.Tensor, g: int) -> torch.Tensor:
    """Replicate-pad on the last two axes (zero gradient)."""
    orig_shape = q.shape
    q4 = q.reshape(-1, 1, orig_shape[-2], orig_shape[-1])
    padded = F.pad(q4, (g, g, g, g), mode="replicate")
    return padded.reshape(*orig_shape[:-2], orig_shape[-2] + 2 * g, orig_shape[-1] + 2 * g)


def _pad_mirror_x(q: torch.Tensor, g: int) -> torch.Tensor:
    """Mirror-pad only on the last (x) axis: left/right ghosts are flipped
    slices of the nearest g interior cells."""
    left = q[..., :, :g].flip(-1)
    right = q[..., :, -g:].flip(-1)
    return torch.cat([left, q, right], dim=-1)


def _pad_mirror_y(q: torch.Tensor, g: int) -> torch.Tensor:
    """Mirror-pad only on the second-to-last (y) axis."""
    bottom = q[..., :g, :].flip(-2)
    top = q[..., -g:, :].flip(-2)
    return torch.cat([bottom, q, top], dim=-2)


def _pad_mirror_xy(q: torch.Tensor, g: int) -> torch.Tensor:
    """Mirror-pad both axes independently."""
    return _pad_mirror_y(_pad_mirror_x(q, g), g)


def apply_ghost_cells(
    rho: torch.Tensor,
    rho_u: torch.Tensor,
    rho_v: torch.Tensor,
    E: torch.Tensor,
    bc_type: str,
    *,
    n_ghost: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad ``(rho, rho_u, rho_v, E)`` with n_ghost cells on every edge.

    Parameters
    ----------
    rho, rho_u, rho_v, E : tensors of shape (..., ny, nx).
    bc_type : "extrap", "periodic", or "wall".
    n_ghost : number of ghost layers.
    """
    g = n_ghost

    if bc_type == "periodic":
        return (
            _pad_periodic(rho, g),
            _pad_periodic(rho_u, g),
            _pad_periodic(rho_v, g),
            _pad_periodic(E, g),
        )

    if bc_type == "wall":
        # Density and energy: plain mirror on both axes.
        rho_g = _pad_mirror_xy(rho, g)
        E_g = _pad_mirror_xy(E, g)
        # rho_u: mirror in y first, then sign-flipped mirror in x (x-wall flips u).
        rho_u_g = _pad_mirror_y(rho_u, g)
        left = -rho_u_g[..., :, :g].flip(-1)
        right = -rho_u_g[..., :, -g:].flip(-1)
        rho_u_g = torch.cat([left, rho_u_g, right], dim=-1)
        # rho_v: mirror in x first, then sign-flipped mirror in y (y-wall flips v).
        rho_v_g = _pad_mirror_x(rho_v, g)
        bottom = -rho_v_g[..., :g, :].flip(-2)
        top = -rho_v_g[..., -g:, :].flip(-2)
        rho_v_g = torch.cat([bottom, rho_v_g, top], dim=-2)
        return rho_g, rho_u_g, rho_v_g, E_g

    # extrap / zero_gradient
    return (
        _pad_extrap(rho, g),
        _pad_extrap(rho_u, g),
        _pad_extrap(rho_v, g),
        _pad_extrap(E, g),
    )
