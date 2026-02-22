"""Boundary computation for trajectory-conditioned models."""

import torch


def compute_boundaries(
    positions: torch.Tensor,
    x_coords: torch.Tensor,
    disc_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute left and right boundary positions for each grid point.

    For each spatial point x at time t, finds:
    - left_bound: position of the nearest discontinuity to the left (or 0.0)
    - right_bound: position of the nearest discontinuity to the right (or 1.0)

    Args:
        positions: Predicted discontinuity positions (B, D, nt).
        x_coords: Spatial coordinates (B, nt, nx).
        disc_mask: Validity mask (B, D).

    Returns:
        Tuple of (left_bound, right_bound), each (B, nt, nx).
    """
    B, D, nt = positions.shape
    nx = x_coords.shape[-1]

    # Expand + contiguous to avoid MPS broadcasting bug with size-1 dims.
    # MPS cannot handle stride-0 views from expand(), so contiguous() is needed.
    pos = positions.unsqueeze(-1).expand(B, D, nt, nx).contiguous()  # (B, D, nt, nx)
    x = x_coords.unsqueeze(1).expand(B, D, nt, nx).contiguous()  # (B, D, nt, nx)
    mask = disc_mask[:, :, None, None].bool()  # (B, D, 1, 1)

    # Left boundary: largest position <= x among valid discs
    is_left = (pos <= x) & mask
    neg_inf = torch.tensor(-float("inf"), device=pos.device, dtype=pos.dtype)
    left_vals = torch.where(is_left, pos, neg_inf)
    left_bound = left_vals.max(dim=1).values  # (B, nt, nx)
    left_bound = torch.where(
        left_bound.isinf(), torch.zeros_like(left_bound), left_bound
    )

    # Right boundary: smallest position > x among valid discs
    is_right = (pos > x) & mask
    pos_inf = torch.tensor(float("inf"), device=pos.device, dtype=pos.dtype)
    right_vals = torch.where(is_right, pos, pos_inf)
    right_bound = right_vals.min(dim=1).values  # (B, nt, nx)
    right_bound = torch.where(
        right_bound.isinf(), torch.ones_like(right_bound), right_bound
    )

    return left_bound, right_bound
