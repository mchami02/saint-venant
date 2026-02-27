"""Grid reconstruction from wave pattern.

Reconstructs the full density grid from a wave data dictionary.
Supports three wave types:
- Type 0 (shock): step function at linear position
- Type 1 (rarefaction): entropy solution fan using inverse_derivative
- Type 2 (bent shock): step function at polynomial position
"""

import torch

from .flux import Flux


def reconstruct_grid(
    waves: dict[str, torch.Tensor],
    base_density: torch.Tensor,
    t_coords: torch.Tensor,
    x_coords: torch.Tensor,
    flux: Flux,
    sigma: float | None,
) -> torch.Tensor:
    """Reconstruct density grid from wave pattern.

    density(t, x) = base + sum_w [jump_w * fraction_w * active_w]

    where fraction depends on wave type:
    - Shock: sigmoid step at linear position
    - Rarefaction: entropy solution fan (smooth transition)
    - Bent shock: sigmoid step at polynomial position

    Args:
        waves: Dictionary of wave tensors (B, W).
        base_density: (B,) leftmost density value.
        t_coords: (B, 1, nt, nx) time coordinates.
        x_coords: (B, 1, nt, nx) space coordinates.
        flux: Flux instance for entropy solution.
        sigma: Sigmoid sharpness. None uses Heaviside (eval).

    Returns:
        output_grid: (B, 1, nt, nx) predicted density.
    """
    B, _, nt, nx = t_coords.shape
    W = waves["wave_origin_x"].shape[1]

    if W == 0:
        return base_density[:, None, None, None].expand(B, 1, nt, nx)

    # Extract wave tensors and expand for broadcasting
    ox = waves["wave_origin_x"][:, :, None, None]  # (B, W, 1, 1)
    ot = waves["wave_origin_t"][:, :, None, None]
    rho_L = waves["wave_rho_L"][:, :, None, None]
    rho_R = waves["wave_rho_R"][:, :, None, None]
    left_sp = waves["wave_left_speed"][:, :, None, None]
    right_sp = waves["wave_right_speed"][:, :, None, None]
    active = waves["wave_active"][:, :, None, None]
    wtype = waves["wave_type"][:, :, None, None]
    poly_c2 = waves["wave_poly_c2"][:, :, None, None]
    poly_c3 = waves["wave_poly_c3"][:, :, None, None]
    poly_dur = waves["wave_poly_duration"][:, :, None, None]

    # Expand coordinates: (B, 1, nt, nx)
    t = t_coords[:, 0:1, :, :]  # (B, 1, nt, nx)
    x = x_coords[:, 0:1, :, :]  # (B, 1, nt, nx)

    jump = rho_R - rho_L  # (B, W, 1, 1)

    # Type masks
    is_shock = (wtype < 0.5).float()
    is_rar = ((wtype > 0.5) & (wtype < 1.5)).float()
    is_bent = (wtype > 1.5).float()

    eps = 1e-8

    # --- Type 0: Shock — step at linear position ---
    dt_shock = t - ot  # (B, W, nt, nx) via broadcast
    shock_pos = ox + left_sp * dt_shock
    if sigma is None:
        frac_shock = (x >= shock_pos).float()
    else:
        frac_shock = torch.sigmoid((x - shock_pos) / sigma)

    # --- Type 1: Rarefaction — entropy solution fan ---
    dt_rar = (t - ot).clamp(min=eps)
    xi = (x - ox) / dt_rar  # similarity variable

    # Fraction: linear ramp in characteristic speed space
    # For Greenshields: f'(rho) = 1-2*rho, so inverse_derivative(xi) = (1-xi)/2
    # fraction = clamp((xi - s_L) / (s_R - s_L), 0, 1)
    speed_range = right_sp - left_sp + eps
    frac_rar = ((xi - left_sp) / speed_range).clamp(0.0, 1.0)

    # At t ≈ origin: fallback to step function (fan hasn't developed)
    at_origin = ((t - ot).abs() < eps).float()
    if sigma is None:
        step_fallback = (x >= ox).float()
    else:
        step_fallback = torch.sigmoid((x - ox) / sigma)
    frac_rar = frac_rar * (1.0 - at_origin) + step_fallback * at_origin

    # --- Type 2: Bent shock — polynomial position ---
    dt_bent = t - ot
    c1 = left_sp  # initial speed = shock speed at collision
    pos_curved = ox + c1 * dt_bent + poly_c2 * dt_bent**2 + poly_c3 * dt_bent**3
    exit_speed = c1 + 2.0 * poly_c2 * poly_dur + 3.0 * poly_c3 * poly_dur**2
    pos_at_dur = ox + c1 * poly_dur + poly_c2 * poly_dur**2 + poly_c3 * poly_dur**3
    pos_after = pos_at_dur + exit_speed * (dt_bent - poly_dur)
    in_curve = (dt_bent <= poly_dur).float()
    bent_pos = pos_curved * in_curve + pos_after * (1.0 - in_curve)

    if sigma is None:
        frac_bent = (x >= bent_pos).float()
    else:
        frac_bent = torch.sigmoid((x - bent_pos) / sigma)

    # --- Combine by type ---
    fraction = is_shock * frac_shock + is_rar * frac_rar + is_bent * frac_bent

    # Density contribution per wave
    contributions = jump * fraction * active  # (B, W, nt, nx)

    # Sum all wave contributions + base density
    density = base_density[:, None, None, None] + contributions.sum(dim=1, keepdim=True)

    return density  # (B, 1, nt, nx)
