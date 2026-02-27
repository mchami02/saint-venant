"""Initial wave construction from discontinuity predictions.

Builds a wave data dictionary from encoded discontinuity parameters.
Each discontinuity produces either a shock or rarefaction wave, determined
by the classifier head with straight-through estimator (STE) for training.
"""

import torch
import torch.nn.functional as F

from .flux import Flux


def build_initial_waves(
    discontinuities: torch.Tensor,
    disc_mask: torch.Tensor,
    is_shock: torch.Tensor,
    shock_speed: torch.Tensor,
    flux: Flux,
    training: bool,
) -> dict[str, torch.Tensor]:
    """Build initial wave tensors from discontinuity predictions.

    During training, uses STE: 2 wave slots per disc (1 shock + 1 rarefaction)
    with hard threshold forward, soft gradient backward.
    During eval, uses hard threshold: 1 wave slot per disc.

    Args:
        discontinuities: (B, D, 3) [x, rho_L, rho_R] per discontinuity.
        disc_mask: (B, D) validity mask.
        is_shock: (B, D) sigmoid probability of shock.
        shock_speed: (B, D) predicted shock speed.
        flux: Flux instance for computing characteristic speeds.
        training: Whether model is in training mode.

    Returns:
        Dictionary of wave tensors, all shape (B, W):
            wave_origin_x, wave_origin_t, wave_rho_L, wave_rho_R,
            wave_left_speed, wave_right_speed, wave_active, wave_type,
            wave_poly_c2, wave_poly_c3, wave_poly_duration
    """
    B, D, _ = discontinuities.shape
    device = discontinuities.device

    x_pos = discontinuities[:, :, 0]  # (B, D)
    rho_L = discontinuities[:, :, 1]  # (B, D)
    rho_R = discontinuities[:, :, 2]  # (B, D)

    # STE: hard threshold forward, soft gradient backward
    is_shock_hard = (is_shock > 0.5).float()
    is_shock_ste = is_shock + (is_shock_hard - is_shock).detach()

    # Rarefaction edge speeds from flux (analytical, IC-level physics)
    rar_left_speed = flux.derivative(rho_L)  # (B, D)
    rar_right_speed = flux.derivative(rho_R)  # (B, D)

    if training:
        # 2 slots per disc: [shock_0, rar_0, shock_1, rar_1, ...]
        W = 2 * D

        # Shock slots (even indices)
        shock_active = is_shock_ste * disc_mask  # (B, D)
        # Rarefaction slots (odd indices)
        rar_active = (1.0 - is_shock_ste) * disc_mask  # (B, D)

        # Interleave: shock, rar, shock, rar, ...
        wave_origin_x = torch.stack(
            [x_pos[:, d:d + 1] for d in range(D) for _ in range(2)], dim=1
        ).squeeze(-1) if D > 0 else torch.zeros(B, 0, device=device)

        # Build via interleaving
        origin_x_list = []
        left_speed_list = []
        right_speed_list = []
        rho_L_list = []
        rho_R_list = []
        active_list = []
        type_list = []

        for d in range(D):
            # Shock slot
            origin_x_list.append(x_pos[:, d])
            left_speed_list.append(shock_speed[:, d])
            right_speed_list.append(shock_speed[:, d])
            rho_L_list.append(rho_L[:, d])
            rho_R_list.append(rho_R[:, d])
            active_list.append(shock_active[:, d])
            type_list.append(0.0)

            # Rarefaction slot
            origin_x_list.append(x_pos[:, d])
            left_speed_list.append(rar_left_speed[:, d])
            right_speed_list.append(rar_right_speed[:, d])
            rho_L_list.append(rho_L[:, d])
            rho_R_list.append(rho_R[:, d])
            active_list.append(rar_active[:, d])
            type_list.append(1.0)

        if D > 0:
            wave_origin_x = torch.stack(origin_x_list, dim=1)
            wave_left_speed = torch.stack(left_speed_list, dim=1)
            wave_right_speed = torch.stack(right_speed_list, dim=1)
            wave_rho_L = torch.stack(rho_L_list, dim=1)
            wave_rho_R = torch.stack(rho_R_list, dim=1)
            wave_active = torch.stack(active_list, dim=1)
        else:
            wave_origin_x = torch.zeros(B, 0, device=device)
            wave_left_speed = torch.zeros(B, 0, device=device)
            wave_right_speed = torch.zeros(B, 0, device=device)
            wave_rho_L = torch.zeros(B, 0, device=device)
            wave_rho_R = torch.zeros(B, 0, device=device)
            wave_active = torch.zeros(B, 0, device=device)

        wave_type = torch.tensor(type_list, device=device).unsqueeze(0).expand(B, -1)
    else:
        # Eval: 1 slot per disc, hard threshold
        W = D
        wave_origin_x = x_pos
        wave_rho_L = rho_L
        wave_rho_R = rho_R

        # Type determines speeds
        wave_type = (1.0 - is_shock_hard) * disc_mask  # 0=shock, 1=rar
        is_s = is_shock_hard  # (B, D)

        wave_left_speed = is_s * shock_speed + (1.0 - is_s) * rar_left_speed
        wave_right_speed = is_s * shock_speed + (1.0 - is_s) * rar_right_speed
        wave_active = disc_mask

    wave_origin_t = torch.zeros(B, W, device=device)
    wave_poly_c2 = torch.zeros(B, W, device=device)
    wave_poly_c3 = torch.zeros(B, W, device=device)
    wave_poly_duration = torch.zeros(B, W, device=device)

    return {
        "wave_origin_x": wave_origin_x,
        "wave_origin_t": wave_origin_t,
        "wave_rho_L": wave_rho_L,
        "wave_rho_R": wave_rho_R,
        "wave_left_speed": wave_left_speed,
        "wave_right_speed": wave_right_speed,
        "wave_active": wave_active,
        "wave_type": wave_type,
        "wave_poly_c2": wave_poly_c2,
        "wave_poly_c3": wave_poly_c3,
        "wave_poly_duration": wave_poly_duration,
    }
