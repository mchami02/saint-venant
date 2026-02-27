"""Type-aware collision processing for wave interactions.

Handles three collision types:
- Shock-shock → new shock (analytical Rankine-Hugoniot)
- Rarefaction-rarefaction → merged rarefaction (wider fan)
- Shock-rarefaction → bent shock (MLP-predicted polynomial trajectory)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flux import Flux


def process_collisions(
    waves: dict[str, torch.Tensor],
    flux: Flux,
    bent_shock_head: nn.Module,
    T_max: float,
    max_rounds: int,
) -> dict[str, torch.Tensor]:
    """Iterative collision processing with type-aware outcomes.

    For each round:
    1. Sort active waves by position at reference time
    2. Find earliest collision between adjacent converging waves
    3. Determine collision type and compute outcome
    4. Deactivate colliding waves, spawn new wave

    Args:
        waves: Dictionary of wave tensors (B, W).
        flux: Flux instance for analytical shock/rarefaction speeds.
        bent_shock_head: MLP predicting (c2, c3, log_duration) from 7 features.
        T_max: Maximum simulation time.
        max_rounds: Maximum collision processing iterations.

    Returns:
        Updated waves dictionary with new waves appended.
    """
    # Unpack (we'll modify these in-place via reassignment)
    wave_ox = waves["wave_origin_x"]
    wave_ot = waves["wave_origin_t"]
    wave_rho_L = waves["wave_rho_L"]
    wave_rho_R = waves["wave_rho_R"]
    wave_left_sp = waves["wave_left_speed"]
    wave_right_sp = waves["wave_right_speed"]
    wave_active = waves["wave_active"]
    wave_type = waves["wave_type"]
    wave_c2 = waves["wave_poly_c2"]
    wave_c3 = waves["wave_poly_c3"]
    wave_dur = waves["wave_poly_duration"]

    B = wave_ox.shape[0]
    device = wave_ox.device

    for _ in range(max_rounds):
        W = wave_ox.shape[1]
        if W < 2:
            break

        # Position at reference time (use center of each wave's speed range)
        t_ref = T_max / 2.0
        center_speed = (wave_left_sp + wave_right_sp) / 2.0
        pos_at_ref = wave_ox + center_speed * (t_ref - wave_ot)

        # Sort by position (inactive waves pushed to end)
        sort_pos = pos_at_ref + (1.0 - wave_active) * 1e6
        sort_idx = torch.argsort(sort_pos, dim=1)  # (B, W)

        # Gather sorted properties
        s_ox = torch.gather(wave_ox, 1, sort_idx)
        s_ot = torch.gather(wave_ot, 1, sort_idx)
        s_rho_L = torch.gather(wave_rho_L, 1, sort_idx)
        s_rho_R = torch.gather(wave_rho_R, 1, sort_idx)
        s_left_sp = torch.gather(wave_left_sp, 1, sort_idx)
        s_right_sp = torch.gather(wave_right_sp, 1, sort_idx)
        s_active = torch.gather(wave_active, 1, sort_idx)
        s_type = torch.gather(wave_type, 1, sort_idx)

        # Collision detection: right edge of left wave vs left edge of right wave
        speed_diff = s_right_sp[:, :-1] - s_left_sp[:, 1:]  # (B, W-1)

        # Collision time: when right edge of wave i meets left edge of wave i+1
        pos_diff = (
            s_ox[:, 1:]
            - s_ox[:, :-1]
            + s_right_sp[:, :-1] * s_ot[:, :-1]
            - s_left_sp[:, 1:] * s_ot[:, 1:]
        )
        speed_diff_safe = speed_diff + (speed_diff.abs() < 1e-10).float() * 1e-10
        t_coll = pos_diff / speed_diff_safe  # (B, W-1)

        # Validity: both active, converging, within time domain
        both_active = s_active[:, :-1] * s_active[:, 1:]
        converging = (speed_diff > 1e-8).float()
        in_time = ((t_coll > 0) & (t_coll < T_max)).float()
        valid = both_active * converging * in_time

        # Find earliest valid collision per batch
        t_coll_masked = t_coll + (1.0 - valid) * 1e10
        earliest_idx = torch.argmin(t_coll_masked, dim=1)  # (B,)
        earliest_t = torch.gather(
            t_coll_masked, 1, earliest_idx.unsqueeze(1)
        ).squeeze(1)
        has_collision = (earliest_t < 1e9).float()  # (B,)

        if has_collision.sum() < 0.5:
            break

        # Gather collision properties
        idx_left = earliest_idx
        idx_right = earliest_idx + 1

        def _gather_1d(tensor: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            return torch.gather(tensor, 1, idx.unsqueeze(1)).squeeze(1)

        coll_t = _gather_1d(t_coll, idx_left)
        left_ox = _gather_1d(s_ox, idx_left)
        left_ot = _gather_1d(s_ot, idx_left)
        left_right_sp = _gather_1d(s_right_sp, idx_left)
        left_type = _gather_1d(s_type, idx_left)
        right_type = _gather_1d(s_type, idx_right)

        # Collision position (where right edge of left wave meets left edge of right)
        coll_x = left_ox + left_right_sp * (coll_t - left_ot)

        # Outer states
        rho_L_outer = _gather_1d(s_rho_L, idx_left)
        rho_R_outer = _gather_1d(s_rho_R, idx_right)
        rho_R_left = _gather_1d(s_rho_R, idx_left)
        rho_L_right = _gather_1d(s_rho_L, idx_right)

        # Determine collision type
        is_ss = ((left_type < 0.5) & (right_type < 0.5)).float()
        is_rr = ((left_type > 0.5) & (left_type < 1.5) &
                 (right_type > 0.5) & (right_type < 1.5)).float()
        is_sr = (1.0 - is_ss - is_rr).clamp(0, 1)

        # --- Shock-shock: analytical new shock ---
        ss_speed = flux.shock_speed(rho_L_outer, rho_R_outer)

        # --- Rar-rar: merged rarefaction ---
        rr_left_speed = flux.derivative(rho_L_outer)
        rr_right_speed = flux.derivative(rho_R_outer)

        # --- Shock-rar: bent shock via MLP ---
        # Determine which side is shock, which is rarefaction
        left_is_shock = (left_type < 0.5).float()
        shock_rho_L = left_is_shock * rho_L_outer + (1.0 - left_is_shock) * rho_L_right
        shock_rho_R = left_is_shock * rho_R_left + (1.0 - left_is_shock) * rho_R_outer
        rar_rho_L = left_is_shock * rho_L_right + (1.0 - left_is_shock) * rho_L_outer
        rar_rho_R = left_is_shock * rho_R_outer + (1.0 - left_is_shock) * rho_R_left

        shock_sp = _gather_1d(s_left_sp, idx_left) * left_is_shock + \
                   _gather_1d(s_left_sp, idx_right) * (1.0 - left_is_shock)
        rar_left_sp_val = flux.derivative(rar_rho_L)
        rar_right_sp_val = flux.derivative(rar_rho_R)

        features = torch.stack([
            shock_rho_L, shock_rho_R, rar_rho_L, rar_rho_R,
            shock_sp, rar_left_sp_val, rar_right_sp_val,
        ], dim=-1)  # (B, 7)

        bent_params = bent_shock_head(features)  # (B, 3)
        sr_c2 = bent_params[:, 0]
        sr_c3 = bent_params[:, 1]
        sr_duration = F.softplus(bent_params[:, 2])

        # The initial speed of bent shock = shock speed at collision
        sr_c1 = shock_sp  # stored in wave_left_speed

        # --- Blend all outcomes ---
        new_rho_L = rho_L_outer
        new_rho_R = rho_R_outer

        new_left_speed = (
            is_ss * ss_speed
            + is_rr * rr_left_speed
            + is_sr * sr_c1
        )
        new_right_speed = (
            is_ss * ss_speed
            + is_rr * rr_right_speed
            + is_sr * sr_c1  # bent shock: single trajectory, same as left
        )
        new_type = is_ss * 0.0 + is_rr * 1.0 + is_sr * 2.0
        new_c2 = is_sr * sr_c2
        new_c3 = is_sr * sr_c3
        new_dur = is_sr * sr_duration

        # --- Spawn new wave ---
        new_ox = (coll_x * has_collision).unsqueeze(1)  # (B, 1)
        new_ot = (coll_t * has_collision).unsqueeze(1)
        new_rho_L_w = (new_rho_L * has_collision).unsqueeze(1)
        new_rho_R_w = (new_rho_R * has_collision).unsqueeze(1)
        new_left_sp_w = (new_left_speed * has_collision).unsqueeze(1)
        new_right_sp_w = (new_right_speed * has_collision).unsqueeze(1)
        new_active_w = has_collision.unsqueeze(1)
        new_type_w = (new_type * has_collision).unsqueeze(1)
        new_c2_w = (new_c2 * has_collision).unsqueeze(1)
        new_c3_w = (new_c3 * has_collision).unsqueeze(1)
        new_dur_w = (new_dur * has_collision).unsqueeze(1)

        # --- Deactivate colliding waves ---
        orig_idx_left = torch.gather(sort_idx, 1, idx_left.unsqueeze(1)).squeeze(1)
        orig_idx_right = torch.gather(sort_idx, 1, idx_right.unsqueeze(1)).squeeze(1)
        deact_left = F.one_hot(orig_idx_left, W).float()
        deact_right = F.one_hot(orig_idx_right, W).float()
        deact = (deact_left + deact_right) * has_collision.unsqueeze(1)
        wave_active = wave_active * (1.0 - deact).clamp(0, 1)

        # --- Concatenate new wave ---
        wave_ox = torch.cat([wave_ox, new_ox], dim=1)
        wave_ot = torch.cat([wave_ot, new_ot], dim=1)
        wave_rho_L = torch.cat([wave_rho_L, new_rho_L_w], dim=1)
        wave_rho_R = torch.cat([wave_rho_R, new_rho_R_w], dim=1)
        wave_left_sp = torch.cat([wave_left_sp, new_left_sp_w], dim=1)
        wave_right_sp = torch.cat([wave_right_sp, new_right_sp_w], dim=1)
        wave_active = torch.cat([wave_active, new_active_w], dim=1)
        wave_type = torch.cat([wave_type, new_type_w], dim=1)
        wave_c2 = torch.cat([wave_c2, new_c2_w], dim=1)
        wave_c3 = torch.cat([wave_c3, new_c3_w], dim=1)
        wave_dur = torch.cat([wave_dur, new_dur_w], dim=1)

    return {
        "wave_origin_x": wave_ox,
        "wave_origin_t": wave_ot,
        "wave_rho_L": wave_rho_L,
        "wave_rho_R": wave_rho_R,
        "wave_left_speed": wave_left_sp,
        "wave_right_speed": wave_right_sp,
        "wave_active": wave_active,
        "wave_type": wave_type,
        "wave_poly_c2": wave_c2,
        "wave_poly_c3": wave_c3,
        "wave_poly_duration": wave_dur,
    }
