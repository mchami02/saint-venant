"""ARZ-aware per-segment attention bias.

Computes a physics-informed attention bias for each IC segment based on
the ARZ Riemann problem at interfaces. The ARZ system has two wave families:

- **lambda_2-wave** (genuinely nonlinear: v - rho * p'(rho))
  → shock or rarefaction. Determines the LEFT segment's RIGHT boundary.
- **lambda_1-wave** (linearly degenerate: v)
  → contact discontinuity. Determines the RIGHT segment's LEFT boundary.

Therefore each interface produces *asymmetric* boundary speeds:
- Left segment boundary: lambda_2 shock speed or far fan edge speed
- Right segment boundary: contact speed v_R

The bias at each query point is ``-relu(dist)``: zero where a segment
has influence, negative distance elsewhere. When ``use_damping=True``,
the bias is multiplied by a sigmoid that fades it after the estimated
collision time:
``-relu(dist) * sigmoid(damping_sharpness * (t_coll - t))``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .arz_physics import ARZPhysics


class ARZBias(nn.Module):
    """Per-segment attention bias from ARZ interface dynamics.

    For each interface between segments k and k+1, the module uses the
    two-wave ARZ Riemann structure to compute asymmetric one-sided
    distance penalties:

    - Left segment (k): bounded by lambda_2-wave (shock speed or far
      rarefaction edge)
    - Right segment (k+1): bounded by lambda_1-contact (speed = v_R)

    Args:
        initial_damping_sharpness: Initial sharpness of the sigmoid
            collision-time damping (learnable).
        arz_physics: ``ARZPhysics`` instance for eigenvalues and shock
            speeds. Defaults to ``ARZPhysics()``.
        use_damping: If True, multiply the bias by a sigmoid that fades
            it after the estimated collision time. Default True.
    """

    def __init__(
        self,
        initial_damping_sharpness: float = 5.0,
        arz_physics: ARZPhysics | None = None,
        use_damping: bool = True,
    ):
        super().__init__()
        self.use_damping = use_damping
        if use_damping:
            self.damping_sharpness = nn.Parameter(
                torch.tensor(initial_damping_sharpness)
            )
        self.physics = arz_physics if arz_physics is not None else ARZPhysics()

    def forward(
        self,
        ic_data: dict[str, torch.Tensor],
        query_points: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Compute per-segment attention bias.

        Args:
            ic_data: Dictionary with:
                - ``xs``: (B, K+1) breakpoint positions.
                - ``ks``: (B, K) segment density values.
                - ``ks_v``: (B, K) segment velocity values.
                - ``pieces_mask``: (B, K) validity mask (1=valid).
            query_points: Tuple ``(t_coords, x_coords)`` each of shape
                ``(B, *spatial)``.

        Returns:
            Bias tensor ``(B, *spatial, K)``.
        """
        xs = ic_data["xs"]  # (B, K+1)
        ks = ic_data["ks"]  # (B, K)
        ks_v = ic_data["ks_v"]  # (B, K)
        pieces_mask = ic_data["pieces_mask"]  # (B, K)
        t_coords, x_coords = query_points  # each (B, *spatial)

        K = ks.shape[1]
        spatial_dims = t_coords.shape[1:]
        n_expand = len(spatial_dims)

        # -- eigenvalues at each segment ------------------------------------
        _lam1, lam2 = self.physics.eigenvalues(ks, ks_v)  # each (B, K)

        # -- interface quantities (K-1 interfaces) --------------------------
        rho_L = ks[:, :-1]  # (B, K-1)
        rho_R = ks[:, 1:]  # (B, K-1)
        v_L = ks_v[:, :-1]  # (B, K-1)
        v_R = ks_v[:, 1:]  # (B, K-1)
        lam2_L = lam2[:, :-1]  # (B, K-1)
        lam2_R = lam2[:, 1:]  # (B, K-1)

        # Lambda_2 wave: shock vs rarefaction
        is_shock = lam2_L > lam2_R  # (B, K-1)

        # Shock speed from mass-conservation RH condition
        s = self.physics.shock_speed_rh(rho_L, v_L, rho_R, v_R)  # (B, K-1)

        # -- boundary speed selection ---------------------------------------
        # LEFT segment's RIGHT boundary (lambda_2 family):
        #   shock → shock speed s
        #   rarefaction → far fan edge speed lam2_R
        speed_lam2 = torch.where(is_shock, s, lam2_R)  # (B, K-1)

        # RIGHT segment's LEFT boundary (lambda_1 contact):
        #   contact speed = v_R
        speed_contact = v_R  # (B, K-1)

        # -- interface positions --------------------------------------------
        x_d = xs[:, 1:K]  # (B, K-1)

        # Expand to (B, *spatial, K-1)
        for _ in range(n_expand):
            x_d = x_d.unsqueeze(1)
            speed_lam2 = speed_lam2.unsqueeze(1)
            speed_contact = speed_contact.unsqueeze(1)

        t_exp = t_coords.unsqueeze(-1)  # (B, *spatial, 1)
        x_exp = x_coords.unsqueeze(-1)  # (B, *spatial, 1)

        # -- one-sided distance penalties -----------------------------------
        # Left segment past lambda_2-wave boundary
        boundary_right = x_d + speed_lam2 * t_exp  # (B, *spatial, K-1)
        penalty_left_seg = torch.relu(x_exp - boundary_right)

        # Right segment past contact boundary
        boundary_left = x_d + speed_contact * t_exp  # (B, *spatial, K-1)
        penalty_right_seg = torch.relu(boundary_left - x_exp)

        # -- accumulate onto K segments with F.pad -------------------------
        bias = -(
            F.pad(penalty_left_seg, (0, 1)) + F.pad(penalty_right_seg, (1, 0))
        )  # (B, *spatial, K)

        # -- per-segment collision-time damping (optional) ------------------
        if self.use_damping:
            t_coll = self._compute_collision_times(xs, lam2, K)  # (B, K)
            for _ in range(n_expand):
                t_coll = t_coll.unsqueeze(1)
            damping = torch.sigmoid(
                self.damping_sharpness.abs() * (t_coll - t_exp)
            )
            bias = bias * damping

        # -- mask padded segments ------------------------------------------
        mask = pieces_mask
        for _ in range(n_expand):
            mask = mask.unsqueeze(1)
        bias = bias * mask + (~mask.bool()).float() * (-1e9)

        return bias

    @staticmethod
    def _compute_collision_times(
        xs: torch.Tensor,
        lam2: torch.Tensor,
        K: int,
    ) -> torch.Tensor:
        """Analytical collision time per segment using lambda_2 speeds.

        ``t_coll[k] = min(width_left / |lam2_{k-1} - lam2_k|,
                          width_right / |lam2_k - lam2_{k+1}|)``

        Args:
            xs: (B, K+1) breakpoint positions.
            lam2: (B, K) lambda_2 eigenvalues.
            K: Number of segments.

        Returns:
            (B, K) positive collision times.
        """
        widths = xs[:, 1:] - xs[:, :-1]  # (B, K)

        # Left-neighbor collision
        lam_left = F.pad(lam2[:, :-1], (1, 0))
        lam_left[:, 0] = lam2[:, 0]
        dx_left = F.pad(widths[:, :-1], (1, 0))
        dx_left[:, 0] = widths[:, 0]
        speed_diff_left = (lam_left - lam2).abs().clamp(min=1e-3)

        # Right-neighbor collision
        lam_right = F.pad(lam2[:, 1:], (0, 1))
        lam_right[:, -1] = lam2[:, -1]
        dx_right = F.pad(widths[:, 1:], (0, 1))
        dx_right[:, -1] = widths[:, -1]
        speed_diff_right = (lam2 - lam_right).abs().clamp(min=1e-3)

        t_coll = torch.minimum(
            dx_left / speed_diff_left,
            dx_right / speed_diff_right,
        )

        return t_coll
