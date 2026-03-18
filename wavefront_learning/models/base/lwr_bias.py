"""LWR-aware per-segment attention bias (similarity-variable formulation).

Computes a physics-informed attention bias for each IC segment based on
shock/rarefaction classification at interfaces. Unlike the backward-
characteristic bias in ``biased_cross_attention.py`` (which traces a
single characteristic foot per segment), this module uses the actual
interface dynamics:

- **Shock** (λ_L > λ_R): one-sided penalty from the Rankine-Hugoniot
  shock speed ``s``.
- **Rarefaction** (λ_L ≤ λ_R): one-sided penalty from the far fan edge
  speed, leaving the interior of the fan unpenalized (both segments attend).

Penalties are computed in the self-similar coordinate ``ξ = (x - x_d) / (t + ε)``
rather than physical space. This makes the bias dimensionless, time-invariant,
and scale-invariant — improving OOD generalization across domain sizes and
time horizons.

The bias at each query point is ``-relu(ξ - speed)``: zero where a segment
has influence, negative elsewhere. When ``use_damping=True``, the bias is
multiplied by a sigmoid that fades it after the estimated collision time:
``-relu(ξ - speed) * sigmoid(damping_sharpness * (t_coll - t))``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flux import Flux, GreenshieldsFlux


class LWRBias(nn.Module):
    """Per-segment attention bias from LWR interface dynamics.

    For each interface between segments k and k+1 the module classifies
    the wave (shock vs rarefaction) and computes a one-sided penalty in
    the self-similar coordinate ``ξ = (x - x_d) / (t + ε)``.  Penalties
    are accumulated across all interfaces so that each segment's bias is
    the sum of contributions from its left and right boundaries.

    The bias for a segment at query point (t, x) is::

        bias = -relu(ξ - speed)                      # use_damping=False
        bias = -relu(ξ - speed) * σ(β * (t_coll - t)) # use_damping=True

    where ``ξ`` is the similarity variable and ``t_coll`` is the
    per-segment analytical collision time.

    Args:
        initial_damping_sharpness: Initial sharpness of the sigmoid
            collision-time damping (learnable).
        flux: ``Flux`` instance for characteristic / shock speeds.
            Defaults to ``GreenshieldsFlux()``.
        use_damping: If True, multiply the bias by a sigmoid that fades
            it after the estimated collision time.  Default True.
        eps: Small constant added to ``t`` to avoid division by zero
            in the similarity variable computation.  Default 1e-6.
    """

    def __init__(
        self,
        initial_damping_sharpness: float = 5.0,
        flux: Flux | None = None,
        use_damping: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.use_damping = use_damping
        self.eps = eps
        if use_damping:
            self.damping_sharpness = nn.Parameter(
                torch.tensor(initial_damping_sharpness)
            )
        self.flux = flux if flux is not None else GreenshieldsFlux()

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
                - ``pieces_mask``: (B, K) validity mask (1=valid).
            query_points: Tuple ``(t_coords, x_coords)`` each of shape
                ``(B, *spatial)``, where ``*spatial`` can be any number
                of dimensions (e.g. ``(nt, nx)``).

        Returns:
            Bias tensor ``(B, *spatial, K)``.  Zero means full attention;
            large negative values suppress attention.
        """
        xs = ic_data["xs"]  # (B, K+1)
        ks = ic_data["ks"]  # (B, K)
        pieces_mask = ic_data["pieces_mask"]  # (B, K)
        t_coords, x_coords = query_points  # each (B, *spatial)

        K = ks.shape[1]
        spatial_dims = t_coords.shape[1:]  # e.g. (nt, nx)
        n_expand = len(spatial_dims)  # number of dims to unsqueeze

        # -- characteristic and shock speeds ----------------------------------
        lam = self.flux.derivative(ks)  # (B, K)
        lam_L = lam[:, :-1]  # (B, K-1)
        lam_R = lam[:, 1:]  # (B, K-1)
        is_shock = lam_L > lam_R  # (B, K-1)
        s = self.flux.shock_speed(ks[:, :-1], ks[:, 1:])  # (B, K-1)

        # -- boundary speed selection -----------------------------------------
        # For the LEFT segment (k): penalty from its RIGHT interface
        #   shock  → boundary moves at shock speed s
        #   rarefaction → boundary moves at the far (right) fan edge speed λ_R
        speed_right = torch.where(is_shock, s, lam_R)  # (B, K-1)

        # For the RIGHT segment (k+1): penalty from its LEFT interface
        #   shock  → boundary moves at shock speed s
        #   rarefaction → boundary moves at the far (left) fan edge speed λ_L
        speed_left = torch.where(is_shock, s, lam_L)  # (B, K-1)

        # -- interface positions ----------------------------------------------
        x_d = xs[:, 1:K]  # (B, K-1) interior breakpoints

        # Expand interface quantities to (B, *spatial, K-1)
        for _ in range(n_expand):
            x_d = x_d.unsqueeze(1)
            speed_right = speed_right.unsqueeze(1)
            speed_left = speed_left.unsqueeze(1)

        t_exp = t_coords.unsqueeze(-1)  # (B, *spatial, 1)
        x_exp = x_coords.unsqueeze(-1)  # (B, *spatial, 1)

        # -- one-sided penalties in similarity variable ξ = (x - x_d)/(t + ε) --
        xi = (x_exp - x_d) / (t_exp + self.eps)  # (B, *spatial, K-1)
        penalty_left_seg = torch.relu(xi - speed_right)
        penalty_right_seg = torch.relu(speed_left - xi)

        # -- accumulate onto K segments with F.pad ----------------------------
        # penalty_left_seg  affects segments 0..K-2 → pad right by 1
        # penalty_right_seg affects segments 1..K-1 → pad left  by 1
        bias = -(
            F.pad(penalty_left_seg, (0, 1)) + F.pad(penalty_right_seg, (1, 0))
        )  # (B, *spatial, K)

        # -- per-segment collision-time damping (optional) ---------------------
        if self.use_damping:
            t_coll = self._compute_collision_times(xs, ks, lam, K)  # (B, K)
            for _ in range(n_expand):
                t_coll = t_coll.unsqueeze(1)  # (B, 1, ..., 1, K)
            damping = torch.sigmoid(
                self.damping_sharpness.abs() * (t_coll - t_exp)
            )  # (B, *spatial, K) — ~1 before collision, ~0 after
            bias = bias * damping

        # -- mask padded segments ---------------------------------------------
        mask = pieces_mask
        for _ in range(n_expand):
            mask = mask.unsqueeze(1)  # (B, 1, ..., 1, K)
        bias = bias * mask + (~mask.bool()).float() * (-1e9)

        return bias

    @staticmethod
    def _compute_collision_times(
        xs: torch.Tensor,
        ks: torch.Tensor,
        lam: torch.Tensor,
        K: int,
    ) -> torch.Tensor:
        """Analytical collision time per segment.

        ``t_coll[k] = min(width_left / |λ_{k-1} - λ_k|,
                          width_right / |λ_k - λ_{k+1}|)``

        At domain edges the segment's own width and speed difference are
        reused so that edge segments get a finite (non-zero) collision
        time.

        Args:
            xs: (B, K+1) breakpoint positions.
            ks: (B, K) segment densities (unused, kept for API clarity).
            lam: (B, K) characteristic speeds.
            K: Number of segments.

        Returns:
            (B, K) positive collision times.
        """
        widths = xs[:, 1:] - xs[:, :-1]  # (B, K)

        # Left-neighbor collision
        lam_left = F.pad(lam[:, :-1], (1, 0))
        lam_left[:, 0] = lam[:, 0]
        dx_left = F.pad(widths[:, :-1], (1, 0))
        dx_left[:, 0] = widths[:, 0]
        speed_diff_left = (lam_left - lam).abs().clamp(min=1e-3)

        # Right-neighbor collision
        lam_right = F.pad(lam[:, 1:], (0, 1))
        lam_right[:, -1] = lam[:, -1]
        dx_right = F.pad(widths[:, 1:], (0, 1))
        dx_right[:, -1] = widths[:, -1]
        speed_diff_right = (lam - lam_right).abs().clamp(min=1e-3)

        t_coll = torch.minimum(
            dx_left / speed_diff_left,
            dx_right / speed_diff_right,
        )  # (B, K)

        return t_coll
