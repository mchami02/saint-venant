"""LWR-aware per-segment attention bias.

Computes a physics-informed attention bias for each IC segment based on
shock/rarefaction classification at interfaces. Unlike the backward-
characteristic bias in ``biased_cross_attention.py`` (which traces a
single characteristic foot per segment), this module uses the actual
interface dynamics:

- **Shock** (λ_L > λ_R): one-sided penalty from the Rankine-Hugoniot
  trajectory ``x_d + s·t``.
- **Rarefaction** (λ_L ≤ λ_R): one-sided penalty from the far fan edge,
  leaving the interior of the fan unpenalized (both segments attend).

The bias at each query point is:
``-(margin + growth_rate * relu(dist)) * exp(-time_spread * t)``.
The exponential damping smoothly fades the bias from full strength
near t=0 toward zero at later times.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flux import Flux, GreenshieldsFlux


class LWRBias(nn.Module):
    """Per-segment attention bias from LWR interface dynamics.

    For each interface between segments k and k+1 the module classifies
    the wave (shock vs rarefaction) and computes a one-sided penalty on
    each adjacent segment.  Penalties are accumulated across all
    interfaces so that each segment's bias is the sum of contributions
    from its left and right boundaries.

    The bias for a segment at query point (t, x) is::

        bias = -(margin + growth_rate * relu(dist)) * exp(-time_spread * t)

    where ``dist`` is the one-sided distance past the boundary
    trajectory.  The exponential damping is 1 at t=0 (full bias)
    and smoothly fades toward 0 at later times.

    Args:
        margin: Constant offset applied at the boundary (dist=0).
            ``0.0`` means zero penalty right at the shock/fan edge.
        growth_rate: Scale for the distance-proportional term.
        time_spread: Decay rate for exponential time damping.  Higher
            values give faster decay; ``0.0`` disables damping
            entirely (exp(0)=1).
        flux: ``Flux`` instance for characteristic / shock speeds.
            Defaults to ``GreenshieldsFlux()``.
    """

    def __init__(
        self,
        margin: float = 0.0,
        growth_rate: float = 100.0,
        time_spread: float = 3.0,
        flux: Flux | None = None,
    ):
        super().__init__()
        self.margin = margin
        self.growth_rate = growth_rate
        self.time_spread = time_spread
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

        # -- time damping (exponential decay) ------------------------------------
        # exp(-time_spread * t) → 1 at t=0, smoothly fades toward 0
        damping = torch.exp(-self.time_spread * t_exp)  # (B, *spatial, 1)

        # -- one-sided penalties at each interface ----------------------------
        boundary_right = x_d + speed_right * t_exp  # (B, *spatial, K-1)
        dist_left = torch.relu(x_exp - boundary_right)  # (B, *spatial, K-1)
        penalty_left_seg = (self.margin + self.growth_rate * dist_left) * damping

        boundary_left = x_d + speed_left * t_exp  # (B, *spatial, K-1)
        dist_right = torch.relu(boundary_left - x_exp)  # (B, *spatial, K-1)
        penalty_right_seg = (self.margin + self.growth_rate * dist_right) * damping

        # -- accumulate onto K segments with F.pad ----------------------------
        # penalty_left_seg  affects segments 0..K-2 → pad right by 1
        # penalty_right_seg affects segments 1..K-1 → pad left  by 1
        bias = -(
            F.pad(penalty_left_seg, (0, 1)) + F.pad(penalty_right_seg, (1, 0))
        )  # (B, *spatial, K)

        # -- mask padded segments ---------------------------------------------
        mask = pieces_mask
        for _ in range(n_expand):
            mask = mask.unsqueeze(1)  # (B, 1, ..., 1, K)
        bias = bias * mask + (~mask.bool()).float() * (-1e9)

        return bias
