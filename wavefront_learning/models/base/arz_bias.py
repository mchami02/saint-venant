"""ARZ-aware per-segment attention bias (similarity-variable formulation).

Computes a physics-informed attention bias for each IC segment based on
the two-wave structure of the Aw-Rascle-Zhang (ARZ) traffic flow system.

The ARZ system has two wave families at each interface:

- **2-wave** (genuinely nonlinear, slower): connects left state to
  intermediate state. The Lagrangian marker ``w = v + p(ρ)`` is constant
  across it. Can be a shock or rarefaction.
- **1-wave** (linearly degenerate, faster): a contact discontinuity
  connecting intermediate to right state. Velocity ``v`` is constant
  across it, propagating at speed ``v_R``.

Penalties are computed in the self-similar coordinate ``ξ = (x - x_d) / (t + ε)``
(same as LWR bias). Each interface generates two one-sided penalties:

- Left segment bounded by the 2-wave (shock speed or rarefaction fan edge)
- Right segment bounded by the 1-contact (speed ``v_R``)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ARZBias(nn.Module):
    """Per-segment attention bias from ARZ interface dynamics.

    For each interface between segments k and k+1 the module solves the
    ARZ Riemann problem to find the 2-wave speed (shock or rarefaction
    fan edge) and the 1-contact speed, then computes one-sided penalties
    in the self-similar coordinate ``ξ = (x - x_d) / (t + ε)``.

    The bias for a segment at query point (t, x) is::

        bias = -(penalty_from_left_interface + penalty_from_right_interface)

    Args:
        gamma: Pressure exponent in ``p(ρ) = ρ^γ``. Default 1.0.
        eps: Small constant for numerical stability. Default 1e-6.
    """

    def __init__(self, gamma: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def _pressure(self, rho: torch.Tensor) -> torch.Tensor:
        """p(ρ) = ρ^γ."""
        return rho.pow(self.gamma)

    def _dp_drho(self, rho: torch.Tensor) -> torch.Tensor:
        """p'(ρ) = γ·ρ^(γ-1)."""
        if self.gamma == 1.0:
            return torch.ones_like(rho)
        return self.gamma * rho.pow(self.gamma - 1)

    def _lam2(self, rho: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Second eigenvalue: λ₂ = v − ρ·p'(ρ)."""
        return v - rho * self._dp_drho(rho)

    def _shock_speed_2(
        self,
        rho_L: torch.Tensor,
        rho_star: torch.Tensor,
        w_L: torch.Tensor,
    ) -> torch.Tensor:
        """2-shock speed from the Rankine-Hugoniot condition for mass.

        σ₂ = w_L − (ρ*^{γ+1} − ρ_L^{γ+1}) / (ρ* − ρ_L)
        """
        gp1 = self.gamma + 1.0
        numer = rho_star.pow(gp1) - rho_L.pow(gp1)
        denom = rho_star - rho_L
        ratio = numer / (denom + self.eps * denom.sign().clamp(min=1e-30))
        # Stable fallback: when ρ* ≈ ρ_L the ratio → (γ+1)·ρ_L^γ
        near = denom.abs() < 1e-6
        fallback = gp1 * rho_L.pow(self.gamma)
        ratio = torch.where(near, fallback, ratio)
        return w_L - ratio

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
        rho = ic_data["ks"]  # (B, K)
        v = ic_data["ks_v"]  # (B, K)
        pieces_mask = ic_data["pieces_mask"]  # (B, K)
        t_coords, x_coords = query_points  # each (B, *spatial)

        K = rho.shape[1]
        spatial_dims = t_coords.shape[1:]
        n_expand = len(spatial_dims)

        # -- per-segment quantities -----------------------------------------------
        rho_L = rho[:, :-1]  # (B, K-1)
        rho_R = rho[:, 1:]  # (B, K-1)
        v_L = v[:, :-1]  # (B, K-1)
        v_R = v[:, 1:]  # (B, K-1)

        w_L = v_L + self._pressure(rho_L)  # (B, K-1)

        # -- intermediate state ---------------------------------------------------
        p_star_val = (w_L - v_R).clamp(min=self.eps)  # clamp avoids vacuum
        rho_star = p_star_val.pow(1.0 / self.gamma)  # (B, K-1)

        # -- 2-wave classification ------------------------------------------------
        is_shock = rho_star > rho_L  # (B, K-1); equivalently v_L > v_R

        # -- 2-wave speeds --------------------------------------------------------
        lam2_L = self._lam2(rho_L, v_L)  # (B, K-1)
        lam2_star = self._lam2(rho_star, v_R)  # v_* = v_R; (B, K-1)
        sigma2 = self._shock_speed_2(rho_L, rho_star, w_L)  # (B, K-1)

        # -- speed selection per interface ----------------------------------------
        # The ARZ wave structure at each interface is (left to right):
        #   2-wave (slower, genuinely nonlinear) → 1-contact (faster, at v_R)
        #
        # Left segment (k): bounded by the outermost RIGHT wave = 1-contact
        #   at speed v_R. This holds for both shock and rarefaction since
        #   v_R > λ₂_* always (the 1-contact is always to the right of the
        #   2-wave).
        speed_right = v_R  # (B, K-1)

        # Right segment (k+1): bounded by the outermost LEFT wave = 2-wave
        #   shock → 2-shock speed σ₂
        #   rarefaction → far fan edge λ₂_L (left edge of 2-rarefaction)
        speed_left = torch.where(is_shock, sigma2, lam2_L)  # (B, K-1)

        # -- interface positions --------------------------------------------------
        x_d = xs[:, 1:K]  # (B, K-1)

        # Expand interface quantities to (B, *spatial, K-1)
        for _ in range(n_expand):
            x_d = x_d.unsqueeze(1)
            speed_right = speed_right.unsqueeze(1)
            speed_left = speed_left.unsqueeze(1)

        t_exp = t_coords.unsqueeze(-1)  # (B, *spatial, 1)
        x_exp = x_coords.unsqueeze(-1)  # (B, *spatial, 1)

        # -- one-sided penalties in similarity variable ξ = (x - x_d)/(t + ε) ----
        xi = (x_exp - x_d) / (t_exp + self.eps)  # (B, *spatial, K-1)
        penalty_left_seg = torch.relu(xi - speed_right)
        penalty_right_seg = torch.relu(speed_left - xi)

        # -- accumulate onto K segments with F.pad --------------------------------
        bias = -(
            F.pad(penalty_left_seg, (0, 1)) + F.pad(penalty_right_seg, (1, 0))
        )  # (B, *spatial, K)

        # -- mask padded segments -------------------------------------------------
        mask = pieces_mask
        for _ in range(n_expand):
            mask = mask.unsqueeze(1)
        bias = bias * mask + (~mask.bool()).float() * (-1e9)

        return bias
