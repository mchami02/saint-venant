"""Euler-equation per-segment attention bias (similarity-variable formulation).

Computes a physics-informed attention bias for each IC segment based on
the three-wave structure of the 1D compressible Euler equations.

The Euler system has three wave families at each interface:

- **1-wave** (genuinely nonlinear, leftward): rarefaction or shock with
  eigenvalue ``u - c`` where ``c = sqrt(gamma * p / rho)`` is the sound speed.
- **2-wave** (linearly degenerate): contact discontinuity at speed ``u*``.
- **3-wave** (genuinely nonlinear, rightward): rarefaction or shock with
  eigenvalue ``u + c``.

Only the outermost waves are used for the bias:

- Left segment bounded by the 3-wave (rightward, fastest from interface)
- Right segment bounded by the 1-wave (leftward, fastest from interface)
- The 2-contact is ignored and learned by the network.

No collision-time damping is applied.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_encoders import FourierFeatures


class EulerBias(nn.Module):
    """Per-segment attention bias from Euler interface dynamics.

    For each interface between segments k and k+1 the module solves the
    Euler Riemann problem via Newton's method to find the star-state
    pressure and velocity, then computes wave speeds for the 1-wave and
    3-wave. One-sided penalties are accumulated in the self-similar
    coordinate ``xi = (x - x_d) / (t + eps)``.

    Args:
        gamma: Heat capacity ratio (default 1.4 for ideal gas).
        eps: Small constant for numerical stability (default 1e-6).
        n_newton: Number of Newton iterations for Riemann solver (default 20).
    """

    def __init__(
        self,
        gamma: float = 1.4,
        eps: float = 1e-6,
        n_newton: int = 20,
    ):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        self.n_newton = n_newton

    def _sound_speed(self, rho: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """c = sqrt(gamma * p / rho), clamped for stability."""
        return (self.gamma * p / rho.clamp(min=self.eps)).clamp(min=self.eps).sqrt()

    def _solve_riemann(
        self,
        rho_L: torch.Tensor,
        u_L: torch.Tensor,
        p_L: torch.Tensor,
        rho_R: torch.Tensor,
        u_R: torch.Tensor,
        p_R: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve the Euler Riemann problem for (p_star, u_star).

        Uses Newton's method on f(p) = f_L(p) + f_R(p) + (u_R - u_L) = 0
        where f_K is the wave-curve function (shock or rarefaction branch).

        All operations are vectorized and differentiable via torch.where.

        Returns:
            Tuple of (p_star, u_star), same shape as inputs.
        """
        gamma = self.gamma
        eps = self.eps
        gm1 = gamma - 1.0
        gp1 = gamma + 1.0

        c_L = self._sound_speed(rho_L, p_L)
        c_R = self._sound_speed(rho_R, p_R)

        # A_K = 2 / ((gamma+1) * rho_K)
        A_L = 2.0 / (gp1 * rho_L.clamp(min=eps))
        A_R = 2.0 / (gp1 * rho_R.clamp(min=eps))
        # B_K = (gamma-1)/(gamma+1) * p_K
        B_L = gm1 / gp1 * p_L
        B_R = gm1 / gp1 * p_R

        # Exponents
        exp_rare = gm1 / (2.0 * gamma)  # (gamma-1)/(2*gamma)

        def _wave_f_df(p, p_K, A_K, B_K, c_K):
            """Wave curve f and f' for one side."""
            denom_s = (p + B_K).clamp(min=eps)
            sqrt_AoD = (A_K / denom_s).clamp(min=eps).sqrt()
            f_shock = (p - p_K) * sqrt_AoD
            df_shock = sqrt_AoD * (1.0 - (p - p_K) / (2.0 * denom_s))

            p_ratio = (p / p_K.clamp(min=eps)).clamp(min=eps)
            f_rare = 2.0 * c_K / gm1 * (p_ratio.pow(exp_rare) - 1.0)
            df_rare = c_K / (gamma * p_K.clamp(min=eps)) * p_ratio.pow(
                -(gp1) / (2.0 * gamma)
            )

            is_shock = p > p_K
            return torch.where(is_shock, f_shock, f_rare), torch.where(
                is_shock, df_shock, df_rare
            )

        # Initial guess: two-rarefaction approximation (good for most cases)
        p_star = (
            (c_L + c_R - gm1 / 2.0 * (u_R - u_L))
            / (
                c_L / p_L.clamp(min=eps).pow(exp_rare)
                + c_R / p_R.clamp(min=eps).pow(exp_rare)
            )
        ).pow(1.0 / exp_rare).clamp(min=eps)

        # Newton iterations
        for _ in range(self.n_newton):
            f_L, df_L = _wave_f_df(p_star, p_L, A_L, B_L, c_L)
            f_R, df_R = _wave_f_df(p_star, p_R, A_R, B_R, c_R)

            residual = f_L + f_R + (u_R - u_L)
            jacobian = (df_L + df_R).clamp(min=eps)
            p_star = (p_star - residual / jacobian).clamp(min=eps)

        # Compute u_star from converged p_star
        f_L, _ = _wave_f_df(p_star, p_L, A_L, B_L, c_L)
        f_R, _ = _wave_f_df(p_star, p_R, A_R, B_R, c_R)
        u_star = 0.5 * (u_L + u_R) + 0.5 * (f_R - f_L)

        return p_star, u_star

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
                - ``ks_p``: (B, K) segment pressure values.
                - ``pieces_mask``: (B, K) validity mask (1=valid).
            query_points: Tuple ``(t_coords, x_coords)`` each of shape
                ``(B, nt, nx)``.

        Returns:
            Bias tensor ``(B, nt, nx, K)``.
        """
        xs = ic_data["xs"]  # (B, K+1)
        rho = ic_data["ks"]  # (B, K)
        u = ic_data["ks_v"]  # (B, K)
        p = ic_data["ks_p"]  # (B, K)
        pieces_mask = ic_data["pieces_mask"]  # (B, K)
        t_coords, x_coords = query_points  # each (B, *spatial)

        gamma = self.gamma
        eps = self.eps
        K = rho.shape[1]

        # -- left/right states at each interface --------------------------------
        rho_L = rho[:, :-1]  # (B, K-1)
        rho_R = rho[:, 1:]
        u_L = u[:, :-1]
        u_R = u[:, 1:]
        p_L = p[:, :-1]
        p_R = p[:, 1:]

        # -- solve Riemann problem at each interface ----------------------------
        p_star, u_star = self._solve_riemann(rho_L, u_L, p_L, rho_R, u_R, p_R)

        # -- sound speeds -------------------------------------------------------
        c_L = self._sound_speed(rho_L, p_L)
        c_R = self._sound_speed(rho_R, p_R)

        # -- wave classification ------------------------------------------------
        is_1_shock = p_star > p_L  # 1-wave is a shock
        is_3_shock = p_star > p_R  # 3-wave is a shock

        # -- 1-wave speeds (bounding right segment from the left) ---------------
        # 1-shock: sigma_1 = u_L - c_L * sqrt(1 + (gamma+1)/(2*gamma) * (p*/p_L - 1))
        gp1_o_2g = (gamma + 1.0) / (2.0 * gamma)
        sigma_1 = u_L - c_L * (
            1.0 + gp1_o_2g * (p_star / p_L.clamp(min=eps) - 1.0)
        ).clamp(min=eps).sqrt()
        # 1-rarefaction: leftmost fan edge = u_L - c_L
        lam1_L = u_L - c_L
        speed_left = torch.where(is_1_shock, sigma_1, lam1_L)  # (B, K-1)

        # -- 3-wave speeds (bounding left segment from the right) ---------------
        # 3-shock: sigma_3 = u_R + c_R * sqrt(1 + (gamma+1)/(2*gamma) * (p*/p_R - 1))
        sigma_3 = u_R + c_R * (
            1.0 + gp1_o_2g * (p_star / p_R.clamp(min=eps) - 1.0)
        ).clamp(min=eps).sqrt()
        # 3-rarefaction: rightmost fan edge = u_R + c_R
        lam3_R = u_R + c_R
        speed_right = torch.where(is_3_shock, sigma_3, lam3_R)  # (B, K-1)

        # -- interface positions ------------------------------------------------
        x_d = xs[:, 1:K]  # (B, K-1)

        # Expand interface quantities to (B, 1, 1, K-1)
        x_d = x_d[:, None, None, :]
        speed_right = speed_right[:, None, None, :]
        speed_left = speed_left[:, None, None, :]

        t_exp = t_coords.unsqueeze(-1)  # (B, *spatial, 1)
        x_exp = x_coords.unsqueeze(-1)  # (B, *spatial, 1)

        # -- one-sided penalties in similarity variable -------------------------
        xi = (x_exp - x_d) / (t_exp + eps)  # (B, *spatial, K-1)
        penalty_left_seg = torch.relu(xi - speed_right)
        penalty_right_seg = torch.relu(speed_left - xi)

        # -- accumulate onto K segments with F.pad ------------------------------
        bias = -(
            F.pad(penalty_left_seg, (0, 1)) + F.pad(penalty_right_seg, (1, 0))
        )  # (B, *spatial, K)

        # -- mask padded segments -----------------------------------------------
        mask = pieces_mask[:, None, None, :]  # (B, 1, 1, K)
        bias = bias * mask + (~mask.bool()).float() * (-1e9)

        return bias


class EulerSegmentEncoder(nn.Module):
    """Encodes IC segments with Euler-specific physics features.

    For each segment k with values (rho_k, u_k, p_k) on [xs[k], xs[k+1]]:
      - x_center, width (Fourier-encoded)
      - rho_k, u_k, p_k (primitive variables)
      - c_k = sqrt(gamma * p_k / rho_k) (sound speed)
      - lam1_k = u_k - c_k (1-characteristic speed)
      - lam3_k = u_k + c_k (3-characteristic speed)
      - Mach_k = u_k / c_k (Mach number)
      - N_k (optional cumulative mass)

    Args:
        hidden_dim: Output embedding dimension.
        num_frequencies: Fourier frequency bands for spatial features.
        num_layers: MLP depth.
        gamma: Heat capacity ratio (default 1.4).
        include_cumulative_mass: Include prefix-sum mass feature.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_frequencies: int | None = 8,
        num_layers: int = 2,
        gamma: float = 1.4,
        include_cumulative_mass: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.eps = 1e-6
        self.include_cumulative_mass = include_cumulative_mass

        if num_frequencies is not None:
            self.fourier_x = FourierFeatures(
                num_frequencies=num_frequencies, include_input=True
            )
            spatial_dim = self.fourier_x.output_dim
        else:
            self.fourier_x = None
            spatial_dim = 1

        # Scalar features: rho, u, p, c, lam1, lam3, Mach [+ N_k] = 7 or 8
        num_scalars = 8 if include_cumulative_mass else 7
        input_dim = 2 * spatial_dim + num_scalars

        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
                layers.append(nn.LayerNorm(out_dim))
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dim

    def forward(
        self,
        xs: torch.Tensor,
        ks: torch.Tensor,
        ks_v: torch.Tensor,
        ks_p: torch.Tensor,
        pieces_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode IC segments with Euler physics features.

        Args:
            xs: Breakpoint positions (B, K+1).
            ks: Density piece values (B, K).
            ks_v: Velocity piece values (B, K).
            ks_p: Pressure piece values (B, K).
            pieces_mask: Validity mask (B, K), 1 = valid.

        Returns:
            Segment embeddings (B, K, hidden_dim).
        """
        B, K = ks.shape

        x_center = (xs[:, :-1] + xs[:, 1:]) / 2  # (B, K)
        width = xs[:, 1:] - xs[:, :-1]  # (B, K)

        # Physics features
        c_k = (
            (self.gamma * ks_p / ks.clamp(min=self.eps))
            .clamp(min=self.eps)
            .sqrt()
        )  # (B, K)
        lam1_k = ks_v - c_k  # (B, K)
        lam3_k = ks_v + c_k  # (B, K)
        mach_k = ks_v / c_k.clamp(min=self.eps)  # (B, K)

        # Encode spatial features
        if self.fourier_x is not None:
            x_center_enc = self.fourier_x(x_center.reshape(-1)).reshape(
                B, K, -1
            )
            width_enc = self.fourier_x(width.reshape(-1)).reshape(B, K, -1)
        else:
            x_center_enc = x_center.unsqueeze(-1)
            width_enc = width.unsqueeze(-1)

        # Concatenate all features
        if self.include_cumulative_mass:
            rho_width = ks * width
            N_k = torch.cumsum(rho_width, dim=1) - rho_width
            total_mass = (
                (rho_width * pieces_mask).sum(dim=1, keepdim=True).clamp(min=1e-8)
            )
            N_k_normalized = N_k / total_mass
            scalar_features = torch.stack(
                [ks, ks_v, ks_p, c_k, lam1_k, lam3_k, mach_k, N_k_normalized],
                dim=-1,
            )  # (B, K, 8)
        else:
            scalar_features = torch.stack(
                [ks, ks_v, ks_p, c_k, lam1_k, lam3_k, mach_k], dim=-1
            )  # (B, K, 7)

        features = torch.cat(
            [x_center_enc, width_enc, scalar_features], dim=-1
        )

        output = self.mlp(features)  # (B, K, H)
        output = output * pieces_mask.unsqueeze(-1)
        return output
