"""Unified PDE interface for WaveNO.

Provides an abstract ``PDE`` base class and concrete implementations for
LWR, ARZ, and Euler equations.  Each PDE encapsulates:

- **Physics features** for the segment encoder (``physics_features``).
- **Boundary speeds** at each interface for the attention bias (``boundary_speeds``).
- **Collision times** for optional damping (``collision_times``).
- **Output metadata** (``output_dim``, ``output_clamp``) so the model
  auto-adapts to the equation.

Adding a new PDE requires implementing a single subclass of ``PDE``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flux import DEFAULT_FLUX, Flux


class PDE(nn.Module):
    """Abstract PDE interface for WaveNO.

    Subclasses must implement :pymethod:`physics_features` and
    :pymethod:`boundary_speeds`.  Optional: override
    :pymethod:`collision_times` to enable damping.

    Class-level attributes (set by subclasses):
        num_vars: Number of IC state variables per segment.
        num_physics_features: Number of scalar features returned by
            ``physics_features``.
        output_dim: Number of output channels (1 for scalar, 2-3 for systems).
        output_clamp: Optional ``(min, max)`` clamp for output values.
    """

    num_vars: int
    num_physics_features: int
    output_dim: int
    output_clamp: tuple[float, float] | None

    def physics_features(
        self, ic_data: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute per-segment physics scalars.

        Args:
            ic_data: Dictionary with at least ``ks`` (B, K).
                May also contain ``ks_v``, ``ks_p`` for systems.

        Returns:
            ``(B, K, num_physics_features)`` tensor.
        """
        raise NotImplementedError

    def boundary_speeds(
        self, ic_data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute boundary speeds at each interface.

        Args:
            ic_data: Dictionary with segment data.

        Returns:
            ``(speed_right, speed_left)`` each ``(B, K-1)``.
            ``speed_right[i]`` bounds left segment *k=i* from its right
            interface.  ``speed_left[i]`` bounds right segment *k=i+1*
            from its left interface.
        """
        raise NotImplementedError

    def collision_times(
        self, ic_data: dict[str, torch.Tensor]
    ) -> torch.Tensor | None:
        """Compute per-segment collision times (optional).

        Returns:
            ``(B, K)`` positive collision times, or ``None`` if not
            supported by this PDE.
        """
        return None


# ---------------------------------------------------------------------------
# LWR (scalar conservation law with pluggable flux)
# ---------------------------------------------------------------------------


class LWRPDE(PDE):
    """LWR traffic flow: d(rho)/dt + d(f(rho))/dx = 0.

    Physics features: ``[rho, f'(rho), f(rho)]``.
    Boundary speeds: shock speed or rarefaction fan edge.
    Collision times: analytical estimate from characteristic speed
    differences.

    Args:
        flux: Flux function instance (default ``GreenshieldsFlux``).
    """

    num_vars = 1
    num_physics_features = 3
    output_dim = 1
    output_clamp = (0.0, 1.0)

    def __init__(self, flux: Flux | None = None):
        super().__init__()
        self.flux = flux or DEFAULT_FLUX()

    def physics_features(
        self, ic_data: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        ks = ic_data["ks"]  # (B, K)
        lambda_k = self.flux.derivative(ks)  # (B, K)
        flux_k = self.flux(ks)  # (B, K)
        return torch.stack([ks, lambda_k, flux_k], dim=-1)  # (B, K, 3)

    def boundary_speeds(
        self, ic_data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ks = ic_data["ks"]  # (B, K)
        lam = self.flux.derivative(ks)  # (B, K)
        lam_L = lam[:, :-1]  # (B, K-1)
        lam_R = lam[:, 1:]  # (B, K-1)
        is_shock = lam_L > lam_R  # (B, K-1)
        s = self.flux.shock_speed(ks[:, :-1], ks[:, 1:])  # (B, K-1)

        speed_right = torch.where(is_shock, s, lam_R)  # (B, K-1)
        speed_left = torch.where(is_shock, s, lam_L)  # (B, K-1)
        return speed_right, speed_left

    def collision_times(
        self, ic_data: dict[str, torch.Tensor]
    ) -> torch.Tensor | None:
        xs = ic_data["xs"]  # (B, K+1)
        ks = ic_data["ks"]  # (B, K)
        lam = self.flux.derivative(ks)  # (B, K)
        K = ks.shape[1]

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


# ---------------------------------------------------------------------------
# ARZ (Aw-Rascle-Zhang traffic system)
# ---------------------------------------------------------------------------


class ARZPDE(PDE):
    """Aw-Rascle-Zhang traffic flow system.

    Two-wave structure: 2-wave (genuinely nonlinear) + 1-contact
    (linearly degenerate).

    Physics features: ``[rho, v, p(rho), lambda_2]``.
    Boundary speeds: 2-wave speed (shock or rarefaction) and 1-contact
    speed.

    Args:
        gamma: Pressure exponent in ``p(rho) = rho^gamma``.
    """

    num_vars = 2
    num_physics_features = 4
    output_dim = 2
    output_clamp = None

    def __init__(self, gamma: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def _pressure(self, rho: torch.Tensor) -> torch.Tensor:
        return rho.pow(self.gamma)

    def _dp_drho(self, rho: torch.Tensor) -> torch.Tensor:
        if self.gamma == 1.0:
            return torch.ones_like(rho)
        return self.gamma * rho.pow(self.gamma - 1)

    def _lam2(self, rho: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return v - rho * self._dp_drho(rho)

    def _shock_speed_2(
        self,
        rho_L: torch.Tensor,
        rho_star: torch.Tensor,
        w_L: torch.Tensor,
    ) -> torch.Tensor:
        gp1 = self.gamma + 1.0
        numer = rho_star.pow(gp1) - rho_L.pow(gp1)
        denom = rho_star - rho_L
        ratio = numer / (denom + self.eps * denom.sign().clamp(min=1e-30))
        near = denom.abs() < 1e-6
        fallback = gp1 * rho_L.pow(self.gamma)
        ratio = torch.where(near, fallback, ratio)
        return w_L - ratio

    def physics_features(
        self, ic_data: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        rho = ic_data["ks"]  # (B, K)
        v = ic_data["ks_v"]  # (B, K)
        p = self._pressure(rho)  # (B, K)
        lam2 = self._lam2(rho, v)  # (B, K)
        return torch.stack([rho, v, p, lam2], dim=-1)  # (B, K, 4)

    def boundary_speeds(
        self, ic_data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rho = ic_data["ks"]  # (B, K)
        v = ic_data["ks_v"]  # (B, K)

        rho_L = rho[:, :-1]  # (B, K-1)
        rho_R = rho[:, 1:]
        v_L = v[:, :-1]
        v_R = v[:, 1:]

        w_L = v_L + self._pressure(rho_L)  # (B, K-1)

        # Intermediate state
        p_star_val = (w_L - v_R).clamp(min=self.eps)
        rho_star = p_star_val.pow(1.0 / self.gamma)  # (B, K-1)

        # 2-wave classification
        is_shock = rho_star > rho_L  # (B, K-1)

        # 2-wave speeds
        lam2_L = self._lam2(rho_L, v_L)  # (B, K-1)
        sigma2 = self._shock_speed_2(rho_L, rho_star, w_L)  # (B, K-1)

        # Left segment bounded by 1-contact at speed v_R
        speed_right = v_R  # (B, K-1)
        # Right segment bounded by 2-wave
        speed_left = torch.where(is_shock, sigma2, lam2_L)  # (B, K-1)

        return speed_right, speed_left


# ---------------------------------------------------------------------------
# Euler (1D compressible Euler equations)
# ---------------------------------------------------------------------------


def _wave_f_df(
    p_star: torch.Tensor,
    p_K: torch.Tensor,
    A_K: torch.Tensor,
    B_K: torch.Tensor,
    c_K: torch.Tensor,
    gamma: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wave-curve function and derivative for one side (K = L or R)."""
    gm1_o_2g = (gamma - 1.0) / (2.0 * gamma)
    is_shock = p_star > p_K
    # Shock branch
    f_shock = (p_star - p_K) * (A_K / (p_star + B_K).clamp(min=eps)).sqrt()
    df_shock = (
        (A_K / (p_star + B_K).clamp(min=eps)).sqrt()
        * (1.0 - 0.5 * (p_star - p_K) / (p_star + B_K).clamp(min=eps))
    )
    # Rarefaction branch
    ratio = (p_star / p_K.clamp(min=eps)).clamp(min=eps)
    f_rare = (2.0 * c_K / (gamma - 1.0)) * (ratio.pow(gm1_o_2g) - 1.0)
    df_rare = (c_K / (gamma * p_K.clamp(min=eps))) * ratio.pow(-(gamma + 1.0) / (2.0 * gamma))
    f = torch.where(is_shock, f_shock, f_rare)
    df = torch.where(is_shock, df_shock, df_rare)
    return f, df


class EulerPDE(PDE):
    """1D compressible Euler equations.

    Three-wave structure: 1-wave (genuinely nonlinear), 2-contact
    (linearly degenerate), 3-wave (genuinely nonlinear).

    Physics features: ``[rho, u, p, c, lam1, lam3, Mach]``.
    Boundary speeds: 1-wave speed (bounding right segment) and 3-wave
    speed (bounding left segment), computed by solving the Riemann
    problem via Newton's method.

    Args:
        gamma: Heat capacity ratio (default 1.4 for ideal gas).
        eps: Numerical stability constant.
        n_newton: Number of Newton iterations for Riemann solver.
    """

    num_vars = 3
    num_physics_features = 7
    output_dim = 3
    output_clamp = None

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

    def _sound_speed(
        self, rho: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        return (
            (self.gamma * p / rho.clamp(min=self.eps))
            .clamp(min=self.eps)
            .sqrt()
        )

    def _solve_riemann(
        self,
        rho_L: torch.Tensor,
        u_L: torch.Tensor,
        p_L: torch.Tensor,
        rho_R: torch.Tensor,
        u_R: torch.Tensor,
        p_R: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve the Euler Riemann problem for (p_star, u_star)."""
        gamma = self.gamma
        eps = self.eps

        c_L = self._sound_speed(rho_L, p_L)
        c_R = self._sound_speed(rho_R, p_R)

        A_L = 2.0 / ((gamma + 1.0) * rho_L.clamp(min=eps))
        B_L = ((gamma - 1.0) / (gamma + 1.0)) * p_L
        A_R = 2.0 / ((gamma + 1.0) * rho_R.clamp(min=eps))
        B_R = ((gamma - 1.0) / (gamma + 1.0)) * p_R

        # Initial guess: two-rarefaction approximation
        p_star = (
            (c_L + c_R - 0.5 * (gamma - 1.0) * (u_R - u_L))
            / (
                c_L / p_L.clamp(min=eps).pow((gamma - 1.0) / (2.0 * gamma))
                + c_R / p_R.clamp(min=eps).pow((gamma - 1.0) / (2.0 * gamma))
            )
        ).pow(2.0 * gamma / (gamma - 1.0))
        p_star = p_star.clamp(min=eps)

        # Newton iterations
        for _ in range(self.n_newton):
            f_L, df_L = _wave_f_df(p_star, p_L, A_L, B_L, c_L, gamma, eps)
            f_R, df_R = _wave_f_df(p_star, p_R, A_R, B_R, c_R, gamma, eps)
            residual = f_L + f_R + (u_R - u_L)
            jacobian = (df_L + df_R).clamp(min=eps)
            p_star = (p_star - residual / jacobian).clamp(min=eps)

        # Compute u_star from converged p_star
        f_L, _ = _wave_f_df(p_star, p_L, A_L, B_L, c_L, gamma, eps)
        f_R, _ = _wave_f_df(p_star, p_R, A_R, B_R, c_R, gamma, eps)
        u_star = 0.5 * (u_L + u_R) + 0.5 * (f_R - f_L)

        return p_star, u_star

    def physics_features(
        self, ic_data: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        rho = ic_data["ks"]  # (B, K)
        u = ic_data["ks_v"]  # (B, K)
        p = ic_data["ks_p"]  # (B, K)
        c = self._sound_speed(rho, p)  # (B, K)
        lam1 = u - c  # (B, K)
        lam3 = u + c  # (B, K)
        mach = u / c.clamp(min=self.eps)  # (B, K)
        return torch.stack([rho, u, p, c, lam1, lam3, mach], dim=-1)

    def boundary_speeds(
        self, ic_data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rho = ic_data["ks"]  # (B, K)
        u = ic_data["ks_v"]  # (B, K)
        p = ic_data["ks_p"]  # (B, K)

        gamma = self.gamma
        eps = self.eps

        rho_L, rho_R = rho[:, :-1], rho[:, 1:]
        u_L, u_R = u[:, :-1], u[:, 1:]
        p_L, p_R = p[:, :-1], p[:, 1:]

        p_star, u_star = self._solve_riemann(
            rho_L, u_L, p_L, rho_R, u_R, p_R
        )

        c_L = self._sound_speed(rho_L, p_L)
        c_R = self._sound_speed(rho_R, p_R)

        # Wave classification
        is_1_shock = p_star > p_L
        is_3_shock = p_star > p_R

        # 1-wave speeds (bounding right segment from the left)
        gp1_o_2g = (gamma + 1.0) / (2.0 * gamma)
        sigma_1 = u_L - c_L * (
            1.0 + gp1_o_2g * (p_star / p_L.clamp(min=eps) - 1.0)
        ).clamp(min=eps).sqrt()
        lam1_L = u_L - c_L
        speed_left = torch.where(is_1_shock, sigma_1, lam1_L)

        # 3-wave speeds (bounding left segment from the right)
        sigma_3 = u_R + c_R * (
            1.0 + gp1_o_2g * (p_star / p_R.clamp(min=eps) - 1.0)
        ).clamp(min=eps).sqrt()
        lam3_R = u_R + c_R
        speed_right = torch.where(is_3_shock, sigma_3, lam3_R)

        return speed_right, speed_left
