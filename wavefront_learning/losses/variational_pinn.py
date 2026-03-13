"""Variational (weak-form) PINN loss for conservation laws.

Enforces the weak form of the conservation law:

    ∫∫ [ρ ∂φ/∂t + f(ρ) ∂φ/∂x] dt dx = 0

for smooth test functions φ. Unlike the strong-form PDE residual which
fails at shocks, the weak form is valid everywhere including discontinuities
(it IS the definition of a weak solution).

We use localized Gaussian test functions centered at random points in the
domain, with derivatives computed analytically. No finite differences needed.
"""

import torch

from .base import BaseLoss
from .flux import greenshields_flux


class VariationalPINNLoss(BaseLoss):
    """Weak-form PDE loss using Gaussian test functions.

    For each sample in the batch, generates random Gaussian test functions
    and evaluates the weak residual. The loss is the mean squared residual
    across all test functions.

    Args:
        n_test_fns: Number of random test functions per sample.
        sigma_t: Std dev of Gaussian in time direction (relative to domain).
        sigma_x: Std dev of Gaussian in space direction (relative to domain).
        dt: Time step size (for coordinate grid).
        dx: Spatial step size (for coordinate grid).
    """

    def __init__(
        self,
        n_test_fns: int = 16,
        sigma_t: float = 0.15,
        sigma_x: float = 0.15,
        dt: float = 0.004,
        dx: float = 0.02,
    ):
        super().__init__()
        self.n_test_fns = n_test_fns
        self.sigma_t = sigma_t
        self.sigma_x = sigma_x
        self.dt = dt
        self.dx = dx

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        output_dict: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute variational PINN loss.

        Args:
            input_dict: Not used.
            output_dict: Must contain 'output_grid' (B, 1, nt, nx).
            target: Not used (self-supervised).

        Returns:
            Tuple of (loss, components dict).
        """
        pred = output_dict["output_grid"]
        if pred.dim() == 4:
            pred = pred.squeeze(1)  # (B, nt, nx)

        B, nt, nx = pred.shape
        device = pred.device

        # Build coordinate grids
        t_grid = torch.arange(nt, device=device, dtype=pred.dtype) * self.dt  # (nt,)
        x_grid = torch.arange(nx, device=device, dtype=pred.dtype) * self.dx  # (nx,)

        t_max = t_grid[-1].item()
        x_max = x_grid[-1].item()

        # Random test function centers — avoid edges where test fn has no support
        margin_t = self.sigma_t
        margin_x = self.sigma_x
        t_centers = torch.rand(B, self.n_test_fns, device=device) * (t_max - 2 * margin_t) + margin_t
        x_centers = torch.rand(B, self.n_test_fns, device=device) * (x_max - 2 * margin_x) + margin_x

        # Compute Gaussian test functions and their derivatives
        # t_grid: (nt,) -> (1, 1, nt, 1) for broadcasting
        # t_centers: (B, K) -> (B, K, 1, 1)
        t_diff = t_grid.view(1, 1, nt, 1) - t_centers.unsqueeze(-1).unsqueeze(-1)  # (B, K, nt, 1)
        x_diff = x_grid.view(1, 1, 1, nx) - x_centers.unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, nx)

        sigma_t2 = self.sigma_t ** 2
        sigma_x2 = self.sigma_x ** 2

        # φ(t,x) = exp(-0.5 * ((t-t0)²/σ_t² + (x-x0)²/σ_x²))
        gauss_t = torch.exp(-0.5 * t_diff ** 2 / sigma_t2)  # (B, K, nt, 1)
        gauss_x = torch.exp(-0.5 * x_diff ** 2 / sigma_x2)  # (B, K, 1, nx)

        # ∂φ/∂t = -(t-t0)/σ_t² * φ
        dphi_dt = -(t_diff / sigma_t2) * gauss_t * gauss_x  # (B, K, nt, nx)

        # ∂φ/∂x = -(x-x0)/σ_x² * φ
        dphi_dx = -(x_diff / sigma_x2) * gauss_t * gauss_x  # (B, K, nt, nx)

        # Compute flux
        flux = greenshields_flux(pred)  # (B, nt, nx)

        # Expand pred and flux for broadcasting with K test functions
        rho_exp = pred.unsqueeze(1)  # (B, 1, nt, nx)
        flux_exp = flux.unsqueeze(1)  # (B, 1, nt, nx)

        # Weak residual: ∫∫ [ρ ∂φ/∂t + f(ρ) ∂φ/∂x] dt dx
        # Approximate integral via summation * dt * dx
        integrand = rho_exp * dphi_dt + flux_exp * dphi_dx  # (B, K, nt, nx)
        residual = integrand.sum(dim=(-2, -1)) * self.dt * self.dx  # (B, K)

        # Loss: mean squared residual
        loss = (residual ** 2).mean()

        return loss, {"variational_pinn": loss.item()}
