"""Shock-aware DeepONet with dual output heads.

Standard DeepONet augmented with a second output head that predicts a
shock proximity field. The shared trunk (SpaceTimeEncoder) learns
shock-aware basis functions, improving solution sharpness.

Input: dict with "grid_input" key containing (B, 3, nt, nx)
       from ToGridInputTransform. Channels are [ic_masked, t_coords, x_coords].
Output: dict {
    "output_grid": (B, 1, nt, nx),
    "shock_proximity": (B, 1, nt, nx),
}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base.feature_encoders import SpaceTimeEncoder


class ShockAwareDeepONet(nn.Module):
    """DeepONet with shared trunk and dual branch heads.

    A shared IC encoder produces a hidden representation. Two separate
    branch heads produce coefficients for the solution and proximity
    fields respectively. A shared SpaceTimeEncoder trunk provides the
    basis functions. Output is formed via einsum + per-head bias.

    Args:
        nx: Number of spatial grid points (IC input size).
        hidden_dim: Width of IC encoder and branch heads.
        latent_dim: Dimension of the branch-trunk dot product.
        num_ic_layers: Number of hidden layers in the IC encoder.
        num_branch_layers: Number of hidden layers per branch head.
        num_trunk_layers: Number of MLP layers in SpaceTimeEncoder.
    """

    def __init__(
        self,
        nx: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_ic_layers: int = 3,
        num_branch_layers: int = 2,
        num_trunk_layers: int = 3,
    ):
        super().__init__()
        self.nx = nx
        self.latent_dim = latent_dim

        # Shared IC encoder: (nx,) -> (hidden_dim,)
        ic_layers = [nn.Linear(nx, hidden_dim), nn.GELU()]
        for _ in range(num_ic_layers - 1):
            ic_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        self.ic_encoder = nn.Sequential(*ic_layers)

        # Solution branch: (hidden_dim,) -> (latent_dim,)
        sol_layers = []
        for _ in range(num_branch_layers - 1):
            sol_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        sol_layers.append(nn.Linear(hidden_dim, latent_dim))
        self.solution_branch = nn.Sequential(*sol_layers)

        # Proximity branch: (hidden_dim,) -> (latent_dim,)
        prox_layers = []
        for _ in range(num_branch_layers - 1):
            prox_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        prox_layers.append(nn.Linear(hidden_dim, latent_dim))
        self.proximity_branch = nn.Sequential(*prox_layers)

        # Shared trunk: SpaceTimeEncoder
        self.trunk = SpaceTimeEncoder(
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            num_layers=num_trunk_layers,
        )

        # Per-head bias parameters
        self.bias_sol = nn.Parameter(torch.zeros(1))
        self.bias_prox = nn.Parameter(torch.zeros(1))

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input dict with "grid_input" of shape (B, 3, nt, nx).

        Returns:
            Dict with "output_grid" (B, 1, nt, nx) and
            "shock_proximity" (B, 1, nt, nx).
        """
        grid = x["grid_input"]
        B, _, nt, nx = grid.shape

        # Extract IC and coordinates
        ic = grid[:, 0, 0, :]  # (B, nx_actual)
        t_coords = grid[:, 1, :, :]  # (B, nt, nx)
        x_coords = grid[:, 2, :, :]  # (B, nt, nx)

        # Adaptive pooling for resolution invariance: (B, nx_actual) -> (B, self.nx)
        if ic.shape[-1] != self.nx:
            ic = F.adaptive_avg_pool1d(ic.unsqueeze(1), self.nx).squeeze(1)

        # Shared IC encoding
        ic_emb = self.ic_encoder(ic)  # (B, hidden_dim)

        # Branch heads
        sol_coeffs = self.solution_branch(ic_emb)  # (B, latent_dim)
        prox_coeffs = self.proximity_branch(ic_emb)  # (B, latent_dim)

        # Trunk: encode all (t, x) points
        trunk_out = self.trunk(t_coords, x_coords)  # (B, nt, nx, latent_dim)

        # Dot products
        sol_out = torch.einsum("bp,btnp->btn", sol_coeffs, trunk_out)  # (B, nt, nx)
        sol_out = sol_out + self.bias_sol
        sol_out = sol_out.unsqueeze(1)  # (B, 1, nt, nx)

        prox_out = torch.einsum("bp,btnp->btn", prox_coeffs, trunk_out)  # (B, nt, nx)
        prox_out = torch.sigmoid(prox_out + self.bias_prox)
        prox_out = prox_out.unsqueeze(1)  # (B, 1, nt, nx)

        return {
            "output_grid": sol_out,
            "shock_proximity": prox_out,
        }


def build_shock_aware_deeponet(args: dict) -> ShockAwareDeepONet:
    """Build ShockAwareDeepONet from configuration dict.

    Args:
        args: Configuration dictionary. Supported keys:
            - nx: Spatial grid points (default: 50)
            - hidden_dim: Hidden layer width (default: 128)
            - latent_dim: Latent/dot-product dimension (default: 64)
            - num_ic_layers: IC encoder depth (default: 3)
            - num_branch_layers: Branch head depth (default: 2)
            - num_trunk_layers: Trunk MLP depth (default: 3)
    """
    return ShockAwareDeepONet(
        nx=args.get("nx", 50),
        hidden_dim=args.get("hidden_dim", 128),
        latent_dim=args.get("latent_dim", 64),
        num_ic_layers=args.get("num_ic_layers", 3),
        num_branch_layers=args.get("num_branch_layers", 2),
        num_trunk_layers=args.get("num_trunk_layers", 3),
    )
