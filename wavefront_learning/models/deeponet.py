"""Classic DeepONet baseline for wavefront learning.

Standard DeepONet with branch (IC encoder) and trunk (coordinate encoder)
networks connected via dot product. Used as a baseline for comparison
against wavefront-specific architectures.

Input: dict with "xs" (B, K+1), "ks" (B, K), "pieces_mask" (B, K),
       "t_coords" (B, 1, nt, nx), "x_coords" (B, 1, nt, nx).
Output: dict {"output_grid": tensor of shape (B, 1, nt, nx)}
"""

import torch
import torch.nn as nn


class DeepONet(nn.Module):
    """Classic DeepONet with branch-trunk dot product.

    Branch network encodes the piecewise-constant IC (breakpoints + values).
    Trunk network encodes query (t, x) coordinates.
    Output is the dot product of branch and trunk embeddings.

    Args:
        max_discontinuities: Maximum number of discontinuities (determines
            branch input size: 2*max_discontinuities + 1).
        hidden_dim: Hidden layer width for both networks.
        latent_dim: Dimension of the latent space (dot product size).
        num_branch_layers: Number of hidden layers in branch network.
        num_trunk_layers: Number of hidden layers in trunk network.
    """

    def __init__(
        self,
        max_discontinuities: int = 10,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_branch_layers: int = 4,
        num_trunk_layers: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Branch input: xs (max_disc+1) + ks*mask (max_disc) = 2*max_disc + 1
        branch_input_dim = 2 * max_discontinuities + 1

        # Branch network: flat IC -> (latent_dim,)
        branch_layers = [nn.Linear(branch_input_dim, hidden_dim), nn.GELU()]
        for _ in range(num_branch_layers - 1):
            branch_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        branch_layers.append(nn.Linear(hidden_dim, latent_dim))
        self.branch = nn.Sequential(*branch_layers)

        # Trunk network: (t, x) -> (latent_dim,)
        trunk_layers = [nn.Linear(2, hidden_dim), nn.GELU()]
        for _ in range(num_trunk_layers - 1):
            trunk_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        trunk_layers.append(nn.Linear(hidden_dim, latent_dim))
        self.trunk = nn.Sequential(*trunk_layers)

        # Bias term
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input dict with:
                - "xs": (B, K+1) breakpoint positions
                - "ks": (B, K) piece values
                - "pieces_mask": (B, K) validity mask
                - "t_coords": (B, 1, nt, nx) time coordinates
                - "x_coords": (B, 1, nt, nx) space coordinates

        Returns:
            Dict with "output_grid" of shape (B, 1, nt, nx).
        """
        xs = x["xs"]  # (B, K+1)
        ks = x["ks"]  # (B, K)
        pieces_mask = x["pieces_mask"]  # (B, K)
        t_coords = x["t_coords"].squeeze(1)  # (B, nt, nx)
        x_coords = x["x_coords"].squeeze(1)  # (B, nt, nx)

        B, nt, nx = t_coords.shape

        # Branch: encode IC from breakpoints and masked values
        branch_input = torch.cat([xs, ks * pieces_mask], dim=-1)  # (B, 2K+1)
        branch_out = self.branch(branch_input)  # (B, latent_dim)

        # Trunk: encode all (t, x) query points
        coords = torch.stack([t_coords, x_coords], dim=-1)  # (B, nt, nx, 2)
        coords_flat = coords.reshape(B, -1, 2)  # (B, nt*nx, 2)
        trunk_out = self.trunk(coords_flat)  # (B, nt*nx, latent_dim)

        # Dot product: branch (B, latent_dim) x trunk (B, nt*nx, latent_dim)
        out = torch.einsum("bp,bnp->bn", branch_out, trunk_out)  # (B, nt*nx)
        out = out + self.bias

        # Reshape to grid
        out = out.reshape(B, 1, nt, nx)

        return {"output_grid": out}


def build_deeponet(args: dict) -> DeepONet:
    """Build DeepONet model from configuration dict.

    Args:
        args: Configuration dictionary. Supported keys:
            - max_discontinuities: Max IC discontinuities (default: 10)
            - hidden_dim: Hidden layer width (default: 128)
            - latent_dim: Latent/dot-product dimension (default: 64)
            - num_branch_layers: Branch MLP depth (default: 4)
            - num_trunk_layers: Trunk MLP depth (default: 4)
    """
    max_discontinuities = args.get("max_discontinuities", 10)
    hidden_dim = args.get("hidden_dim", 128)
    latent_dim = args.get("latent_dim", 64)
    num_branch_layers = args.get("num_branch_layers", 4)
    num_trunk_layers = args.get("num_trunk_layers", 4)

    return DeepONet(
        max_discontinuities=max_discontinuities,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_branch_layers=num_branch_layers,
        num_trunk_layers=num_trunk_layers,
    )
