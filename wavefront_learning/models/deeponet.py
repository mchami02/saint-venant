"""Classic DeepONet baseline for wavefront learning.

Standard DeepONet with branch (IC encoder) and trunk (coordinate encoder)
networks connected via dot product. Used as a baseline for comparison
against wavefront-specific architectures.

Input: dict with "grid_input" key containing tensor of shape (B, 3, nt, nx)
       from ToGridInputTransform. Channels are [ic_masked, t_coords, x_coords].
Output: dict {"output_grid": tensor of shape (B, 1, nt, nx)}
"""

import torch
import torch.nn as nn


class DeepONet(nn.Module):
    """Classic DeepONet with branch-trunk dot product.

    Branch network encodes the initial condition (flattened IC at t=0).
    Trunk network encodes query (t, x) coordinates.
    Output is the dot product of branch and trunk embeddings.

    Args:
        nx: Number of spatial grid points (branch input size).
        nt: Number of time steps.
        hidden_dim: Hidden layer width for both networks.
        latent_dim: Dimension of the latent space (dot product size).
        num_branch_layers: Number of hidden layers in branch network.
        num_trunk_layers: Number of hidden layers in trunk network.
    """

    def __init__(
        self,
        nx: int,
        nt: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_branch_layers: int = 4,
        num_trunk_layers: int = 4,
    ):
        super().__init__()
        self.nx = nx
        self.nt = nt
        self.latent_dim = latent_dim

        # Branch network: IC (nx,) -> (latent_dim,)
        branch_layers = [nn.Linear(nx, hidden_dim), nn.GELU()]
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
            x: Input dict with "grid_input" tensor of shape (B, 3, nt, nx)
               where channels are [ic_masked, t_coords, x_coords].

        Returns:
            Dict with "output_grid" of shape (B, 1, nt, nx).
        """
        grid = x["grid_input"]
        B = grid.shape[0]

        # Extract initial condition from first channel at t=0
        ic = grid[:, 0, 0, :]  # (B, nx)

        # Extract coordinate grids
        t_coords = grid[:, 1, :, :]  # (B, nt, nx)
        x_coords = grid[:, 2, :, :]  # (B, nt, nx)

        # Branch: encode IC
        branch_out = self.branch(ic)  # (B, latent_dim)

        # Trunk: encode all (t, x) query points
        # Stack coordinates: (B, nt, nx, 2)
        coords = torch.stack([t_coords, x_coords], dim=-1)
        coords_flat = coords.reshape(B, -1, 2)  # (B, nt*nx, 2)
        trunk_out = self.trunk(coords_flat)  # (B, nt*nx, latent_dim)

        # Dot product: branch (B, 1, latent_dim) * trunk (B, nt*nx, latent_dim)
        out = torch.einsum("bp,bnp->bn", branch_out, trunk_out)  # (B, nt*nx)
        out = out + self.bias

        # Reshape to grid
        out = out.reshape(B, 1, self.nt, self.nx)

        return {"output_grid": out}


def build_deeponet(args: dict) -> DeepONet:
    """Build DeepONet model from configuration dict.

    Args:
        args: Configuration dictionary. Supported keys:
            - nx: Spatial grid points (default: 50)
            - nt: Time steps (default: 250)
            - hidden_dim: Hidden layer width (default: 128)
            - latent_dim: Latent/dot-product dimension (default: 64)
            - num_branch_layers: Branch MLP depth (default: 4)
            - num_trunk_layers: Trunk MLP depth (default: 4)
    """
    nx = args.get("nx", 50)
    nt = args.get("nt", 250)
    hidden_dim = args.get("hidden_dim", 128)
    latent_dim = args.get("latent_dim", 64)
    num_branch_layers = args.get("num_branch_layers", 4)
    num_trunk_layers = args.get("num_trunk_layers", 4)

    return DeepONet(
        nx=nx,
        nt=nt,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_branch_layers=num_branch_layers,
        num_trunk_layers=num_trunk_layers,
    )
