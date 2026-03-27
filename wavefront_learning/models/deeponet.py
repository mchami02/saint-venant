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
        nx: Number of spatial grid points (branch input size per channel).
        nt: Number of time steps.
        hidden_dim: Hidden layer width for both networks.
        latent_dim: Dimension of the latent space (dot product size).
        num_branch_layers: Number of hidden layers in branch network.
        num_trunk_layers: Number of hidden layers in trunk network.
        output_dim: Number of output channels (1 for LWR, 2 for ARZ, 3 for Euler).
        branch_input_channels: Number of IC channels fed to the branch.
    """

    def __init__(
        self,
        nx: int,
        nt: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_branch_layers: int = 4,
        num_trunk_layers: int = 4,
        output_dim: int = 1,
        branch_input_channels: int = 1,
    ):
        super().__init__()
        self.nx = nx
        self.nt = nt
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.branch_input_channels = branch_input_channels

        # Branch network: IC (nx * branch_input_channels,) -> (latent_dim * output_dim,)
        branch_in = nx * branch_input_channels
        branch_out = latent_dim * output_dim
        branch_layers = [nn.Linear(branch_in, hidden_dim), nn.GELU()]
        for _ in range(num_branch_layers - 1):
            branch_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        branch_layers.append(nn.Linear(hidden_dim, branch_out))
        self.branch = nn.Sequential(*branch_layers)

        # Trunk network: (t, x) -> (latent_dim,)
        trunk_layers = [nn.Linear(2, hidden_dim), nn.GELU()]
        for _ in range(num_trunk_layers - 1):
            trunk_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        trunk_layers.append(nn.Linear(hidden_dim, latent_dim))
        self.trunk = nn.Sequential(*trunk_layers)

        # Bias term (per output channel)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input dict with "grid_input" tensor of shape
               (B, n_ic+2, nt, nx) where the last 2 channels are
               [t_coords, x_coords] and the first n_ic are IC channels.

        Returns:
            Dict with "output_grid" of shape (B, output_dim, nt, nx).
        """
        grid = x["grid_input"]
        B, _, cur_nt, cur_nx = grid.shape
        n_ic = self.branch_input_channels

        # Extract IC channels at t=0
        ic = grid[:, :n_ic, 0, :]  # (B, n_ic, cur_nx)

        # Interpolate IC to training resolution if grid size differs
        if cur_nx != self.nx:
            ic = torch.nn.functional.interpolate(
                ic, size=self.nx, mode="linear", align_corners=False
            )
        ic = ic.reshape(B, -1)  # (B, n_ic * nx)

        # Coordinate channels are always the last 2
        t_coords = grid[:, n_ic, :, :]  # (B, cur_nt, cur_nx)
        x_coords = grid[:, n_ic + 1, :, :]  # (B, cur_nt, cur_nx)

        # Branch: encode IC -> (B, latent_dim * output_dim)
        branch_out = self.branch(ic)

        # Trunk: encode all (t, x) query points
        coords = torch.stack([t_coords, x_coords], dim=-1)
        coords_flat = coords.reshape(B, -1, 2)  # (B, cur_nt*cur_nx, 2)
        trunk_out = self.trunk(coords_flat)  # (B, cur_nt*cur_nx, latent_dim)

        # Dot product per output channel
        branch_reshaped = branch_out.reshape(
            B, self.output_dim, self.latent_dim
        )  # (B, C, latent_dim)
        out = torch.einsum(
            "bcp,bnp->bcn", branch_reshaped, trunk_out
        )  # (B, C, cur_nt*cur_nx)
        out = out + self.bias.reshape(1, self.output_dim, 1)

        # Reshape to actual grid dimensions
        out = out.reshape(B, self.output_dim, cur_nt, cur_nx)

        return {"output_grid": out}


def _equation_channels(args: dict) -> int:
    """Return number of IC/output channels for the equation type."""
    eq = args.get("equation", "LWR")
    if eq == "Euler":
        return 3
    if eq == "ARZ":
        return 2
    return 1


def build_deeponet(args: dict) -> DeepONet:
    """Build DeepONet model from configuration dict.

    Automatically adapts branch input and output channels to the equation type.

    Args:
        args: Configuration dictionary. Supported keys:
            - nx: Spatial grid points (default: 50)
            - nt: Time steps (default: 250)
            - hidden_dim: Hidden layer width (default: 128)
            - latent_dim: Latent/dot-product dimension (default: 64)
            - num_branch_layers: Branch MLP depth (default: 4)
            - num_trunk_layers: Trunk MLP depth (default: 4)
            - equation: Equation type ("LWR", "ARZ", or "Euler")
    """
    nx = args.get("nx", 50)
    nt = args.get("nt", 250)
    hidden_dim = args.get("hidden_dim", 128)
    latent_dim = args.get("latent_dim", 64)
    num_branch_layers = args.get("num_branch_layers", 4)
    num_trunk_layers = args.get("num_trunk_layers", 4)
    n_channels = _equation_channels(args)

    return DeepONet(
        nx=nx,
        nt=nt,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_branch_layers=num_branch_layers,
        num_trunk_layers=num_trunk_layers,
        output_dim=n_channels,
        branch_input_channels=n_channels,
    )
