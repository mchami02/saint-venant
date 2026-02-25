"""DeepONet decoder for latent diffusion model.

Resolution-invariant decoder that maps a latent vector z to function
values at arbitrary (t, x) query points via a branch-trunk architecture.
"""

import torch
import torch.nn as nn

from .feature_encoders import FourierFeatures


class DeepONetDecoder(nn.Module):
    """DeepONet-style decoder: branch net processes z, trunk net processes coordinates.

    The output at each query point is the dot product of branch and trunk
    outputs plus a learned bias. Since the trunk evaluates at arbitrary
    (t, x) points, the decoder is resolution-invariant.

    Args:
        latent_dim: Dimension of the latent input z.
        num_basis: Number of basis functions (branch-trunk dot product dimension).
        trunk_hidden_dim: Hidden dimension of the trunk MLP.
        trunk_num_layers: Number of layers in the trunk MLP.
        branch_hidden_dim: Hidden dimension of the branch MLP.
        branch_num_layers: Number of layers in the branch MLP.
        num_frequencies_t: Fourier frequency bands for time coordinates.
        num_frequencies_x: Fourier frequency bands for space coordinates.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        num_basis: int = 64,
        trunk_hidden_dim: int = 128,
        trunk_num_layers: int = 3,
        branch_hidden_dim: int = 128,
        branch_num_layers: int = 3,
        num_frequencies_t: int = 16,
        num_frequencies_x: int = 16,
    ):
        super().__init__()
        self.num_basis = num_basis

        # Branch net: latent z -> basis coefficients
        branch_layers = []
        in_dim = latent_dim
        for i in range(branch_num_layers):
            out_dim = num_basis if i == branch_num_layers - 1 else branch_hidden_dim
            branch_layers.append(nn.Linear(in_dim, out_dim))
            if i < branch_num_layers - 1:
                branch_layers.append(nn.GELU())
                branch_layers.append(nn.LayerNorm(out_dim))
            in_dim = out_dim
        self.branch_net = nn.Sequential(*branch_layers)

        # Trunk net: (t, x) coordinates -> basis values
        self.fourier_t = FourierFeatures(num_frequencies=num_frequencies_t)
        self.fourier_x = FourierFeatures(num_frequencies=num_frequencies_x)
        trunk_input_dim = self.fourier_t.output_dim + self.fourier_x.output_dim

        trunk_layers = []
        in_dim = trunk_input_dim
        for i in range(trunk_num_layers):
            out_dim = num_basis if i == trunk_num_layers - 1 else trunk_hidden_dim
            trunk_layers.append(nn.Linear(in_dim, out_dim))
            if i < trunk_num_layers - 1:
                trunk_layers.append(nn.GELU())
                trunk_layers.append(nn.LayerNorm(out_dim))
            in_dim = out_dim
        self.trunk_net = nn.Sequential(*trunk_layers)

        # Learned bias
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        z: torch.Tensor,
        t_coords: torch.Tensor,
        x_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Decode latent vector at given query points.

        Args:
            z: Latent vector of shape (B, latent_dim).
            t_coords: Time coordinates of shape (B, nt, nx).
            x_coords: Space coordinates of shape (B, nt, nx).

        Returns:
            Output grid of shape (B, 1, nt, nx).
        """
        B, nt, nx = t_coords.shape

        # Branch: (B, latent_dim) -> (B, num_basis)
        branch_out = self.branch_net(z)

        # Trunk: encode (t, x) coordinates
        t_flat = t_coords.reshape(-1)  # (B*nt*nx,)
        x_flat = x_coords.reshape(-1)  # (B*nt*nx,)

        t_encoded = self.fourier_t(t_flat)  # (B*nt*nx, fourier_t_dim)
        x_encoded = self.fourier_x(x_flat)  # (B*nt*nx, fourier_x_dim)
        trunk_input = torch.cat([t_encoded, x_encoded], dim=-1)
        trunk_out = self.trunk_net(trunk_input)  # (B*nt*nx, num_basis)
        trunk_out = trunk_out.reshape(B, nt * nx, self.num_basis)

        # Dot product: sum over basis functions
        # branch_out: (B, num_basis) -> (B, 1, num_basis)
        # trunk_out: (B, nt*nx, num_basis)
        output = torch.sum(branch_out.unsqueeze(1) * trunk_out, dim=-1)  # (B, nt*nx)
        output = output + self.bias

        return output.reshape(B, 1, nt, nx)
