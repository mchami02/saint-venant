"""Conditional VAE DeepONet for wavefront learning.

When the input IC is on a coarse grid, the exact shock location within a cell
is ambiguous -- the true high-res output is multimodal. A deterministic model
averages over modes, smearing shocks. The CVAE DeepONet learns a latent space
capturing this ambiguity: at test time, different z samples from the prior
produce distinct sharp solutions, with variance peaking at shock locations.

Architecture:
    IC Encoder: shared MLP, IC (nx,) -> condition c (condition_dim,)
    Inference Network (training only): (c, target_flat) -> (z_mean_q, z_logvar_q)
    Prior Network: c -> (z_mean_p, z_logvar_p)
    Branch: [z, c] -> coefficients (dot_product_dim,)
    Trunk: SpaceTimeEncoder with Fourier features, (t,x) -> basis (nt, nx, dot_product_dim)
    Output: einsum(branch, trunk) + bias -> (1, nt, nx)

Loss: MSE(recon) + beta * KL(q(z|a,u) || p(z|a)) with KL annealing + free bits.

Input: dict with "grid_input" key of shape (B, 3, nt, nx) from ToGridInputTransform.
       Optionally "target_grid" of shape (B, 1, nt, nx) during training.
Output: dict with "output_grid", "z_mean_p", "z_logvar_p",
        and optionally "z_mean_q", "z_logvar_q" during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base.feature_encoders import SpaceTimeEncoder


def _build_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int,
    use_layer_norm: bool = True,
) -> nn.Sequential:
    """Build an MLP with GELU activations and optional LayerNorm."""
    layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
    if use_layer_norm:
        layers.append(nn.LayerNorm(hidden_dim))
    for _ in range(num_layers - 2):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class CVAEDeepONet(nn.Module):
    """Conditional VAE DeepONet.

    Uses a conditional VAE to model the distribution of solutions given
    a coarse initial condition, combined with a DeepONet trunk for
    resolution-invariant coordinate encoding.

    Args:
        nx: Number of spatial grid points.
        nt: Number of time steps.
        hidden_dim: Hidden layer width.
        condition_dim: Dimension of the condition embedding.
        latent_dim: Dimension of the latent variable z.
        dot_product_dim: Dimension of the branch-trunk dot product.
        num_branch_layers: Number of layers in branch MLP.
        num_trunk_layers: Number of layers in trunk SpaceTimeEncoder.
    """

    needs_target_input = True

    def __init__(
        self,
        nx: int,
        nt: int,
        hidden_dim: int = 128,
        condition_dim: int = 64,
        latent_dim: int = 32,
        dot_product_dim: int = 64,
        num_branch_layers: int = 4,
        num_trunk_layers: int = 4,
    ):
        super().__init__()
        self.nx = nx
        self.nt = nt
        self.latent_dim = latent_dim
        self.dot_product_dim = dot_product_dim

        # IC Encoder: IC (nx,) -> condition c (condition_dim,)
        self.ic_encoder = _build_mlp(
            nx, hidden_dim, condition_dim, num_layers=3,
        )

        # Prior Network: c -> (z_mean_p, z_logvar_p)
        self.prior_net = _build_mlp(
            condition_dim, hidden_dim, 2 * latent_dim, num_layers=3,
        )

        # Inference Network (training only): (c, target_flat) -> (z_mean_q, z_logvar_q)
        # Target is downsampled to fixed size for resolution independence
        self._pool_size = (16, 16)
        inference_input_dim = 16 * 16  # flattened pooled target
        self.inference_proj = nn.Sequential(
            nn.Linear(inference_input_dim, hidden_dim),
            nn.GELU(),
        )
        self.inference_net = _build_mlp(
            condition_dim + hidden_dim, hidden_dim, 2 * latent_dim, num_layers=3,
        )

        # Branch: [z, c] -> coefficients (dot_product_dim,)
        self.branch = _build_mlp(
            latent_dim + condition_dim, hidden_dim, dot_product_dim,
            num_layers=num_branch_layers,
        )

        # Trunk: SpaceTimeEncoder (t, x) -> (nt, nx, dot_product_dim)
        self.trunk = SpaceTimeEncoder(
            hidden_dim=hidden_dim,
            output_dim=dot_product_dim,
            num_frequencies_t=16,
            num_frequencies_x=16,
            num_layers=num_trunk_layers,
        )

        # Bias term
        self.bias = nn.Parameter(torch.zeros(1))

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z using the reparameterization trick.

        Args:
            mean: Mean of the Gaussian, shape (..., latent_dim).
            logvar: Log variance of the Gaussian, shape (..., latent_dim).

        Returns:
            Sampled z of shape (..., latent_dim).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input dict with:
                - "grid_input": (B, 3, nt, nx) from ToGridInputTransform
                - "target_grid": (B, 1, nt, nx) optional, present during training

        Returns:
            Dict with "output_grid", "z_mean_p", "z_logvar_p",
            and optionally "z_mean_q", "z_logvar_q".
        """
        grid = x["grid_input"]
        B = grid.shape[0]
        nt_in = grid.shape[2]
        nx_in = grid.shape[3]

        # Extract IC from first channel at t=0
        ic = grid[:, 0, 0, :]  # (B, nx_in)

        # Interpolate IC to training resolution if needed (for high-res generalization)
        if nx_in != self.nx:
            ic = F.interpolate(
                ic.unsqueeze(1), size=self.nx, mode="linear", align_corners=False
            ).squeeze(1)  # (B, nx)

        # Extract coordinate grids
        t_coords = grid[:, 1, :, :]  # (B, nt_in, nx_in)
        x_coords = grid[:, 2, :, :]  # (B, nt_in, nx_in)

        # Encode IC -> condition
        condition = self.ic_encoder(ic)  # (B, condition_dim)

        # Prior: condition -> (z_mean_p, z_logvar_p)
        prior_params = self.prior_net(condition)  # (B, 2*latent_dim)
        z_mean_p, z_logvar_p = prior_params.chunk(2, dim=-1)

        result = {
            "z_mean_p": z_mean_p,
            "z_logvar_p": z_logvar_p,
        }

        # Inference or prior sampling
        if "target_grid" in x:
            # Training: use inference network
            target = x["target_grid"]  # (B, 1, nt, nx)
            # Use interpolate instead of AdaptiveAvgPool2d (MPS-compatible)
            target_pooled = F.interpolate(
                target, size=self._pool_size, mode="bilinear", align_corners=False
            )  # (B, 1, 16, 16)
            target_flat = target_pooled.reshape(B, -1)  # (B, 256)
            target_proj = self.inference_proj(target_flat)  # (B, hidden_dim)

            inf_input = torch.cat([condition, target_proj], dim=-1)
            inf_params = self.inference_net(inf_input)  # (B, 2*latent_dim)
            z_mean_q, z_logvar_q = inf_params.chunk(2, dim=-1)

            z = self.reparameterize(z_mean_q, z_logvar_q)
            result["z_mean_q"] = z_mean_q
            result["z_logvar_q"] = z_logvar_q
        else:
            # Inference: sample from prior
            z = self.reparameterize(z_mean_p, z_logvar_p)

        # Branch: [z, condition] -> coefficients
        branch_input = torch.cat([z, condition], dim=-1)  # (B, latent_dim + condition_dim)
        branch_out = self.branch(branch_input)  # (B, dot_product_dim)

        # Trunk: encode (t, x) coordinates
        trunk_out = self.trunk(t_coords, x_coords)  # (B, nt_in, nx_in, dot_product_dim)
        trunk_flat = trunk_out.reshape(B, -1, self.dot_product_dim)  # (B, nt_in*nx_in, dot_product_dim)

        # Dot product + bias
        out = torch.einsum("bp,bnp->bn", branch_out, trunk_flat)  # (B, nt_in*nx_in)
        out = out + self.bias

        # Reshape to grid
        out = out.reshape(B, 1, nt_in, nx_in)
        result["output_grid"] = out

        return result


def build_cvae_deeponet(args: dict) -> CVAEDeepONet:
    """Build CVAEDeepONet model from configuration dict.

    Args:
        args: Configuration dictionary. Supported keys:
            - nx: Spatial grid points (default: 50)
            - nt: Time steps (default: 250)
            - hidden_dim: Hidden layer width (default: 128)
            - condition_dim: Condition embedding dimension (default: 64)
            - latent_dim: Latent variable dimension (default: 32)
            - dot_product_dim: Branch-trunk dot product dimension (default: 64)
            - num_branch_layers: Branch MLP depth (default: 4)
            - num_trunk_layers: Trunk encoder depth (default: 4)
    """
    return CVAEDeepONet(
        nx=args.get("nx", 50),
        nt=args.get("nt", 250),
        hidden_dim=args.get("hidden_dim", 128),
        condition_dim=args.get("condition_dim", 64),
        latent_dim=args.get("latent_dim", 32),
        dot_product_dim=args.get("dot_product_dim", 64),
        num_branch_layers=args.get("num_branch_layers", 4),
        num_trunk_layers=args.get("num_trunk_layers", 4),
    )
