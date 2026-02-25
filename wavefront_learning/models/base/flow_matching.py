"""Flow matching components for latent diffusion.

Contains:
- ConditionEncoder: Encodes piecewise IC (xs, ks, pieces_mask) into a condition vector.
- FlowMatchingDenoiser: Predicts velocity field v(z_t, t, c) for OT flow matching.
- HeunODESolver: Solves the ODE dz/dt = v(z, t, c) from t=0 to t=1 using Heun's method.
"""

import math

import torch
import torch.nn as nn

from .blocks import ResidualBlock


class ConditionEncoder(nn.Module):
    """Encodes piecewise constant IC parameters into a condition vector.

    Concatenates (xs, ks, pieces_mask) into a flat vector and maps it
    through an MLP to produce a fixed-size condition embedding.

    Args:
        max_pieces: Maximum number of pieces (determines input size).
        condition_dim: Output condition dimension.
        hidden_dim: Hidden dimension of the MLP.
        num_layers: Number of MLP layers.
    """

    def __init__(
        self,
        max_pieces: int = 10,
        condition_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        # Input: xs (max_pieces+1) + ks (max_pieces) + pieces_mask (max_pieces)
        input_dim = (max_pieces + 1) + max_pieces + max_pieces

        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = condition_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.LayerNorm(out_dim))
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, batch_input: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode IC parameters to condition vector.

        Args:
            batch_input: Dict with keys "xs", "ks", "pieces_mask",
                each of shape (B, *).

        Returns:
            Condition vector of shape (B, condition_dim).
        """
        xs = batch_input["xs"]
        ks = batch_input["ks"]
        mask = batch_input["pieces_mask"]
        flat = torch.cat([xs, ks, mask], dim=-1)
        return self.mlp(flat)


class FlowMatchingDenoiser(nn.Module):
    """Predicts the velocity field for OT flow matching.

    Takes noisy latent z_t, diffusion time t, and condition c,
    and predicts the velocity v such that dz/dt = v.

    Uses sinusoidal time embedding and ResidualBlock layers.

    Args:
        latent_dim: Dimension of the latent space.
        condition_dim: Dimension of the condition embedding.
        hidden_dim: Hidden dimension of the MLP.
        num_residual_blocks: Number of residual blocks.
        time_embed_dim: Dimension of the time embedding.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        condition_dim: int = 64,
        hidden_dim: int = 256,
        num_residual_blocks: int = 4,
        time_embed_dim: int = 64,
    ):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        # Sinusoidal time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input projection: z_t + time_embed + condition -> hidden
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim + condition_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Residual blocks
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout=0.1) for _ in range(num_residual_blocks)]
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def _sinusoidal_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal time embedding.

        Args:
            t: Diffusion time of shape (B,) in [0, 1].

        Returns:
            Embedding of shape (B, time_embed_dim).
        """
        half_dim = self.time_embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=t.device, dtype=t.dtype)
            / half_dim
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity field.

        Args:
            z_t: Noisy latent of shape (B, latent_dim).
            t: Diffusion time of shape (B,) in [0, 1].
            condition: Condition vector of shape (B, condition_dim).

        Returns:
            Predicted velocity of shape (B, latent_dim).
        """
        # Time embedding
        t_emb = self._sinusoidal_embedding(t)  # (B, time_embed_dim)
        t_emb = self.time_mlp(t_emb)  # (B, hidden_dim)

        # Concatenate inputs
        h = torch.cat([z_t, t_emb, condition], dim=-1)
        h = self.input_proj(h)

        # Residual blocks
        for block in self.blocks:
            h = block(h)

        return self.output_proj(h)


class HeunODESolver:
    """Heun's method ODE solver for flow matching inference.

    Solves dz/dt = v(z, t, c) from t=0 to t=1 starting from noise.
    """

    @staticmethod
    @torch.no_grad()
    def solve(
        denoiser: FlowMatchingDenoiser,
        noise: torch.Tensor,
        condition: torch.Tensor,
        num_steps: int = 100,
    ) -> torch.Tensor:
        """Solve the flow ODE using Heun's method (2nd order).

        Args:
            denoiser: The velocity predictor network.
            noise: Initial noise of shape (B, latent_dim).
            condition: Condition vector of shape (B, condition_dim).
            num_steps: Number of integration steps.

        Returns:
            Denoised latent of shape (B, latent_dim).
        """
        dt = 1.0 / num_steps
        z = noise.clone()
        B = noise.shape[0]

        for i in range(num_steps):
            t_val = i * dt
            t = torch.full((B,), t_val, device=noise.device, dtype=noise.dtype)

            # Euler predictor
            v1 = denoiser(z, t, condition)
            z_euler = z + dt * v1

            # Heun corrector (trapezoidal)
            t_next = torch.full(
                (B,), t_val + dt, device=noise.device, dtype=noise.dtype
            )
            v2 = denoiser(z_euler, t_next, condition)
            z = z + dt * 0.5 * (v1 + v2)

        return z
