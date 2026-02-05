"""Decoder modules for wavefront learning."""

import torch
import torch.nn as nn

from .blocks import ResidualBlock


class TrajectoryDecoder(nn.Module):
    """Decodes trajectory predictions from branch and trunk embeddings.

    Uses bilinear fusion to combine discontinuity (branch) and time (trunk)
    embeddings, then applies residual blocks and separate heads for
    position and existence prediction.

    Args:
        branch_dim: Dimension of branch (discontinuity) embeddings.
        trunk_dim: Dimension of trunk (time) embeddings.
        hidden_dim: Hidden dimension for decoder.
        num_res_blocks: Number of residual blocks.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        branch_dim: int = 128,
        trunk_dim: int = 128,
        hidden_dim: int = 128,
        num_res_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Bilinear fusion layer
        # Maps (branch_dim, trunk_dim) -> hidden_dim
        self.bilinear = nn.Bilinear(branch_dim, trunk_dim, hidden_dim)

        # Also add a linear combination path for better gradient flow
        self.linear_branch = nn.Linear(branch_dim, hidden_dim)
        self.linear_trunk = nn.Linear(trunk_dim, hidden_dim)

        # Fusion normalization
        self.fusion_norm = nn.LayerNorm(hidden_dim)

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_res_blocks)
        ])

        # Output heads
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.existence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        branch_emb: torch.Tensor,
        trunk_emb: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Decode trajectories.

        Args:
            branch_emb: Discontinuity embeddings of shape (B, D, branch_dim).
            trunk_emb: Time embeddings of shape (B, T, trunk_dim).
            mask: Validity mask of shape (B, D).

        Returns:
            Dict with:
                - 'positions': (B, D, T) predicted x-positions for each shock at each time
                - 'existence': (B, D, T) probability that each shock exists at each time
        """
        B, D, _ = branch_emb.shape
        _, T, _ = trunk_emb.shape

        # Expand embeddings for all (discontinuity, time) pairs
        # branch: (B, D, 1, branch_dim) -> (B, D, T, branch_dim)
        # trunk: (B, 1, T, trunk_dim) -> (B, D, T, trunk_dim)
        branch_exp = branch_emb.unsqueeze(2).expand(-1, -1, T, -1)
        trunk_exp = trunk_emb.unsqueeze(1).expand(-1, D, -1, -1)

        # Reshape for bilinear: (B*D*T, dim)
        branch_flat = branch_exp.reshape(-1, branch_exp.shape[-1])
        trunk_flat = trunk_exp.reshape(-1, trunk_exp.shape[-1])

        # Bilinear fusion + linear paths
        fused = self.bilinear(branch_flat, trunk_flat)
        fused = fused + self.linear_branch(branch_flat) + self.linear_trunk(trunk_flat)
        fused = self.fusion_norm(fused)

        # Residual blocks
        for block in self.res_blocks:
            fused = block(fused)

        # Reshape back: (B, D, T, hidden_dim)
        fused = fused.reshape(B, D, T, -1)

        # Apply output heads
        positions = self.position_head(fused).squeeze(-1)  # (B, D, T)
        positions = torch.clamp(positions, 0.0, 1.0)  # Constrain to grid domain
        existence = self.existence_head(fused).squeeze(-1)  # (B, D, T)

        # Mask out invalid discontinuities
        mask_exp = mask.unsqueeze(-1)  # (B, D, 1)
        positions = positions * mask_exp
        existence = existence * mask_exp

        return {
            "positions": positions,
            "existence": existence,
        }
