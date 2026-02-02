"""ShockTrajectoryNet: DeepONet-like model for shock trajectory prediction.

This model predicts shock (discontinuity) trajectories in LWR traffic flow
using a branch-trunk architecture inspired by DeepONet. It is trained
unsupervised using Rankine-Hugoniot physics loss.

Architecture:
    Branch Network (DiscontinuityEncoder): Encodes discontinuity information
        using transformer self-attention to handle variable numbers of shocks.
    Trunk Network (TimeEncoder): Encodes query times using Fourier features
        and an MLP.
    TrajectoryDecoder: Fuses branch and trunk outputs using bilinear interaction
        and residual blocks, then predicts position and existence for each shock.
"""

import math

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """Fourier feature encoding for positional information.

    Maps scalar inputs to higher-dimensional space using sinusoidal features,
    which helps MLPs learn high-frequency functions.

    Args:
        num_frequencies: Number of frequency bands.
        include_input: Whether to include the original input.
    """

    def __init__(self, num_frequencies: int = 8, include_input: bool = True):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        # Frequencies from 2^0 to 2^(num_frequencies-1)
        frequencies = 2.0 ** torch.arange(num_frequencies).float()
        self.register_buffer("frequencies", frequencies)

    @property
    def output_dim(self) -> int:
        """Output dimension of the encoding."""
        dim = 2 * self.num_frequencies  # sin and cos
        if self.include_input:
            dim += 1
        return dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier encoding.

        Args:
            x: Input tensor of shape (..., 1) or (...,).

        Returns:
            Encoded tensor of shape (..., output_dim).
        """
        if x.dim() == 1 or x.shape[-1] != 1:
            x = x.unsqueeze(-1)

        # Scale input to [0, 2*pi] range for frequencies
        scaled = x * math.pi * self.frequencies  # (..., num_frequencies)

        # Compute sin and cos features
        features = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)

        if self.include_input:
            features = torch.cat([x, features], dim=-1)

        return features


class TimeEncoder(nn.Module):
    """Trunk network: encodes query times using Fourier features + MLP.

    Args:
        hidden_dim: Hidden dimension of the MLP.
        output_dim: Output dimension (latent space dimension).
        num_frequencies: Number of Fourier frequency bands.
        num_layers: Number of MLP layers.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_frequencies: int = 32,
        num_layers: int = 3,
    ):
        super().__init__()
        self.fourier = FourierFeatures(num_frequencies=num_frequencies)

        layers = []
        in_dim = self.fourier.output_dim
        for i in range(num_layers):
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.LayerNorm(out_dim))
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        """Encode query times.

        Args:
            times: Query times of shape (B, T).

        Returns:
            Encoded times of shape (B, T, output_dim).
        """
        # Fourier encode each time
        B, T = times.shape
        times_flat = times.reshape(-1)  # (B*T,)
        encoded = self.fourier(times_flat)  # (B*T, fourier_dim)
        encoded = self.mlp(encoded)  # (B*T, output_dim)
        return encoded.reshape(B, T, -1)  # (B, T, output_dim)


class DiscontinuityEncoder(nn.Module):
    """Branch network: encodes discontinuities using transformer self-attention.

    Handles variable numbers of discontinuities through masking.

    Args:
        input_dim: Dimension of discontinuity features (default 3: x, rho_L, rho_R).
        hidden_dim: Hidden dimension of the transformer.
        output_dim: Output dimension (latent space dimension).
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Project input discontinuities to hidden dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Transformer encoder layers for self-attention between discontinuities
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Project to output dimension
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        discontinuities: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode discontinuities.

        Args:
            discontinuities: Discontinuity features of shape (B, D, 3)
                where each row is [x_position, rho_L, rho_R].
            mask: Validity mask of shape (B, D) where 1 = valid, 0 = padding.

        Returns:
            Encoded discontinuities of shape (B, D, output_dim).
        """
        # Project input
        x = self.input_proj(discontinuities)  # (B, D, hidden_dim)

        # Create attention mask (True = ignore)
        # PyTorch transformer uses True to mask out positions
        attn_mask = mask == 0  # (B, D)

        # Apply transformer with masking
        x = self.transformer(x, src_key_padding_mask=attn_mask)  # (B, D, hidden_dim)

        # Project to output
        x = self.output_proj(x)  # (B, D, output_dim)

        # Zero out padded positions
        x = x * mask.unsqueeze(-1)

        return x


class ResidualBlock(nn.Module):
    """Residual block with two linear layers and GELU activation."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


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
        existence = self.existence_head(fused).squeeze(-1)  # (B, D, T)

        # Mask out invalid discontinuities
        mask_exp = mask.unsqueeze(-1)  # (B, D, 1)
        positions = positions * mask_exp
        existence = existence * mask_exp

        return {
            "positions": positions,
            "existence": existence,
        }


class ShockTrajectoryNet(nn.Module):
    """DeepONet-like model for shock trajectory prediction.

    Predicts the space-time trajectories of shock waves (discontinuities)
    in LWR traffic flow. The model takes discontinuity information at t=0
    and predicts their positions and existence probability at query times.

    Architecture:
        - Branch network: DiscontinuityEncoder (transformer-based)
        - Trunk network: TimeEncoder (Fourier features + MLP)
        - Decoder: TrajectoryDecoder (bilinear fusion + residual blocks)

    Args:
        hidden_dim: Hidden dimension for all networks.
        num_frequencies: Number of Fourier frequency bands for time encoding.
        num_disc_layers: Number of transformer layers in discontinuity encoder.
        num_time_layers: Number of MLP layers in time encoder.
        num_res_blocks: Number of residual blocks in decoder.
        num_heads: Number of attention heads in discontinuity encoder.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_frequencies: int = 32,
        num_disc_layers: int = 2,
        num_time_layers: int = 3,
        num_res_blocks: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Branch network: discontinuity encoder
        self.branch = DiscontinuityEncoder(
            input_dim=3,  # [x, rho_L, rho_R]
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_disc_layers,
            dropout=dropout,
        )

        # Trunk network: time encoder
        self.trunk = TimeEncoder(
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_frequencies=num_frequencies,
            num_layers=num_time_layers,
        )

        # Trajectory decoder
        self.decoder = TrajectoryDecoder(
            branch_dim=hidden_dim,
            trunk_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
        )

    def forward(
        self,
        batch_input: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            batch_input: Dict with 'discontinuities', 'disc_mask', and 't_coords' tensors.

        Returns:
            Dict containing:
                - 'positions': (B, D, T) predicted x-positions for each shock
                - 'existence': (B, D, T) probability that each shock exists [0, 1]
        """
        discontinuities = batch_input["discontinuities"]
        disc_mask = batch_input["disc_mask"]
        query_times = batch_input["t_coords"]
        # Encode discontinuities (branch)
        branch_emb = self.branch(discontinuities, disc_mask)  # (B, D, hidden_dim)

        # Encode times (trunk)
        trunk_emb = self.trunk(query_times[:, 0, :, 0])  # (B, T, hidden_dim)

        # Decode trajectories
        output = self.decoder(branch_emb, trunk_emb, disc_mask)

        return output

    def predict_from_dict(
        self,
        input_data: dict,
        n_times: int = 100,
        t_max: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Convenience method to predict from standard wavefront data format.

        Args:
            input_data: Dict with 'discontinuities' and 'disc_mask' keys
                (as returned by WavefrontDataset).
            n_times: Number of query time points.
            t_max: Maximum time to predict to.

        Returns:
            Dict with 'positions' and 'existence' predictions.
        """
        discontinuities = input_data["discontinuities"]
        disc_mask = input_data["disc_mask"]

        # Handle batched or single sample
        if discontinuities.dim() == 2:
            discontinuities = discontinuities.unsqueeze(0)
            disc_mask = disc_mask.unsqueeze(0)

        B = discontinuities.shape[0]
        device = discontinuities.device

        # Generate query times
        query_times = torch.linspace(0, t_max, n_times, device=device)
        query_times = query_times.unsqueeze(0).expand(B, -1)

        return self.forward(discontinuities, disc_mask, query_times)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def build_shock_net(args):
    hidden_dim = args.get("hidden_dim", 128)
    num_frequencies = args.get("num_frequencies", 32)
    num_disc_layers = args.get("num_disc_layers", 2)
    num_time_layers = args.get("num_time_layers", 3)
    num_res_blocks = args.get("num_res_blocks", 2)
    num_heads = args.get("num_heads", 4)
    dropout = args.get("dropout", 0.1)
    return ShockTrajectoryNet(hidden_dim, num_frequencies, num_disc_layers, num_time_layers, num_res_blocks, num_heads, dropout)