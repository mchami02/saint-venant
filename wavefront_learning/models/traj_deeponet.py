"""TrajDeepONet: Trajectory-conditioned DeepONet for wavefront learning.

This model predicts shock trajectories and uses them to condition a single
density trunk network. Unlike HybridDeepONet which uses K separate region
trunks and soft sigmoid assembly, TrajDeepONet uses ONE trunk that receives
the left and right boundary positions as extra input features.

Key differences from HybridDeepONet:
- No existence head: all discontinuities persist through time
- Single trunk: one network conditioned on (t, x, x_left, x_right)
- No GridAssembler: the trunk directly outputs the final density

Architecture:
    Branch: DiscontinuityEncoder (Fourier features + MLP)
    Trajectory: TimeEncoder + PositionDecoder (position-only, no existence)
    Trunk: BoundaryConditionedTrunk (single trunk with boundary context)
"""

import torch
import torch.nn as nn

from .base import (
    DiscontinuityEncoder,
    FourierFeatures,
    ResidualBlock,
    TimeEncoder,
)
from .base.transformer_encoder import EncoderLayer


class PositionDecoder(nn.Module):
    """Decodes trajectory positions from branch and trunk embeddings.

    Similar to TrajectoryDecoder but without the existence head.
    Assumes all discontinuities persist through time.

    Args:
        branch_dim: Dimension of branch (discontinuity) embeddings.
        trunk_dim: Dimension of trunk (time) embeddings.
        hidden_dim: Hidden dimension for decoder.
        num_res_blocks: Number of residual blocks.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        branch_dim: int = 64,
        trunk_dim: int = 64,
        hidden_dim: int = 64,
        num_res_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.bilinear = nn.Bilinear(branch_dim, trunk_dim, hidden_dim)
        self.linear_branch = nn.Linear(branch_dim, hidden_dim)
        self.linear_trunk = nn.Linear(trunk_dim, hidden_dim)
        self.fusion_norm = nn.LayerNorm(hidden_dim)

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_res_blocks)]
        )

        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        branch_emb: torch.Tensor,
        trunk_emb: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode trajectory positions.

        Args:
            branch_emb: Discontinuity embeddings (B, D, branch_dim).
            trunk_emb: Time embeddings (B, T, trunk_dim).
            mask: Validity mask (B, D).

        Returns:
            Predicted positions (B, D, T) clamped to [0, 1].
        """
        B, D, _ = branch_emb.shape
        _, T, _ = trunk_emb.shape

        branch_exp = branch_emb.unsqueeze(2).expand(-1, -1, T, -1)
        trunk_exp = trunk_emb.unsqueeze(1).expand(-1, D, -1, -1)

        branch_flat = branch_exp.reshape(-1, branch_exp.shape[-1])
        trunk_flat = trunk_exp.reshape(-1, trunk_exp.shape[-1])

        fused = self.bilinear(branch_flat, trunk_flat)
        fused = fused + self.linear_branch(branch_flat) + self.linear_trunk(trunk_flat)
        fused = self.fusion_norm(fused)

        for block in self.res_blocks:
            fused = block(fused)

        fused = fused.reshape(B, D, T, -1)
        positions = self.position_head(fused).squeeze(-1)  # (B, D, T)
        positions = torch.clamp(positions, 0.0, 1.0)
        positions = positions * mask.unsqueeze(-1)

        return positions


def compute_boundaries(
    positions: torch.Tensor,
    x_coords: torch.Tensor,
    disc_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute left and right boundary positions for each grid point.

    For each spatial point x at time t, finds:
    - left_bound: position of the nearest discontinuity to the left (or 0.0)
    - right_bound: position of the nearest discontinuity to the right (or 1.0)

    Args:
        positions: Predicted discontinuity positions (B, D, nt).
        x_coords: Spatial coordinates (B, nt, nx).
        disc_mask: Validity mask (B, D).

    Returns:
        Tuple of (left_bound, right_bound), each (B, nt, nx).
    """
    B, D, nt = positions.shape

    # (B, D, nt, 1) vs (B, 1, nt, nx)
    pos = positions.unsqueeze(-1)  # (B, D, nt, 1)
    x = x_coords.unsqueeze(1)  # (B, 1, nt, nx)
    mask = disc_mask[:, :, None, None].bool()  # (B, D, 1, 1)

    # Left boundary: largest position <= x among valid discs
    is_left = (pos <= x) & mask
    left_vals = torch.where(is_left, pos, torch.full_like(pos, -float("inf")))
    left_bound = left_vals.max(dim=1).values  # (B, nt, nx)
    left_bound = torch.where(
        left_bound.isinf(), torch.zeros_like(left_bound), left_bound
    )

    # Right boundary: smallest position > x among valid discs
    is_right = (pos > x) & mask
    right_vals = torch.where(is_right, pos, torch.full_like(pos, float("inf")))
    right_bound = right_vals.min(dim=1).values  # (B, nt, nx)
    right_bound = torch.where(
        right_bound.isinf(), torch.ones_like(right_bound), right_bound
    )

    return left_bound, right_bound


class BoundaryConditionedTrunk(nn.Module):
    """Single trunk network conditioned on boundary positions.

    Takes (t, x, x_left, x_right) as input, encodes them with Fourier
    features, fuses with branch embedding, and predicts density.

    Args:
        branch_dim: Dimension of branch embeddings.
        hidden_dim: Hidden dimension.
        num_frequencies_t: Fourier frequencies for time.
        num_frequencies_x: Fourier frequencies for spatial coords.
        num_layers: MLP layers for coordinate encoding.
        num_res_blocks: Residual blocks after fusion.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        branch_dim: int = 64,
        hidden_dim: int = 64,
        num_frequencies_t: int = 8,
        num_frequencies_x: int = 8,
        num_layers: int = 2,
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        with_boundaries: bool = True,
    ):
        super().__init__()
        self.with_boundaries = with_boundaries

        self.fourier_t = FourierFeatures(num_frequencies=num_frequencies_t)
        self.fourier_x = FourierFeatures(num_frequencies=num_frequencies_x)

        # Input: fourier(t) + fourier(x) [+ fourier(x_left) + fourier(x_right)]
        num_spatial = 3 if with_boundaries else 1
        input_dim = self.fourier_t.output_dim + num_spatial * self.fourier_x.output_dim

        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(out_dim))
            in_dim = out_dim
        self.coord_mlp = nn.Sequential(*layers)

        # Bilinear fusion with branch
        self.bilinear = nn.Bilinear(branch_dim, hidden_dim, hidden_dim)
        self.linear_branch = nn.Linear(branch_dim, hidden_dim)
        self.linear_coord = nn.Linear(hidden_dim, hidden_dim)
        self.fusion_norm = nn.LayerNorm(hidden_dim)

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_res_blocks)]
        )

        self.density_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        branch_emb: torch.Tensor,
        t_coords: torch.Tensor,
        x_coords: torch.Tensor,
        left_bound: torch.Tensor | None = None,
        right_bound: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict density conditioned on boundary positions.

        Args:
            branch_emb: Pooled branch embedding (B, branch_dim).
            t_coords: Time coordinates (B, nt, nx).
            x_coords: Space coordinates (B, nt, nx).
            left_bound: Left boundary positions (B, nt, nx). Required if with_boundaries=True.
            right_bound: Right boundary positions (B, nt, nx). Required if with_boundaries=True.

        Returns:
            Predicted density (B, nt, nx) in [0, 1].
        """
        B, nt, nx = t_coords.shape

        # Flatten to (B*nt*nx,)
        t_flat = t_coords.reshape(-1)
        x_flat = x_coords.reshape(-1)

        # Fourier encode
        t_enc = self.fourier_t(t_flat)
        x_enc = self.fourier_x(x_flat)

        if self.with_boundaries:
            left_flat = left_bound.reshape(-1)
            right_flat = right_bound.reshape(-1)
            left_enc = self.fourier_x(left_flat)
            right_enc = self.fourier_x(right_flat)
            coord_features = torch.cat([t_enc, x_enc, left_enc, right_enc], dim=-1)
        else:
            coord_features = torch.cat([t_enc, x_enc], dim=-1)
        coord_emb = self.coord_mlp(coord_features)  # (B*nt*nx, hidden_dim)

        # Expand branch for all grid points
        branch_exp = branch_emb.unsqueeze(1).unsqueeze(1).expand(-1, nt, nx, -1)
        branch_flat = branch_exp.reshape(-1, branch_exp.shape[-1])

        # Bilinear fusion
        fused = self.bilinear(branch_flat, coord_emb)
        fused = fused + self.linear_branch(branch_flat) + self.linear_coord(coord_emb)
        fused = self.fusion_norm(fused)

        for block in self.res_blocks:
            fused = block(fused)

        density = self.density_head(fused).squeeze(-1)  # (B*nt*nx,)
        density = torch.clamp_max(torch.clamp_min(density, 0.0), 1.0)
        return density.reshape(B, nt, nx)


class TrajDeepONet(nn.Module):
    """Trajectory-conditioned DeepONet for wavefront learning.

    Predicts shock trajectories and uses boundary positions to condition
    a single density trunk. Optionally includes a classifier head that
    predicts per-discontinuity existence (shock vs rarefaction).

    Args:
        hidden_dim: Hidden dimension for all networks.
        num_frequencies_t: Fourier frequencies for time encoding.
        num_frequencies_x: Fourier frequencies for space encoding.
        num_disc_frequencies: Fourier frequencies for discontinuity encoder.
        num_disc_layers: MLP layers in discontinuity encoder.
        num_time_layers: MLP layers in time encoder.
        num_coord_layers: MLP layers in coordinate encoder.
        num_res_blocks: Residual blocks in decoders.
        dropout: Dropout rate.
        classifier: If True, add a classifier head predicting shock (1) vs
            rarefaction (0) per discontinuity, constant across time.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_frequencies_t: int = 8,
        num_frequencies_x: int = 8,
        num_disc_frequencies: int = 8,
        num_disc_layers: int = 2,
        num_time_layers: int = 2,
        num_coord_layers: int = 2,
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        with_traj: bool = True,
        classifier: bool = False,
        num_interaction_layers: int = 2,
        num_attention_heads: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.with_traj = with_traj
        self.has_classifier = classifier

        # Optional classifier: predicts shock (1) vs rarefaction (0) per discontinuity
        if self.has_classifier:
            self.classifier_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

        # Branch: encode discontinuities
        self.branch = DiscontinuityEncoder(
            input_dim=3,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_frequencies=num_disc_frequencies,
            num_layers=num_disc_layers,
            dropout=dropout,
        )

        # Cross-discontinuity self-attention
        self.disc_interaction = nn.ModuleList(
            [
                EncoderLayer(hidden_dim, num_heads=num_attention_heads)
                for _ in range(num_interaction_layers)
            ]
        )

        # Trajectory: time encoder + position decoder (only when with_traj)
        if self.with_traj:
            self.time_encoder = TimeEncoder(
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_frequencies=num_frequencies_t,
                num_layers=num_time_layers,
            )

            self.position_decoder = PositionDecoder(
                branch_dim=hidden_dim,
                trunk_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_res_blocks=num_res_blocks,
                dropout=dropout,
            )

        # Single trunk (conditioned on boundaries only when with_traj)
        self.trunk = BoundaryConditionedTrunk(
            branch_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_frequencies_t=num_frequencies_t,
            num_frequencies_x=num_frequencies_x,
            num_layers=num_coord_layers,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            with_boundaries=with_traj,
        )

    def forward(
        self,
        batch_input: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            batch_input: Dict containing:
                - 'discontinuities': (B, D, 3) with [x, rho_L, rho_R]
                - 'disc_mask': (B, D) validity mask
                - 't_coords': (B, 1, nt, nx) time coordinates
                - 'x_coords': (B, 1, nt, nx) space coordinates

        Returns:
            Dict containing:
                - 'positions': (B, D, T) predicted trajectory positions
                - 'output_grid': (B, 1, nt, nx) predicted density grid
                - 'existence': (B, D, T) shock/rarefaction probability
                  (only when classifier=True, constant across T)
        """
        discontinuities = batch_input["discontinuities"]
        disc_mask = batch_input["disc_mask"]
        t_coords = batch_input["t_coords"].squeeze(1)  # (B, nt, nx)
        x_coords = batch_input["x_coords"].squeeze(1)  # (B, nt, nx)

        # === BRANCH ===
        branch_emb = self.branch(discontinuities, disc_mask)  # (B, D, hidden)

        # === CROSS-DISCONTINUITY INTERACTION ===
        key_padding_mask = ~disc_mask.bool()  # True = ignore
        # Unmask all-masked rows to avoid NaN from softmax (output is zeroed anyway)
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked] = False
        for layer in self.disc_interaction:
            branch_emb = layer(branch_emb, key_padding_mask=key_padding_mask)
        branch_emb = branch_emb * disc_mask.unsqueeze(-1)  # re-zero padded

        # === CLASSIFIER (optional) ===
        if self.has_classifier:
            existence = (
                self.classifier_head(branch_emb).squeeze(-1) * disc_mask
            )  # (B, D)

        # Pooled branch for trunk
        mask_exp = disc_mask.unsqueeze(-1)  # (B, D, 1)
        branch_sum = (branch_emb * mask_exp).sum(dim=1)  # (B, hidden)
        n_valid = disc_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        branch_pooled = branch_sum / n_valid  # (B, hidden)

        if self.with_traj:
            # === TRAJECTORY ===
            query_times = t_coords[:, :, 0]  # (B, nt)
            trunk_emb = self.time_encoder(query_times)  # (B, T, hidden)
            positions = self.position_decoder(
                branch_emb, trunk_emb, disc_mask
            )  # (B, D, T)

            # === BOUNDARIES ===
            # When classifier is active, exclude rarefactions from boundary computation
            if self.has_classifier:
                effective_mask = disc_mask * (existence > 0.5).float()
            else:
                effective_mask = disc_mask
            left_bound, right_bound = compute_boundaries(
                positions, x_coords, effective_mask
            )

            # === TRUNK ===
            density = self.trunk(
                branch_pooled, t_coords, x_coords, left_bound, right_bound
            )  # (B, nt, nx)

            output_grid = density.unsqueeze(1)  # (B, 1, nt, nx)

            output = {
                "positions": positions,
                "output_grid": output_grid,
            }
            if self.has_classifier:
                # Expand to (B, D, T) — constant across time
                output["existence"] = existence.unsqueeze(-1).expand_as(positions)
            return output

        # === TRUNK (no trajectory conditioning) ===
        density = self.trunk(branch_pooled, t_coords, x_coords)  # (B, nt, nx)
        output_grid = density.unsqueeze(1)  # (B, 1, nt, nx)

        output = {"output_grid": output_grid}
        return output

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_traj_deeponet(args: dict) -> TrajDeepONet:
    """Build TrajDeepONet from configuration dict.

    Args:
        args: Configuration dictionary with optional keys:
            - hidden_dim (default 32)
            - num_frequencies_t (default 8)
            - num_frequencies_x (default 8)
            - num_disc_frequencies (default 8)
            - num_disc_layers (default 2)
            - num_time_layers (default 2)
            - num_coord_layers (default 2)
            - num_res_blocks (default 2)
            - dropout (default 0.05)

    Returns:
        Configured TrajDeepONet instance.
    """
    return TrajDeepONet(
        hidden_dim=args.get("hidden_dim", 32),
        num_frequencies_t=args.get("num_frequencies_t", 8),
        num_frequencies_x=args.get("num_frequencies_x", 8),
        num_disc_frequencies=args.get("num_disc_frequencies", 8),
        num_disc_layers=args.get("num_disc_layers", 2),
        num_time_layers=args.get("num_time_layers", 2),
        num_coord_layers=args.get("num_coord_layers", 2),
        num_res_blocks=args.get("num_res_blocks", 2),
        dropout=args.get("dropout", 0.0),
        num_interaction_layers=args.get("num_interaction_layers", 2),
        num_attention_heads=args.get("num_attention_heads", 4),
    )


def build_classifier_traj_deeponet(args: dict) -> TrajDeepONet:
    """Build TrajDeepONet with classifier head (ClassifierTrajDeepONet).

    Same as TrajDeepONet but with an additional binary classifier that
    predicts per-discontinuity existence (shock vs rarefaction).

    Args:
        args: Configuration dictionary (same keys as build_traj_deeponet).

    Returns:
        Configured TrajDeepONet instance with classifier=True.
    """
    return TrajDeepONet(
        hidden_dim=args.get("hidden_dim", 32),
        num_frequencies_t=args.get("num_frequencies_t", 8),
        num_frequencies_x=args.get("num_frequencies_x", 8),
        num_disc_frequencies=args.get("num_disc_frequencies", 8),
        num_disc_layers=args.get("num_disc_layers", 2),
        num_time_layers=args.get("num_time_layers", 2),
        num_coord_layers=args.get("num_coord_layers", 2),
        num_res_blocks=args.get("num_res_blocks", 2),
        dropout=args.get("dropout", 0.0),
        classifier=True,
        num_interaction_layers=args.get("num_interaction_layers", 2),
        num_attention_heads=args.get("num_attention_heads", 4),
    )


def build_no_traj_deeponet(args: dict) -> TrajDeepONet:
    """Build TrajDeepONet without trajectory conditioning (NoTrajDeepONet).

    Same architecture as TrajDeepONet but with trajectory prediction and
    boundary conditioning disabled. The trunk operates on (t, x) only.

    Args:
        args: Configuration dictionary with optional keys:
            - hidden_dim (default 32)
            - num_frequencies_t (default 8)
            - num_frequencies_x (default 8)
            - num_disc_frequencies (default 8)
            - num_disc_layers (default 2)
            - num_coord_layers (default 2)
            - num_res_blocks (default 2)
            - dropout (default 0.05)

    Returns:
        Configured TrajDeepONet instance with with_traj=False.
    """
    return TrajDeepONet(
        hidden_dim=args.get("hidden_dim", 32),
        num_frequencies_t=args.get("num_frequencies_t", 8),
        num_frequencies_x=args.get("num_frequencies_x", 8),
        num_disc_frequencies=args.get("num_disc_frequencies", 8),
        num_disc_layers=args.get("num_disc_layers", 2),
        num_coord_layers=args.get("num_coord_layers", 2),
        num_res_blocks=args.get("num_res_blocks", 2),
        dropout=args.get("dropout", 0.0),
        with_traj=False,
        num_interaction_layers=args.get("num_interaction_layers", 2),
        num_attention_heads=args.get("num_attention_heads", 4),
    )
