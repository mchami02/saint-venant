"""ShockTrajectoryNet: DeepONet-like model for shock trajectory prediction.

This model predicts shock (discontinuity) trajectories in LWR traffic flow
using a branch-trunk architecture inspired by DeepONet. It is trained
unsupervised using Rankine-Hugoniot physics loss.

Architecture:
    Branch Network (DiscontinuityEncoder): Encodes discontinuity information
        using Fourier features + MLP, processing each discontinuity independently.
    Trunk Network (TimeEncoder): Encodes query times using Fourier features
        and an MLP.
    TrajectoryDecoder: Fuses branch and trunk outputs using bilinear interaction
        and residual blocks, then predicts position and existence for each shock.
"""

import torch
import torch.nn as nn

from .base import DiscontinuityEncoder, TimeEncoder, TrajectoryDecoder


class ShockTrajectoryNet(nn.Module):
    """DeepONet-like model for shock trajectory prediction.

    Predicts the space-time trajectories of shock waves (discontinuities)
    in LWR traffic flow. The model takes discontinuity information at t=0
    and predicts their positions and existence probability at query times.

    Architecture:
        - Branch network: DiscontinuityEncoder (Fourier features + MLP)
        - Trunk network: TimeEncoder (Fourier features + MLP)
        - Decoder: TrajectoryDecoder (bilinear fusion + residual blocks)

    Args:
        hidden_dim: Hidden dimension for all networks.
        num_frequencies: Number of Fourier frequency bands for time encoding.
        num_disc_frequencies: Number of Fourier frequency bands for x in discontinuity encoder.
        num_disc_layers: Number of MLP layers in discontinuity encoder.
        num_time_layers: Number of MLP layers in time encoder.
        num_res_blocks: Number of residual blocks in decoder.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_frequencies: int = 32,
        num_disc_frequencies: int = 16,
        num_disc_layers: int = 3,
        num_time_layers: int = 3,
        num_res_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Branch network: discontinuity encoder
        self.branch = DiscontinuityEncoder(
            input_dim=3,  # [x, rho_L, rho_R]
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_frequencies=num_disc_frequencies,
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
    num_disc_frequencies = args.get("num_disc_frequencies", 16)
    num_disc_layers = args.get("num_disc_layers", 3)
    num_time_layers = args.get("num_time_layers", 3)
    num_res_blocks = args.get("num_res_blocks", 2)
    dropout = args.get("dropout", 0.1)
    return ShockTrajectoryNet(
        hidden_dim,
        num_frequencies,
        num_disc_frequencies,
        num_disc_layers,
        num_time_layers,
        num_res_blocks,
        dropout,
    )
