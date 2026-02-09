"""Encoder-Decoder model for wavefront learning.

Transformer encoder processes masked IC conditions, decoder reconstructs
full space-time solution via axial or cross-attention.

Input: tensor of shape (B, 3, nt, nx) from ToGridInputTransform
       channels are [ic_masked, t_coords, x_coords]
Output: dict {"output_grid": tensor of shape (B, 1, nt, nx)}
"""

import torch
import torch.nn as nn

from models.base.axial_decoder import AxialDecoder
from models.base.cross_decoder import CrossDecoder
from models.base.transformer_encoder import Encoder


class EncoderDecoder(nn.Module):
    """Encoder-Decoder model adapted for wavefront learning.

    Args:
        hidden_dim: Hidden dimension for all components.
        layers_encoder: Number of encoder transformer layers.
        decoder_type: "axial" or "cross".
        layers_decoder: Number of decoder layers.
        layers_gnn: Number of GNN layers for shock correction (0 to disable).
        nt, nx, dt, dx, device: Required if layers_gnn > 0.
    """

    def __init__(
        self,
        hidden_dim: int,
        layers_encoder: int,
        decoder_type: str,
        layers_decoder: int,
        layers_gnn: int,
        nt: int = None,
        nx: int = None,
        dt: float = None,
        dx: float = None,
        device=None,
    ):
        super().__init__()

        self.encoder = Encoder(
            input_dim=3, hidden_dim=hidden_dim, num_layers=layers_encoder
        )

        if decoder_type == "axial":
            self.decoder = AxialDecoder(
                input_dim=2, hidden_dim=hidden_dim, num_layers=layers_decoder
            )
        elif decoder_type == "cross":
            self.decoder = CrossDecoder(
                hidden_dim=hidden_dim, num_layers=layers_decoder
            )
        else:
            raise ValueError(
                f"Unknown decoder_type: {decoder_type}. Use 'axial' or 'cross'."
            )

        if layers_gnn > 0:
            if any(v is None for v in [nt, nx, dt, dx, device]):
                raise ValueError(
                    "nt, nx, dt, dx, device are required when layers_gnn > 0"
                )
            from models.base.shock_gnn import ShockGNN

            self.shock_gnn = ShockGNN(
                nx=nx,
                nt=nt,
                dx=dx,
                dt=dt,
                hidden_dim=hidden_dim,
                num_layers=layers_gnn,
                device=device,
            )
        else:
            self.shock_gnn = None

        self.frozen_shock_gnn = False
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.xavier_uniform_(module.out_proj.weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
                if module.out_proj.bias is not None:
                    nn.init.zeros_(module.out_proj.bias)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def freeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = False

    def freeze_shock_gnn(self):
        if self.shock_gnn is not None:
            for param in self.shock_gnn.parameters():
                param.requires_grad = False
            self.frozen_shock_gnn = True

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def unfreeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = True

    def unfreeze_shock_gnn(self):
        if self.shock_gnn is not None:
            for param in self.shock_gnn.parameters():
                param.requires_grad = True
            self.frozen_shock_gnn = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, nt, nx) where channels are
               [ic_masked, t_coords, x_coords].

        Returns:
            Dict with "output_grid" of shape (B, 1, nt, nx).
        """
        B, C, T, N = x.shape

        # Extract conditions where first channel is not -1
        mask = x[:, 0, :, :] != -1  # (B, T, N)
        x_permuted = x.permute(0, 2, 3, 1)  # (B, T, N, C)

        # Count valid elements per batch
        counts = mask.view(B, -1).sum(dim=1)  # (B,)
        max_length = int(counts.max().item())

        # Extract and pad conditions
        conds = torch.full((B, max_length, C), 0.0, device=x.device, dtype=x.dtype)
        key_padding_mask = torch.ones(B, max_length, device=x.device, dtype=torch.bool)
        for b in range(B):
            valid_elements = x_permuted[b][mask[b]]
            conds[b, : counts[b]] = valid_elements
            key_padding_mask[b, : counts[b]] = False

        # Encode
        encoder_output = self.encoder(conds, key_padding_mask=key_padding_mask)

        # Decode
        all_coords = x[:, 1:].permute(0, 2, 3, 1)  # (B, T, N, 2)
        u = self.decoder(all_coords, encoder_output, key_padding_mask=key_padding_mask)

        # Shock correction
        if not self.frozen_shock_gnn and self.shock_gnn is not None:
            delta_u, gate_values = self.shock_gnn(all_coords, u.detach())
        else:
            delta_u = torch.zeros_like(u)

        # (B, T, N, 1) -> (B, 1, T, N)
        output = (u + delta_u).permute(0, 3, 1, 2)

        return {"output_grid": output}


def build_encoder_decoder(args: dict) -> EncoderDecoder:
    """Build EncoderDecoder model with axial decoder.

    Args:
        args: Configuration dictionary. Supported keys:
            - hidden_dim: Hidden dimension (default: 64)
            - layers_encoder: Encoder transformer layers (default: 2)
            - layers_decoder: Decoder layers (default: 2)
            - layers_gnn: GNN correction layers (default: 0)
            - nt, nx, dt, dx: Grid config (required if layers_gnn > 0)
    """
    return EncoderDecoder(
        hidden_dim=args.get("hidden_dim", 64),
        layers_encoder=args.get("layers_encoder", 2),
        decoder_type="axial",
        layers_decoder=args.get("layers_decoder", 2),
        layers_gnn=args.get("layers_gnn", 0),
        nt=args.get("nt"),
        nx=args.get("nx"),
        dt=args.get("dt"),
        dx=args.get("dx"),
    )


def build_encoder_decoder_cross(args: dict) -> EncoderDecoder:
    """Build EncoderDecoder model with cross-attention decoder.

    Args:
        args: Configuration dictionary. Same keys as build_encoder_decoder.
    """
    return EncoderDecoder(
        hidden_dim=args.get("hidden_dim", 64),
        layers_encoder=args.get("layers_encoder", 2),
        decoder_type="cross",
        layers_decoder=args.get("layers_decoder", 2),
        layers_gnn=args.get("layers_gnn", 0),
        nt=args.get("nt"),
        nx=args.get("nx"),
        dt=args.get("dt"),
        dx=args.get("dx"),
    )
