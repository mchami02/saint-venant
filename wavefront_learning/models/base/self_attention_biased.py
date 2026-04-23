"""Self-attention with additive physics bias.

Pre-norm self-attention block that accepts an additive ``attn_mask`` on
the attention logits (same semantics as
``BiasedCrossDecoderLayer`` but for a single token sequence).
"""

import torch
import torch.nn as nn


class BiasedSelfAttentionLayer(nn.Module):
    """Pre-norm self-attention block with additive attention bias.

    Forward:
        x' = x + MHA(LN(x), LN(x), LN(x), attn_mask=bias)
        x  = x' + MLP(LN(x'))

    Args:
        hidden_dim: Token embedding dimension.
        num_heads: Attention heads.
        dropout: Dropout rate applied to residuals.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, batch_first=True
        )

        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, N, H) token features.
            key_padding_mask: (B, N) boolean mask, True = ignore.
            attn_mask: (B*num_heads, N, N) additive bias added to logits
                before softmax. Negative values suppress attention.

        Returns:
            Updated token features (B, N, H).
        """
        x_norm = self.norm_attn(x)
        att = self.attention(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )[0]
        x = x + self.drop(att)

        ff = self.feedforward(self.norm_ff(x))
        x = x + self.drop(ff)

        return x
