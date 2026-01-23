import torch
import torch.nn as nn


class Tokenizer(nn.Module):
    """Projects input features to hidden dimension."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x):
        return self.projection(x)


class EncoderLayer(nn.Module):
    """Transformer encoder layer with self-attention and feedforward."""
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, key_padding_mask=None):
        att = self.attention(x, x, x, key_padding_mask=key_padding_mask)[0]
        x = self.norm1(x + att)
        ff = self.feedforward(x)
        x = self.norm2(x + ff)
        return x


class Encoder(nn.Module):
    """Transformer encoder with tokenizer and stacked encoder layers."""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_heads: int = 8):
        super().__init__()
        self.tokenizer = Tokenizer(input_dim=input_dim, hidden_dim=hidden_dim)
        self.layers = nn.ModuleList([
            EncoderLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])

    def forward(self, x, key_padding_mask=None):
        x = self.tokenizer(x)
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x


if __name__ == "__main__":
    B, S, C = 4, 50, 3
    x = torch.randn(B, S, C)
    key_padding_mask = torch.zeros(B, S, dtype=torch.bool)
    key_padding_mask[:, 30:] = True
    
    encoder = Encoder(input_dim=C, hidden_dim=128, num_layers=4)
    output = encoder(x, key_padding_mask=key_padding_mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.4f}, std: {output.std().item():.4f}")
