import math
import torch
import torch.nn as nn


class FourierTokenizer(nn.Module):
    """Tokenizer with Fourier positional encoding for coordinate inputs."""
    def __init__(self, input_dim: int, hidden_dim: int, num_frequencies: int = 4):
        super().__init__()
        self.num_frequencies = num_frequencies
        fourier_dim = input_dim * (2 * num_frequencies + 1)
        self.projection = nn.Linear(fourier_dim, hidden_dim)
        
        frequencies = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)
        self.register_buffer('frequencies', frequencies)
        
    def forward(self, x):
        B, T, N, input_dim = x.shape
        x_scaled = (x.unsqueeze(-1) * self.frequencies * math.pi).reshape(B, T, N, -1)
        fourier_features = torch.cat([x, torch.sin(x_scaled), torch.cos(x_scaled)], dim=-1)
        return self.projection(fourier_features)


class AxialAttention(nn.Module):
    """Memory-efficient axial attention: attend over T and N dimensions separately."""
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.time_attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.space_attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm_t = nn.LayerNorm(hidden_dim)
        self.norm_s = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        # x: (B, T, N, D)
        B, T, N, D = x.shape
        
        # Attention over time dimension
        x_t = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        att_t = self.time_attention(x_t, x_t, x_t)[0]
        att_t = att_t.reshape(B, N, T, D).permute(0, 2, 1, 3)
        x = self.norm_t(x + att_t)
        
        # Attention over space dimension
        x_s = x.reshape(B * T, N, D)
        att_s = self.space_attention(x_s, x_s, x_s)[0]
        att_s = att_s.reshape(B, T, N, D)
        x = self.norm_s(x + att_s)
        
        return x


class AxialDecoderLayer(nn.Module):
    """Decoder layer with axial self-attention and cross-attention."""
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.self_attention = AxialAttention(hidden_dim, num_heads=num_heads)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm_cross = nn.LayerNorm(hidden_dim)
        self.norm_ff = nn.LayerNorm(hidden_dim)

    def forward(self, x, z, key_padding_mask=None):
        B, T, N, D = x.shape
        
        # Axial self-attention
        x = self.self_attention(x).reshape(B, T * N, D)
        
        # Cross-attention to encoder output
        cross_att = self.cross_attention(x, z, z, key_padding_mask=key_padding_mask)[0]
        x = self.norm_cross(x + cross_att)
        
        # Feedforward
        ff = self.feedforward(x)
        x = self.norm_ff(x + ff)
        
        return x.reshape(B, T, N, D)


class AxialDecoder(nn.Module):
    """Decoder with Fourier tokenizer, axial self-attention, and cross-attention."""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_heads: int = 8, num_frequencies: int = 4):
        super().__init__()
        self.tokenizer = FourierTokenizer(input_dim=input_dim, hidden_dim=hidden_dim, num_frequencies=num_frequencies)
        self.layers = nn.ModuleList([
            AxialDecoderLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.to_out = nn.Linear(hidden_dim, 1)

    def forward(self, coords, encoder_output, key_padding_mask=None):
        """
        Args:
            coords: (B, T, N, input_dim) - query coordinates
            encoder_output: (B, S, hidden_dim) - encoder output for cross-attention
            key_padding_mask: (B, S) - mask for encoder output (True = ignore)
        
        Returns:
            output: (B, T, N, 1)
        """
        x = self.tokenizer(coords)
        for layer in self.layers:
            x = layer(x, encoder_output, key_padding_mask=key_padding_mask)
        return self.to_out(x)


if __name__ == "__main__":
    B, T, N = 4, 10, 25
    hidden_dim = 128
    
    coords = torch.randn(B, T, N, 2)
    encoder_output = torch.randn(B, 50, hidden_dim)
    key_padding_mask = torch.zeros(B, 50, dtype=torch.bool)
    key_padding_mask[:, 30:] = True
    
    decoder = AxialDecoder(input_dim=2, hidden_dim=hidden_dim, num_layers=4)
    output = decoder(coords, encoder_output, key_padding_mask=key_padding_mask)
    
    print(f"Input coords shape: {coords.shape}")
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.4f}, std: {output.std().item():.4f}")
