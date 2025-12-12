import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Tokenizer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(Tokenizer, self).__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x):
        return self.projection(x)


class FourierTokenizer(nn.Module):
    """Tokenizer with Fourier positional encoding for coordinate inputs."""
    def __init__(self, input_dim: int, hidden_dim: int, num_frequencies: int = 32):
        super(FourierTokenizer, self).__init__()
        self.num_frequencies = num_frequencies
        # Fourier features: original + sin/cos of scaled coordinates
        fourier_dim = input_dim * (2 * num_frequencies + 1)
        self.projection = nn.Linear(fourier_dim, hidden_dim)
        
        # Frequency scales (exponentially spaced)
        frequencies = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)
        self.register_buffer('frequencies', frequencies)
        
    def forward(self, x):
        # x: (B, N, input_dim)
        # Scale coordinates by frequencies: (B, N, input_dim, num_freq)
        x_scaled = x.unsqueeze(-1) * self.frequencies * math.pi
        x_scaled = x_scaled.reshape(*x.shape[:-1], -1)  # (B, N, input_dim * num_freq)
        # Concatenate original coords + sin + cos features
        fourier_features = torch.cat([x, torch.sin(x_scaled), torch.cos(x_scaled)], dim=-1)
        return self.projection(fourier_features)

class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super(EncoderLayer, self).__init__()

        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, key_padding_mask=None):
        att = self.attention(x, x, x, key_padding_mask=key_padding_mask)[0]
        att = self.norm1(x + att)
        ff = self.feedforward(att)
        ff = self.norm2(att + ff)
        return ff

class Encoder(nn.Module):
    def __init__(self, hidden_dim: int, layers: int):
        super(Encoder, self).__init__()
        self.tokenizer = Tokenizer(input_dim=3, hidden_dim=hidden_dim)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(hidden_dim) for _ in range(layers)
            ]
        )

    def forward(self, x, key_padding_mask=None):
        x = self.tokenizer(x)
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        return x


class AxialAttention(nn.Module):
    """Memory-efficient axial attention: attend over T and N dimensions separately."""
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super(AxialAttention, self).__init__()
        self.time_attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.space_attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm_t = nn.LayerNorm(hidden_dim)
        self.norm_s = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, T, N):
        # x: (B, T*N, D) -> reshape to (B, T, N, D)
        B, _, D = x.shape
        x = x.view(B, T, N, D)
        
        # Attention over time dimension: (B*N, T, D)
        x_t = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        att_t = self.time_attention(x_t, x_t, x_t)[0]
        att_t = att_t.view(B, N, T, D).permute(0, 2, 1, 3)  # (B, T, N, D)
        x = self.norm_t(x + att_t)
        
        # Attention over space dimension: (B*T, N, D)
        x_s = x.reshape(B * T, N, D)
        att_s = self.space_attention(x_s, x_s, x_s)[0]
        att_s = att_s.view(B, T, N, D)
        x = self.norm_s(x + att_s)
        
        return x.view(B, T * N, D)


class DecoderLayer(nn.Module):
    """Decoder layer with axial self-attention (memory-efficient) and cross-attention."""
    def __init__(self, hidden_dim: int):
        super(DecoderLayer, self).__init__()
        # Axial self-attention for memory efficiency
        self.self_attention = AxialAttention(hidden_dim, num_heads=8)
        # Cross-attention to encoder output
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm_cross = nn.LayerNorm(hidden_dim)
        self.norm_ff = nn.LayerNorm(hidden_dim)

    def forward(self, x, z, key_padding_mask=None, T=None, N=None):
        # Axial self-attention (T and N must be provided)
        if T is not None and N is not None:
            x = self.self_attention(x, T, N)
        
        # Cross-attention to encoder output
        cross_att = self.cross_attention(x, z, z, key_padding_mask=key_padding_mask)[0]
        x = self.norm_cross(cross_att + x)
        
        # Feedforward
        ff = self.feedforward(x)
        x = self.norm_ff(ff + x)
        return x

class Decoder(nn.Module):
    def __init__(self, hidden_dim: int, layers: int):
        super(Decoder, self).__init__()
        self.tokenizer = FourierTokenizer(input_dim=2, hidden_dim=hidden_dim)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(hidden_dim) for _ in range(layers)
            ]
        )
        self.to_out = nn.Linear(hidden_dim, 1)

    def forward(self, x, z, key_padding_mask=None, T=None, N=None):
        x = self.tokenizer(x)
        for layer in self.layers:
            x = layer(x, z, key_padding_mask=key_padding_mask, T=T, N=N)
        x = self.to_out(x)
        return x

class EncoderDecoder(nn.Module):
    def __init__(self, hidden_dim: int, layers_encoder: int, layers_decoder: int):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(hidden_dim=hidden_dim, layers=layers_encoder)
        self.decoder = Decoder(hidden_dim=hidden_dim, layers=layers_decoder)
        
        self.apply(self._init_weights)
        # Initialize output projection with small weights for stable training
        nn.init.normal_(self.decoder.to_out.weight, std=0.02)
        nn.init.zeros_(self.decoder.to_out.bias)
    
    def _init_weights(self, module):
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

    def forward(self, x):
        B, C, T, N = x.shape
        
        # Select conditions where first channel is not -1
        mask = x[:, 0, :, :] != -1  # Shape: (B, T, N)
        x_permuted = x.permute(0, 2, 3, 1)  # (B, T, N, C)

        # Count valid elements per batch and find max length
        counts = mask.view(B, -1).sum(dim=1)  # (B,)
        max_length = int(counts.max().item())

        # Extract and pad conditions for each batch
        conds = torch.full((B, max_length, C), 0.0, device=x.device, dtype=x.dtype)
        # Create key_padding_mask: True for positions to ignore (padded positions)
        key_padding_mask = torch.ones(B, max_length, device=x.device, dtype=torch.bool)
        for b in range(B):
            valid_elements = x_permuted[b][mask[b]]  # (counts[b], C)
            conds[b, :counts[b]] = valid_elements
            key_padding_mask[b, :counts[b]] = False  # Don't mask valid positions

        encoder_output = self.encoder(conds, key_padding_mask=key_padding_mask)
        all_coords = x[:, 1:].permute(0, 2, 3, 1).reshape(B, T * N, 2)
        decoder_output = self.decoder(all_coords, encoder_output, key_padding_mask=key_padding_mask, T=T, N=N)
        return decoder_output.reshape(B, 1, T, N)


if __name__ == "__main__":
    B, C, T, N = 5, 3, 10, 25
    x = torch.randn(B, C, T, N)
    x[:, 0, 1:, 1:-1] = -1
    model = EncoderDecoder(hidden_dim=128, layers_encoder=4, layers_decoder=4)
    output = model(x)
    print(output.shape)
    print(output[0].mean(), output[0].std())

