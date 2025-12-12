import torch
import torch.nn as nn


class Tokenizer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(Tokenizer, self).__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x):
        return self.projection(x)

class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super(EncoderLayer, self).__init__()

        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        att = self.attention(x, x, x)[0]
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

    def forward(self, x):
        x = self.tokenizer(x)
        for layer in self.layers:
            x = layer(x)

        return x


class CrossAttentionDecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super(CrossAttentionDecoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, z):
        att = self.attention(x, z, z)[0]
        att = self.norm1(att + x)
        att = self.feedforward(att)
        att = self.norm2(att)
        return att

class Decoder(nn.Module):
    def __init__(self, hidden_dim: int, layers: int):
        super(Decoder, self).__init__()
        self.tokenizer = Tokenizer(input_dim=2, hidden_dim=hidden_dim)
        self.layers = nn.ModuleList(
            [
                CrossAttentionDecoderLayer(hidden_dim) for _ in range(layers)
            ]
        )
        self.to_out = nn.Linear(hidden_dim, 1)

    def forward(self, x, z):
        x = self.tokenizer(x)
        for layer in self.layers:
            x = layer(x, z)
        x = self.to_out(x)
        return x

class EncoderDecoder(nn.Module):
    def __init__(self, hidden_dim: int, layers_encoder: int, layers_decoder: int):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(hidden_dim=hidden_dim, layers=layers_encoder)
        self.decoder = Decoder(hidden_dim=hidden_dim, layers=layers_decoder)

    def forward(self, x):
        B, C, T, N = x.shape
        
        # Select conditions where first channel is not -1
        mask = x[:, 0, :, :] != -1  # Shape: (B, T, N)
        x_permuted = x.permute(0, 2, 3, 1)  # (B, T, N, C)
        conds = x_permuted[mask].reshape(B, -1, C)
        encoder_output = self.encoder(conds)
        all_coords = x[:, 1:].reshape(B, -1, 2)
        decoder_output = self.decoder(all_coords, encoder_output)
        return decoder_output.reshape(B, 1, T, N)


if __name__ == "__main__":
    B, C, T, N = 5, 3, 10, 25
    x = torch.randn(B, C, T, N)
    x[:, 0, 1:, 1:-1] = -1
    model = EncoderDecoder(hidden_dim=128, layers_encoder=4, layers_decoder=4)
    print(model(x).shape)

