import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, MessagePassing

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
        B, T, N, input_dim = x.shape
        # Scale coordinates by frequencies: (B, T, N, input_dim, num_freq)
        x_scaled = (x.unsqueeze(-1) * self.frequencies * math.pi).reshape(B, T, N, -1)

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
    
    def forward(self, x):
        # x: (B, T, N, D)
        B, T, N, D = x.shape
        
        # Attention over time dimension: group by spatial location
        # (B, T, N, D) -> (B, N, T, D) -> (B*N, T, D)
        x_t = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        att_t = self.time_attention(x_t, x_t, x_t)[0]
        # (B*N, T, D) -> (B, N, T, D) -> (B, T, N, D)
        att_t = att_t.reshape(B, N, T, D).permute(0, 2, 1, 3)
        x = self.norm_t(x + att_t)
        
        # Attention over space dimension: group by timestep
        # (B, T, N, D) -> (B*T, N, D) - this reshape is correct as T and N are adjacent
        x_s = x.reshape(B * T, N, D)
        att_s = self.space_attention(x_s, x_s, x_s)[0]
        att_s = att_s.reshape(B, T, N, D)
        x = self.norm_s(x + att_s)
        
        return x  # (B, T, N, D)


class GatedMPNNLayer(MessagePassing):
    """
    Local gated message-passing layer for shock correction.
    """

    def __init__(self, hidden_dim: int, edge_dim: int, gate_hidden_dim: int = 64):
        super().__init__(aggr="add")

        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # gate uses |u_i - u_j|, |du_dx_i|, |du_dx_j|, |du_dt_i|, |du_dt_j|
        self.gate_mlp = nn.Sequential(
            nn.Linear(7, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 1),
            nn.Sigmoid(),
        )
        # Initialize gate bias to negative so sigmoid starts near 0 (gate closed by default)
        nn.init.constant_(self.gate_mlp[2].bias, -3.0)

        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)
        
        # Store gate values during forward pass
        self._gate_values = None

    def forward(self, x, edge_index, edge_attr, u, du_dx, du_dt):
        # Reset gate values
        self._gate_values = None
        
        out = self.propagate(
            edge_index=edge_index,
            x=x,
            edge_attr=edge_attr,
            u=u,
            du_dx=du_dx,
            du_dt=du_dt,
        )

        x = x + self.update_mlp(torch.cat([x, out], dim=-1))
        return self.norm(x), self._gate_values

    def message(self, x_i, x_j, edge_attr, u_i, u_j, du_dx_i, du_dx_j, du_dt_i, du_dt_j):
        gate_input = torch.cat(
            [
                u_i,
                u_j,
                torch.abs(u_i - u_j),
                du_dx_i,
                du_dx_j,
                du_dt_i,
                du_dt_j,
            ],
            dim=-1,
        )

        gate = self.gate_mlp(gate_input)
        
        # Store gate values for loss computation
        self._gate_values = gate

        msg = self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
        return gate * msg

class ShockCorrector(nn.Module):
    """
    Shock corrector GNN with 2D grid connectivity.

    Forward:
      x : (B, T, N, 2)  grid coordinates (t, x)
      u : (B, T, N, 1)  cross-attention output

    Output:
      delta_u : (B, T, N, 1)
    
    Edges connect each cell to its 4 neighbors (temporal and spatial):
      - (t-1, n), (t+1, n) for temporal neighbors
      - (t, n-1), (t, n+1) for spatial neighbors
    Edge features are (dx, dt) displacements.
    """

    def __init__(
        self,
        nx: int,
        nt: int,
        dx: float,
        dt: float,
        hidden_dim,
        num_layers,
        device,
    ):
        super().__init__()

        self.dx = dx
        self.dt = dt
        self.nx = nx
        self.nt = nt

        # node feature projection (from u only)
        self.in_proj = nn.Linear(1, hidden_dim)

        # build 2D grid edges (spatial + temporal)
        edge_index, edge_attr = self._build_2d_grid_edges(nt, nx, dt, dx, device=device)

        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_attr", edge_attr)

        self.layers = nn.ModuleList(
            [
                GatedMPNNLayer(
                    hidden_dim=hidden_dim,
                    edge_dim=2,
                )
                for _ in range(num_layers)
            ]
        )

        self.out = nn.Linear(hidden_dim, 1)

    def _build_2d_grid_edges(self, nt: int, nx: int, dt: float, dx: float, device=None):
        """
        Build edges for a 2D grid of shape (T, N).
        
        Node indexing: node at (t, n) has index t * nx + n
        
        Each cell connects to its 8 neighbors (4 cardinal + 4 diagonal):
          - Temporal: (t-1, n) and (t+1, n)
          - Spatial: (t, n-1) and (t, n+1)
          - Diagonal: (t-1, n-1), (t-1, n+1), (t+1, n-1), (t+1, n+1)
        
        Edge features: (dx, dt) displacement from source to target
        """
        src, dst = [], []
        edge_attrs = []
        
        for t in range(nt):
            for n in range(nx):
                node_idx = t * nx + n
                
                # Temporal neighbor: previous timestep (t-1, n)
                if t > 0:
                    neighbor_idx = (t - 1) * nx + n
                    src.append(node_idx)
                    dst.append(neighbor_idx)
                    edge_attrs.append([0.0, -dt])  # dx=0, dt=-dt
                
                # Temporal neighbor: next timestep (t+1, n)
                if t < nt - 1:
                    neighbor_idx = (t + 1) * nx + n
                    src.append(node_idx)
                    dst.append(neighbor_idx)
                    edge_attrs.append([0.0, dt])  # dx=0, dt=+dt
                
                # Spatial neighbor: left (t, n-1)
                if n > 0:
                    neighbor_idx = t * nx + (n - 1)
                    src.append(node_idx)
                    dst.append(neighbor_idx)
                    edge_attrs.append([-dx, 0.0])  # dx=-dx, dt=0
                
                # Spatial neighbor: right (t, n+1)
                if n < nx - 1:
                    neighbor_idx = t * nx + (n + 1)
                    src.append(node_idx)
                    dst.append(neighbor_idx)
                    edge_attrs.append([dx, 0.0])  # dx=+dx, dt=0
                
                # Diagonal neighbor: upper-left (t-1, n-1)
                if t > 0 and n > 0:
                    neighbor_idx = (t - 1) * nx + (n - 1)
                    src.append(node_idx)
                    dst.append(neighbor_idx)
                    edge_attrs.append([-dx, -dt])  # dx=-dx, dt=-dt
                
                # Diagonal neighbor: upper-right (t-1, n+1)
                if t > 0 and n < nx - 1:
                    neighbor_idx = (t - 1) * nx + (n + 1)
                    src.append(node_idx)
                    dst.append(neighbor_idx)
                    edge_attrs.append([dx, -dt])  # dx=+dx, dt=-dt
                
                # Diagonal neighbor: lower-left (t+1, n-1)
                if t < nt - 1 and n > 0:
                    neighbor_idx = (t + 1) * nx + (n - 1)
                    src.append(node_idx)
                    dst.append(neighbor_idx)
                    edge_attrs.append([-dx, dt])  # dx=-dx, dt=+dt
                
                # Diagonal neighbor: lower-right (t+1, n+1)
                if t < nt - 1 and n < nx - 1:
                    neighbor_idx = (t + 1) * nx + (n + 1)
                    src.append(node_idx)
                    dst.append(neighbor_idx)
                    edge_attrs.append([dx, dt])  # dx=+dx, dt=+dt
        
        edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32, device=device)
        
        return edge_index, edge_attr

    def _du_dx(self, u):
        # Compute spatial gradient using finite differences
        # Use torch.cat to maintain differentiability throughout
        du_center = (u[..., 2:, 0] - u[..., :-2, 0]) / (2 * self.dx)  # Central diff
        du_left = (u[..., 1:2, 0] - u[..., 0:1, 0]) / self.dx  # Forward diff at left
        du_right = (u[..., -1:, 0] - u[..., -2:-1, 0]) / self.dx  # Backward diff at right
        du = torch.cat([du_left, du_center, du_right], dim=-1).unsqueeze(-1)
        # Use log-scaling for better gradient behavior (avoids division by mean)
        return du

    def _du_dt(self, u):
        # Compute temporal gradient using finite differences
        # Use torch.cat to maintain differentiability throughout
        du_center = (u[:, 2:, :, 0] - u[:, :-2, :, 0]) / (2 * self.dt)  # Central diff
        du_first = (u[:, 1:2, :, 0] - u[:, 0:1, :, 0]) / self.dt  # Forward diff at t=0
        du_last = (u[:, -1:, :, 0] - u[:, -2:-1, :, 0]) / self.dt  # Backward diff at t=end
        du = torch.cat([du_first, du_center, du_last], dim=1).unsqueeze(-1)
        # Use log-scaling for better gradient behavior (avoids division by mean)
        return du
        
    def forward(self, x: torch.Tensor, u: torch.Tensor):
        """
        x : (B, T, N, 2)   coordinates (unused except for shape consistency)
        u : (B, T, N, 1)

        returns:
          delta_u : (B, T, N, 1)
          gate_values : list of gate tensors from each layer, each of shape (num_edges * B,)
        """

        B, T, N, _ = u.shape
        num_nodes_per_sample = T * N  # Full 2D grid per sample

        du_dx = self._du_dx(u).reshape(B * num_nodes_per_sample, 1)
        du_dt = self._du_dt(u).reshape(B * num_nodes_per_sample, 1)
        # flatten: (B, T, N, 1) -> (B * T * N, 1)
        u_flat = u.reshape(B * num_nodes_per_sample, 1)
        
        # project u to hidden_dim for GNN layers
        h = self.in_proj(u_flat)

        # batch edges: replicate edge structure for each sample in batch
        E = self.edge_index.size(1)
        offsets = torch.arange(B, device=h.device).repeat_interleave(E) * num_nodes_per_sample

        edge_index = self.edge_index.repeat(1, B) + offsets
        edge_attr = self.edge_attr.repeat(B, 1)

        # message passing - accumulate gate values from all layers
        all_gate_values = []
        for layer in self.layers:
            h, gate_values = layer(
                x=h,
                edge_index=edge_index,
                edge_attr=edge_attr,
                u=u_flat,
                du_dx=du_dx,
                du_dt=du_dt,
            )
            if gate_values is not None:
                all_gate_values.append(gate_values)

        # output
        delta_u = self.out(h)
        return delta_u.view(B, T, N, 1), all_gate_values

    
class DecoderAttentionLayer(nn.Module):
    """Decoder layer with axial self-attention (memory-efficient) and cross-attention."""
    def __init__(self, hidden_dim: int):
        super(DecoderAttentionLayer, self).__init__()
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


class Decoder(nn.Module):
    def __init__(self, hidden_dim: int, attention_layers: int):
        super(Decoder, self).__init__()
        self.tokenizer = FourierTokenizer(input_dim=2, hidden_dim=hidden_dim)
        self.attention_layers = nn.ModuleList(
            [
                DecoderAttentionLayer(hidden_dim) for _ in range(attention_layers)
            ]
        )
        self.to_out = nn.Linear(hidden_dim, 1)

    def forward(self, x, z, key_padding_mask=None):
        x = self.tokenizer(x)
        for att_layer in self.attention_layers:
            x = att_layer(x, z, key_padding_mask=key_padding_mask)
        x = self.to_out(x)
        return x

class EncoderDecoder(nn.Module):
    def __init__(self, hidden_dim: int, layers_encoder: int, layers_decoder_attention: int, layers_decoder_gcn: int, nt, nx, dx, dt, device):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(hidden_dim=hidden_dim, layers=layers_encoder)
        self.decoder = Decoder(hidden_dim=hidden_dim, attention_layers=layers_decoder_attention)
        self.shock_corrector = ShockCorrector(nx=nx, nt=nt, dx=dx, dt=dt, hidden_dim=hidden_dim, num_layers=layers_decoder_gcn, device=device)
        self.apply(self._init_weights)
        # Initialize output projection with small weights for stable training
        nn.init.normal_(self.decoder.to_out.weight, std=0.02)
        nn.init.zeros_(self.decoder.to_out.bias)

        self.frozen_shock_corrector = False
    
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

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def freeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = False
    
    def freeze_shock_corrector(self):
        for param in self.shock_corrector.parameters():
            param.requires_grad = False
        self.frozen_shock_corrector = True
    
    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def unfreeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = True
    
    def unfreeze_shock_corrector(self):
        for param in self.shock_corrector.parameters():
            param.requires_grad = True
        self.frozen_shock_corrector = False
    
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
        all_coords = x[:, 1:].permute(0, 2, 3, 1)
        u = self.decoder(all_coords, encoder_output, key_padding_mask=key_padding_mask)
        if not self.frozen_shock_corrector:
            delta_u, gate_values = self.shock_corrector(all_coords, u.detach())
        else:
            delta_u = torch.zeros_like(u)
            gate_values = []

        return (u + delta_u).permute(0, 3, 1, 2), delta_u.permute(0, 3, 1, 2), gate_values


if __name__ == "__main__":
    B, C, T, N = 5, 3, 10, 25
    x = torch.randn(B, C, T, N)
    x[:, 0, 1:, 1:-1] = -1
    model = EncoderDecoder(hidden_dim=128, layers_encoder=4, layers_decoder_attention=4, layers_decoder_gcn=1, nt=T, nx=N, dx=1.0, dt=1.0, device=torch.device("cpu"))
    output, delta_u, gate_values = model(x)
    print(output.shape)
    print(output[0].mean(), output[0].std())
    print(f"Number of gate value tensors: {len(gate_values)}")
    if gate_values:
        print(f"Gate values shape: {gate_values[0].shape}, mean: {gate_values[0].mean().item():.4f}")

