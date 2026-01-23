import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossDecoderLayer(nn.Module):
    """
    Single layer of the cross-attention decoder with pre-norm architecture.
    
    Implements equations (8) and (9):
        x'_k = x_{k-1} + MHA(LN(x_{k-1}), LN(z_L), LN(z_L))
        x_k = x'_k + MLP(LN(x'_k))
    """
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
    
    def forward(self, x, z, key_padding_mask=None):
        """
        Args:
            x: (B, Q, D) - query features (interpolated grid features)
            z: (B, S, D) - encoder output (keys/values)
            key_padding_mask: (B, S) - mask for encoder output (True = ignore)
        
        Returns:
            x: (B, Q, D) - updated features
        """
        # Cross-attention with pre-norm (Eq. 8)
        x_norm = self.norm_q(x)
        z_norm = self.norm_kv(z)
        att = self.cross_attention(x_norm, z_norm, z_norm, key_padding_mask=key_padding_mask)[0]
        x = x + att
        
        # Feedforward with pre-norm (Eq. 9)
        ff = self.feedforward(self.norm_ff(x))
        x = x + ff
        
        return x


class CrossDecoder(nn.Module):
    """
    Cross-Attention Decoder with Nadaraya-Watson interpolation over a trainable latent grid.
    
    For each query point y ∈ R², computes a weighted average of learnable grid features
    using exponential distance-based weights (Eq. 7), then passes through K layers of
    cross-attention with encoder output.
    
    Architecture:
        1. Trainable latent grid features x ∈ R^{Nx×Ny×C} on uniform grid in [0,1]²
        2. Nadaraya-Watson interpolation for query points (Eq. 7)
        3. K cross-attention decoder layers (Eqs. 8-9)
        4. Output MLP projection
    
    Args:
        hidden_dim: Dimension of latent features (C)
        num_layers: Number of cross-attention layers (K)
        grid_nx: Number of grid points in first dimension (Nx)
        grid_ny: Number of grid points in second dimension (Ny)
        beta: Locality hyperparameter for Nadaraya-Watson interpolation
              Larger β → more localized weights → higher-frequency interpolant
              Smaller β → smoother interpolant → broader neighborhood averaging
        num_heads: Number of attention heads
    """
    def __init__(
        self, 
        hidden_dim: int, 
        num_layers: int, 
        grid_nx: int = 16, 
        grid_ny: int = 16,
        beta: float = 10.0,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.grid_nx = grid_nx
        self.grid_ny = grid_ny
        self.beta = beta
        
        # Trainable latent grid features: (Nx, Ny, C)
        self.grid_features = nn.Parameter(torch.randn(grid_nx, grid_ny, hidden_dim) * 0.02)
        
        # Create uniform grid points {y_ij} ⊂ [0, 1]² for i=1...Nx, j=1...Ny
        grid_i = torch.linspace(0, 1, grid_nx)
        grid_j = torch.linspace(0, 1, grid_ny)
        # Create meshgrid: grid_ii[i,j] = grid_i[i], grid_jj[i,j] = grid_j[j]
        grid_ii, grid_jj = torch.meshgrid(grid_i, grid_j, indexing='ij')
        # Stack to get grid points: (Nx, Ny, 2) -> (Nx*Ny, 2)
        grid_points = torch.stack([grid_ii, grid_jj], dim=-1).reshape(-1, 2)
        self.register_buffer('grid_points', grid_points)  # (Nx*Ny, 2)
        
        # Cross-attention decoder layers
        self.layers = nn.ModuleList([
            CrossDecoderLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        
        # Final layer norm (post-norm for final output)
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # Output projection MLP
        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following best practices for transformers."""
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
        
        # Initialize output projection with small weights for stable training
        nn.init.normal_(self.to_out[-1].weight, std=0.02)
        nn.init.zeros_(self.to_out[-1].bias)
    
    def _normalize_coords(self, coords):
        """
        Normalize coordinates to [0, 1]² range for Nadaraya-Watson interpolation.
        
        Args:
            coords: (B, Q, 2) - query coordinates
        
        Returns:
            coords_normalized: (B, Q, 2) - normalized to [0, 1]²
        """
        # Compute min/max per batch for each dimension
        coords_min = coords.min(dim=1, keepdim=True)[0]  # (B, 1, 2)
        coords_max = coords.max(dim=1, keepdim=True)[0]  # (B, 1, 2)
        coords_range = coords_max - coords_min
        # Avoid division by zero (constant coordinate)
        coords_range = torch.clamp(coords_range, min=1e-6)
        coords_normalized = (coords - coords_min) / coords_range
        return coords_normalized
    
    def interpolate_features(self, query_points):
        """
        Nadaraya-Watson interpolation of grid features for query points (Eq. 7).
        
        Computes:
            x' = Σ_{i,j} w_{ij} * x_{ij}
            w_{ij} = exp(-β||y - y_{ij}||²) / Σ_{i,j} exp(-β||y - y_{ij}||²)
        
        Args:
            query_points: (B, Q, 2) - query coordinates in [0, 1]²
        
        Returns:
            interpolated: (B, Q, hidden_dim) - interpolated features
        """
        B, Q, _ = query_points.shape
        G = self.grid_nx * self.grid_ny  # Number of grid points
        
        # Compute squared distances: ||y - y_{ij}||²
        # query_points: (B, Q, 2) -> (B, Q, 1, 2)
        # grid_points: (G, 2) -> (1, 1, G, 2)
        diff = query_points.unsqueeze(2) - self.grid_points.unsqueeze(0).unsqueeze(0)  # (B, Q, G, 2)
        sq_dist = (diff ** 2).sum(dim=-1)  # (B, Q, G)
        
        # Compute unnormalized weights: exp(-β * ||y - y_{ij}||²)
        # Use log-sum-exp trick for numerical stability
        log_weights = -self.beta * sq_dist  # (B, Q, G)
        weights = F.softmax(log_weights, dim=-1)  # (B, Q, G) - normalized weights
        
        # Flatten grid features: (Nx, Ny, C) -> (G, C)
        grid_features_flat = self.grid_features.reshape(G, self.hidden_dim)  # (G, C)
        
        # Interpolate: weighted sum over grid features
        # (B, Q, G) @ (G, C) -> (B, Q, C)
        interpolated = torch.matmul(weights, grid_features_flat)
        
        return interpolated
    
    def forward(self, coords, encoder_output, key_padding_mask=None):
        """
        Forward pass of the cross-attention decoder.
        
        Args:
            coords: (B, T, N, 2) - query coordinates (t, x)
            encoder_output: (B, S, hidden_dim) - encoder output z_L for cross-attention
            key_padding_mask: (B, S) - mask for encoder output (True = ignore position)
        
        Returns:
            output: (B, T, N, 1) - decoded output at query points
        """
        B, T, N, _ = coords.shape
        Q = T * N  # Total number of query points
        
        # Flatten spatial-temporal dimensions: (B, T, N, 2) -> (B, T*N, 2)
        coords_flat = coords.reshape(B, Q, 2)
        
        # Normalize coordinates to [0, 1]² for interpolation
        coords_normalized = self._normalize_coords(coords_flat)
        
        # Get interpolated features via Nadaraya-Watson: (B, Q, hidden_dim)
        x = self.interpolate_features(coords_normalized)
        
        # Apply K cross-attention decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, key_padding_mask=key_padding_mask)
        
        # Final layer norm
        x = self.final_norm(x)
        
        # Project to output dimension: (B, Q, hidden_dim) -> (B, Q, 1)
        output = self.to_out(x)
        
        # Reshape to (B, T, N, 1)
        output = output.reshape(B, T, N, 1)
        
        return output


if __name__ == "__main__":
    # Test the CrossDecoder
    B, T, N = 4, 10, 25
    hidden_dim = 128
    num_layers = 4
    
    # Create test inputs
    coords = torch.randn(B, T, N, 2)  # Query coordinates
    encoder_output = torch.randn(B, 50, hidden_dim)  # Encoder output
    key_padding_mask = torch.zeros(B, 50, dtype=torch.bool)  # No masking
    key_padding_mask[:, 30:] = True  # Mask last 20 positions
    
    # Create model
    decoder = CrossDecoder(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        grid_nx=16,
        grid_ny=16,
        beta=10.0,
        num_heads=8
    )
    
    # Forward pass
    output = decoder(coords, encoder_output, key_padding_mask=key_padding_mask)
    
    print(f"Input coords shape: {coords.shape}")
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.4f}, std: {output.std().item():.4f}")
    
    # Count parameters
    num_params = sum(p.numel() for p in decoder.parameters())
    print(f"Number of parameters: {num_params:,}")
