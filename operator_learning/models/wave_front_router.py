"""
Wave Front Router

A router inspired by wave front tracking methods. Detects characteristic
fronts from initial conditions and uses them to partition the (t, x) domain
into regions. Each query point is routed based on which region it falls into.

The key intuition is that discontinuities in initial conditions propagate
along characteristic curves (fronts), dividing the solution space into
distinct regions that may require different modeling approaches.
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# Suppress neuralop metadata warning when creating multiple FNO instances
warnings.filterwarnings(
    "ignore", 
    message="Attempting to update metadata for a module with metadata already in self.state_dict()",
    module="neuralop.models.base_model"
)

from neuralop.models import FNO

from .encoder import Encoder


class FrontPredictor(nn.Module):
    """
    Predicts wave fronts from encoded initial condition features.
    
    Each front is characterized by:
    - x0: starting position at t=0
    - speed: propagation speed (dx/dt)
    - strength: confidence/importance of this front (soft existence)
    
    Fronts are predicted as a set (order-invariant), allowing the model
    to learn the relevant discontinuities.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        max_fronts: int = 8,
    ):
        """
        Args:
            hidden_dim: Dimension of encoded features
            max_fronts: Maximum number of fronts to predict
        """
        super().__init__()
        
        self.max_fronts = max_fronts
        
        # Cross-attention to find front locations
        # Query: learnable front queries
        # Key/Value: encoded spatial features
        self.front_queries = nn.Parameter(torch.randn(1, max_fronts, hidden_dim))
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Predict front parameters from attended features
        self.front_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),  # (x0, speed, strength)
        )
    
    def forward(
        self, 
        encoded_features: torch.Tensor,
        x_coords: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict fronts from encoded initial condition.
        
        Args:
            encoded_features: Encoded IC features of shape (B, W, hidden_dim)
            x_coords: Spatial coordinates of shape (W,) for position reference
            
        Returns:
            front_x0: Starting positions of shape (B, max_fronts)
            front_speed: Propagation speeds of shape (B, max_fronts)
            front_strength: Confidence weights of shape (B, max_fronts)
        """
        B = encoded_features.size(0)
        
        # Expand front queries for batch
        queries = self.front_queries.expand(B, -1, -1)  # (B, max_fronts, hidden_dim)
        
        # Cross-attention: queries attend to encoded features
        attended, _ = self.cross_attention(
            queries, encoded_features, encoded_features
        )
        attended = self.norm(queries + attended)  # (B, max_fronts, hidden_dim)
        
        # Predict front parameters
        front_params = self.front_head(attended)  # (B, max_fronts, 3)
        
        # Split into components
        front_x0 = front_params[..., 0]  # (B, max_fronts) - will be in [-1, 1] after tanh
        front_speed = front_params[..., 1]  # (B, max_fronts) - unbounded
        front_strength = torch.sigmoid(front_params[..., 2])  # (B, max_fronts) - in [0, 1]
        
        # Normalize x0 to spatial domain (assume normalized to [0, 1])
        front_x0 = torch.sigmoid(front_x0)
        
        return front_x0, front_speed, front_strength


class WaveFrontRouter(nn.Module):
    """
    Router based on wave front tracking intuition.
    
    Given initial conditions, detects wave fronts and their propagation.
    For each query point (t, x), determines which region it falls into
    by counting how many fronts have been crossed, then routes accordingly.
    
    The routing is soft and differentiable, with smooth transitions
    near front boundaries.
    """
    
    def __init__(
        self,
        in_channels: int,
        n_experts: int,
        hidden_dim: int = 64,
        num_encoder_layers: int = 2,
        num_heads: int = 4,
        max_fronts: int = 8,
        boundary_sharpness: float = 10.0,
    ):
        """
        Args:
            in_channels: Number of input channels
            n_experts: Number of experts to route between
            hidden_dim: Hidden dimension for encoder
            num_encoder_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            max_fronts: Maximum number of fronts to detect
            boundary_sharpness: Controls smoothness of region transitions
                (higher = sharper boundaries)
        """
        super().__init__()
        
        self.n_experts = n_experts
        self.max_fronts = max_fronts
        self.boundary_sharpness = boundary_sharpness
        
        # Encode initial condition
        self.encoder = Encoder(
            input_dim=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
        )
        
        # Predict fronts from encoded features
        self.front_predictor = FrontPredictor(
            hidden_dim=hidden_dim,
            max_fronts=max_fronts,
        )
        
        # Map region indices to expert routing weights
        # For n_experts experts, we have up to (max_fronts + 1) regions
        # This network learns how to map regions to experts
        self.region_to_expert = nn.Sequential(
            nn.Linear(max_fronts + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_experts),
        )
    
    def compute_region_membership(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        front_x0: torch.Tensor,
        front_speed: torch.Tensor,
        front_strength: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute soft region membership for query points.
        
        For each query (t, x), compute which side of each front it lies on.
        The region is characterized by the pattern of front crossings.
        
        Args:
            t: Time coordinates of shape (B, T) or (B, T, X)
            x: Spatial coordinates of shape (B, X) or (B, T, X)
            front_x0: Front starting positions (B, max_fronts)
            front_speed: Front speeds (B, max_fronts)
            front_strength: Front strengths (B, max_fronts)
            
        Returns:
            region_features: Soft region membership of shape (B, ..., max_fronts + 1)
        """
        # Handle different input shapes
        if t.dim() == 2 and x.dim() == 2:
            # t: (B, T), x: (B, X) -> need to create grid
            B, T = t.shape
            X = x.shape[1]
            t_grid = t.unsqueeze(-1).expand(B, T, X)  # (B, T, X)
            x_grid = x.unsqueeze(1).expand(B, T, X)   # (B, T, X)
        elif t.dim() == 3 and x.dim() == 3:
            # Already in grid form
            t_grid = t
            x_grid = x
            B, T, X = t_grid.shape
        else:
            raise ValueError(f"Unexpected shapes: t={t.shape}, x={x.shape}")
        
        # Compute front positions at each time
        # front_pos(t) = x0 + speed * t
        # front_x0: (B, max_fronts), front_speed: (B, max_fronts)
        # t_grid: (B, T, X)
        front_x0_exp = front_x0.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, max_fronts)
        front_speed_exp = front_speed.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, max_fronts)
        front_strength_exp = front_strength.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, max_fronts)
        
        t_exp = t_grid.unsqueeze(-1)  # (B, T, X, 1)
        x_exp = x_grid.unsqueeze(-1)  # (B, T, X, 1)
        
        # Position of each front at each time
        front_pos = front_x0_exp + front_speed_exp * t_exp  # (B, T, X, max_fronts)
        
        # Signed distance to each front (positive = right of front)
        signed_dist = x_exp - front_pos  # (B, T, X, max_fronts)
        
        # Soft indicator: which side of front (with sharpness control)
        # 1 = right of front, 0 = left of front
        right_of_front = torch.sigmoid(self.boundary_sharpness * signed_dist)
        
        # Weight by front strength (weak fronts don't divide regions)
        right_of_front = right_of_front * front_strength_exp + (1 - front_strength_exp) * 0.5
        
        # Count fronts crossed (soft sum)
        # This gives a soft "region index"
        fronts_crossed = right_of_front.sum(dim=-1)  # (B, T, X)
        
        # Create one-hot-ish region features
        # Map continuous fronts_crossed to soft region membership
        region_indices = torch.arange(
            self.max_fronts + 1, device=fronts_crossed.device, dtype=fronts_crossed.dtype
        )
        region_indices = region_indices.view(1, 1, 1, -1)  # (1, 1, 1, max_fronts+1)
        fronts_crossed = fronts_crossed.unsqueeze(-1)  # (B, T, X, 1)
        
        # Soft region membership using Gaussian-like weighting
        region_dist = (fronts_crossed - region_indices).abs()
        region_membership = F.softmax(-region_dist * self.boundary_sharpness, dim=-1)
        
        return region_membership  # (B, T, X, max_fronts + 1)
    
    def forward(
        self,
        x: torch.Tensor,
        query_t: torch.Tensor | None = None,
        query_x: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute routing weights for query points based on wave front regions.
        
        Args:
            x: Input tensor of shape (B, C, T, X) containing initial conditions
            query_t: Time coordinates for queries (B, T) or None to use all
            query_x: Spatial coordinates for queries (B, X) or None to use all
            
        Returns:
            routing_weights: Expert weights of shape (B, T, X, n_experts)
            front_info: Dictionary with detected front information
        """
        B, C, T, X = x.shape
        device = x.device
        
        # Extract initial condition (first time slice)
        ic = x[:, :, 0, :]  # (B, C, X)
        
        # Reshape for encoder: (B, C, X) -> (B, X, C)
        ic_seq = ic.permute(0, 2, 1)  # (B, X, C)
        
        # Encode initial condition
        encoded = self.encoder(ic_seq)  # (B, X, hidden_dim)
        
        # Predict fronts
        front_x0, front_speed, front_strength = self.front_predictor(encoded)
        
        # Create coordinate grids if not provided
        if query_t is None:
            # Normalized time coordinates [0, 1]
            query_t = torch.linspace(0, 1, T, device=device).unsqueeze(0).expand(B, -1)
        if query_x is None:
            # Normalized spatial coordinates [0, 1]
            query_x = torch.linspace(0, 1, X, device=device).unsqueeze(0).expand(B, -1)
        
        # Compute region membership for all query points
        region_membership = self.compute_region_membership(
            query_t, query_x, front_x0, front_speed, front_strength
        )  # (B, T, X, max_fronts + 1)
        
        # Map regions to expert routing weights
        routing_logits = self.region_to_expert(region_membership)  # (B, T, X, n_experts)
        routing_weights = F.softmax(routing_logits, dim=-1)
        
        front_info = {
            'front_x0': front_x0,
            'front_speed': front_speed,
            'front_strength': front_strength,
            'region_membership': region_membership,
        }
        
        return routing_weights, front_info
    
    def get_expert_for_point(
        self,
        x: torch.Tensor,
        t_query: float,
        x_query: float,
    ) -> tuple[int, torch.Tensor]:
        """
        Get the expert routing for a single query point.
        
        Convenience method for inference.
        
        Args:
            x: Input tensor of shape (B, C, T, X)
            t_query: Time coordinate (scalar, normalized to [0, 1])
            x_query: Spatial coordinate (scalar, normalized to [0, 1])
            
        Returns:
            expert_idx: Index of the selected expert
            weights: Routing weights for all experts
        """
        B = x.size(0)
        device = x.device
        
        query_t = torch.tensor([[t_query]], device=device).expand(B, 1)
        query_x = torch.tensor([[x_query]], device=device).expand(B, 1)
        
        routing_weights, _ = self.forward(x, query_t, query_x)
        routing_weights = routing_weights.squeeze(1).squeeze(1)  # (B, n_experts)
        
        expert_idx = routing_weights.argmax(dim=-1)
        
        return expert_idx, routing_weights


class WaveFrontFNO(nn.Module):
    """
    Mixture of Experts model using wave front routing.
    
    Combines WaveFrontRouter with a set of expert networks. Each spatial
    location in the output is computed as a weighted combination of expert
    outputs, where weights are determined by wave front region membership.
    """
    
    def __init__(
        self,
        n_experts: int,
        n_modes=(16, 8),
        in_channels: int = 3,
        out_channels: int = 1,
        n_layers=4,
        hidden_dim: int = 64,
        num_encoder_layers: int = 2,
        num_heads: int = 4,
        max_fronts: int = 8,
        boundary_sharpness: float = 10.0,
    ):
        """
        Args:
            n_experts: Number of experts
            n_modes: Number of Fourier modes to keep per dimension, e.g., (16, 16)
            in_channels: Number of input channels
            out_channels: Number of output channels
            n_layers: Number of FNO layers per expert
            hidden_dim: Hidden dimension for router encoder
            num_encoder_layers: Number of transformer encoder layers in router
            num_heads: Number of attention heads in router
            max_fronts: Maximum number of fronts to detect
            boundary_sharpness: Controls smoothness of region transitions
        """
        super().__init__()
        
        self.experts = nn.ModuleList([FNO(
            n_modes=n_modes,
            hidden_channels=hidden_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=n_layers,
        ) for _ in range(n_experts)])
        self.n_experts = n_experts
        
        # Create the wave front router
        self.router = WaveFrontRouter(
            in_channels=in_channels,
            n_experts=self.n_experts,
            hidden_dim=hidden_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            max_fronts=max_fronts,
            boundary_sharpness=boundary_sharpness,
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.xavier_uniform_(module.out_proj.weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
                if module.out_proj.bias is not None:
                    nn.init.zeros_(module.out_proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        return_routing_info: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Forward pass through the wave front MoE model.
        
        Args:
            x: Input tensor of shape (B, in_channels, T, X)
            return_routing_info: If True, return routing information
            
        Returns:
            output: Weighted combination of expert outputs (B, out_channels, T, X)
            routing_info: (optional) Dictionary with routing details
        """
        B, C, T, X = x.shape
        
        # Get routing weights for all spatial locations
        routing_weights, front_info = self.router(x)  # (B, T, X, n_experts)
        
        # Run all experts and collect outputs
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)  # (B, out_channels, T, X)
            expert_outputs.append(expert_out)
        
        # Stack expert outputs: (n_experts, B, out_channels, T, X)
        expert_outputs = torch.stack(expert_outputs, dim=0)
        
        # Rearrange for selection
        # expert_outputs: (n_experts, B, out_channels, T, X) -> (B, T, X, n_experts, out_channels)
        expert_outputs = expert_outputs.permute(1, 3, 4, 0, 2)  # (B, T, X, n_experts, out_channels)
        
        # Hard routing: select one expert per location using argmax
        dominant_expert = routing_weights.argmax(dim=-1)  # (B, T, X)
        
        # Create one-hot mask for hard selection
        hard_weights = F.one_hot(dominant_expert, num_classes=self.n_experts)  # (B, T, X, n_experts)
        hard_weights = hard_weights.float().unsqueeze(-1)  # (B, T, X, n_experts, 1)
        
        # Select output from dominant expert only
        output = (expert_outputs * hard_weights).sum(dim=3)  # (B, T, X, out_channels)
        
        # Rearrange back to (B, out_channels, T, X)
        output = output.permute(0, 3, 1, 2)  # (B, out_channels, T, X)
        
        if return_routing_info:
            routing_info = {
                'routing_weights': routing_weights,
                **front_info,
            }
            return output, routing_info
        
        return output
    
    def forward_efficient(
        self,
        x: torch.Tensor,
        top_k: int = 1,
        return_routing_info: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Efficient forward pass that only runs top-k experts per location.
        
        For large numbers of experts, this avoids running all experts for
        every input. Only the top-k experts by routing weight are evaluated.
        
        Args:
            x: Input tensor of shape (B, in_channels, T, X)
            top_k: Number of top experts to use per location
            return_routing_info: If True, return routing information
            
        Returns:
            output: Weighted combination of expert outputs (B, out_channels, T, X)
            routing_info: (optional) Dictionary with routing details
        """
        B, C, T, X = x.shape
        
        # Get routing weights for all spatial locations
        routing_weights, front_info = self.router(x)  # (B, T, X, n_experts)
        
        # Get top-k experts per location
        top_k = min(top_k, self.n_experts)
        top_k_weights, top_k_indices = torch.topk(routing_weights, top_k, dim=-1)
        
        # Renormalize top-k weights
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Initialize output - need to get output channels from a dummy forward
        with torch.no_grad():
            dummy_out = self.experts[0](x[:1])
            out_channels = dummy_out.shape[1]
        
        output = torch.zeros(B, out_channels, T, X, device=x.device, dtype=x.dtype)
        
        # Process each expert
        for expert_idx in range(self.n_experts):
            # Find locations where this expert is in top-k
            # top_k_indices: (B, T, X, top_k)
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)  # (B, T, X)
            
            if not expert_mask.any():
                continue
            
            # Get the weight for this expert at locations where it's used
            # Find which top-k slot this expert occupies
            expert_in_topk = (top_k_indices == expert_idx)  # (B, T, X, top_k)
            expert_weight = (top_k_weights * expert_in_topk.float()).sum(dim=-1)  # (B, T, X)
            
            # Run expert on full input
            expert_out = self.experts[expert_idx](x)  # (B, out_channels, T, X)
            
            # Weight and accumulate
            # expert_weight: (B, T, X) -> (B, 1, T, X)
            output = output + expert_out * expert_weight.unsqueeze(1)
        
        if return_routing_info:
            routing_info = {
                'routing_weights': routing_weights,
                'top_k_indices': top_k_indices,
                'top_k_weights': top_k_weights,
                **front_info,
            }
            return output, routing_info
        
        return output
