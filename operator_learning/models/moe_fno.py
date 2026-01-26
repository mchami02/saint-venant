"""
Mixture of Experts FNO (MoE-FNO)

A Mixture of Experts architecture using FNO as expert networks.
Implements soft top-k routing where each input is processed by 
a weighted combination of experts for smoother expert transitions.
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


class Router(nn.Module):
    """
    Router network with soft top-k expert selection.
    
    Uses a Transformer encoder to extract global features, followed by
    a linear layer to produce expert logits. Soft routing uses a
    temperature-scaled softmax over the top-k experts for smooth blending.
    """
    
    def __init__(
        self,
        in_channels: int,
        n_experts: int,
        hidden_dim: int = 64,
        num_encoder_layers: int = 2,
        num_heads: int = 4,
        top_k: int = 2,
    ):
        """
        Initialize the router.
        
        Args:
            in_channels: Number of input channels
            n_experts: Number of experts to route between
            hidden_dim: Hidden dimension for the encoder
            num_encoder_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            top_k: Number of top experts to use per sample (for soft blending)
        """
        super().__init__()
        
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        
        # Transformer encoder to extract global features from input
        self.encoder = Encoder(
            input_dim=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
        )
        
        # Linear layer to produce expert logits
        self.fc = nn.Linear(hidden_dim, n_experts)
    
    def forward(
        self, 
        x: torch.Tensor, 
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute routing weights for each input based on initial condition.
        
        Uses soft top-k routing: selects top-k experts and applies softmax
        over their logits to get smooth blending weights.
        
        Args:
            x: Input tensor of shape (B, C, H, W) where H is time and W is space
            temperature: Temperature for softmax (lower = sharper routing)
            
        Returns:
            routing_weights: Soft routing weights of shape (B, n_experts)
            top_k_indices: Indices of top-k experts per sample of shape (B, top_k)
            router_logits: Raw logits of shape (B, n_experts)
        """
        B, C, H, W = x.shape
        
        # Extract initial condition (first row, t=0)
        # x[:, :, 0, :] has shape (B, C, W)
        ic = x[:, :, 0, :]  # (B, C, W)
        
        # Reshape from (B, C, W) to (B, W, C) for transformer encoder
        x_seq = ic.permute(0, 2, 1)  # (B, W, C)
        
        # Extract features using transformer encoder
        features = self.encoder(x_seq)  # (B, W, hidden_dim)
        
        # Global max pooling over spatial dimension
        features = features.max(dim=1).values  # (B, hidden_dim)
        
        # Compute expert logits
        router_logits = self.fc(features)  # (B, n_experts)
        
        # Get top-k experts
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (B, top_k)
        
        # Apply temperature-scaled softmax to top-k logits for smooth blending
        top_k_weights = F.softmax(top_k_logits / temperature, dim=-1)  # (B, top_k)
        
        # Scatter top-k weights into full routing weight tensor
        routing_weights = torch.zeros_like(router_logits)  # (B, n_experts)
        routing_weights.scatter_(dim=-1, index=top_k_indices, src=top_k_weights)
        
        return routing_weights, top_k_indices, router_logits


class MoEFNO(nn.Module):
    """
    Mixture of Experts FNO.
    
    Uses multiple FNO experts with a learned router for soft top-k routing.
    Each input sample is processed by a weighted combination of the top-k
    experts, enabling smoother expert transitions and better gradient flow.
    
    Features:
    - Soft top-k routing with temperature-controlled blending
    - Weighted combination of expert outputs for smooth transitions
    - No load balancing loss - experts specialize naturally
    """
    
    def __init__(
        self,
        n_modes: tuple[int, ...],
        hidden_channels: int,
        n_experts: int,
        in_channels: int = 1,
        out_channels: int = 1,
        n_layers: int = 4,
        router_hidden_dim: int = 64,
        router_num_layers: int = 2,
        router_num_heads: int = 4,
        top_k: int = 2,
    ):
        """
        Initialize the MoE-FNO model.
        
        Args:
            n_modes: Number of Fourier modes to keep per dimension, e.g., (16, 16)
            hidden_channels: Hidden channel width for each FNO expert
            n_experts: Number of FNO experts
            in_channels: Number of input channels
            out_channels: Number of output channels
            n_layers: Number of FNO layers per expert
            router_hidden_dim: Hidden dimension for the router's transformer encoder
            router_num_layers: Number of transformer encoder layers in the router
            router_num_heads: Number of attention heads in the router
            top_k: Number of top experts to blend per sample (default: 2)
        """
        super().__init__()
        
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        
        # Router network with transformer encoder
        self.router = Router(
            in_channels=in_channels,
            n_experts=n_experts,
            hidden_dim=router_hidden_dim,
            num_encoder_layers=router_num_layers,
            num_heads=router_num_heads,
            top_k=self.top_k,
        )
        
        # Create FNO experts
        self.experts = nn.ModuleList([
            FNO(
                n_modes=n_modes,
                hidden_channels=hidden_channels,
                in_channels=in_channels,
                out_channels=out_channels,
                n_layers=n_layers,
            )
            for _ in range(n_experts)
        ])
    
    def forward(
        self, 
        x: torch.Tensor,
        temperature: float = 1.0,
        return_routing_info: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Forward pass through MoE-FNO with soft top-k routing.
        
        Each sample is processed by the top-k experts, and outputs are
        combined using the routing weights for smooth blending.
        
        Args:
            x: Input tensor of shape (B, in_channels, H, W)
            temperature: Temperature for softmax routing (lower = sharper)
            return_routing_info: If True, return additional routing information
            
        Returns:
            output: Output tensor of shape (B, out_channels, H, W)
            routing_info: (optional) Dictionary containing:
                - top_k_indices: Which top-k experts processed each sample
                - routing_weights: Soft routing weights for all experts
                - router_logits: Raw router logits
        """
        batch_size = x.size(0)
        
        # Get routing decisions
        routing_weights, top_k_indices, router_logits = self.router(
            x, temperature=temperature
        )
        
        # Initialize output tensor
        output = torch.zeros_like(self.experts[0](x[:1])).repeat(batch_size, 1, 1, 1)
        output = output.to(x.device)
        
        # Process through each expert and accumulate weighted outputs
        # Only process experts that have non-zero weights for at least one sample
        for expert_idx in range(self.n_experts):
            # Get the routing weight for this expert for all samples
            expert_weight = routing_weights[:, expert_idx]  # (B,)
            
            # Find samples that use this expert (non-zero weight)
            active_mask = expert_weight > 0
            if not active_mask.any():
                continue
            
            # Get indices and weights of active samples
            active_indices = active_mask.nonzero(as_tuple=True)[0]
            active_weights = expert_weight[active_indices]  # (n_active,)
            
            # Batch process through expert
            expert_input = x[active_indices]
            expert_output = self.experts[expert_idx](expert_input)  # (n_active, C, H, W)
            
            # Weight the expert output and accumulate
            # Reshape weights for broadcasting: (n_active,) -> (n_active, 1, 1, 1)
            weighted_output = expert_output * active_weights.view(-1, 1, 1, 1)
            
            # Accumulate into output at the correct positions
            output[active_indices] = output[active_indices] + weighted_output
        
        if return_routing_info:
            routing_info = {
                'top_k_indices': top_k_indices,
                'routing_weights': routing_weights,
                'router_logits': router_logits,
            }
            return output, routing_info
        
        return output
    
    def get_expert_usage_stats(self, routing_weights: torch.Tensor) -> dict:
        """
        Compute expert usage statistics from soft routing weights.
        
        Args:
            routing_weights: Soft routing weights of shape (B, n_experts)
            
        Returns:
            Dictionary with usage statistics per expert
        """
        # Sum of routing weights per expert (soft usage measure)
        weight_sums = routing_weights.sum(dim=0)  # (n_experts,)
        total_weight = weight_sums.sum().item()
        
        # Count how many samples have each expert in their top-k
        active_counts = (routing_weights > 0).sum(dim=0)  # (n_experts,)
        
        # Compute entropy of the weight distribution
        weight_probs = weight_sums / (total_weight + 1e-8)
        entropy = -(weight_probs * torch.log(weight_probs + 1e-8)).sum().item()
        
        return {
            'weight_sums': weight_sums.tolist(),
            'weight_fractions': (weight_sums / (total_weight + 1e-8)).tolist(),
            'active_counts': active_counts.tolist(),
            'entropy': entropy,
        }
