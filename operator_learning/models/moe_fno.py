"""
Mixture of Experts FNO (MoE-FNO)

A Mixture of Experts architecture using FNO as expert networks.
Implements hard routing where each input is processed by a single expert.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from neuralop.models import FNO

from .encoder import Encoder


class Router(nn.Module):
    """
    Router network for hard expert selection.
    
    Uses a Transformer encoder to extract global features, followed by
    a linear layer to produce expert logits. Hard routing is achieved 
    via argmax during inference and Gumbel-Softmax during training.
    """
    
    def __init__(
        self,
        in_channels: int,
        n_experts: int,
        hidden_dim: int = 64,
        num_encoder_layers: int = 2,
        num_heads: int = 4,
    ):
        """
        Initialize the router.
        
        Args:
            in_channels: Number of input channels
            n_experts: Number of experts to route between
            hidden_dim: Hidden dimension for the encoder
            num_encoder_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.n_experts = n_experts
        
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
        hard: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute routing weights for each input based on initial condition.
        
        Args:
            x: Input tensor of shape (B, C, H, W) where H is time and W is space
            temperature: Temperature for Gumbel-Softmax (lower = harder)
            hard: If True, use hard routing (argmax). If False, use soft routing.
            
        Returns:
            routing_weights: One-hot or soft routing weights of shape (B, n_experts)
            expert_indices: Selected expert indices of shape (B,)
            router_logits: Raw logits for load balancing loss of shape (B, n_experts)
        """
        B, C, H, W = x.shape
        
        # Extract initial condition (first row, t=0)
        # x[:, :, 0, :] has shape (B, C, W)
        ic = x[:, :, 0, :]  # (B, C, W)
        
        # Reshape from (B, C, W) to (B, W, C) for transformer encoder
        x_seq = ic.permute(0, 2, 1)  # (B, W, C)
        
        # Extract features using transformer encoder
        features = self.encoder(x_seq)  # (B, W, hidden_dim)
        
        # Global average pooling over spatial dimension
        features = features.mean(dim=1)  # (B, hidden_dim)
        
        # Compute expert logits
        router_logits = self.fc(features)  # (B, n_experts)
        
        if hard and not self.training:
            # Hard routing during inference: use argmax
            expert_indices = router_logits.argmax(dim=-1)  # (B,)
            routing_weights = F.one_hot(expert_indices, num_classes=self.n_experts).float()
        else:
            # During training: use Gumbel-Softmax for differentiable hard routing
            routing_weights = F.gumbel_softmax(router_logits, tau=temperature, hard=hard)
            expert_indices = routing_weights.argmax(dim=-1)
        
        return routing_weights, expert_indices, router_logits


class MoEFNO(nn.Module):
    """
    Mixture of Experts FNO.
    
    Uses multiple FNO experts with a learned router for hard routing.
    Each input sample is processed by exactly one expert based on
    the router's decision.
    
    Features:
    - Hard routing via Gumbel-Softmax during training
    - Load balancing auxiliary loss to encourage expert utilization
    - Efficient batched computation when samples share the same expert
    """
    
    def __init__(
        self,
        n_modes: Tuple[int, ...],
        hidden_channels: int,
        n_experts: int,
        in_channels: int = 1,
        out_channels: int = 1,
        n_layers: int = 4,
        router_hidden_dim: int = 64,
        router_num_layers: int = 2,
        router_num_heads: int = 4,
        load_balance_weight: float = 0.01,
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
            load_balance_weight: Weight for the load balancing auxiliary loss
        """
        super().__init__()
        
        self.n_experts = n_experts
        self.load_balance_weight = load_balance_weight
        
        # Router network with transformer encoder
        self.router = Router(
            in_channels=in_channels,
            n_experts=n_experts,
            hidden_dim=router_hidden_dim,
            num_encoder_layers=router_num_layers,
            num_heads=router_num_heads,
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
    
    def compute_load_balance_loss(
        self, 
        router_logits: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute load balancing auxiliary loss.
        
        Encourages uniform expert utilization by penalizing imbalanced
        routing. Uses the formulation from Switch Transformer paper.
        
        Args:
            router_logits: Router logits of shape (B, n_experts)
            expert_indices: Selected expert indices of shape (B,)
            
        Returns:
            Load balancing loss scalar
        """
        batch_size = router_logits.size(0)
        
        # Fraction of samples routed to each expert
        # f_i = (1/B) * sum_j 1(expert_j == i)
        expert_counts = torch.bincount(
            expert_indices, 
            minlength=self.n_experts
        ).float()
        fraction_routed = expert_counts / batch_size  # (n_experts,)
        
        # Average routing probability for each expert
        # P_i = (1/B) * sum_j softmax(router_logits_j)_i
        router_probs = F.softmax(router_logits, dim=-1)  # (B, n_experts)
        avg_prob = router_probs.mean(dim=0)  # (n_experts,)
        
        # Load balance loss: n_experts * sum(f_i * P_i)
        # This is minimized when routing is uniform
        load_balance_loss = self.n_experts * (fraction_routed * avg_prob).sum()
        
        return load_balance_loss
    
    def forward(
        self, 
        x: torch.Tensor,
        temperature: float = 1.0,
        return_routing_info: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        """
        Forward pass through MoE-FNO.
        
        Args:
            x: Input tensor of shape (B, in_channels, H, W)
            temperature: Temperature for Gumbel-Softmax routing
            return_routing_info: If True, return additional routing information
            
        Returns:
            output: Output tensor of shape (B, out_channels, H, W)
            routing_info: (optional) Dictionary containing:
                - expert_indices: Which expert processed each sample
                - routing_weights: Routing weights
                - load_balance_loss: Auxiliary loss for load balancing
        """
        batch_size = x.size(0)
        
        # Get routing decisions
        routing_weights, expert_indices, router_logits = self.router(
            x, temperature=temperature, hard=True
        )
        
        # Initialize output tensor
        output = torch.zeros_like(self.experts[0](x[:1])).repeat(batch_size, 1, 1, 1)
        output = output.to(x.device)
        
        # Process samples through their assigned experts
        # Group samples by expert for efficient batched computation
        for expert_idx in range(self.n_experts):
            # Find samples assigned to this expert
            mask = expert_indices == expert_idx
            if not mask.any():
                continue
            
            # Get indices of samples for this expert
            sample_indices = mask.nonzero(as_tuple=True)[0]
            
            # Batch process through expert
            expert_input = x[sample_indices]
            expert_output = self.experts[expert_idx](expert_input)
            
            # Place outputs in correct positions
            output[sample_indices] = expert_output
        
        if return_routing_info:
            load_balance_loss = self.compute_load_balance_loss(
                router_logits, expert_indices
            )
            
            routing_info = {
                'expert_indices': expert_indices,
                'routing_weights': routing_weights,
                'router_logits': router_logits,
                'load_balance_loss': load_balance_loss,
            }
            return output, routing_info
        
        return output
    
    def get_expert_usage_stats(self, expert_indices: torch.Tensor) -> dict:
        """
        Compute expert usage statistics.
        
        Args:
            expert_indices: Expert indices from a batch of shape (B,)
            
        Returns:
            Dictionary with usage statistics per expert
        """
        counts = torch.bincount(expert_indices, minlength=self.n_experts)
        total = counts.sum().item()
        
        return {
            'counts': counts.tolist(),
            'fractions': (counts.float() / total).tolist() if total > 0 else [0] * self.n_experts,
            'entropy': -(F.softmax(counts.float(), dim=0) * 
                        F.log_softmax(counts.float() + 1e-8, dim=0)).sum().item(),
        }
