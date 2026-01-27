"""
Wave Front Predictor

A model that uses wave front detection to decompose the problem into simpler
sub-problems. Instead of multiple experts, it uses a single FNO but feeds it
different synthetic initial conditions based on detected fronts:
- Constant regions: IC with constant value for each region between fronts
- Riemann problems: IC with the actual discontinuity at each front

The output is assembled using hard routing based on which region each query
point falls into relative to the detected fronts.

Supports multiple fronts for piecewise constant initial conditions.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO


class FrontSpeedPredictor(nn.Module):
    """
    Neural network that predicts front speeds from uL/uR values.

    Takes left and right state values at a discontinuity and outputs:
    - front1_speed: speed of the first (or only) front
    - front2_speed: speed of the second front
        - If front2_speed < 0: shock (single front behavior)
        - If front2_speed > 0: rarefaction (two fronts)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input features computed from uL, uR: uL, uR, diff, abs_diff, avg, sign
        input_dim = 6

        # Feature encoder: project input features to hidden dimension
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Deeper network with residual blocks for better gradient flow
        self.res_block1 = self._make_res_block(hidden_dim)
        self.res_block2 = self._make_res_block(hidden_dim)
        self.res_block3 = self._make_res_block(hidden_dim)

        # Output head: predicts (front1_speed, front2_speed)
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )
    
    def _make_res_block(self, dim: int) -> nn.Module:
        """Create a residual block with LayerNorm and GELU."""
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
    
    def _compute_features(self, uL: torch.Tensor, uR: torch.Tensor) -> torch.Tensor:
        """Compute rich features from left/right values."""
        diff = uL - uR
        abs_diff = diff.abs()
        avg = (uL + uR) / 2
        sign = torch.sign(diff)  # +1 if uL > uR (compression), -1 if uL < uR (expansion)
        
        # Stack features: (..., 6)
        features = torch.stack([uL, uR, diff, abs_diff, avg, sign], dim=-1)
        return features
    
    def forward(self, uL: torch.Tensor, uR: torch.Tensor) -> torch.Tensor:
        """
        Predict front speeds from left/right state values.

        Args:
            uL: Left state values, shape (...) - any shape, will be processed element-wise
            uR: Right state values, shape (...) - same shape as uL

        Returns:
            front_params: Shape (..., 2) with channels:
                          [front1_speed, front2_speed]
                          front2_speed < 0: shock, front2_speed > 0: rarefaction
        """
        # Compute features from uL, uR
        features = self._compute_features(uL, uR)  # (..., 6)

        # Forward through network with residual connections
        h = self.input_proj(features)  # (..., hidden_dim)
        h = h + self.res_block1(h)  # Residual block 1
        h = h + self.res_block2(h)  # Residual block 2
        h = h + self.res_block3(h)  # Residual block 3
        output = self.output_head(h)  # (..., 2): [front1_speed, front2_speed]

        return output


class FrontDetector(nn.Module):
    """
    Detects discontinuities in initial conditions and predicts front parameters.
    
    Uses FrontSpeedPredictor to predict speeds/confidence for each detected discontinuity.
    """
    
    def __init__(self, hidden_dim: int, max_fronts: int = 8, diff_threshold: float = 1e-6):
        super().__init__()
        self.diff_threshold = diff_threshold
        self.max_fronts = max_fronts
        self.hidden_dim = hidden_dim
        
        # Neural network for predicting front speeds
        self.speed_predictor = FrontSpeedPredictor(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Takes as input the initial conditions of shape (B, C, X) where:
        - Channel 0: density values
        - Channel -1 (last): spatial coordinates

        Returns:
            front_params: (B, 6, max_fronts) where the 6 channels are:
                          (front1_speed, front2_speed, uL, uR, front_coords, valid_front)
                          Sorted by front_coords (left to right).
                          Invalid fronts have valid_front = False.
            front_count: (B,) - number of valid fronts per batch item
        """
        B, C, X = x.shape
        K = self.max_fronts
        
        # Get density values (channel 0)
        density = x[:, 0, :]  # (B, X)

        # Get spatial coordinates (last channel)
        x_coords = x[:, -1, :]  # (B, X)

        # Get left and right cell values at each interface
        left_vals = density[:, :-1]  # (B, X-1)
        right_vals = density[:, 1:]  # (B, X-1)

        # Detect where adjacent cells have different values
        has_discontinuity = (
            left_vals - right_vals
        ).abs() > self.diff_threshold  # (B, X-1)

        # Compute front_coords as midpoint of spatial coordinates at each interface
        left_coords = x_coords[:, :-1]  # (B, X-1)
        right_coords = x_coords[:, 1:]  # (B, X-1)
        all_front_coords = (left_coords + right_coords) / 2  # (B, X-1)

        # Count valid fronts per batch
        front_count = has_discontinuity.sum(dim=-1)  # (B,)

        # Create score for topk selection: use front_coords for valid, large negative for invalid
        # This way topk by ascending coords gives us leftmost valid fronts
        selection_score = torch.where(
            has_discontinuity,
            all_front_coords,
            torch.full_like(all_front_coords, float("inf")),
        )  # (B, X-1)

        # Select top K fronts by position (smallest coords = leftmost)
        # topk with largest=False gives smallest values
        _, topk_indices = torch.topk(
            selection_score, k=min(K, X - 1), dim=-1, largest=False
        )  # (B, K)

        # Pad if K > X-1
        if K > X - 1:
            pad_size = K - (X - 1)
            topk_indices = F.pad(topk_indices, (0, pad_size), value=0)

        # Gather values at selected indices
        # left_vals, right_vals: (B, X-1) -> (B, K)
        uL = torch.gather(left_vals, dim=-1, index=topk_indices)  # (B, K)
        uR = torch.gather(right_vals, dim=-1, index=topk_indices)  # (B, K)
        front_coords = torch.gather(
            all_front_coords, dim=-1, index=topk_indices
        )  # (B, K)
        is_valid = torch.gather(has_discontinuity, dim=-1, index=topk_indices)  # (B, K)

        # Use FrontSpeedPredictor to get speeds
        speed_params = self.speed_predictor(uL, uR)  # (B, K, 2)

        # Transpose to (B, 2, K)
        front_params = speed_params.permute(0, 2, 1)

        # Concatenate all channels: (B, 6, K)
        front_params = torch.cat(
            [
                front_params,  # (B, 2, K): speeds
                uL.unsqueeze(1),  # (B, 1, K): uL
                uR.unsqueeze(1),  # (B, 1, K): uR
                front_coords.unsqueeze(1),  # (B, 1, K): front_coords
                is_valid.unsqueeze(1),  # (B, 1, K): valid_front
            ],
            dim=1,
        )

        return front_params, front_count

class WavePredictor(nn.Module):
    def __init__(self, n_modes=(16, 8), hidden_dim=64, in_channels=3, out_channels=1, n_layers=4):
        super().__init__()
                # Single FNO expert
        self.model = FNO(
            n_modes=n_modes,
            hidden_channels=hidden_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=n_layers,
        )

    def forward(self, ul, ur, coord, x):
        '''
        Takes as input the riemann problem initial conditions and the coordinates of the riemann problem and returns the predicted wave.
        
        Creates a Riemann problem IC where:
        - All positions left of coord have value ul
        - All positions right of coord have value ur
        
        Args:
            ul: Left initial condition (B, 1)
            ur: Right initial condition (B, 1)
            coord: Coordinates of the riemann problem discontinuity (B, 1)
            x: Input tensor with coordinates in last channel (B, C, T, X)
        Returns:
            wave: Predicted wave (B, out_C, T, X)
        '''
        B, C, T, X = x.shape
        
        # Extract spatial coordinates from the last channel at t=0
        x_coords_t0 = x[:, -1, 0, :]  # (B, X) - spatial coordinates at t=0
        
        # Create mask: True where x < coord (left of discontinuity)
        coord_exp = coord.view(B, 1)  # (B, 1)
        left_mask = x_coords_t0 < coord_exp  # (B, X)
        
        # Expand ul and ur to match spatial dimension at t=0
        ul_exp = ul.view(B, 1).expand(B, X)  # (B, X)
        ur_exp = ur.view(B, 1).expand(B, X)  # (B, X)
        
        # Create Riemann IC at t=0: ul where left of coord, ur where right
        riemann_ic_t0 = torch.where(left_mask, ul_exp, ur_exp)  # (B, X)
        
        # Start with original x and only modify timestep 0
        model_input = x.clone()
        
        # Replace density channel with Riemann IC for all timesteps
        model_input[:, 0, :, :] = riemann_ic_t0.unsqueeze(1).expand(B, T, X)  # (B, T, X)
        
        # Run model
        wave = self.model(model_input)
        
        return wave

class RegionIndexer(nn.Module):
    def __init__(self, sharpness: float = 50.0):
        super().__init__()
        self.sharpness = sharpness  # Controls sigmoid sharpness for soft routing

    def forward(
        self, front_params: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Takes as input the front parameters and the coordinates of the query points.
        Returns SOFT region weights that are differentiable w.r.t. front speeds.

        Uses soft sigmoid instead of hard boolean for differentiability.

        Indexing logic based on front2_speed:
        - front2_speed == 0: Not a valid discontinuity, skip entirely
        - front2_speed < 0: Shock (single front), crossing adds +2 (even → even)
        - front2_speed > 0: Rarefaction (two fronts), each crossing adds +1
          So between front1 and front2 is an odd index (Riemann region)

        Args:
            front_params: Front parameters (B, 6, K) where channels are:
                          [front1_speed, front2_speed, uL, uR, front_coords, is_valid]
            x: Query coordinates (B, C, T, X) where:
               - Channel -2 contains time coordinates
               - Channel -1 contains spatial coordinates

        Returns:
            region_idx: Hard region index for each query point (B, T, X) - for gathering
            is_riemann_soft: Soft probability of being in Riemann region (B, T, X) - differentiable
        """
        B, C, T, X = x.shape
        K = front_params.size(2)

        # Extract front parameters
        front1_speed = front_params[:, 0, :]  # (B, K)
        front2_speed = front_params[:, 1, :]  # (B, K)
        front_coords = front_params[:, 4, :]  # (B, K) - initial x position
        is_valid = front_params[:, 5, :]  # (B, K) - valid front

        # Soft masks for different front types (differentiable)
        # Use sigmoid for soft thresholding based on front2_speed
        # front2_speed < 0: shock, front2_speed > 0: rarefaction 
        is_rarefaction_soft = torch.sigmoid(
            self.sharpness * front2_speed
        )  # ~1 if front2_speed > 0, ~0 if < 0
        is_shock_soft = torch.sigmoid(
            -self.sharpness * front2_speed
        )  # ~1 if front2_speed < 0, ~0 if > 0
        
        # Get time and spatial coordinates from x
        t_query = x[:, -2, :, :]  # (B, T, X) - time coordinates
        x_query = x[:, -1, :, :]  # (B, T, X) - spatial coordinates

        # Get unique time values (same across X)
        t_unique = t_query[:, :, 0]  # (B, T)

        # Compute front positions at each time: (B, K, T)
        front1_pos = (
            front_coords.unsqueeze(2) + front1_speed.unsqueeze(2) * t_unique.unsqueeze(1)
        )
        front2_pos = (
            front_coords.unsqueeze(2) + front2_speed.unsqueeze(2) * t_unique.unsqueeze(1)
        )

        # Flatten for memory-efficient processing
        x_flat = x_query.reshape(B * T, X, 1)
        front1_flat = front1_pos.permute(0, 2, 1).reshape(B * T, 1, K)
        front2_flat = front2_pos.permute(0, 2, 1).reshape(B * T, 1, K)

        # Expand masks to (B*T, 1, K)
        is_shock_flat = is_shock_soft.unsqueeze(1).expand(B, T, K).reshape(B * T, 1, K)
        is_rarefaction_flat = (
            is_rarefaction_soft.unsqueeze(1).expand(B, T, K).reshape(B * T, 1, K)
        )
        is_valid_flat = is_valid.unsqueeze(1).expand(B, T, K).reshape(B * T, 1, K)

        # SOFT crossing check using sigmoid - this is differentiable!
        # sigmoid(sharpness * (x - front_pos)) gives soft indicator of being right of front
        right_of_front1_soft = torch.sigmoid(
            self.sharpness * (x_flat - front1_flat)
        )  # (B*T, X, K)
        right_of_front2_soft = torch.sigmoid(
            self.sharpness * (x_flat - front2_flat)
        )  # (B*T, X, K)

        # For shocks: crossing front1 adds +2 (only for valid fronts)
        shock_contribution = (right_of_front1_soft * is_shock_flat * is_valid_flat * 2).sum(
            dim=-1
        )  # (B*T, X)

        # For rarefactions: crossing front1 adds +1, crossing front2 adds +1 (only for valid fronts)
        rarefaction_contribution = (
            (right_of_front1_soft * is_rarefaction_flat * is_valid_flat)
            + (right_of_front2_soft * is_rarefaction_flat * is_valid_flat)
        ).sum(dim=-1)  # (B*T, X)

        # Total soft region index (differentiable)
        region_soft = (shock_contribution + rarefaction_contribution).reshape(B, T, X)

        # Hard region index for gathering (non-differentiable, but we don't backprop through this)
        region_idx = region_soft.round().long()

        # Soft probability of being in a Riemann region (odd index)
        # Use sin to detect odd values: sin(pi * x) is 0 for integers, max at x.5
        # For soft detection of "odd", we can use: 0.5 - 0.5 * cos(pi * region_soft)
        is_riemann_soft = 0.5 - 0.5 * torch.cos(torch.pi * region_soft)

        return region_idx, is_riemann_soft

class WaveFront(nn.Module):
    def __init__(
        self,
        n_modes=(16, 8),
        hidden_dim=64,
        in_channels=3,
        out_channels=1,
        n_layers=4,
        max_fronts=8,
    ):
        super().__init__()
        self.max_fronts = max_fronts
        self.wave_predictor = WavePredictor(
            n_modes, hidden_dim, in_channels, out_channels, n_layers
        )
        self.front_detector = FrontDetector(hidden_dim, max_fronts=max_fronts)
        self.region_indexer = RegionIndexer()
        self.out_channels = out_channels
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Takes as input the grid of shape (B, C, T, X) and returns the predicted wave.
        
        Memory efficient: uses max_fronts (K) instead of X-1, so tensors are O(B*T*X*K).
        
        Pipeline:
        1. Detect top-K fronts from IC
        2. Compute region indices for each query point
        3. For constant regions (even index): use IC value
        4. For Riemann regions (odd index): use WavePredictor output
        '''
        B, C, T, X = x.shape
        device = x.device
        K = self.max_fronts
        
        # Step 1: Detect fronts from initial condition
        # front_params: (B, 5, K), front_count: (B,)
        front_params, front_count = self.front_detector(x[:, :, 0, :])

        # Extract front info
        uL = front_params[:, 2, :]  # (B, K)
        uR = front_params[:, 3, :]  # (B, K)
        front_coords = front_params[:, 4, :]  # (B, K)
        
        # Step 2: Compute region indices (soft for gradients, hard for gathering)
        # region_idx: (B, T, X) - hard index for gathering
        # is_riemann_soft: (B, T, X) - soft probability for gradient flow
        region_idx, is_riemann_soft = self.region_indexer(front_params, x)
        
        # Step 3: Prepare constant region values
        # For region 2*k, use value from IC at position between front k-1 and front k
        # We have at most K + 1 constant regions
        ic = x[:, 0, 0, :]  # (B, X) - initial condition values
        
        # Build constant values for each potential region
        # Region 0: leftmost constant (use IC at x=0)
        # After each front: use uR from that front
        constant_values = torch.zeros(B, K + 1, device=device)
        constant_values[:, 0] = ic[:, 0]
        constant_values[:, 1:] = uR  # (B, K)
        
        # Step 4: Run WavePredictor for all K Riemann problems (batched)
        # This is O(B*K) forward passes, not O(B*X)
        uL_flat = uL.reshape(B * K, 1)  # (B*K, 1)
        uR_flat = uR.reshape(B * K, 1)  # (B*K, 1)
        coord_flat = front_coords.reshape(B * K, 1)  # (B*K, 1)
        
        # Expand x for each front: (B, C, T, X) -> (B*K, C, T, X)
        x_expanded = x.unsqueeze(1).expand(B, K, C, T, X)  # (B, K, C, T, X)
        x_flat = x_expanded.reshape(B * K, C, T, X)
        
        # Run WavePredictor on all K Riemann problems
        riemann_outputs = self.wave_predictor(uL_flat, uR_flat, coord_flat, x_flat)  # (B*K, out_C, T, X)
        riemann_outputs = riemann_outputs.reshape(B, K, self.out_channels, T, X)  # (B, K, out_C, T, X)
        
        # Step 5: Assemble output using region indices
        # For constant regions: get value from constant_values based on region_idx // 2
        const_region_idx = (region_idx // 2).clamp(0, K)  # (B, T, X)
        
        # Gather constant values: (B, K+1) -> (B, T, X)
        constant_values_exp = constant_values.unsqueeze(-1).unsqueeze(-1).expand(B, K + 1, T, X)
        constant_output = torch.gather(constant_values_exp, dim=1, index=const_region_idx.unsqueeze(1))  # (B, 1, T, X)
        constant_output = constant_output.expand(B, self.out_channels, T, X)
        
        # For Riemann regions: get value from riemann_outputs based on region_idx // 2
        riemann_region_idx = (region_idx // 2).clamp(0, K - 1)  # (B, T, X)
        
        # Gather Riemann outputs: (B, K, out_C, T, X) -> (B, out_C, T, X)
        riemann_outputs_perm = riemann_outputs.permute(0, 2, 3, 4, 1)  # (B, out_C, T, X, K)
        riemann_idx_exp = riemann_region_idx.unsqueeze(1).expand(B, self.out_channels, T, X).unsqueeze(-1)
        riemann_output = torch.gather(riemann_outputs_perm, dim=-1, index=riemann_idx_exp).squeeze(-1)  # (B, out_C, T, X)
        
        # SOFT blending using is_riemann_soft for gradient flow to FrontSpeedPredictor
        # This allows gradients to flow back through the soft sigmoid comparisons
        is_riemann_soft_exp = is_riemann_soft.unsqueeze(1).expand(B, self.out_channels, T, X)
        output = is_riemann_soft_exp * riemann_output + (1 - is_riemann_soft_exp) * constant_output
        
        return output


if __name__ == "__main__":
    print("=" * 60)
    print("WaveFront Sanity Check")
    print("=" * 60)
    
    # Parameters
    B, C, T, X = 2, 3, 20, 50  # batch, channels (density, t_coord, x_coord), time, space
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create synthetic input with piecewise constant IC
    x = torch.zeros(B, C, T, X, device=device)
    
    # Channel 0: density (piecewise constant IC)
    # Sample 1: single discontinuity at x=0.5 (shock)
    x[0, 0, 0, :X//2] = 0.8
    x[0, 0, 0, X//2:] = 0.2
    
    # Sample 2: two discontinuities (could be rarefaction)
    x[1, 0, 0, :X//3] = 0.9
    x[1, 0, 0, X//3:2*X//3] = 0.5
    x[1, 0, 0, 2*X//3:] = 0.1
    
    # Channel 1: time coordinates (same for all x at each t)
    t_coords = torch.linspace(0, 1, T, device=device)
    x[:, 1, :, :] = t_coords.view(1, T, 1).expand(B, T, X)
    
    # Channel 2: spatial coordinates (same for all t)
    x_coords = torch.linspace(0, 1, X, device=device)
    x[:, 2, :, :] = x_coords.view(1, 1, X).expand(B, T, X)
    
    print(f"\nInput shape: {x.shape}")
    print(f"  - Batch size: {B}")
    print(f"  - Channels: {C} (density, t_coord, x_coord)")
    print(f"  - Time steps: {T}")
    print(f"  - Spatial points: {X}")
    
    # Test FrontDetector
    print("\n--- Testing FrontDetector ---")
    max_fronts = 8
    detector = FrontDetector(hidden_dim=32, max_fronts=max_fronts).to(device)
    ic = x[:, :, 0, :]  # (B, C, X) - full IC with density, t_coord, x_coord
    front_params, front_count = detector(ic)
    print(f"Front params shape: {front_params.shape}")  # Should be (B, 5, max_fronts)
    print(f"  - Expected: ({B}, 5, {max_fronts})")
    print(f"Front count per sample: {front_count.tolist()}")
    
    # Check discontinuity detection
    front2_speed = front_params[:, 1, :]  # front2_speed channel
    n_valid = (front2_speed != 0).sum(dim=-1)
    print(f"Valid fronts per sample (from front2_speed != 0): {n_valid.tolist()}")
    
    # Test RegionIndexer
    print("\n--- Testing RegionIndexer ---")
    indexer = RegionIndexer().to(device)
    region_idx, is_riemann_soft = indexer(front_params, x)
    print(f"Region index shape: {region_idx.shape}")  # Should be (B, T, X)
    print(f"  - Expected: ({B}, {T}, {X})")
    print(f"Is Riemann soft shape: {is_riemann_soft.shape}")  # Should be (B, T, X)
    print(f"Is Riemann soft range: [{is_riemann_soft.min():.3f}, {is_riemann_soft.max():.3f}]")
    print("Unique region indices per sample:")
    for b in range(B):
        unique_regions = torch.unique(region_idx[b])
        print(f"  Sample {b}: {unique_regions.tolist()}")
    
    # Test WavePredictor
    print("\n--- Testing WavePredictor ---")
    predictor = WavePredictor(
        n_modes=(8, 4),
        hidden_dim=16,
        in_channels=C,
        out_channels=1,
        n_layers=2
    ).to(device)
    
    ul = torch.tensor([[0.8], [0.9]], device=device)
    ur = torch.tensor([[0.2], [0.5]], device=device)
    coord = torch.tensor([[0.5], [0.33]], device=device)
    
    wave = predictor(ul, ur, coord, x)
    print(f"Wave output shape: {wave.shape}")  # Should be (B, 1, T, X)
    print(f"  - Expected: ({B}, 1, {T}, {X})")
    
    # Test full WaveFront model
    print("\n--- Testing WaveFront (full model) ---")
    model = WaveFront(
        n_modes=(8, 4),
        hidden_dim=16,
        in_channels=C,
        out_channels=1,
        n_layers=2,
        max_fronts=max_fronts,
    ).to(device)
    
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be (B, 1, T, X)
    print(f"  - Expected: ({B}, 1, {T}, {X})")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # Test backward pass
    print("\n--- Testing Backward Pass ---")
    loss = output.mean()
    loss.backward()
    print("Backward pass successful!")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")
    
    print("\n" + "=" * 60)
    print("All sanity checks passed!")
    print("=" * 60)

