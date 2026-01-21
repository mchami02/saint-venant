import torch
import torch.nn.functional as F
from operator_learning.loss.pde_loss import PDELoss


def Q(rho, vmax=1.0, rhomax=1.0):
    """Flux function for LWR equation: Q(ρ) = v_max * ρ * (1 - ρ/ρ_max)"""
    return vmax * rho * (1.0 - rho / rhomax)


def dQ_drho(rho, vmax=1.0, rhomax=1.0):
    """Derivative of flux function: dQ/dρ = v_max * (1 - 2ρ/ρ_max)"""
    return vmax * (1.0 - 2.0 * rho / rhomax)


def detect_shock_cells(pred, dx, curvature_threshold=5.0, jump_threshold=0.3):
    """
    Detect shock cells using second derivative (curvature) analysis.
    
    Shocks are characterized by:
    1. Very high absolute second derivative (discontinuity creates sharp curvature)
    2. Sign changes in second derivative (curvature flips at shock)
    
    Rarefactions have smooth transitions with low curvature, so they are filtered out.
    
    Args:
        pred: (B, T, X) - predicted density
        dx: spatial step size
        curvature_threshold: threshold for normalized curvature magnitude
        jump_threshold: threshold for detecting jumps in first derivative
    
    Returns:
        shock_mask: (B, T, X-2) - True for cells at shock edges (to be excluded)
    """
    eps = 1e-6
    
    # Second derivative (curvature): d²ρ/dx²
    d2rho_dx2 = (pred[:, :, 2:] - 2*pred[:, :, 1:-1] + pred[:, :, :-2]) / (dx**2)  # (B, T, X-2)
    
    # Normalize by local density scale
    rho_center = pred[:, :, 1:-1]
    rho_scale = torch.abs(rho_center).clamp(min=0.1)  # Avoid division issues
    
    # Normalized curvature (dimensionless) - multiply by dx² to make scale-independent
    normalized_curvature = torch.abs(d2rho_dx2) * dx**2 / rho_scale
    
    # High curvature indicates potential shock
    high_curvature = normalized_curvature > curvature_threshold
    
    # Detect sign changes in second derivative (shock signature)
    # At a shock, curvature rapidly changes sign
    # Check if neighboring cells have opposite signs
    d2_left = F.pad(d2rho_dx2[:, :, :-1], (1, 0), value=0)  # shift right
    d2_right = F.pad(d2rho_dx2[:, :, 1:], (0, 1), value=0)  # shift left
    sign_change = (d2rho_dx2 * d2_left < 0) | (d2rho_dx2 * d2_right < 0)
    
    # First derivative (gradient) - for detecting actual jumps
    drho_dx_left = (pred[:, :, 1:-1] - pred[:, :, :-2]) / dx   # left difference
    drho_dx_right = (pred[:, :, 2:] - pred[:, :, 1:-1]) / dx   # right difference
    
    # Jump in gradient: difference between left and right derivatives
    # At shocks, this is large; at rarefactions, it's small
    gradient_jump = torch.abs(drho_dx_right - drho_dx_left) / (rho_scale + eps)
    
    # Shock detection: high curvature with sign change, OR very large gradient jump
    shock_mask = (high_curvature & sign_change) | (gradient_jump > jump_threshold)
    
    return shock_mask


def finite_difference_pinn_loss(pred, dt, dx, vmax=1.0, rhomax=1.0):
    """
    PINN loss using finite differences for derivative computation.
    
    pred : predicted density grid (B, 1, T, X)
    dt   : time step
    dx   : space step
    """
    # Squeeze channel dimension: (B, 1, T, X) -> (B, T, X)
    pred = pred.squeeze(1)

    # ∂ρ/∂t (forward difference)
    pred_t = (pred[:, 1:, :] - pred[:, :-1, :]) / dt         # (B, T-1, X)

    # Q(ρ)
    q = Q(pred, vmax=vmax, rhomax=rhomax)

    # ∂Q/∂x (central difference)
    q_x = (q[:, :, 2:] - q[:, :, :-2]) / (2.0 * dx)          # (B, T, X-2)

    # Align shapes so pred_t and q_x match
    pred_t = pred_t[:, :, 1:-1]                              # (B, T-1, X-2)
    q_x = q_x[:, :-1, :]                                     # (B, T-1, X-2)

    # PDE residual
    residual = pred_t + q_x

    return torch.mean(residual**2)


def autograd_pinn_loss(pred, gt, input_tensor, dt, dx, vmax=1.0, rhomax=1.0, 
                       curvature_threshold=5.0, jump_threshold=0.3, n_samples=1):
    """
    PINN loss using autograd for derivative computation.
    
    Samples n_samples (batch, t, x) triplets from valid cells across all batches,
    computes local gradients ∂ρ/∂t and ∂ρ/∂x for each, and averages the PDE residual.
    
    NOTE: input_tensor must have requires_grad=True BEFORE the model forward pass.
    
    Args:
        pred: predicted density grid (B, 1, T, X) - must be connected to input_tensor
        gt: ground truth grid (B, 1, T, X) - used for shock detection
        input_tensor: input tensor (B, C, T, X) - last 2 channels are t and x coordinates
        dt: time step
        dx: space step
        vmax: maximum velocity
        rhomax: maximum density
        curvature_threshold: threshold for normalized curvature in shock detection
        jump_threshold: threshold for gradient jump in shock detection
        n_samples: number of (batch, t, x) triplets to sample
    
    Returns:
        Mean squared PDE residual over sampled cells
    """
    B, C, T, X = pred.shape
    device = pred.device
    
    # Get prediction as (B, T, X)
    pred_squeezed = pred.squeeze(1)  # (B, T, X)
    gt_squeezed = gt.squeeze(1)  # (B, T, X)
    
    # Detect shock cells to exclude from loss computation
    shock_mask = detect_shock_cells(gt_squeezed, dx, curvature_threshold, jump_threshold)  # (B, T, X-2)
    
    # Build valid mask (interior cells, not at shocks, not at boundaries)
    full_shock_mask = torch.zeros(B, T, X, dtype=torch.bool, device=device)
    full_shock_mask[:, :, 1:-1] = shock_mask
    full_shock_mask[:, 0, :] = True   # Exclude first timestep
    full_shock_mask[:, -1, :] = True  # Exclude last timestep
    full_shock_mask[:, :, 0] = True   # Exclude left boundary
    full_shock_mask[:, :, -1] = True  # Exclude right boundary
    valid_mask = ~full_shock_mask  # (B, T, X)
    
    # Find all valid (batch, t, x) indices
    valid_indices = valid_mask.nonzero(as_tuple=False)  # (N_valid, 3) where columns are [b, t, x]
    n_valid = valid_indices.shape[0]
    
    if n_valid == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Sample n_samples triplets from valid indices
    n_samples = min(n_samples, n_valid)
    sample_perm = torch.randperm(n_valid, device=device)[:n_samples]
    sampled_indices = valid_indices[sample_perm]  # (n_samples, 3)
    
    # Compute LOCAL gradient for each sampled cell
    residuals_sq = []
    
    for k in range(n_samples):
        bi = sampled_indices[k, 0].item()
        ti = sampled_indices[k, 1].item()
        xi = sampled_indices[k, 2].item()
        
        # Compute gradient of pred at this specific (batch, t, x) cell
        grad_input = torch.autograd.grad(
            outputs=pred_squeezed[bi, ti, xi],
            inputs=input_tensor,
            create_graph=False,
            retain_graph=True
        )[0]  # (B, C, T, X)
        # Extract LOCAL gradients at this cell
        grad_t = grad_input[bi, -2, ti, xi]  # ∂pred[bi,ti,xi]/∂t[bi,ti,xi]
        grad_x = grad_input[bi, -1, ti, xi]  # ∂pred[bi,ti,xi]/∂x[bi,ti,xi]

        # Compute ∂Q/∂x using chain rule
        pred_val = pred_squeezed[bi, ti, xi]
        dQ_drho_val = dQ_drho(pred_val, vmax=vmax, rhomax=rhomax)
        grad_q_x = dQ_drho_val * grad_x
        
        # PDE residual at this cell: ∂ρ/∂t + ∂Q/∂x = 0
        cell_residual = grad_t + grad_q_x
        residuals_sq.append(cell_residual ** 2)
    
    # Average over all sampled cells
    loss = torch.stack(residuals_sq).mean()
    
    return loss


class LWRLoss(PDELoss):
    def __init__(self, *args, pinn_method="autograd", curvature_threshold=5.0, jump_threshold=0.3, **kwargs):
        """
        LWR-specific loss function with PINN components.
        
        The total loss is: grid_loss + pinn_weight * pde_residual_loss
        where pinn_weight is inherited from PDELoss.
        
        Args:
            pinn_method: Method for computing PDE derivatives:
                - "finite_difference": Use finite differences (faster, less accurate)
                - "autograd": Use PyTorch autograd via grid_sample (more accurate, handles shocks)
            curvature_threshold: Threshold for normalized curvature in shock detection
                (higher = fewer cells excluded, default 5.0)
            jump_threshold: Threshold for gradient jump in shock detection
                (higher = fewer cells excluded, default 0.3)
        """
        super(LWRLoss, self).__init__(*args, **kwargs)
        self.pinn_method = pinn_method
        self.curvature_threshold = curvature_threshold
        self.jump_threshold = jump_threshold

    def pinn_loss_func(self, pred, gt, input_tensor) -> torch.Tensor:
        """
        Compute PINN loss for LWR equation (PDE residual only).
        
        Args:
            pred: (B, 1, T, X) - predicted density grid
            gt: (B, 1, T, X) - ground truth density grid (unused, kept for interface)
            input_tensor: (B, C, T, X) - input tensor with coordinates
        
        Returns:
            PDE residual loss value (single tensor)
        """
        if self.pinn_method == "autograd":
            return autograd_pinn_loss(
                pred, gt, input_tensor, self.dt, self.dx, 
                self.vmax, self.rhomax, self.curvature_threshold, self.jump_threshold
            )
        else:  # finite_difference (default)
            return finite_difference_pinn_loss(
                pred, self.dt, self.dx, self.vmax, self.rhomax
            )