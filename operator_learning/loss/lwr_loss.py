import torch
import torch.nn as nn
from operator_learning.loss.pde_loss import PDELoss


def Q(rho, vmax=1.0, rhomax=1.0):
    return vmax * rho * (1.0 - rho / rhomax)


def lpde_loss(pred, dt, dx, vmax=1.0, rhomax=1.0):
    """
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


def ic_loss(pred, gt):
    """
    pred : (B, 1, T, X)
    gt   : (B, 1, T, X)
    """
    # Squeeze channel dimension: (B, 1, T, X) -> (B, T, X)
    pred = pred.squeeze(1)
    gt = gt.squeeze(1)
    return torch.mean((pred[:, 0, :] - gt[:, 0, :])**2)


def bc_loss(pred, gt):
    """
    pred : (B, 1, T, X)
    gt   : (B, 1, T, X)
    """
    # Squeeze channel dimension: (B, 1, T, X) -> (B, T, X)
    pred = pred.squeeze(1)
    gt = gt.squeeze(1)
    left_err = (pred[:, :, 0] - gt[:, :, 0])**2
    right_err = (pred[:, :, -1] - gt[:, :, -1])**2
    return torch.mean(left_err + right_err)

class LWRLoss(PDELoss):
    def __init__(self, *args, lpde_loss_weight=1.0, ic_loss_weight=5e-1, bc_loss_weight=5e-1, grid_loss_weight=1.0, **kwargs):
        super(LWRLoss, self).__init__(*args, **kwargs)
        self.lpde_loss_weight = lpde_loss_weight
        self.ic_loss_weight = ic_loss_weight
        self.bc_loss_weight = bc_loss_weight
        self.grid_loss_weight = grid_loss_weight

    def pinn_loss_func(self, pred, gt):
        """
        pred : (B, 1, T, X)
        gt   : (B, 1, T, X)
        """
        lpde_loss_value = self.lpde_loss_weight * lpde_loss(pred, self.dt, self.dx, self.vmax, self.rhomax)
        ic_loss_value = self.ic_loss_weight * ic_loss(pred, gt)
        bc_loss_value = self.bc_loss_weight * bc_loss(pred, gt)
        grid_loss_value = self.grid_loss_weight * self.grid_loss(pred, gt)
        return {
            "lpde_loss": lpde_loss_value,
            "ic_loss": ic_loss_value,
            "bc_loss": bc_loss_value,
            "grid_loss": grid_loss_value
        }