import torch

import torch

def Q(rho, vmax=1.0, rhomax=1.0):
    return vmax * rho * (1.0 - rho / rhomax)


def lpde_loss(pred, gt, dt, dx, vmax=30.0, rhomax=1.0):
    """
    pred : predicted density grid (T, X)
    gt   : ground truth density grid   (T, X)
           (only included because you asked for this signature)
    dt   : time step
    dx   : space step
    """

    # ∂ρ/∂t (forward difference)
    pred_t = (pred[1:, :] - pred[:-1, :]) / dt         # (T-1, X)

    # Q(ρ)
    q = Q(pred, vmax=vmax, rhomax=rhomax)

    # ∂Q/∂x (central difference)
    q_x = (q[:, 2:] - q[:, :-2]) / (2.0 * dx)          # (T, X-2)

    # Align shapes so pred_t and q_x match
    pred_t = pred_t[:, 1:-1]                           # (T-1, X-2)
    q_x    = q_x[:-1, :]                               # (T-1, X-2)

    # PDE residual
    residual = pred_t + q_x

    return torch.mean(residual**2)


def lpde_loss(pred, dt, dx, vmax=1.0, rhomax=1.0):
    """
    pred : predicted density grid (T, X)
    gt   : ground truth density grid   (T, X)
           (only included because you asked for this signature)
    dt   : time step
    dx   : space step
    """

    # ∂ρ/∂t (forward difference)
    pred_t = (pred[1:, :] - pred[:-1, :]) / dt         # (T-1, X)

    # Q(ρ)
    q = Q(pred, vmax=vmax, rhomax=rhomax)

    # ∂Q/∂x (central difference)
    q_x = (q[:, 2:] - q[:, :-2]) / (2.0 * dx)          # (T, X-2)

    # Align shapes so pred_t and q_x match
    pred_t = pred_t[:, 1:-1]                           # (T-1, X-2)
    q_x    = q_x[:-1, :]                               # (T-1, X-2)

    # PDE residual
    residual = pred_t + q_x

    return torch.mean(residual**2)

def ic_loss(pred, gt):
    """
    pred : (T, X)
    gt   : (T, X)
    """
    return torch.mean((pred[0] - gt[0])**2)

def bc_loss(pred, gt):
    """
    pred : (T, X)
    gt   : (T, X)
    """
    left_err  = (pred[:, 0]  - gt[:, 0])**2
    right_err = (pred[:, -1] - gt[:, -1])**2
    return torch.mean(left_err + right_err)

class LWRLoss(torch.nn.Module):
    def __init__(self, dt, dx, vmax=30.0, rhomax=1.0, lpde_loss_weight=1.0, ic_loss_weight=1.0, bc_loss_weight=1.0):
        super(LWRLoss, self).__init__()
        self.dt = dt
        self.dx = dx
        self.vmax = vmax
        self.rhomax = rhomax
        self.lpde_loss_weight = lpde_loss_weight
        self.ic_loss_weight = ic_loss_weight
        self.bc_loss_weight = bc_loss_weight

    def forward(self, pred, gt):
        return (self.lpde_loss_weight * lpde_loss(pred, self.dt, self.dx, self.vmax, self.rhomax) 
        + self.ic_loss_weight * ic_loss(pred, gt) 
        + self.bc_loss_weight * bc_loss(pred, gt))