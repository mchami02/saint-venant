import torch
import torch.nn as nn

def decaying_loss(pred, targets, nt):
    '''
    Make the loss decay over the timesteps predicted
    The loss is L=∑_{k=1}^K w_k ∥u^t+k - u^{t+k}∥_2^2 where w_k = gamma^(k-1)'''
    timesteps = torch.arange(1, nt, device=pred.device, dtype=pred.dtype)
    weights = torch.linspace(1, 0, nt-1, device=pred.device, dtype=pred.dtype)  # shape: (nt-1,)
    mse = (pred[:, :, 1:] - targets[:, :, 1:]).pow(2).mean(dim=(0, 1)).mean(dim=-1)  # shape: (nt-1,)
    loss = (weights * mse).sum()
    return loss

class PDELoss(torch.nn.Module):
    def __init__(self, nt, nx, dt, dx, vmax=1.0, rhomax=1.0, decaying_loss = False):
        super(PDELoss, self).__init__()
        self.nt = nt
        self.nx = nx
        self.dt = dt
        self.dx = dx
        self.vmax = vmax
        self.rhomax = rhomax
        self.losses = {}
        self.loss_count = 0
        self.decaying_loss = decaying_loss

    def compute_loss(self, pred, gt) -> dict[str, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement this method")

    def grid_loss(self, pred, gt):
        """
        pred : (B, 1, T, X)
        gt   : (B, 1, T, X)
        """
        if self.decaying_loss:
            return decaying_loss(pred, gt, self.nt)
        else:
            return nn.MSELoss()(pred, gt)

    def forward(self, pred, gt):
        loss_values = self.compute_loss(pred, gt)
        for loss_name, loss_value in loss_values.items():
            # Use .item() to avoid keeping the computation graph in memory
            self.losses[loss_name] = self.losses.get(loss_name, 0.0) + loss_value.item()
        self.loss_count += 1
        return sum(loss_values.values())

    def get_loss_value(self):
        loss_value = sum(self.losses.values()) / self.loss_count
        return loss_value

    def get_loss_values(self):
        return {loss_name: loss_value / self.loss_count for loss_name, loss_value in self.losses.items()}

    def show_loss_values(self):
        loss_values = self.get_loss_values()
        loss_str = " | ".join([f"{name} : {value:.6f}" for name, value in loss_values.items()])
        print(f"Loss Values: {loss_str}")
