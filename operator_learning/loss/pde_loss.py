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

def get_loss_function(loss_type, nt):
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "l1":
        return nn.L1Loss()
    elif loss_type == "huber":
        return nn.HuberLoss(delta=0.01)
    elif loss_type == "smooth_l1":
        return nn.SmoothL1Loss(beta=0.2)
    elif loss_type == "decaying_mse":
        return lambda pred, targets: decaying_loss(pred, targets, nt)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

class PDELoss(torch.nn.Module):
    def __init__(self, nt, nx, dt, dx, vmax=1.0, rhomax=1.0, loss_type = "mse", pinn_loss = False, subset = 1.0):
        super(PDELoss, self).__init__()
        self.nt = nt
        self.nx = nx
        self.dt = dt
        self.dx = dx
        self.vmax = vmax
        self.rhomax = rhomax
        self.losses = {}
        self.monitoring_losses = {}
        self.loss_count = 0
        self.loss_type = loss_type
        self.loss_function = get_loss_function(loss_type, nt)
        self.pinn_loss = pinn_loss
        self.subset = subset
    
    def pinn_loss_func(self, pred, gt):
        raise NotImplementedError("Subclasses must implement this method")
        
    def compute_loss(self, pred, gt) -> dict[str, torch.Tensor]:
        if self.pinn_loss:
            return self.pinn_loss_func(pred, gt)
        else:
            return {"grid_loss": self.grid_loss(pred, gt)}

    def _subsample(self, pred, gt):
        """Apply spatial subsampling if subset < 1.0. Returns (pred, gt) tensors."""
        if self.subset < 1.0:
            B, _, T, X = pred.shape
            device = pred.device
            k = int(self.subset * X)
            x_idx = torch.randperm(X, device=device)[:k]
            x_idx = torch.unique(torch.cat([
                x_idx,
                torch.tensor([0, X - 1], device=device)
            ]))
            pred = pred[:, :, :, x_idx]
            gt = gt[:, :, :, x_idx]
        return pred, gt

    def grid_loss(self, pred, gt):
        """
        pred : (B, 1, T, X)
        gt   : (B, 1, T, X)
        """
        pred, gt = self._subsample(pred, gt)
        return self.loss_function(pred, gt)

    def compute_monitoring_losses(self, pred, gt) -> dict[str, torch.Tensor]:
        """Compute all monitoring losses (not used for training, only for logging)."""
        pred_sub, gt_sub = self._subsample(pred, gt)
        
        # Grid losses with different loss functions
        grid_loss_mse = nn.functional.mse_loss(pred_sub, gt_sub)
        grid_loss_l1 = nn.functional.l1_loss(pred_sub, gt_sub)
        grid_loss_huber = nn.functional.huber_loss(pred_sub, gt_sub)
        grid_loss_smooth_l1 = nn.functional.smooth_l1_loss(pred_sub, gt_sub, beta=0.2)
        
        # Derivative losses
        pred_dx = (pred[:, :, :, 1:] - pred[:, :, :, :-1]) / self.dx
        gt_dx = (gt[:, :, :, 1:] - gt[:, :, :, :-1]) / self.dx
        du_dx = torch.mean(torch.abs(pred_dx - gt_dx))
        
        pred_dt = (pred[:, :, 1:, :] - pred[:, :, :-1, :]) / self.dt
        gt_dt = (gt[:, :, 1:, :] - gt[:, :, :-1, :]) / self.dt
        du_dt = torch.mean(torch.abs(pred_dt - gt_dt))
        
        return {
            "mse": grid_loss_mse,
            "l1": grid_loss_l1,
            "huber": grid_loss_huber,
            "smooth_l1": grid_loss_smooth_l1,
            "du_dx": du_dx,
            "du_dt": du_dt,
        }

    def forward(self, pred, gt):
        loss_values = self.compute_loss(pred, gt)
        monitoring = self.compute_monitoring_losses(pred, gt)
        
        for loss_name, loss_value in loss_values.items():
            self.losses[loss_name] = self.losses.get(loss_name, 0.0) + loss_value.item()
        for loss_name, loss_value in monitoring.items():
            self.monitoring_losses[loss_name] = self.monitoring_losses.get(loss_name, 0.0) + loss_value.item()
        
        self.loss_count += 1
        return sum(loss_values.values())

    def get_loss_value(self):
        """Return only the training loss."""
        return sum(self.losses.values()) / self.loss_count

    def get_loss_values(self):
        all_losses = {**self.losses, **self.monitoring_losses}
        return {loss_name: loss_value / self.loss_count for loss_name, loss_value in all_losses.items()}

    def show_loss_values(self):
        if not self.pinn_loss:
            return
        loss_values = self.get_loss_values()
        loss_str = " | ".join([f"{name} : {value:.6f}" for name, value in loss_values.items()])
        print(f"Loss Values: {loss_str}")

    def log_loss_values(self, experiment, stage):
        experiment.log_metric(f"{stage}/loss", self.get_loss_value())
        # Log training losses if pinn_loss is enabled
        if self.pinn_loss:
            for loss_name, loss_value in self.losses.items():
                experiment.log_metric(f"{stage}/loss/{loss_name}", loss_value / self.loss_count)
        # Always log monitoring losses
        for loss_name, loss_value in self.monitoring_losses.items():
            experiment.log_metric(f"{stage}/loss/{loss_name}", loss_value / self.loss_count)
