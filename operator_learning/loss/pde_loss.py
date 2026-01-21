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
    def __init__(self, nt, nx, dt, dx, vmax=1.0, rhomax=1.0, loss_type="mse", pinn_weight=0.0, subset=1.0):
        """
        Base PDE loss class.
        
        Args:
            nt: number of time steps
            nx: number of spatial points
            dt: time step size
            dx: spatial step size
            vmax: maximum velocity
            rhomax: maximum density
            loss_type: type of grid loss ("mse", "l1", "huber", "smooth_l1", "decaying_mse")
            pinn_weight: weight for PINN loss component (0.0 = no PINN loss, >0 = weighted PINN loss)
            subset: fraction of spatial points to use for loss computation
        """
        super(PDELoss, self).__init__()
        self.nt = nt
        self.nx = nx
        self.dt = dt
        self.dx = dx
        self.vmax = vmax
        self.rhomax = rhomax
        self.loss_sum = 0.0  # Accumulated total loss (single number)
        self.monitoring_losses = {}  # Dictionary for monitoring (includes PINN components)
        self.loss_count = 0
        self.loss_type = loss_type
        self.loss_function = get_loss_function(loss_type, nt)
        self.pinn_weight = pinn_weight
        self.subset = subset
    
    def pinn_loss_func(self, pred, gt, input_tensor) -> torch.Tensor:
        """Compute PINN loss. Subclasses must implement this method.
        
        Returns:
            Single tensor value for the PINN loss
        """
        raise NotImplementedError("Subclasses must implement this method")
        
    def compute_loss(self, pred, gt, input_tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute the total loss as: grid_loss + pinn_weight * pinn_loss.
        
        Args:
            pred: predicted grid (B, C, T, X)
            gt: ground truth grid (B, C, T, X)
            input_tensor: input tensor (B, C, T, X) for PINN loss computation
        
        Returns:
            Tuple of (total_loss, monitoring_dict) where monitoring_dict contains
            individual loss components for logging
        """
        grid_loss_value = self.grid_loss(pred, gt)
        monitoring = {"grid_loss": grid_loss_value}
        total_loss = grid_loss_value
        
        if self.pinn_weight > 0:
            pinn_loss_value = self.pinn_loss_func(pred, gt, input_tensor)
            monitoring["pinn_loss"] = pinn_loss_value
            total_loss = total_loss + self.pinn_weight * pinn_loss_value
        
        return total_loss, monitoring

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

    def forward(self, pred, gt, input_tensor=None):
        """
        Compute total loss.
        
        Args:
            pred: predicted grid (B, C, T, X)
            gt: ground truth grid (B, C, T, X)
            input_tensor: input tensor (B, C, T, X) for PINN loss computation.
                Required if pinn_weight > 0.
        
        Returns:
            Total loss value
        """
        if self.pinn_weight > 0 and input_tensor is None:
            raise ValueError("input_tensor is required when pinn_weight > 0")
        
        # Compute main loss and its components for monitoring
        total_loss, loss_components = self.compute_loss(pred, gt, input_tensor)
        
        # Compute additional monitoring metrics
        monitoring = self.compute_monitoring_losses(pred, gt)
        
        # Accumulate total loss (single number)
        self.loss_sum += total_loss.item()
        
        # Accumulate all monitoring values (loss components + other metrics)
        for name, value in loss_components.items():
            self.monitoring_losses[name] = self.monitoring_losses.get(name, 0.0) + value.item()
        for name, value in monitoring.items():
            self.monitoring_losses[name] = self.monitoring_losses.get(name, 0.0) + value.item()
        
        self.loss_count += 1
        return total_loss

    def get_loss_value(self):
        """Return the average training loss."""
        return self.loss_sum / self.loss_count

    def get_loss_values(self):
        """Return all monitoring values (averaged over batches)."""
        return {name: value / self.loss_count for name, value in self.monitoring_losses.items()}

    def show_loss_values(self):
        if self.pinn_weight == 0:
            return
        loss_values = self.get_loss_values()
        loss_str = " | ".join([f"{name} : {value:.6f}" for name, value in loss_values.items()])
        print(f"Loss Values: {loss_str}")

    def log_loss_values(self, experiment, stage):
        # Log total training loss
        experiment.log_metric(f"{stage}/loss", self.get_loss_value())
        # Log all monitoring values (includes grid_loss, pinn_loss if enabled, and other metrics)
        for name, value in self.monitoring_losses.items():
            experiment.log_metric(f"{stage}/loss/{name}", value / self.loss_count)
