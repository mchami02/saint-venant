from deepxde.nn.pytorch import DeepONetCartesianProd, DeepONet
import torch

class DeepONetWrapper(DeepONetCartesianProd):
    def __init__(self, nt, nx, dt, dx, n_features, *args, **kwargs):
        """
        Proper DeepONet wrapper that separates input function from query coordinates.
        
        Args:
            nt: number of time steps
            nx: number of spatial points
            dt: time step size
            dx: spatial step size
            n_features: number of feature channels (excluding coordinates)
        """
        super().__init__(*args, **kwargs)
        self.nt = nt
        self.nx = nx
        self.dt = dt
        self.dx = dx
        self.n_features = n_features
        
        # Create coordinate grid for trunk network (nt*nx, 2) for (t, x)
        t_coords = torch.arange(nt).float() * dt
        x_coords = torch.arange(nx).float() * dx
        t_grid, x_grid = torch.meshgrid(t_coords, x_coords, indexing='ij')
        # Flatten and stack to get (nt*nx, 2)
        self.coords = torch.stack([t_grid.flatten(), x_grid.flatten()], dim=1)
        
    def forward(self, x):
        '''
        Forward pass for DeepONetWrapper with proper separation of function and location inputs.
        
        Args:
            x: Input tensor of shape (B, in_channels, nt, nx) where:
               - First n_features channels are the masked input (initial condition + boundaries)
               - Last 2 channels are time and space coordinates (which we ignore here)
            
        Returns:
            Output tensor of shape (B, out_channels, nt, nx)
        '''
        batch_size = x.shape[0]
        
        # Extract only feature channels (exclude coordinate channels)
        # x has shape (B, n_features+2, nt, nx) - we want (B, n_features, nt, nx)
        x_features = x[:, :self.n_features, :, :]
        
        # BRANCH INPUT: Use only the initial condition and boundaries
        # Extract initial timestep: (B, n_features, nx)
        initial_condition = x_features[:, :, 0, :]  
        
        # Also extract boundaries across all time (for boundary-dependent problems)
        # Left boundary: (B, n_features, nt)
        left_boundary = x_features[:, :, :, 0]
        # Right boundary: (B, n_features, nt)
        right_boundary = x_features[:, :, :, -1]
        
        # Flatten and concatenate for branch network
        # Shape: (B, n_features*nx + n_features*nt*2)
        x_func = torch.cat([
            initial_condition.reshape(batch_size, -1),  # Initial condition
            left_boundary.reshape(batch_size, -1),      # Left boundary
            right_boundary.reshape(batch_size, -1)      # Right boundary
        ], dim=1)
        
        # TRUNK INPUT: Use precomputed coordinate grid
        x_loc = self.coords.to(x.device)  # (nt*nx, 2)
        
        # Forward pass through DeepONetCartesianProd
        out = super().forward([x_func, x_loc])
        
        # Reshape output to (B, out_channels, nt, nx) to match FNO format
        if out.dim() == 2:
            # Single output: (B, nt*nx) -> (B, 1, nt, nx)
            out = out.reshape(batch_size, 1, self.nt, self.nx)
        else:
            # Multiple outputs: (B, nt*nx, num_outputs) -> (B, num_outputs, nt, nx)
            out = out.permute(0, 2, 1).reshape(batch_size, -1, self.nt, self.nx)
        
        return out
