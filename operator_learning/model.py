from neuralop.models import FNO
from deepxde.nn.pytorch import DeepONetCartesianProd
from models.fno_wrapper import fno_custom_freqs
from models.wno import WNO2d
# DeepXDE sets torch default device to cuda, but this breaks DataLoader with num_workers > 0
# Reset it to None/cpu to avoid generator device mismatch
import torch
torch.set_default_device(None)




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

def create_model(args):
    if args.model == "FNO":
        model = FNO(
        n_modes=(128, 64),        # modes in (time, space) dimensions
        hidden_channels=64,       # network width
        in_channels=args.in_channels - 2,           # density + time + space
        out_channels=args.out_channels,          # predicted density
        n_layers=4               # number of FNO layers
        )
    elif args.model == "FNOPersonalized":
        model = fno_custom_freqs(
            n_modes=(8, 16),
            hidden_channels=16,
            in_channels=args.in_channels - 2,
            out_channels=args.out_channels,
            n_layers=4
        )
    elif args.model == "DeepONet":
        # Calculate proper branch network input size
        # Branch receives: initial condition (n_features*nx) + boundaries (2*n_features*nt)
        n_features = args.in_channels - 2  # Exclude coordinate channels
        branch_input_size = n_features * (args.nx + 2 * args.nt)
        
        model = DeepONetWrapper(
        nt=args.nt,
        nx=args.nx,
        dt=args.dt,
        dx=args.dx,
        n_features=n_features,
        layer_sizes_branch = [branch_input_size, 256, 512, 1024, 512, 256, 128],  # Deeper network for better learning
        layer_sizes_trunk = [2, 256, 256, 256, 128],  # 2 for (t, x) coordinates
        activation = "relu",
        kernel_initializer = "Glorot normal",
        num_outputs = args.out_channels
        )
    elif args.model == "WNO":
        model = WNO2d(
            width=8,
            level=3,
            layers=2,
            size=[args.nt, args.nx],
            wavelet='db4',
            in_channel=3,
            grid_range=[0.0, 1.0],
        )
    else:
        raise ValueError(f"Model {args.model} not supported")
    return model
