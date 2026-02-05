import torch
import torch.nn as nn
from neuralop.models import FNO


class CNN2D(nn.Module):
    """
    2D CNN for refining FNO output on a spatiotemporal grid.
    Uses residual connections and batch normalization for stable training.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 32,
        num_layers: int = 4,
        kernel_size: int = 3,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        
        # Input projection
        self.input_proj = nn.Conv2d(
            in_channels, hidden_channels, 
            kernel_size=kernel_size, padding=kernel_size // 2
        )
        
        # Convolutional blocks with residual connections
        self.conv_blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = nn.Sequential(
                nn.Conv2d(
                    hidden_channels, hidden_channels,
                    kernel_size=kernel_size, padding=kernel_size // 2
                ),
                nn.BatchNorm2d(hidden_channels),
                nn.GELU(),
                nn.Conv2d(
                    hidden_channels, hidden_channels,
                    kernel_size=kernel_size, padding=kernel_size // 2
                ),
                nn.BatchNorm2d(hidden_channels),
            )
            self.conv_blocks.append(block)
        
        self.activation = nn.GELU()
        
        # Output projection
        self.output_proj = nn.Conv2d(
            hidden_channels, out_channels,
            kernel_size=kernel_size, padding=kernel_size // 2
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN.
        
        Args:
            x: Input tensor of shape (B, in_channels, nt, nx)
            
        Returns:
            Output tensor of shape (B, out_channels, nt, nx)
        """
        # Input projection
        x = self.input_proj(x)
        
        # Residual convolutional blocks
        for block in self.conv_blocks:
            residual = x
            x = block(x)
            x = self.activation(x + residual)
        
        # Output projection
        x = self.output_proj(x)
        
        return x


class FNOCNN(nn.Module):
    """
    Two-stage model combining FNO and CNN for PDE grid prediction.
    
    Stage 1 (FNO): Learns global patterns in the frequency domain,
                   capturing long-range dependencies efficiently.
    Stage 2 (CNN): Refines the FNO output with local convolutions,
                   improving fine-grained spatial details.
    
    This architecture combines the strengths of both:
    - FNO excels at capturing global structure and long-range correlations
    - CNN excels at refining local patterns and sharp features
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        fno_modes: tuple = (16, 8),
        fno_hidden_channels: int = 32,
        fno_layers: int = 4,
        cnn_hidden_channels: int = 32,
        cnn_layers: int = 4,
        cnn_kernel_size: int = 3,
        skip_connection: bool = True,
    ):
        """
        Initialize the FNO-CNN model.
        
        Args:
            in_channels: Number of input channels (excluding coordinate channels)
            out_channels: Number of output channels
            fno_modes: Number of Fourier modes to keep in (time, space) dimensions
            fno_hidden_channels: Hidden channel width for FNO
            fno_layers: Number of FNO layers
            cnn_hidden_channels: Hidden channel width for CNN
            cnn_layers: Number of CNN layers
            cnn_kernel_size: Kernel size for CNN convolutions
            skip_connection: Whether to add a skip connection from FNO output to final output
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_connection = skip_connection
        
        # Stage 1: FNO for global frequency-domain processing
        self.fno = FNO(
            n_modes=fno_modes,
            hidden_channels=fno_hidden_channels,
            in_channels=in_channels,
            out_channels=fno_hidden_channels,  # Output to intermediate channels
            n_layers=fno_layers,
        )
        
        # Stage 2: CNN for local refinement
        self.cnn = CNN2D(
            in_channels=fno_hidden_channels,
            out_channels=out_channels,
            hidden_channels=cnn_hidden_channels,
            num_layers=cnn_layers,
            kernel_size=cnn_kernel_size,
        )
        
        # Optional skip connection projection (if input/output channels differ)
        if skip_connection:
            self.skip_proj = nn.Conv2d(
                fno_hidden_channels, out_channels, kernel_size=1
            ) if fno_hidden_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the FNO-CNN model.
        
        Args:
            x: Input tensor of shape (B, in_channels, nt, nx)
               Contains the masked input grid with initial conditions and boundaries.
               
        Returns:
            Output tensor of shape (B, out_channels, nt, nx)
            The predicted full spatiotemporal grid.
        """
        # Stage 1: FNO processing
        fno_out = self.fno(x)  # (B, fno_hidden_channels, nt, nx)
        
        # Stage 2: CNN refinement
        cnn_out = self.cnn(fno_out)  # (B, out_channels, nt, nx)
        
        # Optional skip connection from FNO
        if self.skip_connection:
            cnn_out = cnn_out + self.skip_proj(fno_out)
        
        return cnn_out


class FNOCNNWrapper(FNOCNN):
    """
    Wrapper for FNO-CNN that handles coordinate channel stripping,
    matching the interface used by other models in the codebase.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        fno_modes: tuple = (16, 8),
        fno_hidden_channels: int = 32,
        fno_layers: int = 4,
        cnn_hidden_channels: int = 32,
        cnn_layers: int = 4,
        cnn_kernel_size: int = 3,
        skip_connection: bool = True,
        strip_coords: bool = True,
    ):
        """
        Initialize the FNO-CNN wrapper.
        
        Args:
            in_channels: Number of input channels (including coordinate channels if strip_coords=True)
            out_channels: Number of output channels
            strip_coords: If True, removes the last 2 channels (time, space coordinates)
                         from the input, matching FNO behavior in model.py
            ... (other args same as FNOCNN)
        """
        self.strip_coords = strip_coords
        actual_in_channels = in_channels - 2 if strip_coords else in_channels
        
        super().__init__(
            in_channels=actual_in_channels,
            out_channels=out_channels,
            fno_modes=fno_modes,
            fno_hidden_channels=fno_hidden_channels,
            fno_layers=fno_layers,
            cnn_hidden_channels=cnn_hidden_channels,
            cnn_layers=cnn_layers,
            cnn_kernel_size=cnn_kernel_size,
            skip_connection=skip_connection,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional coordinate stripping.
        
        Args:
            x: Input tensor of shape (B, in_channels, nt, nx)
               If strip_coords=True, assumes last 2 channels are coordinates.
               
        Returns:
            Output tensor of shape (B, out_channels, nt, nx)
        """
        if self.strip_coords:
            # Remove coordinate channels (last 2 channels)
            x = x[:, :-2, :, :]
        
        return super().forward(x)
