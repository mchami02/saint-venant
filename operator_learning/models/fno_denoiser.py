import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO


class DilatedResBlock(nn.Module):
    """Residual block with dilated convolutions for multi-scale receptive fields."""
    
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.activation = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Kaiming normal for ReLU activation."""
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.conv1.bias)
        # Initialize conv2 to zero for stable residual learning
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.activation(self.conv1(x))
        y = self.conv2(y)
        return self.activation(x + y)


class ShockPreservingCNN(nn.Module):
    """
    Encoder-decoder CNN with dilated convolutions for refining FNO output.
    Uses gradient magnitude as a shock indicator and skip connections for
    preserving fine details. Designed for PDE solutions with discontinuities.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 48,
        num_layers: int = 4,  # kept for API compatibility, not used
        kernel_size: int = 3,  # kept for API compatibility, not used
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        
        # Input projection (in_channels + gradient channel)
        self.conv_in = nn.Conv2d(in_channels + 1, hidden_channels, 3, padding=1)
        
        # Encoder
        self.block1 = DilatedResBlock(hidden_channels, dilation=1)
        self.down1 = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1)
        self.block2 = DilatedResBlock(hidden_channels, dilation=2)
        
        # Bottleneck
        self.block_mid = DilatedResBlock(hidden_channels, dilation=4)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, stride=2, padding=1)
        self.block3 = DilatedResBlock(hidden_channels, dilation=2)
        
        # Output projection
        self.conv_out = nn.Conv2d(hidden_channels, out_channels, 3, padding=1)
        
        # Residual projection (if in/out channels differ)
        if in_channels == out_channels:
            self.res_proj = nn.Identity()
        else:
            self.res_proj = nn.Conv2d(in_channels, out_channels, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Kaiming normal for ReLU activation."""
        # Input projection
        nn.init.kaiming_normal_(self.conv_in.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.conv_in.bias)
        
        # Encoder downsampling
        nn.init.kaiming_normal_(self.down1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.down1.bias)
        
        # Decoder upsampling
        nn.init.kaiming_normal_(self.up1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.up1.bias)
        
        # Output projection - initialize to small values for residual denoising
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
        
        # Residual projection (if not Identity)
        if isinstance(self.res_proj, nn.Conv2d):
            nn.init.kaiming_normal_(self.res_proj.weight, mode='fan_out', nonlinearity='relu')
            nn.init.zeros_(self.res_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with shock-aware processing.
        
        Args:
            x: Input tensor of shape (B, in_channels, nt, nx)
            
        Returns:
            Output tensor of shape (B, out_channels, nt, nx)
        """
        # Compute gradient magnitude (shock indicator) from first channel
        u = x[:, :1, :, :]
        gx = u[:, :, :, 1:] - u[:, :, :, :-1]
        gy = u[:, :, 1:, :] - u[:, :, :-1, :]
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        g = torch.sqrt(gx**2 + gy**2 + 1e-8)
        
        # Concatenate input with gradient magnitude
        x_aug = torch.cat([x, g], dim=1)
        
        # Encoder
        e1 = self.block1(self.conv_in(x_aug))
        d1 = self.down1(e1)
        e2 = self.block2(d1)
        
        # Bottleneck
        mid = self.block_mid(e2)
        
        # Decoder
        u1 = self.up1(mid)
        u1 = u1 + e1  # Skip connection
        
        out = self.block3(u1)
        res = self.conv_out(out)
        
        # Residual denoising
        return self.res_proj(x) + res


class FNODenoiser(nn.Module):
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
        Initialize the FNO-Denoiser model.
        
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
        
        # Stage 2: Shock-preserving CNN for local refinement
        self.cnn = ShockPreservingCNN(
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
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for skip connection projection."""
        if self.skip_connection and isinstance(self.skip_proj, nn.Conv2d):
            nn.init.kaiming_normal_(self.skip_proj.weight, mode='fan_out', nonlinearity='relu')
            nn.init.zeros_(self.skip_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the FNO-Denoiser model.
        
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


class FNODenoiserWrapper(FNODenoiser):
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
