import torch
import torch.nn as nn
from typing import Optional, Tuple
from neuralop.models import FNO
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.fno_block import FNOBlocks
class SpectralConvWithManualFreqs(nn.Module):
    """
    Wrapper around SpectralConv that, instead of taking the lowest n_modes
    (i.e. contiguous from 0), selects n_modes *manually chosen* frequencies
    in the forward pass.

    Assumptions for simplicity:
      - Supports 1D and 2D convolutions
      - real-valued spatial data (complex_data=False)
      - non-separable convolution (separable=False)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: int,
        **kwargs,
    ):
        super().__init__()
        # Instantiate the original SpectralConv with the same arguments
        self.spectral_conv = SpectralConv(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=n_modes,
            **kwargs,
        )

        if self.spectral_conv.complex_data:
            raise NotImplementedError("Wrapper currently assumes complex_data=False.")
        if self.spectral_conv.separable:
            raise NotImplementedError("Wrapper currently assumes separable=False.")
        if self.spectral_conv.order > 2:
            raise NotImplementedError("Wrapper currently implemented for 1D and 2D only.")

    # ---- Helper: choose manual frequency indices (default) -----------------
    def _default_frequency_indices_1d(self, fft_size: int, n_modes_used: int, device) -> torch.LongTensor:
        """
        Choose which frequency bins to keep for a single dimension.

        Uses the Fibonacci sequence to select frequencies, which provides
        a naturally growing spacing that captures both low and mid frequencies.
        If we need more modes than Fibonacci provides, fill with remaining
        lowest frequencies.
        """
        if n_modes_used >= fft_size:
            # If we ask for more modes than available, just keep everything
            return torch.arange(fft_size, device=device)

        # Generate Fibonacci sequence up to fft_size
        fib_indices = []
        a, b = 0, 1
        while a < fft_size:
            if a not in fib_indices:  # Avoid duplicates (e.g., 0, 1, 1)
                fib_indices.append(a)
            a, b = b, a + b
        
        base = torch.tensor(fib_indices, device=device)

        # If we already have enough, truncate
        if base.numel() >= n_modes_used:
            return base[:n_modes_used]

        # Otherwise, fill with the remaining lowest frequencies not in `base`
        used = set(base.tolist())
        extra = [i for i in range(fft_size) if i not in used]
        extra = torch.tensor(extra, device=device)

        idx = torch.cat([base, extra])[:n_modes_used]
        return idx

    # ----------------------------------------------------------------------
    def forward(self, x: torch.Tensor, output_shape: Optional[Tuple[int]] = None):
        """
        Forward pass where we:
          1) Go to Fourier domain.
          2) Select *manual* frequency indices.
          3) Contract only on those indices.
          4) Scatter the result back into the full spectrum.
          5) Inverse FFT to spatial domain.
        
        Supports both 1D and 2D inputs.
        """
        sc = self.spectral_conv  # shorthand
        order = sc.order  # 1 for 1D, 2 for 2D

        # Precision handling (same as original)
        if sc.fno_block_precision == "half":
            x = x.half()

        # Determine spatial dimensions based on order
        if order == 1:
            batchsize, in_channels, L = x.shape
            assert in_channels == sc.in_channels
            fft_dims = [-1]
            spatial_shape = (L,)
        elif order == 2:
            batchsize, in_channels, H, W = x.shape
            assert in_channels == sc.in_channels
            fft_dims = [-2, -1]
            spatial_shape = (H, W)
        else:
            raise ValueError(f"Unsupported order: {order}")

        # Real FFT along spatial dims
        x_fft = torch.fft.rfftn(x, dim=fft_dims, norm=sc.fft_norm)
        
        # Get FFT sizes for each dimension
        fft_sizes = list(x_fft.shape[-order:])

        # Precision switch after FFT if "mixed"
        if sc.fno_block_precision == "mixed":
            x_fft = x_fft.chalf()

        # Decide output dtype in Fourier domain
        if sc.fno_block_precision in ["half", "mixed"]:
            out_dtype = torch.chalf
        else:
            out_dtype = torch.cfloat

        # ---- MANUAL FREQUENCY SELECTION -----------------------------------
        # Choose which bins to keep for each dimension
        indices = []
        n_kept_list = []
        
        for dim_idx in range(order):
            n_modes_used = sc.n_modes[dim_idx]
            fft_size = fft_sizes[dim_idx]
            
            idx = self._default_frequency_indices_1d(fft_size, n_modes_used, x_fft.device)
            n_kept = idx.numel()
            
            # Make sure it does not exceed what the weight tensor can support
            max_modes = sc.max_n_modes[dim_idx] if sc.max_n_modes is not None else n_kept
            n_kept = min(n_kept, max_modes)
            idx = idx[:n_kept]
            
            indices.append(idx)
            n_kept_list.append(n_kept)

        # Select frequencies from x_fft based on indices
        x_sel = x_fft
        for dim_idx in range(order):
            fft_dim = -(order - dim_idx)  # -1 for last, -2 for second to last
            x_sel = torch.index_select(x_sel, dim=fft_dim, index=indices[dim_idx])

        # Restrict the weight tensor to selected modes
        if order == 1:
            weight = sc.weight[:, :, :n_kept_list[0]]
        elif order == 2:
            weight = sc.weight[:, :, :n_kept_list[0], :n_kept_list[1]]

        # Contract along channels and modes using the original contract function
        out_sel = sc._contract(x_sel, weight, separable=sc.separable)

        # Allocate full Fourier tensor and scatter the selected frequencies back
        if order == 1:
            out_fft = torch.zeros(
                (batchsize, sc.out_channels, fft_sizes[0]),
                device=x_fft.device,
                dtype=out_dtype,
            )
            out_fft.index_copy_(dim=-1, index=indices[0], source=out_sel)
        elif order == 2:
            out_fft = torch.zeros(
                (batchsize, sc.out_channels, fft_sizes[0], fft_sizes[1]),
                device=x_fft.device,
                dtype=out_dtype,
            )
            # Scatter back for 2D: need to handle both dimensions
            # First, create a grid of indices
            for i, idx_h in enumerate(indices[0]):
                for j, idx_w in enumerate(indices[1]):
                    out_fft[:, :, idx_h, idx_w] = out_sel[:, :, i, j]

        # ---- Handle possible resolution scaling / output_shape ------------
        if sc.resolution_scaling_factor is not None and output_shape is None:
            if order == 1:
                mode_sizes = (round(spatial_shape[0] * sc.resolution_scaling_factor[0][0]),)
            elif order == 2:
                mode_sizes = (
                    round(spatial_shape[0] * sc.resolution_scaling_factor[0][0]),
                    round(spatial_shape[1] * sc.resolution_scaling_factor[0][1])
                )
        elif output_shape is not None:
            mode_sizes = output_shape
        else:
            mode_sizes = spatial_shape

        # Inverse real FFT to go back to spatial domain
        x_out = torch.fft.irfftn(
            out_fft, s=mode_sizes, dim=fft_dims, norm=sc.fft_norm
        )

        # Add bias if present
        if sc.bias is not None:
            x_out = x_out + sc.bias

        return x_out

def fno_custom_freqs(n_modes, hidden_channels, in_channels, out_channels, n_layers, **kwargs):
    return FNO(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        conv_module=SpectralConvWithManualFreqs,
        **kwargs
    )