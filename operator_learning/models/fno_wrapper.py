import torch
import torch.nn as nn
from typing import Optional, Tuple
from neuralop.models import FNO
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.fno_block import FNOBlocks

class SpectralConvSparse(SpectralConv):
    """
    Modified SpectralConv that selects non-contiguous frequency modes
    to capture both smooth (low-freq) and non-smooth (high-freq) features.
    """
    
    def __init__(self, *args, frequency_sampling='logarithmic', **kwargs):
        """
        Parameters
        ----------
        frequency_sampling : str or callable
            Strategy for selecting which frequencies to keep:
            - 'contiguous': original behavior (low-pass filter)
            - 'logarithmic': log-spaced frequencies [1, 2, 4, 8, 16, ...]
            - 'fibonacci': Fibonacci spacing [0, 1, 2, 3, 5, 8, 13, ...]
            - 'mixed': half low-freq, half high-freq
            - callable: custom function(fft_size, n_modes) -> indices
        """
        super().__init__(*args, **kwargs)
        self.frequency_sampling = frequency_sampling
    
    def _get_frequency_indices(self, fft_size, n_modes, device):
        """
        Generate non-contiguous frequency indices based on sampling strategy.
        
        Returns
        -------
        indices : torch.LongTensor
            Indices of frequencies to keep (in fftshift-ed coordinates)
        """
        center = fft_size // 2
        
        if self.frequency_sampling == 'contiguous':
            # Original behavior: contiguous around DC
            negative_freqs = n_modes // 2
            positive_freqs = n_modes // 2 + n_modes % 2
            return torch.arange(
                center - negative_freqs, 
                center + positive_freqs, 
                device=device
            )
        
        elif self.frequency_sampling == 'logarithmic':
            # Logarithmic spacing: captures multiple scales
            # Takes: [0, 1, 2, 4, 8, 16, ...] around DC
            indices = []
            # Positive frequencies
            pos_indices = [0]  # DC
            k = 1
            while len(pos_indices) < n_modes // 2 + 1 and center + k < fft_size:
                pos_indices.append(k)
                k = min(k * 2, k + 1)  # Exponential, but cap growth
            
            # Negative frequencies (mirror)
            neg_indices = [-i for i in pos_indices[1:]][::-1]
            
            # Combine and convert to absolute indices
            freq_offsets = neg_indices + pos_indices
            freq_offsets = freq_offsets[:n_modes]  # Truncate to n_modes
            indices = torch.tensor([center + offset for offset in freq_offsets], 
                                  device=device)
            return indices
        
        elif self.frequency_sampling == 'fibonacci':
            # Fibonacci spacing: natural growth rate
            # Similar to your wrapper implementation
            fib_offsets = []
            a, b = 0, 1
            while a < fft_size // 2:
                fib_offsets.append(a)
                a, b = b, a + b
            
            # Take positive and negative
            indices = [center]  # DC
            for offset in fib_offsets[1:]:
                if len(indices) >= n_modes:
                    break
                if center + offset < fft_size:
                    indices.append(center + offset)
                if len(indices) >= n_modes:
                    break
                if center - offset >= 0:
                    indices.append(center - offset)
            
            return torch.tensor(sorted(indices[:n_modes]), device=device)
        
        elif self.frequency_sampling == 'mixed':
            # Half low-frequency (smooth), half high-frequency (non-smooth)
            low_modes = n_modes // 2
            high_modes = n_modes - low_modes
            
            # Low frequencies: contiguous around DC
            low_indices = torch.arange(
                center - low_modes // 2,
                center + (low_modes - low_modes // 2),
                device=device
            )
            
            # High frequencies: from the edges
            high_indices = []
            # Take from positive high frequencies
            for i in range(high_modes):
                idx = center + (low_modes // 2) + i * (fft_size // high_modes)
                if idx < fft_size:
                    high_indices.append(idx)
            
            high_indices = torch.tensor(high_indices, device=device)
            return torch.cat([low_indices, high_indices]).sort()[0]
        
        elif callable(self.frequency_sampling):
            # Custom user-defined function
            offsets = self.frequency_sampling(fft_size, n_modes)
            return torch.tensor([center + o for o in offsets], device=device)
        
        else:
            raise ValueError(f"Unknown frequency_sampling: {self.frequency_sampling}")
    
    def forward(self, x: torch.Tensor, output_shape=None):
        """Modified forward pass with sparse frequency selection."""
        batchsize, channels, *mode_sizes = x.shape
        
        fft_size = list(mode_sizes)
        if not self.complex_data:
            fft_size[-1] = fft_size[-1] // 2 + 1
        fft_dims = list(range(-self.order, 0))
        
        if self.fno_block_precision == "half":
            x = x.half()
        
        # FFT Transform
        if self.complex_data:
            x = torch.fft.fftn(x, norm=self.fft_norm, dim=fft_dims)
            dims_to_fft_shift = fft_dims
        else:
            x = torch.fft.rfftn(x, norm=self.fft_norm, dim=fft_dims)
            dims_to_fft_shift = fft_dims[:-1]
        
        if self.order > 1:
            x = torch.fft.fftshift(x, dim=dims_to_fft_shift)
        
        if self.fno_block_precision == "mixed":
            x = x.chalf()
        
        if self.fno_block_precision in ["half", "mixed"]:
            out_dtype = torch.chalf
        else:
            out_dtype = torch.cfloat
        
        out_fft = torch.zeros(
            [batchsize, self.out_channels, *fft_size], 
            device=x.device, 
            dtype=out_dtype
        )
        
        # ====== KEY MODIFICATION: Non-contiguous frequency selection ======
        
        # Get sparse frequency indices for each dimension
        freq_indices = []
        for dim_idx in range(self.order):
            indices = self._get_frequency_indices(
                fft_size[dim_idx], 
                self.n_modes[dim_idx],
                x.device
            )
            freq_indices.append(indices)
        
        # Select frequencies from input FFT using advanced indexing
        x_selected = x
        for dim_idx, indices in enumerate(freq_indices):
            # Select along spatial dimensions (skip batch and channel dims)
            dim = 2 + dim_idx
            x_selected = torch.index_select(x_selected, dim=dim, index=indices)
        
        # Get corresponding weight slice
        # Weight tensor shape: [in_channels, out_channels, modes_1, modes_2, ...]
        if self.separable:
            weight_slice = [slice(None)]
        else:
            weight_slice = [slice(None), slice(None)]
        
        # For each spatial dimension, take only the modes we're using
        for dim_idx in range(self.order):
            n_kept = len(freq_indices[dim_idx])
            weight_slice.append(slice(None, n_kept))
        
        weight = self.weight[tuple(weight_slice)]
        
        # Contract (multiply in frequency domain)
        out_selected = self._contract(x_selected, weight, separable=self.separable)
        
        # Scatter back to full spectrum at the selected frequencies
        # This is the tricky part - we need to place values at specific indices
        if self.order == 1:
            # 1D case
            out_fft[:, :, freq_indices[0]] = out_selected
        elif self.order == 2:
            # 2D case - need meshgrid
            idx_h, idx_w = freq_indices[0], freq_indices[1]
            for i, h in enumerate(idx_h):
                for j, w in enumerate(idx_w):
                    out_fft[:, :, h, w] = out_selected[:, :, i, j]
        elif self.order == 3:
            # 3D case
            for i, h in enumerate(freq_indices[0]):
                for j, w in enumerate(freq_indices[1]):
                    for k, d in enumerate(freq_indices[2]):
                        out_fft[:, :, h, w, d] = out_selected[:, :, i, j, k]
        
        # ====== End of modification ======
        
        # Resolution scaling
        if self.resolution_scaling_factor is not None and output_shape is None:
            mode_sizes = tuple([round(s * r) for (s, r) in 
                               zip(mode_sizes, self.resolution_scaling_factor)])
        if output_shape is not None:
            mode_sizes = output_shape
        
        # Inverse FFT
        if self.order > 1:
            out_fft = torch.fft.ifftshift(out_fft, dim=dims_to_fft_shift)
        
        if self.complex_data:
            x = torch.fft.ifftn(out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm)
        else:
            x = torch.fft.irfftn(out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm)
        
        if self.bias is not None:
            x = x + self.bias
        
        return x

def fno_custom_freqs(n_modes, hidden_channels, in_channels, out_channels, n_layers, **kwargs):
    return FNO(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        conv_module=SpectralConvSparse,
        **kwargs
    )