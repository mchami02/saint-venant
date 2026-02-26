"""Autoregressive FNO variants for wavefront learning.

Two models that apply a 1D spatial FNO autoregressively in time:

1. **AutoregressiveFNO** — wraps neuralop's FNO. Uses complex-valued spectral
   convolution weights internally. Note: gradient clipping with
   ``clip_grad_norm_`` fails on MPS because PyTorch does not support norm ops
   on complex tensors there.

2. **AutoregressiveRealFNO** — self-contained 1D FNO with real-valued weights
   (stores real/imag parts separately). Works on all backends including MPS.

Both models:
- Take the current state ``(1, nx)`` + a ``dt`` channel ``(1, nx)`` as input
- Predict the state delta via a residual connection
- Condition on ``dt`` so they handle different temporal resolutions (high-res
  testing scales ``dt`` by ``1/res``)

Input: dict with:
    - "grid_input": (B, 1, nt, nx) from ToGridNoCoords (masked IC)
    - "dt": (B,) scalar dt per sample
Output: dict {"output_grid": (B, 1, nt, nx)}
"""

import torch
import torch.nn as nn
from neuralop.models import FNO


# ---------------------------------------------------------------------------
# AutoregressiveFNO — neuralop wrapper
# ---------------------------------------------------------------------------


class AutoregressiveFNO(nn.Module):
    """1D spatial FNO (neuralop) applied autoregressively in time.

    Uses neuralop's ``FNO`` with complex-valued spectral convolution weights.
    Does **not** work with ``clip_grad_norm_`` on MPS (complex norm unsupported).
    Use ``AutoregressiveRealFNO`` for MPS compatibility.
    """

    def __init__(self, **fno_kwargs):
        super().__init__()
        self.fno = FNO(**fno_kwargs)
        self.teacher_forcing_ratio = 0.0

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        grid_input = x["grid_input"]  # (B, 1, nt, nx)
        dt = x["dt"]  # (B,)

        B, _, nt, nx = grid_input.shape
        state = grid_input[:, :, 0, :]  # (B, 1, nx) — initial condition
        dt_channel = dt.view(B, 1, 1).expand(B, 1, nx)

        tf_ratio = self.teacher_forcing_ratio if self.training else 0.0
        target_grid = x.get("target_grid") if tf_ratio > 0 else None

        outputs = [state]
        for t in range(nt - 1):
            fno_input = torch.cat([state, dt_channel], dim=1)  # (B, 2, nx)
            state = state + self.fno(fno_input)  # (B, 1, nx)
            outputs.append(state)
            # Replace input to NEXT step with GT stochastically
            if target_grid is not None and torch.rand(1).item() < tf_ratio:
                state = target_grid[:, :, t + 1, :]

        return {"output_grid": torch.stack(outputs, dim=2)}  # (B, 1, nt, nx)


def build_autoregressive_fno(args: dict) -> AutoregressiveFNO:
    """Build AutoregressiveFNO from configuration dict.

    Args:
        args: Configuration dictionary. Supported keys:
            - n_modes_x: Fourier modes in space (default: 16)
            - hidden_channels: hidden width (default: 32)
            - n_layers: spectral conv layers (default: 2)
            - domain_padding: padding fraction for non-periodic BCs (default: 0.2)
    """
    return AutoregressiveFNO(
        n_modes=(args.get("n_modes_x", 16),),
        hidden_channels=args.get("hidden_channels", 32),
        in_channels=2,
        out_channels=1,
        n_layers=args.get("n_layers", 2),
        domain_padding=args.get("domain_padding", 0.2),
    )


# ---------------------------------------------------------------------------
# AutoregressiveRealFNO — real-valued 1D FNO (MPS-compatible)
# ---------------------------------------------------------------------------


class SpectralConv1d(nn.Module):
    """1D spectral convolution with real-valued weight storage.

    Stores the real and imaginary parts of the spectral weights as separate
    ``nn.Parameter`` tensors so that all gradients are real-valued.
    """

    def __init__(self, in_channels: int, out_channels: int, n_modes: int):
        super().__init__()
        self.n_modes = n_modes
        scale = 1.0 / (in_channels * out_channels)
        self.weight_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, n_modes)
        )
        self.weight_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, n_modes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, nx)
        nx = x.shape[-1]
        x_ft = torch.fft.rfft(x, dim=-1)  # (B, C_in, nx//2+1)

        # Truncate to n_modes
        k = min(self.n_modes, x_ft.shape[-1])
        x_ft_trunc = x_ft[:, :, :k]  # (B, C_in, k)

        # Build complex weight from real parts
        weight = torch.complex(
            self.weight_real[:, :, :k], self.weight_imag[:, :, :k]
        )  # (C_in, C_out, k)

        # Spectral multiply: einsum over input channels
        out_ft = torch.einsum("bik,iok->bok", x_ft_trunc, weight)  # (B, C_out, k)

        # Pad back to full frequency domain and inverse FFT
        out_full = torch.zeros(
            x.shape[0], out_ft.shape[1], nx // 2 + 1,
            dtype=out_ft.dtype, device=out_ft.device,
        )
        out_full[:, :, :k] = out_ft
        return torch.fft.irfft(out_full, n=nx, dim=-1)  # (B, C_out, nx)


class FNO1dBlock(nn.Module):
    """Single FNO layer: spectral conv + linear skip + GELU."""

    def __init__(self, channels: int, n_modes: int):
        super().__init__()
        self.spectral = SpectralConv1d(channels, channels, n_modes)
        self.linear = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.spectral(x) + self.linear(x))


class RealFNO1d(nn.Module):
    """Minimal 1D FNO with real-valued parameters throughout.

    Architecture: Lifting → [SpectralConv1d + Conv1d skip + GELU] × L → Projection
    Optional symmetric zero-padding handles non-periodic boundaries.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_modes: int,
        n_layers: int,
        domain_padding: float = 0.0,
    ):
        super().__init__()
        self.domain_padding = domain_padding
        self.lifting = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.layers = nn.ModuleList(
            [FNO1dBlock(hidden_channels, n_modes) for _ in range(n_layers)]
        )
        self.projection = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, nx)
        nx = x.shape[-1]

        # Pad
        if self.domain_padding > 0:
            pad = int(nx * self.domain_padding)
            x = nn.functional.pad(x, (pad, pad), mode="constant", value=0.0)

        x = self.lifting(x)
        for layer in self.layers:
            x = layer(x)
        x = self.projection(x)

        # Remove padding
        if self.domain_padding > 0:
            x = x[..., pad : pad + nx]

        return x  # (B, C_out, nx)


class AutoregressiveRealFNO(nn.Module):
    """1D spatial FNO (real-valued) applied autoregressively in time.

    Uses ``RealFNO1d`` which stores spectral weights as separate real/imag
    ``nn.Parameter`` tensors — fully compatible with ``clip_grad_norm_`` on
    all backends including MPS.
    """

    def __init__(self, **fno_kwargs):
        super().__init__()
        self.fno = RealFNO1d(**fno_kwargs)
        self.teacher_forcing_ratio = 0.0

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        grid_input = x["grid_input"]  # (B, 1, nt, nx)
        dt = x["dt"]  # (B,)

        B, _, nt, nx = grid_input.shape
        state = grid_input[:, :, 0, :]  # (B, 1, nx) — initial condition
        dt_channel = dt.view(B, 1, 1).expand(B, 1, nx)

        tf_ratio = self.teacher_forcing_ratio if self.training else 0.0
        target_grid = x.get("target_grid") if tf_ratio > 0 else None

        outputs = [state]
        for t in range(nt - 1):
            fno_input = torch.cat([state, dt_channel], dim=1)  # (B, 2, nx)
            state = state + self.fno(fno_input)  # (B, 1, nx)
            outputs.append(state)
            # Replace input to NEXT step with GT stochastically
            if target_grid is not None and torch.rand(1).item() < tf_ratio:
                state = target_grid[:, :, t + 1, :]

        return {"output_grid": torch.stack(outputs, dim=2)}  # (B, 1, nt, nx)


def build_autoregressive_real_fno(args: dict) -> AutoregressiveRealFNO:
    """Build AutoregressiveRealFNO from configuration dict.

    Args:
        args: Configuration dictionary. Supported keys:
            - n_modes_x: Fourier modes in space (default: 16)
            - hidden_channels: hidden width (default: 32)
            - n_layers: spectral conv layers (default: 2)
            - domain_padding: padding fraction for non-periodic BCs (default: 0.2)
    """
    return AutoregressiveRealFNO(
        in_channels=2,
        out_channels=1,
        hidden_channels=args.get("hidden_channels", 32),
        n_modes=args.get("n_modes_x", 16),
        n_layers=args.get("n_layers", 2),
        domain_padding=args.get("domain_padding", 0.2),
    )
