"""Lagrangian Neural Operator (LNO) for LWR traffic flow.

A dual-branch neural operator that replaces FNO's Fourier branch with a
characteristic-aligned Lagrangian transform. The Lagrangian branch uses the
known LWR flux function to compute characteristic speeds, pushes grid points
along characteristics using differentiable ``F.grid_sample``, processes in
the Lagrangian frame via 1D convolution, and interpolates back.

Architecture:
- **LNO** (outer): Autoregressive wrapper conditioned on ``dt``.
- **LNO1d** (core): Lifting → [LNO1dBlock] × L → Projection.
- **LNO1dBlock**: Three parallel branches (skip + Eulerian + Lagrangian).

Input:  dict with ``grid_input`` (B, 1, nt, nx) and ``dt`` (B,).
Output: dict with ``output_grid`` (B, 1, nt, nx).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base.flux import DEFAULT_FLUX, Flux


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def grid_sample_1d(
    signal: torch.Tensor,
    grid_x: torch.Tensor,
) -> torch.Tensor:
    """Differentiable 1D linear interpolation (MPS-compatible).

    Manual implementation that avoids ``F.grid_sample`` whose backward pass
    is not implemented on MPS.  Uses border clamping (out-of-range queries
    return the nearest boundary value).

    Args:
        signal: (B, C, nx) — input signal to resample.
        grid_x: (B, nx_q) — query positions in **normalized** [-1, 1] coords.

    Returns:
        Resampled signal (B, C, nx_q).
    """
    nx = signal.shape[-1]
    # Normalized [-1, 1] -> continuous index [0, nx-1]
    idx = (grid_x + 1.0) * 0.5 * (nx - 1)
    # Border clamp
    idx = idx.clamp(0.0, nx - 1.0)

    idx_lo = idx.long().clamp(max=nx - 2)  # (B, nx_q)
    idx_hi = idx_lo + 1
    w = (idx - idx_lo.float()).unsqueeze(1)  # (B, 1, nx_q) — lerp weight

    # Gather left and right neighbours: (B, C, nx_q)
    lo = torch.gather(signal, 2, idx_lo.unsqueeze(1).expand_as(signal[:, :, : idx_lo.shape[-1]]))
    hi = torch.gather(signal, 2, idx_hi.unsqueeze(1).expand_as(lo))
    return lo + w * (hi - lo)


def _normalize_grid(x: torch.Tensor, nx: int) -> torch.Tensor:
    """Map physical grid indices [0, nx-1] to normalized [-1, 1]."""
    return 2.0 * x / (nx - 1) - 1.0


# ---------------------------------------------------------------------------
# LNO1dBlock
# ---------------------------------------------------------------------------


class LNO1dBlock(nn.Module):
    """Single LNO layer with skip, Eulerian, and Lagrangian branches.

    The Lagrangian branch:
    1. Projects hidden state to density estimate.
    2. Computes characteristic speeds from the flux derivative.
    3. Displaces grid points along characteristics (scaled by learnable alpha).
    4. Processes in the Lagrangian frame via 1D convolution.
    5. Interpolates back to the Eulerian grid.

    Args:
        channels: Hidden channel width.
        kernel_size: Convolution kernel for Lagrangian branch.
        flux: Flux function providing ``derivative(rho)``.
        max_displacement: Absolute clamp on displacement (fraction of domain).
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        flux: Flux | None = None,
        max_displacement: float = 0.1,
    ):
        super().__init__()
        self.flux = flux if flux is not None else DEFAULT_FLUX()
        self.max_displacement = max_displacement

        # Skip branch
        self.skip = nn.Conv1d(channels, channels, kernel_size=1)

        # Eulerian branch
        self.eul_conv1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.eul_conv2 = nn.Conv1d(channels, channels, kernel_size=1)

        # Lagrangian branch
        self.rho_proj = nn.Conv1d(channels, 1, kernel_size=1)
        self.lag_conv = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, padding=kernel_size // 2
        )

        # Learnable displacement scale (per layer)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, h: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            h: (B, C, nx) hidden state.
            dt: (B,) time step size per sample.

        Returns:
            (B, C, nx) updated hidden state.
        """
        B, C, nx = h.shape

        # --- Skip branch ---
        skip_out = self.skip(h)

        # --- Eulerian branch ---
        eul_out = self.eul_conv2(F.gelu(self.eul_conv1(h)))

        # --- Lagrangian branch ---
        # 1. Project to density estimate
        rho = self.rho_proj(h).squeeze(1)  # (B, nx)

        # 2. Characteristic speed from flux derivative
        char_speed = self.flux.derivative(rho)  # (B, nx)

        # 3. Displacement = speed * alpha * dt, clamped
        displacement = (
            char_speed * self.alpha * dt.view(B, 1)
        )  # (B, nx)
        displacement = displacement.clamp(
            -self.max_displacement, self.max_displacement
        )

        # 4. Build Eulerian grid [0, nx-1] and displaced Lagrangian grid
        x_eul = torch.arange(nx, device=h.device, dtype=h.dtype).unsqueeze(
            0
        )  # (1, nx)
        x_lag = x_eul + displacement * (nx - 1)  # scale displacement to grid units

        # 5. Forward warp: sample h at Lagrangian positions
        x_lag_norm = _normalize_grid(x_lag, nx)  # (B, nx)
        h_lag = grid_sample_1d(h, x_lag_norm)  # (B, C, nx)

        # 6. Process in Lagrangian frame
        h_lag = self.lag_conv(h_lag)  # (B, C, nx)

        # 7. Inverse warp: sample back to Eulerian grid
        x_inv = x_eul - displacement * (nx - 1)
        x_inv_norm = _normalize_grid(x_inv, nx)
        lag_out = grid_sample_1d(h_lag, x_inv_norm)  # (B, C, nx)

        # Combine branches
        return F.gelu(skip_out + eul_out + lag_out)


# ---------------------------------------------------------------------------
# LNO1d — core spatial network
# ---------------------------------------------------------------------------


class LNO1d(nn.Module):
    """Core 1D Lagrangian Neural Operator.

    Architecture: Lifting Conv1d → [LNO1dBlock] × L → Projection Conv1d.

    Args:
        in_channels: Input channels (typically 2: state + dt channel).
        out_channels: Output channels (typically 1: state delta).
        hidden_channels: Hidden width.
        n_layers: Number of LNO1dBlock layers.
        kernel_size: Convolution kernel for Lagrangian branch.
        max_displacement: Displacement clamp for Lagrangian branch.
        flux: Flux function instance.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        hidden_channels: int = 32,
        n_layers: int = 4,
        kernel_size: int = 5,
        max_displacement: float = 0.1,
        flux: Flux | None = None,
    ):
        super().__init__()
        flux_instance = flux if flux is not None else DEFAULT_FLUX()

        self.lifting = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.layers = nn.ModuleList(
            [
                LNO1dBlock(
                    hidden_channels,
                    kernel_size=kernel_size,
                    flux=flux_instance,
                    max_displacement=max_displacement,
                )
                for _ in range(n_layers)
            ]
        )
        self.projection = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, in_channels, nx) — concatenated [state, dt_channel].
            dt: (B,) — time step per sample.

        Returns:
            (B, out_channels, nx) — predicted state delta.
        """
        h = self.lifting(x)
        for layer in self.layers:
            h = layer(h, dt)
        return self.projection(h)


# ---------------------------------------------------------------------------
# LNO — autoregressive wrapper
# ---------------------------------------------------------------------------


class LNO(nn.Module):
    """Lagrangian Neural Operator with autoregressive time stepping.

    Wraps ``LNO1d`` in a time loop: at each step the model predicts a state
    delta conditioned on the current state and ``dt``.

    Input:  dict with ``grid_input`` (B, 1, nt, nx) and ``dt`` (B,).
    Output: dict with ``output_grid`` (B, 1, nt, nx).
    """

    def __init__(self, **lno_kwargs):
        super().__init__()
        self.lno = LNO1d(**lno_kwargs)
        self.teacher_forcing_ratio = 0.0
        self.noise_std = 0.0

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        grid_input = x["grid_input"]  # (B, 1, nt, nx)
        dt = x["dt"]  # (B,)

        B, _, nt, nx = grid_input.shape
        state = grid_input[:, :, 0, :]  # (B, 1, nx) — initial condition
        dt_channel = dt.view(B, 1, 1).expand(B, 1, nx)

        tf_ratio = self.teacher_forcing_ratio
        target_grid = x.get("target_grid") if tf_ratio > 0 else None

        outputs = [state]
        for t in range(nt - 1):
            lno_input = torch.cat([state, dt_channel], dim=1)  # (B, 2, nx)
            state = state + self.lno(lno_input, dt)  # (B, 1, nx)
            state = state.clamp(0.0, 1.0)  # Greenshields density bounds
            outputs.append(state)
            # Pushforward noise: perturb input to next step (training only)
            if self.training and self.noise_std > 0:
                state = (state + torch.randn_like(state) * self.noise_std).clamp(0.0, 1.0)
            # Teacher forcing: replace input to NEXT step with GT
            if target_grid is not None and torch.rand(1).item() < tf_ratio:
                state = target_grid[:, :, t + 1, :]

        return {"output_grid": torch.stack(outputs, dim=2)}  # (B, 1, nt, nx)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_lno(args: dict) -> LNO:
    """Build LNO from configuration dict.

    Args:
        args: Configuration dictionary. Supported keys:
            - hidden_channels: hidden width (default: 32)
            - n_layers: number of LNO1dBlock layers (default: 4)
            - kernel_size: Lagrangian conv kernel (default: 5)
            - max_displacement: displacement clamp (default: 0.1)
    """
    return LNO(
        in_channels=2,
        out_channels=1,
        hidden_channels=args.get("hidden_channels", 32),
        n_layers=args.get("n_layers", 4),
        kernel_size=args.get("kernel_size", 5),
        max_displacement=args.get("max_displacement", 0.1),
    )
