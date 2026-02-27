"""NeuralFVSolver: Learned finite volume time-marching for hyperbolic PDEs.

Instead of predicting the full (T, X) solution at once, this model learns a
single-step update operator (a learned Riemann solver) and rolls it forward
in time. Each step gathers local neighborhood features, passes them through
a shared flux MLP, and performs an Euler update. This respects the causal,
local, finite-speed-of-propagation structure of hyperbolic PDEs by design.

Architecture:
    1. Extract IC from grid_input[:, :, 0, :]
    2. Autoregressive rollout (for t = 0..nt-2):
       a. Ghost-cell pad + stencil extraction via unfold
       b. Characteristic speeds from flux.derivative(stencil)
       c. Shock proximity from DifferentiableShockProximity (detached)
       d. Stack features: [stencil, char_speeds, stencil_prox, dt]
       e. FluxNetwork(features) -> update
       f. Euler step: state = state + (dt/dx) * update, clamped to [0, 1]
       g. Store clean prediction
       h. Pushforward noise (training only)
       i. Teacher forcing
    3. Return {"output_grid": (B, 1, nt, nx)}

Input (via ToGridNoCoords transform):
    - grid_input: (B, 1, nt, nx) -- masked IC on grid (only t=0 used)
    - dt: (B,) -- time step scalar
    - dx: (B,) -- spatial step scalar

Output: {"output_grid": (B, 1, nt, nx)}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base.flux import DEFAULT_FLUX, Flux


class DifferentiableShockProximity(nn.Module):
    """Compute shock proximity from the current state using Lax entropy condition.

    Detects shocks at cell interfaces where the Lax entropy condition holds
    (char_L > s > char_R) and computes an exponential proximity field.

    Args:
        flux: Flux function instance.
        sigma: Length scale for proximity decay.
    """

    def __init__(self, flux: Flux, sigma: float = 0.05):
        super().__init__()
        self.flux = flux
        self.sigma = sigma

    def forward(self, state: torch.Tensor, dx: float) -> torch.Tensor:
        """Compute shock proximity field.

        Args:
            state: (B, 1, nx) current density field.
            dx: Spatial step size (scalar).

        Returns:
            (B, 1, nx) detached shock proximity in [0, 1].
        """
        with torch.no_grad():
            B, _, nx = state.shape

            rho_L = state[:, :, :-1]  # (B, 1, nx-1)
            rho_R = state[:, :, 1:]  # (B, 1, nx-1)

            char_L = self.flux.derivative(rho_L)  # (B, 1, nx-1)
            char_R = self.flux.derivative(rho_R)  # (B, 1, nx-1)
            s = self.flux.shock_speed(rho_L, rho_R)  # (B, 1, nx-1)

            # Lax entropy condition: char_L > s > char_R
            is_shock = (char_L > s) & (s > char_R)  # (B, 1, nx-1)

            # Cell center positions: 0.5*dx, 1.5*dx, ..., (nx-0.5)*dx
            cell_centers = torch.arange(nx, device=state.device, dtype=state.dtype)
            cell_centers = (cell_centers + 0.5) * dx  # (nx,)

            # Interface positions: dx, 2*dx, ..., (nx-1)*dx
            interface_pos = torch.arange(
                1, nx, device=state.device, dtype=state.dtype
            )
            interface_pos = interface_pos * dx  # (nx-1,)

            # Distance from each cell center to each interface
            # cell_centers: (nx,), interface_pos: (nx-1,) -> (nx, nx-1)
            dist = (cell_centers.unsqueeze(1) - interface_pos.unsqueeze(0)).abs()

            # Mask non-shock interfaces with large value
            large_val = 1e6
            # is_shock: (B, 1, nx-1) -> expand for broadcasting with (nx, nx-1)
            shock_mask = is_shock.squeeze(1)  # (B, nx-1)

            # dist: (nx, nx-1) -> (1, nx, nx-1)
            dist = dist.unsqueeze(0).expand(B, -1, -1)  # (B, nx, nx-1)
            masked_dist = torch.where(
                shock_mask.unsqueeze(1).expand_as(dist),
                dist,
                torch.full_like(dist, large_val),
            )

            # Min distance to nearest shock
            min_dist = masked_dist.min(dim=2).values  # (B, nx)

            # Check if any shocks exist per sample
            any_shock = shock_mask.any(dim=1, keepdim=True)  # (B, 1)

            proximity = torch.exp(-min_dist / self.sigma)  # (B, nx)
            proximity = proximity * any_shock.float()  # zero if no shocks

            return proximity.unsqueeze(1)  # (B, 1, nx)


class FluxNetwork(nn.Module):
    """Pointwise MLP as Conv1d(kernel_size=1) layers.

    Shared across all cells (translation invariant). Final layer initialized
    to zero for small initial updates.

    Args:
        in_features: Number of input features per cell.
        hidden_dim: Hidden layer width.
        n_layers: Total number of layers (including input and output).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.05,
    ):
        super().__init__()
        layers = [nn.Conv1d(in_features, hidden_dim, 1)]
        for _ in range(n_layers - 1):
            layers.extend([nn.GELU(), nn.Dropout(dropout), nn.Conv1d(hidden_dim, hidden_dim, 1)])
        layers.extend([nn.GELU(), nn.Conv1d(hidden_dim, 1, 1)])
        self.net = nn.Sequential(*layers)

        # Zero-init final layer for small initial updates
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, in_features, nx) input features.

        Returns:
            (B, 1, nx) update field.
        """
        return self.net(x)


class NeuralFVSolver(nn.Module):
    """Learned finite volume solver with autoregressive time-stepping.

    Args:
        stencil_k: Half-width of stencil (full width = 2k+1).
        flux_hidden_dim: Hidden dimension of flux network.
        flux_n_layers: Number of layers in flux network.
        dx: Spatial step size.
        proximity_sigma: Length scale for shock proximity decay.
        dropout: Dropout probability.
        flux: Flux function instance.
    """

    def __init__(
        self,
        stencil_k: int = 3,
        flux_hidden_dim: int = 64,
        flux_n_layers: int = 3,
        dx: float = 0.02,
        proximity_sigma: float = 0.05,
        dropout: float = 0.05,
        flux: Flux | None = None,
    ):
        super().__init__()
        self.stencil_k = stencil_k
        self.dx = dx
        flux = flux or DEFAULT_FLUX()
        stencil_size = 2 * stencil_k + 1

        # in_features = stencil * 3 (value + char_speed + proximity) + 1 (dt)
        in_features = stencil_size * 3 + 1

        self.shock_proximity = DifferentiableShockProximity(
            flux=flux, sigma=proximity_sigma
        )
        self.flux_fn = flux
        self.flux_net = FluxNetwork(
            in_features=in_features,
            hidden_dim=flux_hidden_dim,
            n_layers=flux_n_layers,
            dropout=dropout,
        )

        # Noise and teacher forcing (set by training orchestrator)
        self.noise_std: float = 0.0
        self.teacher_forcing_ratio: float = 0.0

    def forward(
        self,
        batch_input: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            batch_input: Dict containing:
                - 'grid_input': (B, 1, nt, nx) masked IC grid
                - 'dt': (B,) time step scalar
                - 'dx': (B,) spatial step scalar

        Returns:
            Dict containing:
                - 'output_grid': (B, 1, nt, nx) predicted density
        """
        grid_input = batch_input["grid_input"]  # (B, 1, nt, nx)
        dt = batch_input["dt"]  # (B,)

        B, _, nt, nx = grid_input.shape
        k = self.stencil_k
        dx = self.dx

        # Extract IC
        state = grid_input[:, :, 0, :]  # (B, 1, nx)

        tf_ratio = self.teacher_forcing_ratio
        target_grid = batch_input.get("target_grid") if tf_ratio > 0 else None

        # dt as a broadcast channel: (B, 1, 1) for feature stacking
        dt_val = dt[:, None, None]  # (B, 1, 1)

        state_list = [state]

        for t in range(nt - 1):
            # a. Ghost cell boundary + stencil extraction
            padded = F.pad(state, (k, k), mode="replicate")  # (B, 1, nx+2k)
            # unfold: extract sliding windows of size 2k+1
            stencil_vals = padded.unfold(2, 2 * k + 1, 1)  # (B, 1, nx, 2k+1)
            stencil_vals = stencil_vals.squeeze(1).permute(0, 2, 1)  # (B, 2k+1, nx)

            # b. Characteristic speeds at stencil positions
            char_speeds = self.flux_fn.derivative(stencil_vals)  # (B, 2k+1, nx)

            # c. Shock proximity (detached)
            prox = self.shock_proximity(state, dx)  # (B, 1, nx)
            prox_padded = F.pad(prox, (k, k), mode="replicate")  # (B, 1, nx+2k)
            stencil_prox = prox_padded.unfold(2, 2 * k + 1, 1)  # (B, 1, nx, 2k+1)
            stencil_prox = stencil_prox.squeeze(1).permute(0, 2, 1)  # (B, 2k+1, nx)

            # d. Stack features: [stencil_vals, char_speeds, stencil_prox, dt]
            dt_channel = dt_val.expand(B, 1, nx)  # (B, 1, nx)
            features = torch.cat(
                [stencil_vals, char_speeds, stencil_prox, dt_channel], dim=1
            )  # (B, 3*(2k+1)+1, nx)

            # e. Flux network
            update = self.flux_net(features)  # (B, 1, nx)

            # f. Euler step
            state = (state + (dt_val / dx) * update).clamp(0.0, 1.0)

            # g. Store clean prediction
            state_list.append(state)

            # h. Pushforward noise (training only)
            if self.training and self.noise_std > 0:
                state = (state + torch.randn_like(state) * self.noise_std).clamp(0.0, 1.0)

            # i. Teacher forcing
            if target_grid is not None and torch.rand(1).item() < tf_ratio:
                state = target_grid[:, :, t + 1, :]

        output_grid = torch.stack(state_list, dim=2)  # (B, 1, nt, nx)

        return {"output_grid": output_grid}


def build_neural_fv_solver(args: dict) -> NeuralFVSolver:
    """Factory function for NeuralFVSolver.

    Args:
        args: Dict or Namespace with model configuration.

    Returns:
        Configured NeuralFVSolver instance.
    """
    if not isinstance(args, dict):
        args = vars(args)

    return NeuralFVSolver(
        stencil_k=args.get("stencil_k", 3),
        flux_hidden_dim=args.get("flux_hidden_dim", 64),
        flux_n_layers=args.get("flux_n_layers", 3),
        dx=args.get("dx", 0.02),
        proximity_sigma=args.get("proximity_sigma", 0.05),
        dropout=args.get("dropout", 0.05),
    )
