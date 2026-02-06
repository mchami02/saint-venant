# Wavefront Learning - Directory Structure

```
wavefront_learning/
├── ARCHITECTURE.md        # Detailed model and loss architecture documentation
├── CLAUDE.md              # Claude Code guidance for this module
├── Structure.md           # This file - folder structure documentation
├── train.py               # Main training script (standalone)
├── test.py                # Testing/evaluation script (standalone)
├── data.py                # Dataset classes, transforms, get_wavefront_datasets()
├── data_loading.py        # HuggingFace upload/download utilities
├── data_processing.py     # Grid generation, discontinuity extraction, preprocessing
├── model.py               # Model factory and registry
├── loss.py                # Loss function factory, CombinedLoss, presets
├── logger.py              # W&B logging utilities
├── metrics.py             # Shared metrics utilities (MSE, MAE, rel_l2, max_error)
├── plotter.py             # Plotting factory (PLOTS, PLOT_PRESETS, plot_wandb)
├── plotting/              # Visualization subpackage
│   ├── __init__.py        # Re-exports all public plotting functions
│   ├── base.py            # Common setup, helpers (save_figure, _get_extent, etc.)
│   ├── grid_plots.py      # Grid comparison plots + plot_ground_truth_wandb
│   ├── trajectory_plots.py # Core trajectory functions (plot_shock_trajectories, etc.)
│   ├── wandb_trajectory_plots.py # W&B trajectory plots for PLOTS registry
│   └── hybrid_plots.py    # HybridDeepONet plots (plot_region_weights_wandb, etc.)
├── models/
│   ├── __init__.py
│   ├── base.py                    # BaseWavefrontModel abstract class
│   ├── shock_trajectory_net.py    # ShockTrajectoryNet (DeepONet-like)
│   ├── region_trunk.py            # SpaceTimeEncoder, RegionTrunk, RegionTrunkSet
│   ├── hybrid_deeponet.py         # HybridDeepONet (trajectory + grid prediction)
│   └── traj_deeponet.py           # TrajDeepONet (trajectory-conditioned single trunk)
├── losses/
│   ├── __init__.py                # Package exports all loss classes
│   ├── base.py                    # BaseLoss abstract class (unified interface)
│   ├── flux.py                    # Centralized flux functions (greenshields_flux, etc.)
│   ├── mse.py                     # MSELoss for grid predictions
│   ├── ic.py                      # ICLoss for initial condition matching
│   ├── trajectory_consistency.py  # TrajectoryConsistencyLoss (RH trajectory)
│   ├── boundary.py                # BoundaryLoss (shocks outside domain)
│   ├── collision.py               # CollisionLoss (shock merging)
│   ├── existence_regularization.py # ICAnchoringLoss
│   ├── supervised_trajectory.py   # SupervisedTrajectoryLoss (when GT available)
│   ├── pde_residual.py            # PDEResidualLoss + PDEShockResidualLoss
│   ├── rh_residual.py             # RHResidualLoss (RH from sampled densities)
│   ├── acceleration.py            # AccelerationLoss (shock detection via acceleration)
│   └── regularize_traj.py        # RegularizeTrajLoss (penalize erratic trajectory jumps)
└── wandb/                 # W&B run logs (gitignored)
```

## Key Files

### Entry Points
- **train.py**: `uv run python train.py --model ShockNet --loss shock_net --epochs 100`
- **train.py**: `uv run python train.py --model HybridDeepONet --loss hybrid --epochs 100`
- **test.py**: `uv run python test.py --model_path wavefront_model.pth`

### Data Pipeline
- **data.py**: Public interface - `WavefrontDataset`, transforms, `collate_wavefront_batch`, `get_wavefront_datasets()`
- **data_processing.py**: Internal - grid generation with Lax-Hopf solver, discontinuity extraction from discretized ICs, preprocessing
- **data_loading.py**: Internal - HuggingFace upload/download for caching grids (only grids stored, ICs extracted from grid[:, 0, :])

### Models
- **ShockNet** (`models/shock_trajectory_net.py`): DeepONet-like architecture for shock trajectory prediction
  - Branch: Transformer-based discontinuity encoder
  - Trunk: Fourier feature time encoder
  - Output: `{positions: (B, D, T), existence: (B, D, T)}`

- **HybridDeepONet** (`models/hybrid_deeponet.py`): Combined trajectory and density prediction
  - Shared Branch: DiscontinuityEncoder (transformer-based)
  - Trajectory Trunk: TimeEncoder + TrajectoryDecoder
  - Region Trunks: K = max_disc + 1 RegionTrunks with shared SpaceTimeEncoder
  - Grid Assembly: Soft region assignment using sigmoid boundaries
  - Output: `{positions, existence, output_grid, region_densities, region_weights}`

- **TrajDeepONet** (`models/traj_deeponet.py`): Trajectory-conditioned single trunk
  - Branch: DiscontinuityEncoder
  - Trajectory: TimeEncoder + PositionDecoder (no existence head)
  - Single BoundaryConditionedTrunk: takes (t, x, x_left, x_right) + branch
  - No GridAssembler: trunk directly outputs density
  - Output: `{positions, output_grid}`

### Model Components
- **SpaceTimeEncoder** (`models/region_trunk.py`): Encodes (t, x) coordinates using Fourier features
- **RegionTrunk** (`models/region_trunk.py`): Predicts density in a specific region
- **RegionTrunkSet** (`models/region_trunk.py`): Set of K region trunks
- **GridAssembler** (`models/hybrid_deeponet.py`): Assembles grid from region predictions using soft boundaries

### Losses

All losses follow a unified interface:
```python
def forward(
    self,
    input_dict: dict[str, torch.Tensor],
    output_dict: dict[str, torch.Tensor],
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
```

#### Individual Losses (one file per loss)
| File | Class | Purpose |
|------|-------|---------|
| `flux.py` | - | `greenshields_flux`, `greenshields_flux_derivative`, `compute_shock_speed` |
| `mse.py` | `MSELoss` | Grid MSE loss |
| `ic.py` | `ICLoss` | Initial condition matching at t=0 |
| `trajectory_consistency.py` | `TrajectoryConsistencyLoss` | Match predicted to analytical RH trajectory |
| `boundary.py` | `BoundaryLoss` | Penalize shocks outside domain |
| `collision.py` | `CollisionLoss` | Penalize colliding shocks |
| `existence_regularization.py` | `ICAnchoringLoss` | Anchor trajectories to IC positions |
| `supervised_trajectory.py` | `SupervisedTrajectoryLoss` | Supervised trajectory (when GT available) |
| `pde_residual.py` | `PDEResidualLoss`, `PDEShockResidualLoss` | PDE conservation in smooth regions; GT residual penalizing unpredicted shocks |
| `rh_residual.py` | `RHResidualLoss` | RH residual from sampled region densities |
| `acceleration.py` | `AccelerationLoss` | Shock detection via high acceleration + missed shock term |
| `regularize_traj.py` | `RegularizeTrajLoss` | Penalize erratic jumps between consecutive timesteps |

#### CombinedLoss and Presets (`loss.py`)
```python
# Available presets
LOSS_PRESETS = {
    "shock_net": [  # For trajectory-only models
        ("rh_residual", 1.0, {"mode": "gt"}),
        ("boundary", 1.0),
        ("acceleration", 1.0, {"missed_shock_weight": 1.0}),
        ("existence_reg", 0.1),
    ],
    "hybrid": [  # For HybridDeepONet
        ("mse", 1.0),
        ("rh_residual", 1.0),
        ("pde_residual", 0.1),
        ("ic", 10.0),
        ("existence_reg", 0.01),
    ],
}

# Usage
loss = get_loss("shock_net")  # Use preset
loss = get_loss("mse")  # Use individual loss
loss = get_loss("hybrid", loss_kwargs={  # Customize preset
    "pde_residual": {"dt": 0.004, "dx": 0.02},
})
```

### Plotting

The plotting system consists of two parts:
1. **`plotter.py`** - Thin facade with preset system (similar to loss.py)
2. **`plotting/`** - Individual plot functions organized by category

#### `plotter.py` - Plotting Factory

The main plotting module provides a preset-based system similar to `loss.py`. It imports all plot functions from `plotting/` and exposes them via the `PLOTS` registry.

```python
# Available presets
PLOT_PRESETS = {
    "shock_net": [        # For trajectory-only models
        "grid_with_trajectory",
        "grid_with_acceleration",
    ],
    "hybrid": [           # For HybridDeepONet
        "prediction_with_trajectory",
        "mse_error",
        "trajectory_vs_analytical",
        "existence",
        "region_weights",
    ],
}

# Usage in train.py
plot_wandb(traj_data, grid_config, logger, epoch, mode="val", preset=args.plot)
```

| Plot Function | Description | Required Data |
|---------------|-------------|---------------|
| `ground_truth` | GT grid heatmap | `grids` |
| `grid_with_trajectory` | GT grid + predicted trajectories | `grids`, `positions`, `existence`, `masks`, `times` |
| `grid_with_acceleration` | GT grid + trajectories alongside |a| grid | `grids`, `positions`, `existence`, `masks`, `times` |
| `trajectory_vs_analytical` | Predicted vs RH trajectories | `positions`, `existence`, `discontinuities`, `masks`, `times` |
| `existence` | Existence probability heatmap | `existence`, `masks`, `times` |
| `prediction_with_trajectory` | Predicted grid + trajectories | `output_grid`, `positions`, `existence`, `masks`, `times` |
| `mse_error` | MSE error heatmap | `output_grid`, `grids` |
| `region_weights` | Region assignment visualization | `region_weights`, `positions`, `existence`, `masks`, `times` |

Command-line usage:
```bash
# Auto-detect preset based on model
uv run python train.py --model ShockNet --epochs 10

# Explicit preset
uv run python train.py --model ShockNet --plot shock_net --epochs 10
uv run python train.py --model HybridDeepONet --plot hybrid --epochs 10
```

#### `plotting/base.py` - Common Utilities
- `save_figure(fig, path, dpi)` - Save figure to file
- `_get_extent(nx, nt, dx, dt)` - Compute extent for imshow
- `_get_colors(n)` - Get colormap array for n items
- `_plot_heatmap(ax, data, extent, ...)` - Reusable heatmap plotting
- `_create_comparison_animation(gt, pred, ...)` - Animated comparison GIF
- `_log_figure(logger, key, fig, epoch, use_summary)` - Unified W&B logging

#### `plotting/grid_plots.py` - Grid Comparison
All grid plotting functions accept a `grid_config: dict` with keys `{nx, nt, dx, dt}`.

- `plot_prediction_comparison(gt, pred, grid_config, title)` - GT vs prediction heatmaps
- `plot_error_map(gt, pred, grid_config)` - Error heatmap
- `plot_comparison_wandb(gt, pred, grid_config, logger, epoch, mode)` - W&B grid comparison + animations
- `plot_grid_comparison(gt, pred, positions, existence, times, grid_config, sample_idx)` - GT vs prediction with trajectory overlay
- `plot_ground_truth_wandb(traj_data, grid_config, logger, epoch, mode)` - **PLOTS registry**: GT grid heatmaps

#### `plotting/trajectory_plots.py` - Core Trajectory Visualization

Core utility functions (not in PLOTS registry):
- `plot_shock_trajectories(positions, existence, discontinuities, mask, times, ...)` - Trajectories vs analytical RH
- `plot_existence_heatmap(existence, mask, times, sample_idx)` - Existence probability heatmap
- `plot_trajectory_on_grid(grid, positions, existence, discontinuities, mask, times, grid_config, ...)` - Trajectory overlay on solution
- `plot_wavefront_trajectory(prediction, wavefront_positions, ...)` - Wavefront detection plot
- `plot_loss_curves(train_losses, val_losses, title)` - Training progress
- `plot_sample_predictions(model, dataloader, device, num_samples, grid_config)` - Batch sample plots
- `_compute_acceleration_numpy(density, dt)` - Compute temporal acceleration via central difference

#### `plotting/wandb_trajectory_plots.py` - W&B Trajectory Plots

Functions with W&B suffix accept `traj_data: dict` and `grid_config: dict` for simplified signatures.
- `traj_data` keys: `{grids, positions, existence, discontinuities, masks, times}`

**PLOTS registry functions** (compatible with plotter.py):
- `plot_grid_with_trajectory_wandb(...)` - GT grid + predicted trajectory overlay
- `plot_grid_with_acceleration_wandb(...)` - Two-column: GT grid+trajectories | acceleration magnitude
- `plot_trajectory_vs_analytical_wandb(...)` - Predicted vs RH trajectories
- `plot_existence_wandb(...)` - Existence probability heatmap

Other W&B trajectory functions:
- `plot_trajectory_on_grid_wandb(traj_data, grid_config, logger, epoch, mode)` - W&B trajectory on grid
- `plot_trajectory_wandb(traj_data, logger, epoch, mode)` - W&B trajectory plots

#### `plotting/hybrid_plots.py` - HybridDeepONet Visualization

**PLOTS registry functions** (compatible with plotter.py):
- `plot_prediction_with_trajectory_wandb(...)` - Predicted grid + trajectory overlay
- `plot_mse_error_wandb(...)` - MSE error heatmap
- `plot_region_weights_wandb(...)` - Region weight visualization

Other utility functions:
- `plot_hybrid_predictions_wandb(traj_data, grid_config, logger, epoch, mode)` - Comprehensive hybrid visualization
  - `traj_data` keys: `{grids, output_grid, positions, existence, discontinuities, masks, region_densities, region_weights, times}`
  - Logs `{mode}/hybrid_summary`: B rows x 3 cols (GT, Pred+traj, MSE Error)
  - Logs `test/hybrid_comparison_table`: W&B table with Sample, GT, Pred, MSE, Region columns (first 5 samples)

## Logging

### Sanity Check

Before training starts, `run_sanity_check()` verifies all code paths:
1. **[1/4] Forward pass on training batch** - Logs output shapes
2. **[2/4] Forward pass on validation batch** - Verifies consistency
3. **[3/4] Loss computation** - Tests loss function with model output
4. **[4/4] Backward pass** - Verifies gradients are computed

### wandb.watch()

After W&B initialization, `wandb.watch(model, log="all", log_freq=100)` is called to track:
- Model parameter histograms
- Gradient histograms (every 100 forward passes)

View in W&B dashboard under the "Gradients" section.

### Profiler

Run with `--profile` flag to profile training steps before the main loop:
```bash
uv run python train.py --model HybridDeepONet --epochs 1 --profile
```

The profiler:
- Runs warmup steps (default: 2) followed by active profiling steps (default: 10)
- Records CPU and CUDA (if available) activities
- Uploads artifacts to W&B:
  - `trace.json`: Chrome trace for visualization (chrome://tracing)
  - `summary.txt`: Summary table sorted by time

### Metrics Hierarchy

Training logs to W&B with the following metrics hierarchy:

### Loss Components
- `train/loss/*`: Training loss components (trajectory, boundary, collision, etc.)
- `val/loss/*`: Validation loss components

### Grid Metrics (for models with grid output)
- `train/metrics/{mse,mae,rel_l2,max_error}`: Training grid metrics
- `val/metrics/{mse,mae,rel_l2,max_error}`: Validation grid metrics
- `test/metrics/{mse,mae,rel_l2,max_error}`: Test grid metrics

**Note**: Grid metrics are only logged for models that produce grid outputs (e.g., HybridDeepONet). Trajectory-only models (e.g., ShockNet) do not log grid metrics.

### Visualizations
- For ShockNet:
  - `train/trajectory_on_grid_sample_*`: Trajectory overlay on grid (every 5 epochs)
  - `val/trajectory_on_grid_sample_*`: Validation trajectory overlay (every 5 epochs)
- For HybridDeepONet:
  - `{mode}/hybrid_summary`: 3-column summary (GT, Pred+traj, MSE Error)
  - `test/hybrid_comparison_table`: W&B table with GT, Pred, MSE, and Region columns

## Architecture Details

### HybridDeepONet Architecture

```
Input: discontinuities (B, D, 3), t_coords (B, nt), x_coords (B, nx)

┌─────────────────────────────────────────────────────────┐
│                    SHARED BRANCH                        │
│  DiscontinuityEncoder: (B, D, 3) → (B, D, hidden)       │
│  + Pooled branch: masked mean → (B, hidden)             │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
┌───────────────────┐    ┌────────────────────────────────┐
│  TRAJECTORY TRUNK │    │       REGION TRUNKS (K=D+1)    │
│  TimeEncoder      │    │  SpaceTimeEncoder: (t,x)→emb   │
│  TrajectoryDecoder│    │  K RegionTrunks: branch+coord  │
│                   │    │                 → density       │
│  Output:          │    │  Output:                        │
│  - positions(B,D,T)│   │  - region_densities(B,K,nt,nx) │
│  - existence(B,D,T)│   │                                 │
└─────────┬─────────┘    └────────────────┬───────────────┘
          │                               │
          └───────────┬───────────────────┘
                      ▼
         ┌────────────────────────────────┐
         │        GRID ASSEMBLY           │
         │  Soft region assignment using  │
         │  sigmoid boundaries at shocks  │
         │  → output_grid (B, 1, nt, nx)  │
         └────────────────────────────────┘
```

### Loss Computation

**Rankine-Hugoniot Residual (Corrected)**:
```
R_RH(t) = ẋ_s(t) · (u⁺ - u⁻) - (f(u⁺) - f(u⁻))
```
- `ẋ_s(t)` = shock velocity from finite difference of positions
- `u⁻` = density left of shock (from region trunk d)
- `u⁺` = density right of shock (from region trunk d+1)
- `f(u) = u(1-u)` = Greenshields flux

**PDE Residual** (smooth regions):
```
R_PDE = ∂ρ/∂t + ∂f(ρ)/∂x
```
- Computed with central finite differences
- Points within shock_buffer of predicted shocks are excluded

## Usage Examples

### Training ShockNet (trajectory only)
```bash
cd wavefront_learning
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python train.py \
  --model ShockNet --loss shock_net --epochs 100
```

### Training HybridDeepONet (trajectory + grid)
```bash
cd wavefront_learning
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python train.py \
  --model HybridDeepONet --loss hybrid --epochs 100 --n_samples 1000
```

### Testing
```bash
cd wavefront_learning
uv run python test.py --model_path wavefront_model.pth
```
