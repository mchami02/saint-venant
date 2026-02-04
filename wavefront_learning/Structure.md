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
├── loss.py                # Loss function factory and registry
├── logger.py              # W&B logging utilities
├── metrics.py             # Shared metrics utilities (MSE, MAE, rel_l2, max_error)
├── plotter.py             # Backwards-compat shim (imports from plotting/)
├── plotting/              # Visualization subpackage
│   ├── __init__.py        # Re-exports all public plotting functions
│   ├── base.py            # Common setup, helpers (save_figure, _get_extent, etc.)
│   ├── grid_plots.py      # Grid comparison plots (plot_prediction_comparison, etc.)
│   ├── trajectory_plots.py # Trajectory visualization (plot_shock_trajectories, etc.)
│   └── hybrid_plots.py    # HybridDeepONet plots (plot_region_assignments, etc.)
├── models/
│   ├── __init__.py
│   ├── base.py                    # BaseWavefrontModel abstract class
│   ├── shock_trajectory_net.py    # ShockTrajectoryNet (DeepONet-like)
│   ├── region_trunk.py            # SpaceTimeEncoder, RegionTrunk, RegionTrunkSet
│   └── hybrid_deeponet.py         # HybridDeepONet (trajectory + grid prediction)
├── losses/
│   ├── __init__.py
│   ├── base.py                    # BaseLoss abstract class
│   ├── rankine_hugoniot.py        # Rankine-Hugoniot physics loss
│   ├── pde_residual.py            # PDE residual loss for smooth regions
│   └── hybrid_loss.py             # Combined loss for HybridDeepONet
└── wandb/                 # W&B run logs (gitignored)
```

## Key Files

### Entry Points
- **train.py**: `uv run python train.py --model ShockNet --epochs 100`
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

### Model Components
- **SpaceTimeEncoder** (`models/region_trunk.py`): Encodes (t, x) coordinates using Fourier features
- **RegionTrunk** (`models/region_trunk.py`): Predicts density in a specific region
- **RegionTrunkSet** (`models/region_trunk.py`): Set of K region trunks
- **GridAssembler** (`models/hybrid_deeponet.py`): Assembles grid from region predictions using soft boundaries

### Losses
- **RankineHugoniotLoss** (`losses/rankine_hugoniot.py`): Physics-based unsupervised loss
  - Trajectory consistency (Rankine-Hugoniot condition)
  - Boundary loss (shocks exit domain)
  - Collision loss (shock merging)
  - Existence regularization

- **HybridDeepONetLoss** (`losses/hybrid_loss.py`): Combined loss for HybridDeepONet
  - Grid MSE: Match output_grid to target
  - RH Residual (CORRECTED): R_RH = ẋ_s(u+ - u-) - (f(u+) - f(u-))
  - PDE Residual: Enforce conservation in smooth regions
  - Existence Regularization

- **PDEResidualLoss** (`losses/pde_residual.py`): Conservation law loss
  - Computes ∂ρ/∂t + ∂f/∂x = 0 using central finite differences
  - Shock masking to exclude points near predicted discontinuities

### Plotting

The `plotting/` subpackage provides visualization utilities split into focused modules:

#### `plotting/base.py` - Common Utilities
- `save_figure(fig, path, dpi)` - Save figure to file
- `_get_extent(nx, nt, dx, dt)` - Compute extent for imshow
- `_get_colors(n)` - Get colormap array for n items
- `_plot_heatmap(ax, data, extent, ...)` - Reusable heatmap plotting
- `_create_comparison_animation(gt, pred, ...)` - Animated comparison GIF
- `_log_figure(logger, key, fig, epoch, use_summary)` - Unified W&B logging

#### `plotting/grid_plots.py` - Grid Comparison
- `plot_prediction_comparison(gt, pred, nx, nt, dx, dt, title)` - GT vs prediction heatmaps
- `plot_error_map(gt, pred, nx, nt, dx, dt)` - Error heatmap
- `plot_comparison_wandb(gt, pred, ..., logger, epoch, mode)` - W&B grid comparison + animations
- `plot_grid_comparison(gt, pred, positions, existence, times, ...)` - GT vs prediction with trajectory overlay

#### `plotting/trajectory_plots.py` - Trajectory Visualization
- `plot_shock_trajectories(positions, existence, discontinuities, mask, times, ...)` - Trajectories vs analytical RH
- `plot_existence_heatmap(existence, mask, times, sample_idx)` - Existence probability heatmap
- `plot_trajectory_on_grid(grid, positions, existence, ...)` - Trajectory overlay on solution
- `plot_trajectory_on_grid_wandb(grids, positions, ..., logger, epoch)` - W&B trajectory on grid
- `plot_trajectory_wandb(positions, existence, ..., logger, epoch)` - W&B trajectory plots
- `plot_wavefront_trajectory(prediction, wavefront_positions, ...)` - Wavefront detection plot
- `plot_loss_curves(train_losses, val_losses, title)` - Training progress
- `plot_sample_predictions(model, dataloader, device, num_samples, ...)` - Batch sample plots

#### `plotting/hybrid_plots.py` - HybridDeepONet Visualization
- `plot_hybrid_predictions_wandb(ground_truths, predictions, ..., logger, epoch)` - Comprehensive hybrid visualization
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
  --model ShockNet --loss rankine_hugoniot --epochs 100
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
