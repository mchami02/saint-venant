# CLAUDE.md - Wavefront Learning Module

This file provides guidance for Claude Code when working with the `wavefront_learning/` directory.

**IMPORTANT**: After completing any task that modifies the folder structure (adding/removing files), update `structure.md` to reflect the changes.

## Module Purpose

This module implements neural network-based learning of wavefront dynamics for 1D conservation laws (specifically LWR traffic flow / shallow water equations). Unlike the `operator_learning/` module which uses full discretized initial conditions, this module uses **exact discontinuity points** as input.

## Key Concept: Discontinuity-Based Input

The main distinction from `operator_learning/`:
- **Input**: Exact breakpoints (`xs`) and piece values (`ks`) of piecewise constant initial conditions
- **Output**: Shock trajectories (positions and existence over time) OR full space-time solution grid `(nt, nx)`

This compact representation allows the model to learn directly from the mathematical description of discontinuities rather than discretized approximations.

## Runnable Scripts

Only two scripts are meant to be run standalone:

```bash
# Training ShockNet (trajectory prediction)
cd wavefront_learning
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python train.py --model ShockNet --epochs 100

# Testing (requires trained model)
uv run python test.py --model_path wavefront_model.pth
```

**Note**: `PYTORCH_ENABLE_MPS_FALLBACK=1` is required on Apple Silicon due to transformer operations not yet supported on MPS.

## Data Format

The `WavefrontDataset` returns `(input_data, target_grid)` where:
- `input_data` is a dictionary containing:
  - `discontinuities`: `(max_disc, 3)` tensor with `[x, left_val, right_val]`
  - `disc_mask`: `(max_disc,)` mask for valid discontinuities
  - `xs`: `(max_pieces + 1,)` breakpoint positions
  - `ks`: `(max_pieces,)` piece values
  - `pieces_mask`: `(max_pieces,)` mask for valid pieces
  - `t_coords`, `x_coords`: coordinate grids
  - `dx`, `dt`: grid spacing scalars
- `target_grid`: `(1, nt, nx)` solution tensor

## Experiment Tracking

All experiments are logged to **Weights & Biases** (wandb). Logged metrics include:
- Training/validation loss and metrics (MSE, MAE, relative L2, max error)
- Learning rate
- Comparison plots (ground truth vs prediction heatmaps)
- Animated GIFs showing time evolution
- Final model artifacts

Use `--no_wandb` flag to disable logging.

## Available Transforms

- `FlattenDiscontinuitiesTransform`: Converts dict input to flat tensor for simple MLPs
- `ToGridInputTransform`: Reconstructs discretized IC from discontinuities (for grid-based models like FNO)

## Implementation Status

**Fully Implemented:**
- `data.py` - Full data pipeline with discontinuity extraction
- `logger.py` - W&B logging utilities
- `plotter.py` - Visualization (grid comparisons + trajectory plots)
- `train.py` - Training loop with early stopping, LR scheduling, periodic plotting
- `test.py` - Evaluation with metrics and visualization
- `model.py` - Model factory with ShockNet registered
- `loss.py` - Loss factory with RankineHugoniotLoss registered
- `models/shock_trajectory_net.py` - DeepONet-like trajectory prediction model
- `losses/rankine_hugoniot.py` - Physics-based unsupervised loss

## Available Models

| Model | Description | Output |
|-------|-------------|--------|
| **ShockNet** | DeepONet-like architecture with transformer branch | `{positions, existence}` trajectories |

## Available Losses

| Loss | Description | Type |
|------|-------------|------|
| **rankine_hugoniot** | Enforces Rankine-Hugoniot jump conditions | Unsupervised physics |

## Adding a New Model

1. Implement your model class inheriting from `BaseWavefrontModel` in `models/`
2. Register it in `model.py`:
```python
from models.my_model import MyModel
MODELS = {
    "my_model": MyModel,
}
```

## Dependencies

Uses the same environment as `operator_learning/` with:
- PyTorch for models
- `nfv` package for data generation (Lax-Hopf solver)
- Weights & Biases for logging
- matplotlib for plotting
