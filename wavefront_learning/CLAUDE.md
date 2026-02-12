# CLAUDE.md - Wavefront Learning Module

This file provides guidance for Claude Code when working with the `wavefront_learning/` directory.

**IMPORTANT**: Whenever a file is created, deleted, or moved in this directory, you MUST update `Structure.md`:
- Add/remove the file in the directory tree
- Include a short description of what the file contains
- List the public classes and functions available for other files to import
- This applies to ALL file changes, not just structural reorganizations

**IMPORTANT**: After modifying models or losses (adding new ones, changing architecture, or modifying loss formulas), update `ARCHITECTURE.md` with:
- Architecture diagrams and component descriptions for models
- Mathematical formulas and loss component definitions for losses
- Default weights and configuration parameters
- **No ASCII art**: Use compact pseudocode (input sizes → component → output sizes) instead of ASCII box drawings
- **No batch dimension**: Omit the batch dimension `B` from tensor shapes in pseudocode

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
- `plotter.py` - Backwards-compat shim (imports from `plotting/`)
- `plotting/` - Visualization subpackage (grid comparisons + trajectory plots)
- `train.py` - Training loop with early stopping, LR scheduling, periodic plotting
- `test.py` - Evaluation with metrics and visualization
- `model.py` - Model factory with ShockNet registered
- `loss.py` - Loss factory with RankineHugoniotLoss registered
- `models/shock_trajectory_net.py` - DeepONet-like trajectory prediction model
- `losses/rankine_hugoniot.py` - Physics-based unsupervised loss

## Available Models

| Model | Description | Output |
|-------|-------------|--------|
| **ShockNet** | DeepONet-like architecture for trajectory prediction | `{positions, existence}` |
| **HybridDeepONet** | Trajectory + region trunks + grid assembly | `{positions, existence, output_grid, region_densities, region_weights}` |
| **TrajDeepONet** | Boundary-conditioned single trunk (bilinear fusion) | `{positions, output_grid}` |
| **ClassifierTrajDeepONet** | TrajDeepONet + shock/rarefaction classifier | `{positions, existence, output_grid}` |
| **NoTrajDeepONet** | TrajDeepONet without trajectory prediction | `{output_grid}` |
| **TrajTransformer** | Cross-attention variant of TrajDeepONet | `{positions, output_grid}` |
| **ClassifierTrajTransformer** | TrajTransformer + shock/rarefaction classifier | `{positions, existence, output_grid}` |
| **DeepONet** | Classic branch-trunk dot product baseline | `{output_grid}` |
| **FNO** | Fourier Neural Operator baseline (neuralop wrapper) | `{output_grid}` |
| **EncoderDecoder** | Transformer encoder + axial attention decoder | `{output_grid}` |
| **EncoderDecoderCross** | Transformer encoder + cross-attention decoder | `{output_grid}` |

## Available Losses

Individual losses (in `losses/`):
| Loss | Description |
|------|-------------|
| `mse` | Grid MSE loss |
| `ic` | Initial condition matching |
| `trajectory` | Trajectory consistency (analytical RH) |
| `boundary` | Penalize shocks outside domain |
| `collision` | Penalize colliding shocks |
| `ic_anchoring` | Anchor trajectories to IC positions |
| `supervised_trajectory` | Supervised trajectory loss |
| `pde_residual` | PDE conservation in smooth regions |
| `pde_shock_residual` | PDE residual on GT, penalizing unpredicted shocks |
| `rh_residual` | RH residual from region densities |
| `acceleration` | Shock detection via high acceleration + missed shock term |
| `regularize_traj` | Penalize erratic trajectory jumps between timesteps |

Presets (in `loss.py`):
| Preset | Models | Description |
|--------|--------|-------------|
| `shock_net` | ShockNet | boundary + acceleration + ic_anchoring |
| `hybrid` | HybridDeepONet | mse + rh_residual + pde_residual + ic + ic_anchoring |
| `traj_net` | TrajDeepONet, NoTrajDeepONet | mse + ic_anchoring + boundary + regularize_traj |
| `classifier_traj_net` | ClassifierTrajDeepONet | mse + ic_anchoring + boundary + regularize_traj + acceleration |
| `traj_transformer` | TrajTransformer | mse + ic_anchoring + boundary + regularize_traj |
| `classifier_traj_transformer` | ClassifierTrajTransformer | mse + ic_anchoring + boundary + regularize_traj + acceleration |
| `pde_shocks` | Any grid model | mse + pde_shock_residual |
| `mse` | Any grid model | mse only |

## Adding a New Loss

1. Create a new file in `losses/` inheriting from `BaseLoss`
2. Implement the unified interface: `forward(input_dict, output_dict, target) -> (loss, components)`
3. Register it in `loss.py`:
```python
from losses.my_loss import MyLoss
LOSSES = {
    ...
    "my_loss": MyLoss,
}
```
4. Export it in `losses/__init__.py`
5. Update `ARCHITECTURE.md` with the mathematical formula and configuration

## Adding a New Model

1. Implement your model class in `models/` with a `build_*` factory function
2. Register it in `model.py`:
   - Add to `MODELS` dict (name → builder function)
   - Add to `MODEL_TRANSFORM` dict (name → transform string or `None`)
3. Add a loss preset in `loss.py` `LOSS_PRESETS` dict
4. Add a plot preset in `plotter.py` `PLOT_PRESETS` dict

```python
# model.py
from models.my_model import build_my_model
MODELS = { ..., "MyModel": build_my_model }
MODEL_TRANSFORM = { ..., "MyModel": None }  # or "ToGridInput", "FlattenDiscontinuities"

# loss.py
LOSS_PRESETS = { ..., "my_model": [("mse", 1.0), ("ic", 10.0)] }

# plotter.py
PLOT_PRESETS = { ..., "my_model": ["ground_truth", "mse_error"] }
```

## Code Style Guidelines

- **File size limit**: Keep files under 600 lines. If a file grows beyond this, split it into logical modules.
- **Model file placement**: Only standalone models (with their own `build_*` factory, registered in `MODELS`) belong directly in `models/`. All sub-modules, building blocks, and shared components must go in `models/base/`.

## Dependencies

Uses the same environment as `operator_learning/` with:
- PyTorch for models
- `nfv` package for data generation (Lax-Hopf solver)
- Weights & Biases for logging
- matplotlib for plotting
