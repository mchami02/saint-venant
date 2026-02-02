# Wavefront Learning - Directory Structure

```
wavefront_learning/
├── CLAUDE.md              # Claude Code guidance for this module
├── structure.md           # This file - folder structure documentation
├── train.py               # Main training script (standalone)
├── test.py                # Testing/evaluation script (standalone)
├── data.py                # Data generation and loading utilities
├── model.py               # Model factory and registry
├── loss.py                # Loss function factory and registry
├── logger.py              # W&B logging utilities
├── plotter.py             # Visualization and plotting utilities
├── models/
│   ├── __init__.py
│   ├── base.py                    # BaseWavefrontModel abstract class
│   └── shock_trajectory_net.py    # ShockTrajectoryNet (DeepONet-like)
├── losses/
│   ├── __init__.py
│   ├── base.py                    # BaseLoss abstract class
│   └── rankine_hugoniot.py        # Rankine-Hugoniot physics loss
└── wandb/                 # W&B run logs (gitignored)
```

## Key Files

### Entry Points
- **train.py**: `uv run python train.py --model ShockNet --epochs 100`
- **test.py**: `uv run python test.py --model_path wavefront_model.pth`

### Models
- **ShockNet** (`models/shock_trajectory_net.py`): DeepONet-like architecture for shock trajectory prediction
  - Branch: Transformer-based discontinuity encoder
  - Trunk: Fourier feature time encoder
  - Output: `{positions: (B, D, T), existence: (B, D, T)}`

### Losses
- **RankineHugoniotLoss** (`losses/rankine_hugoniot.py`): Physics-based unsupervised loss
  - Trajectory consistency (Rankine-Hugoniot condition)
  - Boundary loss (shocks exit domain)
  - Collision loss (shock merging)
  - Existence regularization

### Data
- **data.py**: Generates piecewise constant ICs, extracts discontinuities, creates datasets
