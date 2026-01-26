# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A research project for solving 1D shallow water (Saint-Venant) equations using both traditional finite volume numerical methods and machine learning-based neural operators. The codebase supports two equation systems:
- **Saint-Venant equations** (shallow water flow)
- **LWR traffic flow equations** (Lighthill-Whitham-Richards)

## Commands

### Running the numerical solver
```bash
uv run python numerical_solver/main.py numerical_solver/parameters/parameters.txt
```

### Training neural operators
```bash
cd operator_learning
uv run python train.py --model FNO --epochs 100 --batch_size 8 --nx 50 --nt 250
```

Key training arguments:
- `--model`: FNO, DeepONet, WNO, LNO, MoEFNO, EncoderDecoder
- `--autoregressive`: Enable autoregressive training with scheduled sampling
- `--pinn_weight`: Weight for physics-informed loss (0 = disabled)
- `--multi-res`: Train on multiple resolutions
- `--test-high-res`: Test on 2x finer resolution grids

### Testing trained models
```bash
cd operator_learning
uv run python test.py --model_path path/to/model.pth
```

### Linting
```bash
uv run ruff check .
uv run ruff format .
```

## Architecture

### Three main packages

1. **numerical_methods/** - Core finite volume solver
   - `solvers/godunov.py`: Godunov scheme with pluggable Riemann solvers
   - `solvers/solver.py`: Base Solver abstract class
   - `flux/`: Flux functions (Greenshields, Triangular for LWR)
   - `boundary_cond/`, `initial_cond/`: Modular BC/IC implementations
   - Riemann solvers: `LWRRiemannSolver`, `SVERiemannSolver`

2. **operator_learning/** - Neural operator implementations
   - `train.py`: Main training script with Comet.ml integration
   - `model.py`: Model factory (`create_model` function)
   - `operator_data_pipeline.py`: Data generation from numerical solver
   - `models/`: FNO, DeepONet, WNO, LNO, MoEFNO, encoder-decoder variants, GNN-based models
   - `loss/`: PDE-based loss (`pde_loss.py`) and LWR-specific loss (`lwr_loss.py`)

3. **learner/** - Simpler sequential encoder-predictor model
   - `model.py`: SVEModel with separate h/u prediction heads
   - `data_pipeline.py`: AutoRegressiveDataset for sequential prediction

### Data flow

1. Numerical solver generates training data (grids with h, u, q values over space-time)
2. Data stored in HDF5 format, can be uploaded/downloaded from Hugging Face via `hf_grids.py`
3. `operator_data_pipeline.py` preprocesses data and adds coordinate channels
4. Neural operators learn the solution operator mapping initial conditions to full space-time solutions

### Model output convention

Models may return either:
- Single tensor: prediction only
- Tuple of 3: (prediction, delta_u, gate_values) for models with correction terms

Use `_unpack_model_output()` from `test.py` for consistent handling.

## Dependencies

Two dependency groups in pyproject.toml:
- `def` (default): PyTorch 2.5.1+ for most models
- `gnn`: PyTorch 2.2.1 + DGL for graph neural network models

Switch groups with: `uv sync --group gnn`

## Key conventions

- Device selection: `torch.device("cuda" if torch.cuda.is_available() else "mps")`
- Experiment tracking: Comet.ml (configured in train.py)
- Ruff for linting: line length 88, double quotes, Python 3.11 target
