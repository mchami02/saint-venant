# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workflow

Follow this workflow for every task, without exception:

1. **Clarify first.** Before doing anything, identify any ambiguities in the task and ask for clarification. Only proceed once the task is clear.
2. **Create a branch from main.** Always create a new branch from `main` (never from the current branch). Name it descriptively based on the task.
3. **Implement the change.**
4. **Run a targeted sanity check.** Design a lightweight command that actually exercises the feature or fix you just implemented. Do NOT blindly run the same default command every time. Think about what flags, arguments, or configuration will verify your change is working. **Explicitly pass every relevant flag** — never rely on auto-selection or defaults for the thing you're testing. Examples:
   - Added a new loss? → Use `--loss` to select it explicitly.
   - Added a new model? → Use `--model` to select it. If the model has an associated loss, **also pass `--loss`** explicitly to confirm the loss works too.
   - Added a new model + loss together? → Pass **both** `--model` and `--loss` explicitly.
   - Fixed a bug in high-res evaluation? → Include `--max_high_res` to trigger that code path.
   - Changed data loading? → Use a small `--n_samples` run that hits the new loading logic.

   The base template (adjust as needed):
   ```bash
   cd wavefront_learning
   uv run python train.py --n_samples 50 --epochs 1 --model WaveNO --max_steps 4 --max_test_steps 5 --max_high_res 2 --no_wandb
   ```
   Always add or change flags so the sanity check covers the code you touched. If the default command doesn't exercise your change at all, it is not a valid sanity check. **Do not rely on auto-selected presets** (`MODEL_LOSS_PRESET`, `MODEL_PLOT_PRESET`) — always pass the flags explicitly so the sanity check proves the feature works end-to-end.
5. **Analyze the result:**
   - **Pass** → push the branch. Before creating the pull request, check for merge conflicts with `main` by running `git merge --no-commit --no-ff main`. If there are conflicts, resolve them locally, commit the merge, and push. Then create a pull request with a clear description of what was done, and stop.
   - **Fail** → diagnose, fix, and re-run. Repeat at most 3 times.
   - **Still failing after 3 attempts** → stop. Write a clear summary of what was tried and what the suspected remaining issue is. Leave the branch as-is without pushing.
6. **Post-PR fixes.** If review comments or CI checks flag issues, fix them on the same branch and push. Repeat until the PR is clean.
7. **Never merge.** Never accept or merge the pull request — that is always the user's decision.

## Scope

Unless explicitly stated otherwise, all learning-related tasks are implemented in `wavefront_learning/`.

## Hard Rules

- Branches are always created from `main`, never from another branch.
- Do not install or modify packages without asking first.
- Keep all experiment outputs and checkpoints in the designated output folder.
- Do not modify files outside the current task's scope.
- Never merge or push directly to `main` under any circumstances.
- **Keep code modular.** Each class or function should do one thing. In particular, loss classes must compute a single loss term — use presets in `loss.py` to compose multiple losses with weights. Never bundle unrelated computations into one class.

## File Maintenance

**IMPORTANT**: After completing any task that modifies a folder's structure (adding/removing files), update the corresponding `structure.md` file to reflect the changes. If no `structure.md` exists in that folder, create one.

**IMPORTANT**: After modifying models or losses:
- Update `ARCHITECTURE.md` with architecture details and mathematical formulas
- Update `Structure.md` if adding/removing files

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

4. **wavefront_learning/** - Discontinuity-based learning (shock trajectories)
   - `train.py`: Training script with W&B integration
   - `models/shock_trajectory_net.py`: DeepONet-like model for trajectory prediction
   - `losses/rankine_hugoniot.py`: Physics-based unsupervised loss (Rankine-Hugoniot)
   - Uses exact discontinuity points as input instead of discretized grids
   - See `wavefront_learning/CLAUDE.md` for detailed documentation

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
