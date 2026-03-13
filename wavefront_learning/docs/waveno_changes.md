# WaveNO Changes: Original → Current Best

Tracking all changes from the original WaveNO (at branch creation, merge base `566e2db`) to the current best model (`7e0643d`).

## Architecture Changes

### Model (`models/waveno.py`)

| Component | Original | Current Best | Impact |
|-----------|----------|-------------|--------|
| `hidden_dim` | 64 | 96 | +50% capacity |
| `num_self_attn_layers` | 2 | 3 | More segment interaction |
| `num_cross_layers` | 2 | 3 | Deeper density decoding |
| `use_damping` | False | True | Collision-time damping in LWR bias |
| Query MLP | Linear→ReLU→Dropout→Linear | Linear→ReLU→**LayerNorm**→Dropout→Linear | Stabilizes query magnitude |
| Density head activation | ReLU | **GELU** | Smoother gradients |
| Density head width | `hidden_dim` intermediate | `hidden_dim * 2` intermediate | 2x wider MLP |
| Pre-density LayerNorm | None | **LayerNorm** before density head | K-invariant feature magnitude |
| Per-head bias scales | Uniform (all heads share same bias) | **Learnable `nn.Parameter(ones(num_heads))`** | Heads can learn to ignore/emphasize bias |
| Eval-only width normalization | None | **`sqrt(ref_width/width)` scaling at eval** | K-invariant attention concentration |

### Cross-Attention (`models/base/biased_cross_attention.py`)

| Component | Original | Current Best |
|-----------|----------|-------------|
| `ff_mult` parameter | Hardcoded 4x | Configurable (default 4) |
| FF dropout | None | **Dropout in feedforward** |

## Training Changes (`configs/training.yaml`)

| Parameter | Original | Current Best |
|-----------|----------|-------------|
| `hidden_dim` | 64 | 96 |
| `epochs` | 100 | 150 |
| `batch_size` | 8 | 16 |
| `lr` | 0.001 | 0.003 |
| `dropout` | 0.05 | 0.01 |
| `scheduler` | ReduceLROnPlateau | **Cosine annealing** |
| `scheduler_factor` | 0.5 | 0.3 |
| `scheduler_patience` | 5 | 3 |
| `grad_clip_max_norm` | 1.0 | 0.5 |
| `ema_decay` | None | **0.85** |

## Key Decisions

- **Damping enabled**: Collision-time damping prevents the LWR bias from guiding attention to segments that have already collided, important for longer-time predictions.
- **Per-head bias scales**: Allow some attention heads to learn physics-free attention patterns while others maintain characteristic-speed guidance. Critical for K-generalization.
- **Eval-only width normalization**: Scales bias by `sqrt(0.33/width)` at eval time only. Narrow segments (high K) get proportionally stronger bias to maintain attention concentration. Training dynamics are preserved since this only activates during inference.
- **EMA**: Exponential moving average of weights (decay 0.85) smooths training. Updated per-step, not per-epoch.
- **Density head improvements**: GELU + wider MLP + LayerNorm stabilizes density prediction across varying numbers of segments.

## What Was Tried and Discarded

Notable failed experiments (see `results.tsv` for full list):
- Fourier segment features (+40.5%)
- Always-on width normalization (+5.9% to +62.3%)
- Wasserstein loss (+98.7%)
- IC density as query feature (+82.6%)
- PDE residual loss (+77% to +254%)
- Variational PINN loss (+33% to +81%)
- Conservation loss (+5.4% on 3-seed)
- Removing FiLM (+86.9%)
- Higher dropout (+18.9%)
- Batch size 32 (+66.5%)
