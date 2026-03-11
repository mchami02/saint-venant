#!/bin/bash

# Ablation study
# 5000 samples, 100 epochs

# === WaveNO component ablation study ===
# Cumulative progression (building up from bare minimum)
uv run train.py --model WaveNOBare --n_samples 5000 --epochs 100 --loss mse --plot grid_residual --exp waveno-modules-pointwise --run_name bare-baseline
uv run train.py --model WaveNOBiasOnly --n_samples 5000 --epochs 100 --loss mse --plot grid_residual --exp waveno-modules-pointwise --run_name +char-bias
uv run train.py --model WaveNOBiasDamp --n_samples 5000 --epochs 100 --loss mse --plot grid_residual --exp waveno-modules-pointwise --run_name +char-bias+damping
uv run train.py --model WaveNODamp --n_samples 5000 --epochs 100 --loss mse --plot grid_residual --exp waveno-modules-pointwise --run_name +char-bias+damping+film
uv run train.py --model WaveNODampCrossAttn --n_samples 5000 --epochs 100 --loss mse --plot grid_residual --exp waveno-modules-pointwise --run_name +char-bias+damping+film+cross-attn
uv run train.py --model WaveNOAll --n_samples 5000 --epochs 100 --loss mse --plot grid_residual --exp waveno-modules-pointwise --run_name +char-bias+damping+film+cross-attn+self-attn

# Isolated additions (each component alone on bare baseline)
uv run train.py --model WaveNOFiLMOnly --n_samples 5000 --epochs 100 --loss mse --plot grid_residual --exp waveno-modules-pointwise --run_name film-only
uv run train.py --model WaveNOCrossAttnOnly --n_samples 5000 --epochs 100 --loss mse --plot grid_residual --exp waveno-modules-pointwise --run_name cross-attn-only
uv run train.py --model WaveNO --n_samples 5000 --epochs 100 --loss mse --plot grid_residual --exp waveno-modules-pointwise --run_name bias+film

# Reference models (existing, full architectures)
uv run train.py --model WaveNOFullBase --n_samples 5000 --epochs 100 --loss mse --plot grid_residual --exp waveno-modules-pointwise --run_name waveno-base
uv run train.py --model WaveNOFull --n_samples 5000 --epochs 100 --loss pde_shocks --plot traj_residual --exp waveno-modules-pointwise --run_name waveno-full
