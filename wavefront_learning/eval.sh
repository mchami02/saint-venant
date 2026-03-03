#!/bin/bash

# Ablation study
# 5000 samples, 100 epochs

# === WaveNO component ablation study ===
# Cumulative progression (building up from bare minimum)
uv run train.py --model WaveNOAblation --n_samples 5000 --epochs 100 --loss mse --plot grid_residual
uv run train.py --model WaveNOAblationBias --n_samples 5000 --epochs 100 --loss mse --plot grid_residual
uv run train.py --model WaveNOAblationDamp --n_samples 5000 --epochs 100 --loss mse --plot grid_residual
uv run train.py --model WaveNOAblationFiLM --n_samples 5000 --epochs 100 --loss mse --plot grid_residual
uv run train.py --model WaveNOAblationCrossAttn --n_samples 5000 --epochs 100 --loss mse --plot grid_residual
uv run train.py --model WaveNOAblationFull --n_samples 5000 --epochs 100 --loss mse --plot grid_residual

# Isolated additions (each component alone on bare baseline)
uv run train.py --model WaveNOAblationFiLMOnly --n_samples 5000 --epochs 100 --loss mse --plot grid_residual
uv run train.py --model WaveNOAblationCrossAttnOnly --n_samples 5000 --epochs 100 --loss mse --plot grid_residual

# Reference models (existing, full architectures)
uv run train.py --model WaveNOBase --n_samples 5000 --epochs 100 --loss mse --plot grid_residual
uv run train.py --model WaveNO --n_samples 5000 --epochs 100 --loss pde_shocks --plot traj_residual
