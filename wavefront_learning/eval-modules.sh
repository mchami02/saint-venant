#!/bin/bash

# Ablation study
# 5000 samples, 100 epochs

# === WaveNO component ablation study ===
# Cumulative progression (building up from bare minimum)
uv run train.py --model WaveNOAblation --n_samples 5000 --epochs 100 --loss mse --plot grid_residual --exp waveno-modules --run_name bare-baseline
uv run train.py --model WaveNOAblationBias --n_samples 5000 --epochs 100 --loss mse --plot grid_residual --exp waveno-modules --run_name +char-bias
uv run train.py --model WaveNOAblationDamp --n_samples 5000 --epochs 100 --loss mse --plot grid_residual --exp waveno-modules --run_name +char-bias+damping
uv run train.py --model WaveNOAblationFiLM --n_samples 5000 --epochs 100 --loss mse --plot grid_residual --exp waveno-modules --run_name +char-bias+damping+film
uv run train.py --model WaveNOAblationCrossAttn --n_samples 5000 --epochs 100 --loss mse --plot grid_residual --exp waveno-modules --run_name +char-bias+damping+film+cross-attn
uv run train.py --model WaveNOAblationFull --n_samples 5000 --epochs 100 --loss mse --plot grid_residual --exp waveno-modules --run_name +char-bias+damping+film+cross-attn+self-attn

# Isolated additions (each component alone on bare baseline)
uv run train.py --model WaveNOAblationFiLMOnly --n_samples 5000 --epochs 100 --loss mse --plot grid_residual --exp waveno-modules --run_name film-only
uv run train.py --model WaveNOAblationCrossAttnOnly --n_samples 5000 --epochs 100 --loss mse --plot grid_residual --exp waveno-modules --run_name cross-attn-only
uv run train.py --model WaveNOAblationBiasFiLM --n_samples 5000 --epochs 100 --loss mse --plot grid_residual --exp waveno-modules --run_name bias+film

# Reference models (existing, full architectures)
uv run train.py --model WaveNOBase --n_samples 5000 --epochs 100 --loss mse --plot grid_residual --exp waveno-modules --run_name waveno-base
uv run train.py --model WaveNO --n_samples 5000 --epochs 100 --loss pde_shocks --plot traj_residual --exp waveno-modules --run_name waveno-full
