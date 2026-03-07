#!/bin/bash
# ARZ traffic flow evaluation: ECARZ baseline vs WaveNO ARZ variants.
set -e
cd "$(dirname "$0")"

# ECARZ baseline
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python train.py \
    --model ECARZ --equation ARZ --n_samples 5000 --epochs 100 \
    --loss mse --plot ecarz --exp arz-waveno --run_name ecarz-baseline

# WaveNO ARZ grid-only (no trajectory prediction)
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python train.py \
    --model WaveNOARZBase --equation ARZ --n_samples 5000 --epochs 100 \
    --loss mse --plot ecarz --exp arz-waveno --run_name waveno-arz-base

# WaveNO ARZ with trajectories, MSE only
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python train.py \
    --model WaveNOARZ --equation ARZ --n_samples 5000 --epochs 100 \
    --loss mse --plot ecarz --exp arz-waveno --run_name waveno-arz-mse

# WaveNO ARZ with trajectories + PDE loss
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python train.py \
    --model WaveNOARZ --equation ARZ --n_samples 5000 --epochs 100 \
    --loss arz_pde_shocks --plot ecarz --exp arz-waveno --run_name waveno-arz-full
