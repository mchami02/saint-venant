#!/bin/bash

# Ablation study on the ARZ equation
# 5000 samples, 100 epochs

# === WaveNO component ablation study ===
# Cumulative progression (building up from bare minimum)
uv run train.py --model WaveNOARZBare --equation ARZ --n_samples 5000 --epochs 100 --loss mse --plot grid_residual --exp waveno-arz-modules --run_name bare-baseline
uv run train.py --model WaveNOARZ --equation ARZ --n_samples 5000 --epochs 100 --loss mse --plot grid_residual --exp waveno-arz-modules --run_name waveno-arz
