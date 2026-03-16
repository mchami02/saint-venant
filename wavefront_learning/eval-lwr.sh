#!/bin/bash

# training_params
n_samples=5000
test_epochs=2
train_epochs=100
seeds=(42 123 8 7 256 99 1337 2024 55 314)
n_runs=3

# logging params
plotting="grid_residual"
exp="lwr-experiments"

# models
all_models=("FNO" "DeepONet" "WaveNOBiasOnly" "WaveNOBare" "Godunov" "WaveNO")
ablation_models=("WaveNOBiasOnly" "WaveNOBare" "WaveNO")
loss_fn="mse"

# ----------------- #
exp_models=("${all_models[@]}") 

# Testing it runs
for model in "${exp_models[@]}"; do
    uv run train.py --model $model --n_samples $n_samples --epochs $test_epochs --loss $loss_fn --no_wandb --seed ${seeds[0]}
done

# LWR experiments
for i in $(seq 0 $((n_runs - 1))); do
    seed=${seeds[$i]}
    for model in "${exp_models[@]}"; do
        uv run train.py --model $model --n_samples $n_samples --epochs $train_epochs --loss $loss_fn --plot $plotting --exp $exp --run_name $model --seed $seed
    done
done