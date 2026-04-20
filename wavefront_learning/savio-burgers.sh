#!/bin/bash
# Run on login node: bash savio-burgers.sh
# Submits 18 jobs: 6 models x 3 seeds

mkdir -p slurm_logs/train

all_models=("FNO" "DeepONet" "WaveNOBiasOnly" "WaveNOBare" "Godunov" "WaveNO")
seeds=(42 123 8)

for seed in "${seeds[@]}"; do
    for model in "${all_models[@]}"; do
        sbatch \
            --job-name="${model}_burgers_s${seed}" \
            --account=ac_mixedav \
            --partition=savio3_gpu \
            --qos=a40_gpu3_normal \
            --nodes=1 --ntasks=1 --cpus-per-task=8 \
            --gres=gpu:A40:1 \
            --time=24:00:00 \
            --output="slurm_logs/train/burgers_${model}_s${seed}_%j.out" \
            --error="slurm_logs/train/burgers_${model}_s${seed}_%j.err" \
            --wrap="cd /global/home/users/mamoun/saint-venant/wavefront_learning && source .env && uv run python train.py \
                --model $model --n_samples 5000 --epochs 100 --loss mse \
                --plot burgers --exp all-experiments --run_name $model \
                --seed $seed --equation burgers --max_steps 8 --max_test_steps 16 \
                --save_path checkpoints/all-experiments/burgers/${model}_seed${seed}.pth"
    done
done

echo "Submitted 18 Burgers jobs. Check with: squeue -u \$USER"
