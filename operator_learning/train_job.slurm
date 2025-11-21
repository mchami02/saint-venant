#!/bin/bash
#SBATCH --job-name=mytrain
#SBATCH --output=out_%j.txt
#SBATCH --error=err_%j.txt
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Run Python script
uv run train.py --epochs 3 --n_samples 50
