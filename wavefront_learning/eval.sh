#!/bin/bash

# Ablation study: WaveNO vs ClassifierTrajTransformer
# 5000 samples, 100 epochs, pde_shocks loss

# === WaveNO ablations (fixing step-generalization) ===
uv run train.py --model WaveNOCls --n_samples 5000 --epochs 100 --loss pde_shocks --plot waveno_cls
uv run train.py --model WaveNOLocal --n_samples 5000 --epochs 100 --loss pde_shocks --plot waveno_local
uv run train.py --model WaveNOIndepTraj --n_samples 5000 --epochs 100 --loss pde_shocks --plot waveno_indep_traj
uv run train.py --model WaveNODisc --n_samples 5000 --epochs 100 --loss pde_shocks --plot waveno_disc

# === CTT ablations (fixing resolution) ===
uv run train.py --model CTTBiased --n_samples 5000 --epochs 100 --loss pde_shocks --plot ctt_biased
uv run train.py --model CTTSegPhysics --n_samples 5000 --epochs 100 --loss pde_shocks --plot ctt_seg_physics
uv run train.py --model CTTFiLM --n_samples 5000 --epochs 100 --loss pde_shocks --plot ctt_film
