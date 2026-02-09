#!/bin/bash

uv run train.py --model EncoderDecoderCross --n_samples 5000 --epochs 100 --loss mse --plot grid_only
uv run train.py --model TrajDeepONet --n_samples 5000 --epochs 100 --loss pde_shocks --plot traj_net
uv run train.py --model NoTrajDeepONet --n_samples 5000 --epochs 100 --loss mse --plot grid_only
