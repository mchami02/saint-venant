#!/bin/bash

uv run train.py --model TrajTransformer --n_samples 50 --epochs 1 --loss pde_shocks --plot traj_net
uv run train.py --model ClassifierTrajTransformer --n_samples 50 --epochs 1 --loss pde_shocks --plot traj_net
uv run train.py --model ClassifierAllTrajTransformer --n_samples 50 --epochs 1 --loss pde_shocks --plot traj_net
uv run train.py --model NoTrajTransformer --n_samples 50 --epochs 1 --loss mse --plot grid_only
uv run train.py --model ClassifierTrajDeepONet --n_samples 50 --epochs 1 --loss pde_shocks --plot traj_net
uv run train.py --model TrajDeepONet --n_samples 50 --epochs 1 --loss pde_shocks --plot traj_net
uv run train.py --model NoTrajDeepONet --n_samples 50 --epochs 1 --loss mse --plot grid_only


