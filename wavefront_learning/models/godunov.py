"""Godunov finite-volume solver wrapped as a non-trainable nn.Module.

Provides a classical numerical baseline that runs through the same
evaluation pipeline as neural operator models.
"""

import torch
import torch.nn as nn
from nfv.flows import Greenshield
from nfv.initial_conditions import PiecewiseConstant
from nfv.problem import Problem
from nfv.solvers.finite_volume import FVM
from nfv.solvers.godunov import Godunov


class GodunovSolver(nn.Module):
    """Godunov FV solver exposed as an nn.Module (no trainable parameters).

    Grid parameters (nx, nt, dx, dt) are read from the input dict at
    inference time, so the model works at any resolution.
    """

    def forward(self, input_dict: dict) -> dict:
        xs = input_dict["xs"]  # (B, max_pieces+1)
        ks = input_dict["ks"]  # (B, max_pieces)
        pieces_mask = input_dict["pieces_mask"]  # (B, max_pieces)
        dx = input_dict["dx"]  # (B,) scalar per sample
        dt = input_dict["dt"]  # (B,) scalar per sample

        # Infer grid shape from coordinate tensors
        t_coords = input_dict["t_coords"]  # (B, 1, nt, nx)
        nt = t_coords.shape[2]
        nx = t_coords.shape[3]

        device = xs.device
        B = xs.shape[0]

        # Use first sample's dx/dt (uniform across batch)
        dx_val = dx[0].item()
        dt_val = dt[0].item()

        # Build one PiecewiseConstant IC per sample
        ics = []
        for i in range(B):
            n_pieces = int(pieces_mask[i].sum().item())
            raw_ks = ks[i, :n_pieces].cpu().numpy().tolist()
            raw_xs = xs[i, : n_pieces + 1].cpu().numpy()
            ic = PiecewiseConstant(raw_ks)
            ic.xs = raw_xs  # override uniform breakpoints
            ics.append(ic)

        flow = Greenshield()
        problem = Problem(
            nx=nx, nt=nt, dx=dx_val, dt=dt_val, ic=ics, flow=flow
        )

        # Construct IC-based boundary conditions (replicate IC across all timesteps)
        ic_grids = torch.stack(
            [torch.from_numpy(ic.discretize(nx)) for ic in ics]
        )  # (B, nx)
        problem.solutions["ic"] = (
            ic_grids.unsqueeze(1).expand(-1, nt, -1).clone()
        )

        solution = problem.solve(
            FVM(Godunov),
            boundaries="ic",
            ghost_cells=1,
            boundary_size=1,
            boundary_pad=0,
            dtype=torch.float64,
            device="cpu",
        )  # (B, nt, nx)

        return {"output_grid": solution.unsqueeze(1).float().to(device)}


def build_godunov(args: dict) -> GodunovSolver:
    """Factory function for GodunovSolver."""
    return GodunovSolver()
