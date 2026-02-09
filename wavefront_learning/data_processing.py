"""Data generation and preprocessing for wavefront learning.

This module handles:
- Grid generation using the nfv Lax-Hopf solver
- Discontinuity extraction from discretized ICs
- Data preprocessing for neural network training
"""

import numpy as np
import torch
from data_loading import download_grids, upload_grids
from nfv.flows import Greenshield
from nfv.initial_conditions import PiecewiseConstant
from nfv.problem import Problem
from nfv.solvers import LaxHopf


class PiecewiseRandom(PiecewiseConstant):
    """Piecewise constant IC with random breakpoint locations."""

    def __init__(self, ks, x_noise=False):
        super().__init__(ks, x_noise)
        self.xs = np.random.rand(len(ks) - 1)
        self.xs = np.sort(self.xs)
        self.xs = np.concatenate([[0], self.xs, [1]])


def get_nfv_dataset(
    n_samples: int,
    nx: int,
    nt: int,
    dx: float,
    dt: float,
    max_steps: int = 3,
    only_shocks: bool = True,
) -> np.ndarray:
    """Generate grids using the Lax-Hopf solver.

    Args:
        n_samples: Number of samples to generate.
        nx: Number of spatial grid points.
        nt: Number of time steps.
        dx: Spatial step size.
        dt: Time step size.
        max_steps: Maximum number of pieces in piecewise constant IC.
        only_shocks: If True, sort ks to ensure only shock waves (no rarefactions).

    Returns:
        Grid data of shape (n_samples, nt, nx).
    """
    ics = [
        PiecewiseRandom(ks=[np.random.rand() for _ in range(max_steps)], x_noise=False)
        for _ in range(n_samples)
    ]
    if only_shocks:
        for ic in ics:
            ic.ks.sort()

    problem = Problem(nx=nx, nt=nt, dx=dx, dt=dt, ic=ics, flow=Greenshield())
    grids = (
        problem.solve(LaxHopf, batch_size=4, dtype=torch.float64, progressbar=True)
        .cpu()
        .numpy()
    )
    return grids


def clean_piecewise_constant_ic(ic_grid: np.ndarray, max_passes: int = 1) -> np.ndarray:
    """Remove discretization artifacts from a piecewise constant initial condition.

    When a piecewise constant function with continuous breakpoints is
    discretized onto a grid, cells at breakpoint locations may receive
    intermediate values. This function removes such isolated cells by
    replacing them with their nearest-valued neighbor.

    Performs a left-to-right sweep: each interior cell that differs from
    both neighbors is replaced by the numerically closest neighbor. The
    sweep cascades, so adjacent artifact cells are also cleaned.

    Args:
        ic_grid: 1D array of shape (nx,). Modified in place.
        max_passes: Number of cleaning passes. Default 1 suffices for
            typical 1-2 cell artifacts. Set to 0 to run until convergence.

    Returns:
        The same array (modified in place).
    """
    n = len(ic_grid)
    if n <= 2:
        return ic_grid

    pass_count = 0
    run_until_convergence = max_passes <= 0

    while True:
        changed = False
        for i in range(1, n - 1):
            left_val = ic_grid[i - 1]
            curr_val = ic_grid[i]
            right_val = ic_grid[i + 1]

            if curr_val != left_val and curr_val != right_val:
                if abs(curr_val - left_val) <= abs(curr_val - right_val):
                    ic_grid[i] = left_val
                else:
                    ic_grid[i] = right_val
                changed = True

        pass_count += 1
        if not run_until_convergence and pass_count >= max_passes:
            break
        if run_until_convergence and not changed:
            break

    return ic_grid


def extract_discontinuities_from_grid(
    ic_grid: np.ndarray,
    dx: float,
    max_discontinuities: int = 10,
    threshold: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract discontinuity points from a discretized initial condition.

    Detects jumps in the IC by looking at the gradient and extracts
    the position, left value, and right value for each discontinuity.

    Args:
        ic_grid: Discretized IC of shape (nx,).
        dx: Spatial step size.
        max_discontinuities: Maximum number of discontinuities to detect.
        threshold: Minimum gradient magnitude to consider as a discontinuity.

    Returns:
        Tuple of (discontinuities, mask) where:
            - discontinuities: tensor of shape (max_discontinuities, 3)
              containing [x, left_val, right_val] for each discontinuity
            - mask: tensor of shape (max_discontinuities,) where 1 indicates
              valid discontinuity, 0 indicates padding
    """
    nx = len(ic_grid)
    grad = np.abs(np.diff(ic_grid))

    # Find indices where gradient exceeds threshold
    disc_indices = np.where(grad > threshold)[0]

    discontinuities = torch.zeros(max_discontinuities, 3, dtype=torch.float32)
    mask = torch.zeros(max_discontinuities, dtype=torch.float32)

    n_found = min(len(disc_indices), max_discontinuities)

    for i in range(n_found):
        idx = disc_indices[i]
        # Position is at the midpoint between the two grid points
        x_pos = (idx + 0.5) * dx
        left_val = ic_grid[idx]
        right_val = ic_grid[idx + 1] if idx + 1 < nx else ic_grid[idx]

        discontinuities[i, 0] = x_pos
        discontinuities[i, 1] = left_val
        discontinuities[i, 2] = right_val
        mask[i] = 1.0

    return discontinuities, mask


def extract_ic_representation_from_grid(
    ic_grid: np.ndarray,
    dx: float,
    max_pieces: int = 10,
    threshold: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract piecewise constant representation from a discretized IC.

    Args:
        ic_grid: Discretized IC of shape (nx,).
        dx: Spatial step size.
        max_pieces: Maximum number of pieces to support.
        threshold: Minimum gradient magnitude to consider as a discontinuity.

    Returns:
        Tuple of (xs, ks, mask) where:
            - xs: tensor of shape (max_pieces + 1,) with breakpoint positions
            - ks: tensor of shape (max_pieces,) with piece values
            - mask: tensor of shape (max_pieces,) where 1 indicates valid piece
    """
    nx = len(ic_grid)
    grad = np.abs(np.diff(ic_grid))

    # Find breakpoint indices
    disc_indices = np.where(grad > threshold)[0]

    # Create breakpoints: start at 0, each discontinuity, end at 1
    xs = torch.zeros(max_pieces + 1, dtype=torch.float32)
    ks = torch.zeros(max_pieces, dtype=torch.float32)
    mask = torch.zeros(max_pieces, dtype=torch.float32)

    # Start breakpoint
    xs[0] = 0.0

    # Number of discontinuities found
    n_disc = min(len(disc_indices), max_pieces - 1)

    # Fill in discontinuity positions and piece values
    for i in range(n_disc):
        idx = disc_indices[i]
        xs[i + 1] = (idx + 0.5) * dx
        ks[i] = ic_grid[idx]
        mask[i] = 1.0

    # Last breakpoint at 1.0
    if n_disc < max_pieces:
        xs[n_disc + 1] = 1.0
        # Last piece value
        if n_disc > 0:
            last_disc_idx = disc_indices[n_disc - 1]
            ks[n_disc] = ic_grid[min(last_disc_idx + 1, nx - 1)]
        else:
            ks[0] = ic_grid[0]
            mask[0] = 1.0
        mask[n_disc] = 1.0

    return xs, ks, mask


def preprocess_wavefront_data(
    grids: np.ndarray,
    nx: int,
    nt: int,
    dx: float,
    dt: float,
    max_discontinuities: int = 10,
) -> list[tuple[dict, torch.Tensor]]:
    """Preprocess grids for wavefront learning.

    Extracts discontinuities from the initial condition (grid[:, 0, :])
    and creates input dictionaries with coordinate grids.

    Args:
        grids: Grid data of shape (n_samples, nt, nx).
        nx, nt: Grid dimensions.
        dx, dt: Grid spacing.
        max_discontinuities: Maximum number of discontinuities to support.

    Returns:
        List of tuples (input_data, target_grid) where:
            - input_data: dict containing 'discontinuities', 'mask', 't_coords', 'x_coords'
            - target_grid: tensor of shape (1, nt, nx)
    """
    processed = []

    for idx in range(len(grids)):
        # Get target grid
        target_grid = torch.from_numpy(grids[idx]).to(torch.float32).unsqueeze(0)

        # Extract IC from first time step
        ic_grid = grids[idx, 0, :].copy()

        # Clean IC: remove discretization artifacts at breakpoints
        clean_piecewise_constant_ic(ic_grid)

        # Extract discontinuity representation
        discontinuities, disc_mask = extract_discontinuities_from_grid(
            ic_grid, dx, max_discontinuities=max_discontinuities
        )

        # Extract full IC representation (xs and ks)
        xs, ks, pieces_mask = extract_ic_representation_from_grid(
            ic_grid, dx, max_pieces=max_discontinuities
        )

        # Create coordinate grids for the output
        t_coords = (
            (torch.arange(nt).float() * dt)[:, None].expand(nt, nx).unsqueeze(0)
        )  # (1, nt, nx)
        x_coords = (
            (torch.arange(nx).float() * dx)[None, :].expand(nt, nx).unsqueeze(0)
        )  # (1, nt, nx)

        input_data = {
            "discontinuities": discontinuities,  # (max_disc, 3): [x, left_val, right_val]
            "disc_mask": disc_mask,  # (max_disc,)
            "xs": xs,  # (max_pieces + 1,): breakpoint positions
            "ks": ks,  # (max_pieces,): piece values
            "pieces_mask": pieces_mask,  # (max_pieces,)
            "t_coords": t_coords,  # (1, nt, nx)
            "x_coords": x_coords,  # (1, nt, nx)
            "dx": torch.tensor(dx, dtype=torch.float32),
            "dt": torch.tensor(dt, dtype=torch.float32),
        }

        processed.append((input_data, target_grid))

    return processed


def get_wavefront_data(
    n_samples: int,
    nx: int,
    nt: int,
    dx: float,
    dt: float,
    max_steps: int = 3,
    only_shocks: bool = True,
    random_seed: int = 42,
    max_discontinuities: int = 10,
    upload_to_hf: bool = True,
) -> list[tuple[dict, torch.Tensor]]:
    """Get wavefront data, downloading from HuggingFace or generating locally.

    This is the main entry point for data loading. It:
    1. Tries to download from HuggingFace first
    2. If not found, generates locally and optionally uploads
    3. Preprocesses and returns

    Args:
        n_samples: Number of samples needed.
        nx: Number of spatial grid points.
        nt: Number of time steps.
        dx: Spatial step size.
        dt: Time step size.
        max_steps: Maximum number of pieces in piecewise constant IC.
        only_shocks: If True, generate only shock waves (no rarefactions).
        random_seed: Random seed for reproducibility.
        max_discontinuities: Maximum number of discontinuities to support.
        upload_to_hf: If True, upload generated data to HuggingFace.

    Returns:
        List of tuples (input_data, target_grid).
    """
    np.random.seed(random_seed)

    # Try to download from HuggingFace (shared mchami/grids repo)
    grids = download_grids(nx, nt, dx, dt, max_steps, only_shocks)

    if grids is not None and len(grids) >= n_samples:
        print(f"Using cached data from mchami/grids ({len(grids)} samples available)")
        grids = grids[:n_samples]
    else:
        # Generate locally
        print(f"Generating {n_samples} samples locally...")
        grids = get_nfv_dataset(n_samples, nx, nt, dx, dt, max_steps, only_shocks)

        # Upload to HuggingFace for caching
        if upload_to_hf:
            try:
                upload_grids(grids, nx, nt, dx, dt, max_steps, only_shocks)
            except Exception as e:
                print(f"Failed to upload to HuggingFace: {e}")

    # Preprocess
    processed = preprocess_wavefront_data(grids, nx, nt, dx, dt, max_discontinuities)

    return processed
