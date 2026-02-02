"""Data generation and loading utilities for wavefront learning.

This module provides data pipelines similar to operator_learning/operator_data_pipeline.py,
but the input representation is the exact discontinuity points of the initial condition
(breakpoints and values) rather than the full discretized grid.
"""

import numpy as np
import torch
from nfv.flows import Greenshield
from nfv.initial_conditions import PiecewiseConstant
from nfv.problem import Problem
from nfv.solvers import LaxHopf
from torch.utils.data import Dataset


class PiecewiseRandom(PiecewiseConstant):
    """Piecewise constant IC with random breakpoint locations."""

    def __init__(self, ks, x_noise=False):
        super().__init__(ks, x_noise)
        self.xs = np.random.rand(len(ks) - 1)
        self.xs = np.sort(self.xs)
        self.xs = np.concatenate([[0], self.xs, [1]])


def get_nfv_dataset_with_ics(
    n_samples: int,
    nx: int,
    nt: int,
    dx: float,
    dt: float,
    max_steps: int = 3,
    only_shocks: bool = False,
):
    """Generate grids and return both solutions and IC parameters.

    Args:
        n_samples: Number of samples to generate.
        nx: Number of spatial grid points.
        nt: Number of time steps.
        dx: Spatial step size.
        dt: Time step size.
        max_steps: Maximum number of pieces in piecewise constant IC.
        only_shocks: If True, sort ks to ensure only shock waves (no rarefactions).

    Returns:
        Tuple of (grids, ics) where:
            - grids: numpy array of shape (n_samples, nt, nx)
            - ics: list of PiecewiseRandom objects containing xs and ks
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
    return grids, ics


def extract_discontinuities(ic: PiecewiseConstant, max_discontinuities: int = 10):
    """Extract discontinuity points from a piecewise constant IC.

    The discontinuity representation is a fixed-size tensor where each row
    contains [x_position, left_value, right_value] for each discontinuity.
    Unused slots are padded with zeros and marked with a mask.

    Args:
        ic: PiecewiseConstant initial condition with xs and ks attributes.
        max_discontinuities: Maximum number of discontinuities to support.

    Returns:
        Tuple of (discontinuities, mask) where:
            - discontinuities: tensor of shape (max_discontinuities, 3)
              containing [x, left_val, right_val] for each discontinuity
            - mask: tensor of shape (max_discontinuities,) where 1 indicates
              valid discontinuity, 0 indicates padding
    """
    xs = ic.xs  # Breakpoints including 0 and 1
    ks = ic.ks  # Values at each piece

    # Discontinuities are at internal breakpoints (exclude 0 and 1)
    n_discontinuities = len(xs) - 2  # Number of internal breakpoints

    discontinuities = torch.zeros(max_discontinuities, 3, dtype=torch.float32)
    mask = torch.zeros(max_discontinuities, dtype=torch.float32)

    for i in range(min(n_discontinuities, max_discontinuities)):
        x_pos = xs[i + 1]  # Internal breakpoint position
        left_val = ks[i]  # Value to the left
        right_val = ks[i + 1]  # Value to the right

        discontinuities[i, 0] = x_pos
        discontinuities[i, 1] = left_val
        discontinuities[i, 2] = right_val
        mask[i] = 1.0

    return discontinuities, mask


def extract_ic_representation(ic: PiecewiseConstant, max_pieces: int = 10):
    """Extract a compact representation of the piecewise constant IC.

    The representation includes:
    - Breakpoints (xs): positions where value changes, shape (max_pieces + 1,)
    - Values (ks): value at each piece, shape (max_pieces,)
    - Mask: indicating valid pieces, shape (max_pieces,)

    Args:
        ic: PiecewiseConstant initial condition.
        max_pieces: Maximum number of pieces to support.

    Returns:
        Tuple of (xs, ks, mask) where:
            - xs: tensor of shape (max_pieces + 1,) with breakpoint positions
            - ks: tensor of shape (max_pieces,) with piece values
            - mask: tensor of shape (max_pieces,) where 1 indicates valid piece
    """
    n_pieces = len(ic.ks)

    xs = torch.zeros(max_pieces + 1, dtype=torch.float32)
    ks = torch.zeros(max_pieces, dtype=torch.float32)
    mask = torch.zeros(max_pieces, dtype=torch.float32)

    # Fill in actual values
    for i in range(min(n_pieces, max_pieces)):
        ks[i] = ic.ks[i]
        mask[i] = 1.0

    for i in range(min(len(ic.xs), max_pieces + 1)):
        xs[i] = ic.xs[i]

    return xs, ks, mask


def preprocess_wavefront_data(
    grids: np.ndarray,
    ics: list,
    nx: int,
    nt: int,
    dx: float,
    dt: float,
    max_discontinuities: int = 10,
):
    """Preprocess grids and ICs for wavefront learning.

    Args:
        grids: numpy array of shape (n_samples, nt, nx) with solution values.
        ics: list of PiecewiseConstant objects with xs and ks attributes.
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

        # Extract discontinuity representation
        discontinuities, mask = extract_discontinuities(
            ics[idx], max_discontinuities=max_discontinuities
        )

        # Extract full IC representation (xs and ks)
        xs, ks, pieces_mask = extract_ic_representation(
            ics[idx], max_pieces=max_discontinuities
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
            "disc_mask": mask,  # (max_disc,)
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


def get_wavefront_datasets(
    n_samples: int,
    nx: int,
    nt: int,
    dx: float,
    dt: float,
    max_steps: int = 3,
    max_discontinuities: int = 10,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    random_seed: int = 42,
    only_shocks: bool = True,
):
    """Get train, val, and test datasets for wavefront learning.

    Args:
        n_samples: Total number of samples.
        nx: Number of spatial grid points.
        nt: Number of time steps.
        dx: Spatial step size.
        dt: Time step size.
        max_steps: Maximum number of pieces in piecewise constant IC.
        max_discontinuities: Maximum number of discontinuities to support.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        random_seed: Random seed for reproducibility.
        only_shocks: If True, generate only shock waves (no rarefactions).

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    np.random.seed(random_seed)

    # Generate data
    print(f"Generating {n_samples} samples for wavefront learning...")
    grids, ics = get_nfv_dataset_with_ics(
        n_samples, nx, nt, dx, dt, max_steps, only_shocks
    )

    # Shuffle indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # Split indices
    n_train = int(train_ratio * n_samples)
    n_val = int(val_ratio * n_samples)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    # Preprocess each split
    train_ics = [ics[i] for i in train_idx]
    val_ics = [ics[i] for i in val_idx]
    test_ics = [ics[i] for i in test_idx]

    train_processed = preprocess_wavefront_data(
        grids[train_idx], train_ics, nx, nt, dx, dt, max_discontinuities
    )
    val_processed = preprocess_wavefront_data(
        grids[val_idx], val_ics, nx, nt, dx, dt, max_discontinuities
    )
    test_processed = preprocess_wavefront_data(
        grids[test_idx], test_ics, nx, nt, dx, dt, max_discontinuities
    )

    train_dataset = WavefrontDataset(train_processed)
    val_dataset = WavefrontDataset(val_processed)
    test_dataset = WavefrontDataset(test_processed)

    print(f"Created datasets: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


class WavefrontDataset(Dataset):
    """Dataset for wavefront prediction training.

    Unlike the operator learning dataset which uses the full discretized IC,
    this dataset uses the exact discontinuity points (breakpoints and values)
    as input.

    Args:
        processed_data: List of tuples (input_data, target_grid) where:
            - input_data: dict with discontinuity representation and coordinates
            - target_grid: tensor of shape (1, nt, nx)
        transform: Optional transform to apply to samples.
    """

    def __init__(self, processed_data: list, transform=None):
        self.processed_data = processed_data
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.processed_data)

    def __getitem__(self, idx: int) -> tuple[dict, torch.Tensor]:
        """Return a single sample (input_data, target) pair.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (input_data, target_grid) where:
                - input_data: dict containing discontinuity info and coordinates
                - target_grid: tensor of shape (1, nt, nx)
        """
        input_data, target_grid = self.processed_data[idx]

        if self.transform is not None:
            input_data, target_grid = self.transform(input_data, target_grid)
        
        return input_data, target_grid


class FlattenDiscontinuitiesTransform:
    """Transform that flattens discontinuity data into a single tensor.

    This is useful for models that expect a simple tensor input rather than
    a dictionary. The output tensor has shape (n_features,) where n_features
    includes all discontinuity information.
    """

    def __call__(self, input_data: dict, target_grid: torch.Tensor):
        # Flatten: [xs (max_pieces+1), ks (max_pieces), pieces_mask (max_pieces)]
        xs = input_data["xs"]
        ks = input_data["ks"]
        mask = input_data["pieces_mask"]

        flat_input = torch.cat([xs, ks, mask], dim=0)

        return flat_input, target_grid


class ToGridInputTransform:
    """Transform that converts discontinuity data to a grid-like input.

    This reconstructs the initial condition on the grid from discontinuities,
    similar to how operator_learning represents inputs, but also includes
    the raw discontinuity information as additional channels.
    """

    def __init__(self, nx: int, nt: int):
        self.nx = nx
        self.nt = nt

    def __call__(self, input_data: dict, target_grid: torch.Tensor):
        xs = input_data["xs"]
        ks = input_data["ks"]
        mask = input_data["pieces_mask"]
        t_coords = input_data["t_coords"]
        x_coords = input_data["x_coords"]

        # Reconstruct IC on grid from piecewise constant representation
        ic_grid = torch.zeros(self.nx, dtype=torch.float32)
        x_positions = torch.linspace(0, 1, self.nx)

        n_pieces = int(mask.sum().item())
        for i in range(n_pieces):
            x_left = xs[i]
            x_right = xs[i + 1]
            val = ks[i]
            ic_grid[(x_positions >= x_left) & (x_positions < x_right)] = val

        # Handle the rightmost piece (include the boundary)
        if n_pieces > 0:
            ic_grid[x_positions >= xs[n_pieces - 1]] = ks[n_pieces - 1]

        # Expand IC to full grid (repeat across time)
        ic_expanded = ic_grid[None, :].expand(self.nt, self.nx).unsqueeze(0)  # (1, nt, nx)

        # Mask everything except initial condition (like GridMaskAllButInitial)
        ic_masked = ic_expanded.clone()
        ic_masked[:, 1:, :] = -1

        # Stack: [ic_masked, t_coords, x_coords]
        full_input = torch.cat([ic_masked, t_coords, x_coords], dim=0)  # (3, nt, nx)

        return full_input, target_grid


def collate_wavefront_batch(batch):
    """Custom collate function for WavefrontDataset.

    Handles batching of dictionary inputs properly.

    Args:
        batch: List of (input_data, target_grid) tuples.

    Returns:
        Tuple of (batched_input_data, batched_target_grid).
    """
    input_data_list, target_grids = zip(*batch, strict=True)

    # Stack target grids
    batched_targets = torch.stack(target_grids, dim=0)

    # Handle input data based on type
    if isinstance(input_data_list[0], dict):
        # Batch dictionary inputs
        batched_input = {}
        for key in input_data_list[0].keys():
            values = [d[key] for d in input_data_list]
            if isinstance(values[0], torch.Tensor):
                batched_input[key] = torch.stack(values, dim=0)
            else:
                batched_input[key] = values
        return batched_input, batched_targets
    else:
        # Simple tensor input
        batched_input = torch.stack(input_data_list, dim=0)
        return batched_input, batched_targets
