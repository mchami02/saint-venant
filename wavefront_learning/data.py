"""Dataset classes and transforms for wavefront learning.

This module provides the train.py-facing interfaces:
- WavefrontDataset: PyTorch Dataset class
- Transforms for different input representations
- Collate function for batching
- get_wavefront_datasets: Main entry point for getting train/val/test datasets

Data generation and HuggingFace integration are handled by:
- data_processing.py: Grid generation and preprocessing
- data_loading.py: HuggingFace upload/download
"""

import numpy as np
import torch
from data_processing import get_wavefront_data
from torch.utils.data import Dataset


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

    def __init__(self, nx: int, nt: int, **kwargs):
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
        ic_expanded = (
            ic_grid[None, :].expand(self.nt, self.nx).unsqueeze(0)
        )  # (1, nt, nx)

        # Mask everything except initial condition (like GridMaskAllButInitial)
        ic_masked = ic_expanded.clone()
        ic_masked[:, 1:, :] = -1

        # Stack: [ic_masked, t_coords, x_coords]
        full_input = torch.cat([ic_masked, t_coords, x_coords], dim=0)  # (3, nt, nx)

        return full_input, target_grid


# Registry of available transforms (string name -> class)
TRANSFORMS = {
    "FlattenDiscontinuities": FlattenDiscontinuitiesTransform,
    "ToGridInput": ToGridInputTransform,
}


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


def get_wavefront_datasets(
    n_samples: int,
    grid_config: dict,
    model_name: str,
    max_steps: int = 3,
    max_discontinuities: int = 10,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    random_seed: int = 42,
    only_shocks: bool = True,
) -> tuple[WavefrontDataset, WavefrontDataset, WavefrontDataset]:
    """Get train, val, and test datasets for wavefront learning.

    This function handles data loading (from HuggingFace or generating locally),
    preprocessing, and splitting into train/val/test sets.

    Args:
        n_samples: Total number of samples.
        grid_config: Dict with keys nx, nt, dx, dt.
        model_name: Model name (key into MODEL_TRANSFORM for per-model transforms).
        max_steps: Maximum number of pieces in piecewise constant IC.
        max_discontinuities: Maximum number of discontinuities to support.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        random_seed: Random seed for reproducibility.
        only_shocks: If True, generate only shock waves (no rarefactions).

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    from model import MODEL_TRANSFORM

    nx = grid_config["nx"]
    nt = grid_config["nt"]
    dx = grid_config["dx"]
    dt = grid_config["dt"]

    np.random.seed(random_seed)

    # Get preprocessed data (handles HF download/upload internally)
    processed = get_wavefront_data(
        n_samples=n_samples,
        nx=nx,
        nt=nt,
        dx=dx,
        dt=dt,
        max_steps=max_steps,
        only_shocks=only_shocks,
        random_seed=random_seed,
        max_discontinuities=max_discontinuities,
    )

    # Resolve per-model transform
    transform_name = MODEL_TRANSFORM.get(model_name)
    transform = None
    if transform_name is not None:
        if transform_name not in TRANSFORMS:
            raise ValueError(
                f"Transform '{transform_name}' not found. "
                f"Available: {list(TRANSFORMS.keys())}"
            )
        transform = TRANSFORMS[transform_name](**grid_config)

    # Shuffle indices
    indices = np.arange(len(processed))
    np.random.shuffle(indices)

    # Split indices
    n_train = int(train_ratio * len(processed))
    n_val = int(val_ratio * len(processed))

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    # Create dataset splits
    train_processed = [processed[i] for i in train_idx]
    val_processed = [processed[i] for i in val_idx]
    test_processed = [processed[i] for i in test_idx]

    train_dataset = WavefrontDataset(train_processed, transform=transform)
    val_dataset = WavefrontDataset(val_processed, transform=transform)
    test_dataset = WavefrontDataset(test_processed, transform=transform)

    print(
        f"Created datasets: train={len(train_dataset)}, "
        f"val={len(val_dataset)}, test={len(test_dataset)}"
    )

    return train_dataset, val_dataset, test_dataset
