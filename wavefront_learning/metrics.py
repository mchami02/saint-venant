"""Shared metrics utilities for wavefront learning.

Provides functions to compute evaluation metrics and extract grid predictions
from various model output formats.
"""

import torch


def compute_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """Compute evaluation metrics for predictions.

    Args:
        prediction: Model predictions.
        target: Ground truth targets.

    Returns:
        Dictionary containing MSE, MAE, relative L2, and max error.
    """
    mse = torch.mean((prediction - target) ** 2).item()
    mae = torch.mean(torch.abs(prediction - target)).item()

    # Relative L2 error
    rel_l2 = torch.norm(prediction - target) / torch.norm(target)
    rel_l2 = rel_l2.item() if not torch.isnan(rel_l2) else float("inf")

    # Max absolute error
    max_error = torch.max(torch.abs(prediction - target)).item()

    return {
        "mse": mse,
        "mae": mae,
        "rel_l2": rel_l2,
        "max_error": max_error,
    }


def cell_average_prediction(
    prediction: torch.Tensor,
    k: int,
    original_nx: int,
) -> torch.Tensor:
    """Average k per-cell query predictions back to cell-level values.

    Reshapes prediction from (..., nt, nx*k) to (..., nt, nx, k) and
    averages over the k dimension.

    Args:
        prediction: Tensor with last dim = nx*k.
        k: Number of query points per cell.
        original_nx: Original number of cells.

    Returns:
        Tensor with last dim = nx (cell-averaged).
    """
    shape = prediction.shape
    *leading, nt, nxk = shape
    return prediction.reshape(*leading, nt, original_nx, k).mean(dim=-1)


def extract_grid_prediction(model_output: dict | torch.Tensor) -> torch.Tensor | None:
    """Extract grid prediction from model output.

    Handles different model output formats:
    - Standard tensor output: returns as-is
    - HybridDeepONet dict with 'output_grid': returns the grid
    - ShockNet dict with only 'positions'/'existence': returns None (no grid)

    Args:
        model_output: Model output (tensor or dict).

    Returns:
        Grid tensor if available, None otherwise.
    """
    if isinstance(model_output, dict):
        if "output_grid" in model_output:
            return model_output["output_grid"]  # HybridDeepONet
        return None  # ShockNet - trajectory only, no grid
    return model_output  # Standard tensor


def can_compute_grid_metrics(model_output: dict | torch.Tensor) -> bool:
    """Check if grid-based metrics can be computed for this model output.

    Args:
        model_output: Model output (tensor or dict).

    Returns:
        True if grid metrics can be computed, False otherwise.
    """
    return extract_grid_prediction(model_output) is not None
