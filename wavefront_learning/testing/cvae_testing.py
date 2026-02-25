"""Multi-sample inference utilities for CVAE models.

Provides functions for generating multiple predictions from different
latent samples and packaging them for visualization.
"""

import numpy as np
import torch
import torch.nn as nn


def multi_sample_predict(
    model: nn.Module,
    input_dict: dict[str, torch.Tensor],
    n_samples: int = 10,
) -> dict[str, torch.Tensor]:
    """Run model n_samples times with different latent draws.

    The model must be in eval mode (no target_grid in input).
    Each forward pass samples a different z from the prior.

    Args:
        model: CVAE model instance.
        input_dict: Input dict (should NOT contain "target_grid").
        n_samples: Number of latent samples to draw.

    Returns:
        Dict with:
            - samples: (B, n_samples, 1, nt, nx) all predictions
            - mean: (B, 1, nt, nx) mean prediction
            - std: (B, 1, nt, nx) std prediction
    """
    # Ensure target_grid is not present (use prior)
    clean_input = {k: v for k, v in input_dict.items() if k != "target_grid"}

    all_preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            output = model(clean_input)
            all_preds.append(output["output_grid"])  # (B, 1, nt, nx)

    # Stack: (B, n_samples, 1, nt, nx)
    samples = torch.stack(all_preds, dim=1)
    mean = samples.mean(dim=1)  # (B, 1, nt, nx)
    std = samples.std(dim=1)  # (B, 1, nt, nx)

    return {"samples": samples, "mean": mean, "std": std}


def collect_cvae_samples(
    model: nn.Module,
    batch_input: dict[str, torch.Tensor],
    batch_target: torch.Tensor,
    n_samples: int = 10,
    num_plot_samples: int = 2,
) -> dict[str, np.ndarray]:
    """Collect CVAE multi-sample predictions for plotting.

    Calls multi_sample_predict and packages results as numpy arrays
    compatible with the plotting functions.

    Args:
        model: CVAE model instance (should be in eval mode).
        batch_input: Input batch dict.
        batch_target: Target batch tensor.
        n_samples: Number of latent samples.
        num_plot_samples: Number of batch elements to collect.

    Returns:
        Dict with cvae_samples, cvae_mean, cvae_std as numpy arrays.
    """
    result = multi_sample_predict(model, batch_input, n_samples)

    # Take first num_plot_samples from batch, squeeze channel dim
    samples = result["samples"][:num_plot_samples].cpu().numpy()  # (P, N, 1, nt, nx)
    samples = samples.squeeze(2)  # (P, N, nt, nx)

    mean = result["mean"][:num_plot_samples].squeeze(1).cpu().numpy()  # (P, nt, nx)
    std = result["std"][:num_plot_samples].squeeze(1).cpu().numpy()  # (P, nt, nx)

    return {
        "cvae_samples": samples,
        "cvae_mean": mean,
        "cvae_std": std,
    }
