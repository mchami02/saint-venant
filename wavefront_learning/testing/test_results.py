"""Functions for measuring model performance.

Includes evaluation metrics, sample collection for visualization,
and test-set evaluation with logging.
"""

import numpy as np
import torch
import torch.nn as nn
from data import collate_wavefront_batch, get_wavefront_datasets
from logger import WandbLogger
from metrics import compute_metrics
from plotter import plot
from torch.utils.data import DataLoader
from tqdm import tqdm


def collect_samples(
    pred: dict | torch.Tensor,
    batch_input: dict | torch.Tensor,
    batch_target: torch.Tensor,
    num_samples: int = 2,
) -> dict[str, np.ndarray]:
    """Collect sample predictions from a batch for plotting.

    Iterates through the model output dict to automatically collect all outputs.

    Args:
        pred: Model prediction (dict or tensor).
        batch_input: Input batch (dict or tensor).
        batch_target: Target batch tensor.
        num_samples: Number of samples to collect.

    Returns:
        Dict with all model outputs, input context, and target as numpy arrays.
    """
    samples = {}

    if isinstance(pred, dict):
        for k, v in pred.items():
            if isinstance(v, torch.Tensor):
                arr = v[:num_samples].detach().cpu().numpy()
                # Squeeze channel dim for grid outputs: (B, 1, H, W) -> (B, H, W)
                if arr.ndim == 4 and arr.shape[1] == 1:
                    arr = arr.squeeze(1)
                samples[k] = arr

    # Add input context for trajectory plots
    if isinstance(batch_input, dict):
        if "discontinuities" in batch_input:
            samples["discontinuities"] = (
                batch_input["discontinuities"][:num_samples].detach().cpu().numpy()
            )
        if "disc_mask" in batch_input:
            samples["masks"] = (
                batch_input["disc_mask"][:num_samples].detach().cpu().numpy()
            )
        if "t_coords" in batch_input:
            samples["times"] = (
                batch_input["t_coords"][0, 0, :, 0].detach().cpu().numpy()
            )

    # Add target (ground truth grid)
    grids = batch_target[:num_samples].detach().cpu().numpy()
    if grids.ndim == 4 and grids.shape[1] == 1:
        grids = grids.squeeze(1)
    samples["grids"] = grids

    return samples


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    logger: WandbLogger | None = None,
    loss_fn: nn.Module | None = None,
    grid_config: dict | None = None,
    num_plots: int = 3,
    epoch: int = 0,  # noqa: ARG001 - kept for API compat, now unused
    plot_preset: str | None = None,
    mode: str = "test",
) -> dict[str, float]:
    """Evaluate model on test dataset.

    Args:
        model: Trained model to evaluate.
        test_loader: DataLoader for test data.
        device: Computation device.
        logger: Optional WandbLogger for logging results.
        loss_fn: Optional loss function for trajectory models.
        grid_config: Dict with {nx, nt, dx, dt} for plotting. Defaults provided if None.
        num_plots: Number of samples to plot.
        mode: Logging mode prefix (e.g. "test", "test_high_res").

    Returns:
        Dictionary of evaluation metrics.
    """
    # Default grid_config if not provided
    if grid_config is None:
        grid_config = {"nx": 50, "nt": 250, "dx": 0.02, "dt": 0.004}

    model.eval()
    batch_metrics = []

    with torch.no_grad():
        for batch_input, batch_target in tqdm(test_loader, desc="Testing"):
            # Move to device
            if isinstance(batch_input, dict):
                batch_input = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch_input.items()
                }
            else:
                batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)

            # Forward pass
            pred = model(batch_input)

            # Compute metrics
            if isinstance(pred, dict):
                if "output_grid" in pred:
                    batch_metrics.append(
                        compute_metrics(pred["output_grid"], batch_target)
                    )
                elif loss_fn is not None:
                    _, components = loss_fn(batch_input, pred, batch_target)
                    batch_metrics.append(components)
            else:
                batch_metrics.append(compute_metrics(pred, batch_target))

    # Aggregate metrics
    if batch_metrics:
        avg_metrics = {
            key: np.mean([m[key] for m in batch_metrics])
            for key in batch_metrics[0].keys()
        }
    else:
        avg_metrics = {}

    # Print results
    print("\nTest Results:")
    print("-" * 40)
    for key, value in avg_metrics.items():
        print(f"  {key}: {value:.6f}")
    print("-" * 40)

    # Log to W&B and plot
    if logger is not None:
        logger.log_summary({f"{mode}/metrics/{k}": v for k, v in avg_metrics.items()})

        samples = collect_samples(pred, batch_input, batch_target, num_samples=num_plots)
        plot(
            samples, grid_config, logger, epoch=None, mode=mode, preset=plot_preset
        )

    return avg_metrics


def test_high_res(
    model: nn.Module,
    args,
    device: torch.device,
    logger: WandbLogger | None = None,
    loss_fn: nn.Module | None = None,
    plot_preset: str | None = None,
) -> dict[str, float]:
    """Test model on 2x higher resolution grids.

    Generates data at double resolution (2*nx, 2*nt, dx/2, dt/2) covering
    the same physical domain but with a finer grid. Since wavefront models
    use coordinate-based inputs (discontinuity positions + coordinate grids),
    they should generalize to arbitrary resolutions.

    Args:
        model: Trained model to evaluate.
        args: Training arguments (needs nx, nt, dx, dt, batch_size, n_samples).
        device: Computation device.
        logger: Optional WandbLogger for logging results.
        loss_fn: Optional loss function for trajectory models.
        plot_preset: Plot preset name.

    Returns:
        Dictionary of evaluation metrics on high-res grids.
    """
    high_res_nx = args.nx * 2
    high_res_nt = args.nt * 2
    high_res_dx = args.dx / 2
    high_res_dt = args.dt / 2
    n_samples = max(100, args.n_samples // 10)

    grid_config = {
        "nx": high_res_nx,
        "nt": high_res_nt,
        "dx": high_res_dx,
        "dt": high_res_dt,
    }

    print(
        f"\nHigh-res test: nx={high_res_nx}, nt={high_res_nt}, "
        f"dx={high_res_dx}, dt={high_res_dt} ({n_samples} samples)"
    )

    # Generate high-res dataset (all samples go to test split)
    _, _, high_res_dataset = get_wavefront_datasets(
        n_samples=n_samples,
        grid_config=grid_config,
        model_name=args.model,
        train_ratio=0.0,
        val_ratio=0.0,
    )

    high_res_loader = DataLoader(
        high_res_dataset,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        collate_fn=collate_wavefront_batch,
    )

    metrics = test_model(
        model=model,
        test_loader=high_res_loader,
        device=device,
        logger=logger,
        loss_fn=loss_fn,
        grid_config=grid_config,
        plot_preset=plot_preset,
        mode="test_high_res",
    )

    return metrics
