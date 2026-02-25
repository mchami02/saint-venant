"""Functions for measuring model performance.

Includes evaluation metrics, sample collection for visualization,
and test-set evaluation with logging.
"""

import time

import numpy as np
import torch
import torch.nn as nn
from data import collate_wavefront_batch, get_wavefront_datasets
from logger import WandbLogger
from metrics import compute_metrics
from plotter import plot
from torch.utils.data import DataLoader
from tqdm import tqdm


def _get_equation_kwargs(args) -> dict | None:
    """Extract ARZ-specific kwargs from args, if applicable."""
    if getattr(args, "equation", "LWR") != "ARZ":
        return None
    return {
        "gamma": getattr(args, "gamma", 1.0),
        "flux_type": getattr(args, "flux_type", "hll"),
        "reconstruction": getattr(args, "reconstruction", "weno5"),
        "bc_type": getattr(args, "bc_type", "zero_gradient"),
    }


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
                # Handle scalar (0-D) tensors (e.g. temperature)
                if v.ndim == 0:
                    samples[k] = v.detach().cpu().item()
                    continue
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
        for key in ("xs", "ks", "pieces_mask"):
            if key in batch_input:
                samples[key] = batch_input[key][:num_samples].detach().cpu().numpy()

    # Add target (ground truth grid)
    grids = batch_target[:num_samples].detach().cpu().numpy()
    if grids.ndim == 4 and grids.shape[1] == 1:
        grids = grids.squeeze(1)
    samples["grids"] = grids

    return samples


def eval_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module | None = None,
    grid_config: dict | None = None,
    num_plots: int = 3,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Evaluate model on a dataset.

    Pure computation: runs inference, computes metrics, collects samples.
    Does not log or plot results.

    Args:
        model: Trained model to evaluate.
        test_loader: DataLoader for evaluation data.
        device: Computation device.
        loss_fn: Optional loss function for trajectory models.
        grid_config: Dict with {nx, nt, dx, dt}. Defaults provided if None.
        num_plots: Number of samples to collect for plotting.

    Returns:
        Tuple of (metrics_dict, samples_dict).
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

    # Collect samples from last batch
    samples = collect_samples(pred, batch_input, batch_target, num_samples=num_plots)

    return avg_metrics, samples


def eval_res(
    model: nn.Module,
    args,
    device: torch.device,
    new_res: int,
    loss_fn: nn.Module | None = None,
    num_plots: int = 3,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Evaluate model on higher resolution grids.

    Generates data at new_res× resolution (new_res*nx, new_res*nt,
    dx/new_res, dt/new_res) covering the same physical domain but with
    a finer grid. Since wavefront models use coordinate-based inputs
    (discontinuity positions + coordinate grids), they should generalize
    to arbitrary resolutions.

    Does not log or plot results.

    Args:
        model: Trained model to evaluate.
        args: Training arguments (needs nx, nt, dx, dt, batch_size, n_samples, model).
        device: Computation device.
        new_res: Resolution multiplier (e.g. 2 for 2× resolution).
        loss_fn: Optional loss function for trajectory models.
        num_plots: Number of samples to collect for plotting.

    Returns:
        Tuple of (metrics_dict, samples_dict).
    """
    high_res_nx = args.nx * new_res
    high_res_nt = args.nt * new_res
    high_res_dx = args.dx / new_res
    high_res_dt = args.dt / new_res
    n_samples = max(100, args.n_samples // 10)

    grid_config = {
        "nx": high_res_nx,
        "nt": high_res_nt,
        "dx": high_res_dx,
        "dt": high_res_dt,
    }

    print(
        f"\n{new_res}x res test: nx={high_res_nx}, nt={high_res_nt}, "
        f"dx={high_res_dx}, dt={high_res_dt} ({n_samples} samples)"
    )

    # Generate high-res dataset (all samples go to test split)
    _, _, high_res_dataset = get_wavefront_datasets(
        n_samples=n_samples,
        grid_config=grid_config,
        model_name=args.model,
        train_ratio=0.0,
        val_ratio=0.0,
        only_shocks=False,
        equation=getattr(args, "equation", "LWR"),
        equation_kwargs=_get_equation_kwargs(args),
    )

    high_res_loader = DataLoader(
        high_res_dataset,
        batch_size=max(2, args.batch_size // new_res),
        shuffle=False,
        collate_fn=collate_wavefront_batch,
    )

    return eval_model(
        model=model,
        test_loader=high_res_loader,
        device=device,
        loss_fn=loss_fn,
        grid_config=grid_config,
        num_plots=num_plots,
    )


def eval_high_res(
    model: nn.Module,
    args,
    device: torch.device,
    loss_fn: nn.Module | None = None,
    num_plots: int = 3,
) -> dict[int, tuple[dict[str, float], dict[str, np.ndarray]]]:
    """Evaluate model on increasing resolution multipliers.

    For each resolution multiplier from 2 to args.max_high_res (inclusive),
    generates a test dataset at that resolution and evaluates the model.

    Args:
        model: Trained model to evaluate.
        args: Training arguments (needs nx, nt, dx, dt, batch_size, n_samples,
              model, and max_high_res).
        device: Computation device.
        loss_fn: Optional loss function for trajectory models.
        num_plots: Number of samples to collect for plotting.

    Returns:
        Dict mapping resolution multiplier to (metrics_dict, samples_dict).
    """
    max_high_res = getattr(args, "max_high_res", 5)
    all_res_results = {}

    for res in range(2, max_high_res + 1):
        metrics, samples = eval_res(
            model=model,
            args=args,
            device=device,
            new_res=res,
            loss_fn=loss_fn,
            num_plots=num_plots,
        )
        all_res_results[res] = (metrics, samples)

    return all_res_results


def eval_steps(
    model: nn.Module,
    args,
    device: torch.device,
    loss_fn: nn.Module | None = None,
    grid_config: dict | None = None,
    num_plots: int = 3,
) -> dict[int, tuple[dict[str, float], dict[str, np.ndarray]]]:
    """Evaluate model on increasing step counts to test generalization.

    For each step count from 2 to args.max_test_steps (inclusive),
    generates a test dataset with that many pieces in the piecewise constant IC
    and evaluates the model.

    Args:
        model: Trained model to evaluate.
        args: Training arguments (needs max_steps, max_test_steps, nx, nt, dx, dt,
              batch_size, n_samples, model, and optionally only_shocks).
        device: Computation device.
        loss_fn: Optional loss function for trajectory models.
        grid_config: Dict with {nx, nt, dx, dt}. Defaults from args if None.
        num_plots: Number of samples to collect for plotting.

    Returns:
        Dict mapping step count to (metrics_dict, samples_dict).
    """
    if grid_config is None:
        grid_config = {"nx": args.nx, "nt": args.nt, "dx": args.dx, "dt": args.dt}

    n_samples = max(100, args.n_samples // 10)
    only_shocks = getattr(args, "only_shocks", False)

    all_step_results = {}

    for n_steps in range(2, args.max_test_steps + 1):
        print(f"\nStep-count test: max_steps={n_steps} ({n_samples} samples)")

        _, _, step_dataset = get_wavefront_datasets(
            n_samples=n_samples,
            grid_config=grid_config,
            model_name=args.model,
            max_steps=n_steps,
            min_steps=n_steps,
            train_ratio=0.0,
            val_ratio=0.0,
            only_shocks=only_shocks,
            equation=getattr(args, "equation", "LWR"),
            equation_kwargs=_get_equation_kwargs(args),
        )

        step_loader = DataLoader(
            step_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_wavefront_batch,
        )

        metrics, samples = eval_model(
            model=model,
            test_loader=step_loader,
            device=device,
            loss_fn=loss_fn,
            grid_config=grid_config,
            num_plots=num_plots,
        )

        all_step_results[n_steps] = (metrics, samples)

    return all_step_results


def _sync_device(device: torch.device) -> None:
    """Synchronize device to ensure all operations are complete."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def eval_inference_time(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_iterations: int = 50,
    warmup_iterations: int = 5,
) -> dict[str, float]:
    """Measure forward pass inference time.

    Runs the model on a single batch repeatedly to measure latency.
    Includes warmup iterations and proper device synchronization.

    Args:
        model: Trained model to benchmark.
        test_loader: DataLoader to grab a batch from.
        device: Computation device.
        num_iterations: Number of timed forward passes.
        warmup_iterations: Number of untimed warmup passes.

    Returns:
        Dict with timing stats in milliseconds:
        {"mean_ms", "std_ms", "min_ms", "max_ms"}.
    """
    model.eval()

    # Grab a single batch and move to device
    batch_input, _ = next(iter(test_loader))
    if isinstance(batch_input, dict):
        batch_input = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch_input.items()
        }
    else:
        batch_input = batch_input.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            model(batch_input)

    # Timed iterations
    _sync_device(device)
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            model(batch_input)
            _sync_device(device)
            times.append((time.perf_counter() - start) * 1000)

    times_arr = np.array(times)
    return {
        "mean_ms": float(times_arr.mean()),
        "std_ms": float(times_arr.std()),
        "min_ms": float(times_arr.min()),
        "max_ms": float(times_arr.max()),
    }


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    args,
    device: torch.device,
    logger: WandbLogger | None = None,
    loss_fn: nn.Module | None = None,
    grid_config: dict | None = None,
    num_plots: int = 3,
    plot_preset: str | None = None,
) -> dict[str, float]:
    """Full test pipeline: evaluate on test set and high-res, log and plot results.

    Orchestrates eval_model and eval_high_res, then handles printing,
    W&B logging, and plotting.

    Args:
        model: Trained model to evaluate.
        test_loader: DataLoader for test data.
        args: Training arguments (needs nx, nt, dx, dt, batch_size, n_samples, model).
        device: Computation device.
        logger: Optional WandbLogger for logging results.
        loss_fn: Optional loss function for trajectory models.
        grid_config: Dict with {nx, nt, dx, dt} for plotting. Defaults provided if None.
        num_plots: Number of samples to plot.
        plot_preset: Plot preset name.

    Returns:
        Dictionary of evaluation metrics (standard resolution).
    """
    if grid_config is None:
        grid_config = {"nx": args.nx, "nt": args.nt, "dx": args.dx, "dt": args.dt}

    # --- Standard resolution evaluation ---
    metrics, samples = eval_model(
        model=model,
        test_loader=test_loader,
        device=device,
        loss_fn=loss_fn,
        grid_config=grid_config,
        num_plots=num_plots,
    )

    # Print results
    print("\nTest Results:")
    print("-" * 40)
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    print("-" * 40)

    # Log to W&B and plot
    if logger is not None:
        logger.log_summary({f"test/metrics/{k}": v for k, v in metrics.items()})
        plot(samples, grid_config, logger, epoch=None, mode="test", preset=plot_preset)

    # --- Inference time ---
    timing = eval_inference_time(model, test_loader, device)

    print("\nInference Time (per batch):")
    print("-" * 40)
    for key, value in timing.items():
        print(f"  {key}: {value:.2f}")
    print("-" * 40)

    if logger is not None:
        logger.log_summary({f"test/inference_time/{k}": v for k, v in timing.items()})

    # --- High-resolution evaluation ---
    max_high_res = getattr(args, "max_high_res", 5)
    print(f"\nRunning high-resolution evaluation (2x..{max_high_res}x)...")
    hr_results = eval_high_res(
        model=model,
        args=args,
        device=device,
        loss_fn=loss_fn,
        num_plots=num_plots,
    )

    for res, (hr_metrics, hr_samples) in hr_results.items():
        print(f"\nHigh-Res Test ({res}x):")
        print("-" * 40)
        for key, value in hr_metrics.items():
            print(f"  {key}: {value:.6f}")
        print("-" * 40)

        if logger is not None:
            logger.log_summary(
                {f"test_high_res/{res}/{k}": v for k, v in hr_metrics.items()}
            )
            hr_grid_config = {
                "nx": args.nx * res,
                "nt": args.nt * res,
                "dx": args.dx / res,
                "dt": args.dt / res,
            }
            plot(
                hr_samples,
                hr_grid_config,
                logger,
                epoch=None,
                mode=f"test_high_res/{res}",
                preset=plot_preset,
            )

    # --- Step-count generalization evaluation ---
    max_test_steps = getattr(args, "max_test_steps", None)
    max_steps = getattr(args, "max_steps", 3)

    if max_test_steps is not None and max_test_steps >= 2:
        print(
            f"\nRunning step-count generalization test "
            f"(max_steps=2..{max_test_steps})..."
        )
        step_results = eval_steps(
            model=model,
            args=args,
            device=device,
            loss_fn=loss_fn,
            grid_config=grid_config,
            num_plots=num_plots,
        )

        for n_steps, (step_metrics, step_samples) in step_results.items():
            print(f"\nStep-Count Test (max_steps={n_steps}):")
            print("-" * 40)
            for key, value in step_metrics.items():
                print(f"  {key}: {value:.6f}")
            print("-" * 40)

            if logger is not None:
                logger.log_summary(
                    {f"test_steps/{n_steps}/{k}": v for k, v in step_metrics.items()}
                )
                plot(
                    step_samples,
                    grid_config,
                    logger,
                    epoch=None,
                    mode=f"test_steps/{n_steps}",
                    preset=plot_preset,
                )

    return metrics
