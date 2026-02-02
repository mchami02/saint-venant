"""Testing utilities for wavefront learning models.

This is a standalone script that can be run with:
    python test.py --model_path path/to/model.pth
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
from data import collate_wavefront_batch, get_wavefront_datasets
from logger import WandbLogger, init_logger
from plotter import plot_comparison_wandb, plot_trajectory_wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "mps")


def compute_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """Compute evaluation metrics for predictions.

    Args:
        prediction: Model predictions.
        target: Ground truth targets.

    Returns:
        Dictionary containing MSE, MAE, relative error.
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


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    logger: WandbLogger | None = None,
    loss_fn: nn.Module | None = None,
    nx: int = 50,
    nt: int = 250,
    dx: float = 0.02,
    dt: float = 0.004,
    num_plots: int = 3,
) -> dict[str, float]:
    """Evaluate model on test dataset.

    Args:
        model: Trained model to evaluate.
        test_loader: DataLoader for test data.
        device: Computation device.
        logger: Optional WandbLogger for logging results.
        loss_fn: Optional loss function for trajectory models.
        nx, nt, dx, dt: Grid parameters for plotting.
        num_plots: Number of samples to plot.

    Returns:
        Dictionary of evaluation metrics.
    """
    model.eval()

    all_preds = []
    all_targets = []
    batch_metrics = []
    is_trajectory_model = False

    # For trajectory models: collect data for plotting
    traj_positions = []
    traj_existence = []
    traj_discontinuities = []
    traj_masks = []
    traj_times = None

    with torch.no_grad():
        for batch_input, batch_target in tqdm(test_loader, desc="Testing"):
            # Store original batch_input for trajectory data
            batch_input_orig = batch_input

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

            # Handle trajectory models (dict output)
            if isinstance(pred, dict):
                is_trajectory_model = True
                if loss_fn is not None:
                    _, components = loss_fn(pred, batch_input, batch_target)
                    batch_metrics.append(components)

                # Collect trajectory data for plotting (only first few samples)
                if len(traj_positions) < num_plots:
                    batch_size = pred["positions"].shape[0]
                    for i in range(min(num_plots - len(traj_positions), batch_size)):
                        traj_positions.append(pred["positions"][i].cpu().numpy())
                        traj_existence.append(pred["existence"][i].cpu().numpy())
                        traj_discontinuities.append(batch_input_orig["discontinuities"][i].cpu().numpy())
                        traj_masks.append(batch_input_orig["disc_mask"][i].cpu().numpy())
                    if traj_times is None:
                        t_coords = batch_input_orig["t_coords"]
                        traj_times = t_coords[0, 0, :, 0].cpu().numpy()
            else:
                # Standard models (tensor output)
                metrics = compute_metrics(pred, batch_target)
                batch_metrics.append(metrics)

                # Store for plotting
                for i in range(batch_target.shape[0]):
                    all_preds.append(pred[i].squeeze(0).cpu().numpy())
                    all_targets.append(batch_target[i].squeeze(0).cpu().numpy())

    # Aggregate metrics
    if batch_metrics:
        avg_metrics = {
            key: np.mean([m[key] for m in batch_metrics]) for key in batch_metrics[0].keys()
        }
    else:
        avg_metrics = {}

    # Print results
    print("\nTest Results:")
    print("-" * 40)
    for key, value in avg_metrics.items():
        print(f"  {key}: {value:.6f}")
    print("-" * 40)

    # Log to W&B if logger provided
    if logger is not None:
        logger.log_metrics({f"test/{k}": v for k, v in avg_metrics.items()})

        if is_trajectory_model and traj_positions:
            # Plot trajectory for trajectory models
            plot_trajectory_wandb(
                np.array(traj_positions),
                np.array(traj_existence),
                np.array(traj_discontinuities),
                np.array(traj_masks),
                traj_times,
                logger,
                epoch=0,
                mode="test",
            )
        elif not is_trajectory_model and all_targets:
            # Plot comparison for standard models
            gts = np.array(all_targets[:num_plots])
            preds = np.array(all_preds[:num_plots])
            plot_comparison_wandb(gts, preds, nx, nt, dx, dt, logger, epoch=0, mode="test")

    return avg_metrics


def run_inference(
    model: nn.Module,
    input_data: dict | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Run inference on input data.

    Args:
        model: Trained model.
        input_data: Input tensor or dict.
        device: Computation device.

    Returns:
        Model prediction tensor.
    """
    model.eval()

    with torch.no_grad():
        if isinstance(input_data, dict):
            input_data = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in input_data.items()
            }
        else:
            input_data = input_data.to(device)

        pred = model(input_data)

    return pred


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test wavefront prediction model")

    # Required argument
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model checkpoint"
    )

    # Data arguments (with defaults matching training)
    parser.add_argument("--n_samples", type=int, default=200, help="Number of test samples")
    parser.add_argument("--nx", type=int, default=50, help="Number of spatial points")
    parser.add_argument("--nt", type=int, default=250, help="Number of time steps")
    parser.add_argument("--dx", type=float, default=0.02, help="Spatial step size")
    parser.add_argument("--dt", type=float, default=0.004, help="Time step size")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")

    # Output arguments
    parser.add_argument("--num_plots", type=int, default=3, help="Number of samples to plot")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")

    return parser.parse_args()


def main():
    """Main entry point for standalone testing."""
    args = parse_args()

    print(f"Using device: {device}")
    print(f"Loading model from: {args.model_path}")

    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)

    # Extract model config from checkpoint if available
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        model_config = checkpoint.get("config", {})
    else:
        state_dict = checkpoint
        model_config = {}

    # Import model creation - this requires model.py to have proper model registry
    from model import MODELS

    # Try to determine model type from checkpoint or use default
    model_name = model_config.get("model", "fno")

    if model_name not in MODELS:
        print(f"Warning: Model '{model_name}' not found in registry. Available: {list(MODELS.keys())}")
        print("Please ensure your model is registered in model.py")
        return

    # Create model with config
    model_class = MODELS[model_name]
    model = model_class(
        in_channels=model_config.get("in_channels", 3),
        out_channels=model_config.get("out_channels", 1),
        hidden_channels=model_config.get("hidden_channels", 64),
    )
    model.load_state_dict(state_dict)
    model = model.to(device)

    print(f"Model loaded: {model_name}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Generate test data
    print("\nGenerating test data...")
    _, _, test_dataset = get_wavefront_datasets(
        n_samples=args.n_samples,
        nx=args.nx,
        nt=args.nt,
        dx=args.dx,
        dt=args.dt,
        train_ratio=0.0,  # All data for testing
        val_ratio=0.0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_wavefront_batch,
    )

    # Initialize logger
    logger = None
    if not args.no_wandb:
        args.model = model_name
        logger = init_logger(args, project="wavefront-learning-test")

    # Run evaluation
    metrics = test_model(
        model=model,
        test_loader=test_loader,
        device=device,
        logger=logger,
        nx=args.nx,
        nt=args.nt,
        dx=args.dx,
        dt=args.dt,
        num_plots=args.num_plots,
    )

    # Cleanup
    if logger is not None:
        logger.finish()

    print("\nTesting complete!")
    return metrics


if __name__ == "__main__":
    main()
