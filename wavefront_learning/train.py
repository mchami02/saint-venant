"""Main training script for wavefront learning.

This is a standalone script that can be run with:
    python train.py --model fno --epochs 100

Minimal arguments are required - sensible defaults are provided for all parameters.
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
from data import collate_wavefront_batch, get_wavefront_datasets
from logger import WandbLogger, init_logger, log_values
from loss import LOSS_PRESETS, LOSSES, get_loss
from metrics import compute_metrics, extract_grid_prediction
from model import MODELS, get_model, load_model, save_model
from plotter import PLOT_PRESETS, plot
from testing import (
    collect_samples,
    run_profiler,
    run_sanity_check,
    test_model,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Minimal set of arguments - most have sensible defaults.
    """
    parser = argparse.ArgumentParser(description="Train wavefront prediction model")

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="shock_net",
        choices=list(MODELS.keys()) if MODELS else ["fno"],
        help="Model architecture",
    )

    # Loss selection
    parser.add_argument(
        "--loss",
        type=str,
        default="shock_net",
        choices=list(LOSSES.keys()) + list(LOSS_PRESETS.keys()),
        help="Loss function or preset (shock_net, hybrid, or individual losses)",
    )

    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    # Data parameters
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--nx", type=int, default=50, help="Spatial grid points")
    parser.add_argument("--nt", type=int, default=250, help="Time steps")
    parser.add_argument("--dx", type=float, default=0.02, help="Spatial step size")
    parser.add_argument("--dt", type=float, default=0.004, help="Time step size")
    parser.add_argument(
        "--only_shocks", action="store_true", help="Generate only shock waves (no rarefactions)"
    )

    # Output
    parser.add_argument(
        "--save_path", type=str, default="wavefront_model.pth", help="Model save path"
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--run_name", type=str, default=None, help="W&B run name")

    # Debugging/profiling
    parser.add_argument(
        "--profile", action="store_true", help="Run profiler before training"
    )

    # Plot preset
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        choices=list(PLOT_PRESETS.keys()),
        help="Plot preset (default: auto-detect based on model)",
    )

    return parser.parse_args()


def detach_output(pred):
    """Detach model output (handles both tensor and dict)."""
    if isinstance(pred, dict):
        return {
            k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in pred.items()
        }
    return pred.detach()


def train_step(
    model: nn.Module,
    batch_input: dict | torch.Tensor,
    batch_target: torch.Tensor,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, dict | torch.Tensor, dict[str, float]]:
    """Execute a single training step.

    Args:
        model: Model to train.
        batch_input: Input tensor or dict.
        batch_target: Target tensor.
        loss_fn: Loss function.
        optimizer: Optimizer.

    Returns:
        Tuple of (loss_value, prediction, loss_components).
    """
    optimizer.zero_grad()

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
    pred = model(
        batch_input,
    )

    # Compute loss (input_dict, output_dict, target)
    loss, components = loss_fn(batch_input, pred, batch_target)

    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item(), detach_output(pred), components


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, dict[str, float], dict[str, float] | None, dict[str, np.ndarray]]:
    """Train for one epoch.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        loss_fn: Loss function.
        optimizer: Optimizer.

    Returns:
        Tuple of (average_loss, loss_components, grid_metrics | None, samples).
        grid_metrics is None for trajectory-only models (ShockNet).
        samples is a dict with all model outputs, input context, and target.
    """
    model.train()
    total_loss = 0.0
    all_components = []
    all_predictions = []
    all_targets = []

    for batch_input, batch_target in tqdm(train_loader, desc="Training", leave=False):
        loss, pred, components = train_step(
            model, batch_input, batch_target, loss_fn, optimizer
        )
        total_loss += loss
        all_components.append(components)

        # Collect grid predictions for metrics (if available)
        grid_pred = extract_grid_prediction(pred)
        if grid_pred is not None:
            all_predictions.append(grid_pred.cpu())
            all_targets.append(batch_target.cpu())

    avg_loss = total_loss / len(train_loader)
    # Average loss components
    avg_loss_components = {
        key: np.mean([c[key] for c in all_components])
        for key in all_components[0].keys()
    }

    # Compute grid metrics if predictions available
    grid_metrics = None
    if all_predictions:
        all_preds_tensor = torch.cat(all_predictions, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)
        grid_metrics = compute_metrics(all_preds_tensor, all_targets_tensor)

    # Collect samples from last batch for plotting
    samples = collect_samples(pred, batch_input, batch_target)

    return avg_loss, avg_loss_components, grid_metrics, samples


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
) -> tuple[float, dict[str, float], dict[str, float] | None, dict[str, np.ndarray]]:
    """Validate for one epoch.

    Args:
        model: Model to validate.
        val_loader: Validation data loader.
        loss_fn: Loss function.

    Returns:
        Tuple of (average_loss, loss_components, grid_metrics | None, samples).
        grid_metrics is None for trajectory-only models (ShockNet).
        samples is a dict with all model outputs, input context, and target.
    """
    model.eval()
    total_loss = 0.0
    all_components = []
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_input, batch_target in tqdm(
            val_loader, desc="Validating", leave=False
        ):
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
            # Compute loss (input_dict, output_dict, target)
            loss, components = loss_fn(batch_input, pred, batch_target)

            total_loss += loss.item()
            all_components.append(components)

            # Collect grid predictions for metrics (if available)
            grid_pred = extract_grid_prediction(pred)
            if grid_pred is not None:
                all_predictions.append(grid_pred.cpu())
                all_targets.append(batch_target.cpu())

    avg_loss = total_loss / len(val_loader)
    avg_loss_components = {
        key: np.mean([c[key] for c in all_components])
        for key in all_components[0].keys()
    }

    # Compute grid metrics if predictions available
    grid_metrics = None
    if all_predictions:
        all_preds_tensor = torch.cat(all_predictions, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)
        grid_metrics = compute_metrics(all_preds_tensor, all_targets_tensor)

    # Collect samples from last batch for plotting
    samples = collect_samples(pred, batch_input, batch_target)

    return avg_loss, avg_loss_components, grid_metrics, samples


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    logger: WandbLogger,
    grid_config: dict | None = None,
) -> nn.Module:
    """Full training loop with early stopping.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        args: Training arguments.
        logger: W&B logger.
        grid_config: Dict with {nx, nt, dx, dt} for plotting.

    Returns:
        Trained model (best checkpoint).
    """
    # Build grid_config if not provided
    if grid_config is None:
        grid_config = {"nx": args.nx, "nt": args.nt, "dx": args.dx, "dt": args.dt}
    # Configure loss function with additional parameters
    loss_kwargs = {
        "pde_residual": {"dt": args.dt, "dx": args.dx},
        "rh_residual": {"dt": args.dt},
    }
    loss_fn = get_loss(args.loss, loss_kwargs=loss_kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 15

    progress_bar = tqdm(range(args.epochs), desc="Training")

    for epoch in progress_bar:
        # Train
        train_loss, train_loss_components, train_metrics, train_samples = train_epoch(
            model, train_loader, loss_fn, optimizer
        )

        # Validate
        val_loss, val_loss_components, val_metrics, val_samples = validate_epoch(
            model, val_loader, loss_fn
        )

        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Log metrics
        ep = epoch + 1
        logger.log_metrics({"train/lr": current_lr}, epoch=ep)
        log_values(logger, train_loss_components, ep, "train", "loss")
        log_values(logger, val_loss_components, ep, "val", "loss")
        log_values(logger, train_metrics, ep, "train", "metrics")
        log_values(logger, val_metrics, ep, "val", "metrics")

        # Plot every 5 epochs
        if (epoch + 1) % 5 == 0:
            plot(
                train_samples,
                grid_config,
                logger,
                epoch + 1,
                mode="train",
                preset=args.plot,
            )
            plot(
                val_samples,
                grid_config,
                logger,
                epoch + 1,
                mode="val",
                preset=args.plot,
            )

        # Check for improvement
        if val_loss < best_val_loss * 0.99:  # 1% improvement threshold
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            save_model(model, args.save_path, vars(args), epoch + 1)
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

        # Update progress bar
        progress_bar.set_postfix(
            {
                "train": f"{train_loss:.4f}",
                "val": f"{val_loss:.4f}",
                "best": f"{best_val_loss:.4f}",
                "lr": f"{current_lr:.2e}",
            }
        )

    # Load best model
    model = load_model(args.save_path, device, vars(args))

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    return model


def main():
    """Main entry point for training."""
    args = parse_args()

    # Create grid_config dict for plotting functions
    grid_config = {"nx": args.nx, "nt": args.nt, "dx": args.dx, "dt": args.dt}

    print(f"Using device: {device}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")

    # Check if model is available
    if args.model not in MODELS:
        print(f"Error: Model '{args.model}' not found in registry.")
        print(f"Available models: {list(MODELS.keys())}")
        print("Please implement your model in model.py and register it in MODELS dict.")
        return

    # Load data
    print("\nGenerating datasets...")
    train_dataset, val_dataset, test_dataset = get_wavefront_datasets(
        n_samples=args.n_samples,
        grid_config=grid_config,
        model_name=args.model,
        only_shocks=args.only_shocks,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_wavefront_batch,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_wavefront_batch,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_wavefront_batch,
        num_workers=4,
    )

    print(
        f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    # Create model
    model = get_model(args.model, vars(args))
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Run sanity check before logging
    run_sanity_check(model, train_loader, val_loader, args, device)

    # Initialize logger (no-ops when args.no_wandb is True)
    logger = init_logger(args)
    logger.log_metrics({"model/parameters": num_params})
    logger.watch_model(model)

    # Run profiler if requested
    if args.profile:
        run_profiler(model, train_loader, args, device, logger)

    # Train
    if args.epochs > 0:
        model = train_model(model, train_loader, val_loader, args, logger, grid_config)

    # Final test (standard + high-res)
    print("\nRunning final evaluation on test set...")
    loss_kwargs = {
        "pde_residual": {"dt": args.dt, "dx": args.dx},
        "rh_residual": {"dt": args.dt},
    }
    loss_fn = get_loss(args.loss, loss_kwargs=loss_kwargs)
    test_model(
        model=model,
        test_loader=test_loader,
        args=args,
        device=device,
        logger=logger,
        loss_fn=loss_fn,
        grid_config=grid_config,
        plot_preset=args.plot,
    )

    # Log final model
    logger.log_model(model, args.model, metadata=vars(args))
    logger.finish()

    print(f"\nModel saved to: {args.save_path}")


if __name__ == "__main__":
    main()
