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
from logger import WandbLogger, init_logger, log_epoch_metrics
from loss import LOSSES, get_loss
from model import MODELS, get_model, load_model, save_model
from plotter import plot_comparison_wandb, plot_trajectory_wandb
from test import test_model
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
        default="rankine_hugoniot",
        choices=list(LOSSES.keys()),
        help="Loss function",
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

    # Output
    parser.add_argument(
        "--save_path", type=str, default="wavefront_model.pth", help="Model save path"
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--run_name", type=str, default=None, help="W&B run name")

    return parser.parse_args()


def detach_output(pred):
    """Detach model output (handles both tensor and dict)."""
    if isinstance(pred, dict):
        return {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in pred.items()}
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

    # Compute loss
    loss, components = loss_fn(pred, batch_input, batch_target)

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
) -> tuple[float, dict[str, float]]:
    """Train for one epoch.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        loss_fn: Loss function.
        optimizer: Optimizer.

    Returns:
        Tuple of (average_loss, metrics_dict).
    """
    model.train()
    total_loss = 0.0
    all_components = []

    for batch_input, batch_target in tqdm(train_loader, desc="Training", leave=False):
        loss, pred, components = train_step(model, batch_input, batch_target, loss_fn, optimizer)
        total_loss += loss
        all_components.append(components)

    avg_loss = total_loss / len(train_loader)
    # Average loss components
    avg_metrics = {
        key: np.mean([c[key] for c in all_components]) for key in all_components[0].keys()
    }

    return avg_loss, avg_metrics


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
) -> tuple[float, dict[str, float]]:
    """Validate for one epoch.

    Args:
        model: Model to validate.
        val_loader: Validation data loader.
        loss_fn: Loss function.

    Returns:
        Tuple of (average_loss, metrics_dict).
    """
    model.eval()
    total_loss = 0.0
    all_components = []

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
            loss, components = loss_fn(pred, batch_input, batch_target)

            total_loss += loss.item()
            all_components.append(components)

    avg_loss = total_loss / len(val_loader)
    avg_metrics = {
        key: np.mean([c[key] for c in all_components]) for key in all_components[0].keys()
    }

    return avg_loss, avg_metrics


def sample_predictions(
    model: nn.Module,
    val_loader: DataLoader,
    num_samples: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample predictions from validation set for visualization.

    Args:
        model: Trained model.
        val_loader: Validation data loader.
        num_samples: Number of samples to collect.

    Returns:
        Tuple of (ground_truths, predictions) as numpy arrays.
    """
    model.eval()
    gts, preds = [], []

    with torch.no_grad():
        for batch_input, batch_target in val_loader:
            if isinstance(batch_input, dict):
                batch_input = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch_input.items()
                }
            else:
                batch_input = batch_input.to(device)

            pred = model(batch_input)

            # Skip if model returns dict (trajectory model)
            if isinstance(pred, dict):
                return None, None

            for i in range(min(num_samples - len(gts), batch_target.shape[0])):
                gts.append(batch_target[i].squeeze(0).cpu().numpy())
                preds.append(pred[i].squeeze(0).cpu().numpy())

            if len(gts) >= num_samples:
                break

    return np.array(gts), np.array(preds)


def sample_trajectory_predictions(
    model: nn.Module,
    val_loader: DataLoader,
    num_samples: int = 3,
) -> dict[str, np.ndarray] | None:
    """Sample trajectory predictions from validation set for visualization.

    Args:
        model: Trained trajectory model.
        val_loader: Validation data loader.
        num_samples: Number of samples to collect.

    Returns:
        Dict with 'positions', 'existence', 'discontinuities', 'masks', 'times'
        as numpy arrays, or None if not a trajectory model.
    """
    model.eval()
    positions_list = []
    existence_list = []
    disc_list = []
    mask_list = []
    times = None

    with torch.no_grad():
        for batch_input, batch_target in val_loader:
            if isinstance(batch_input, dict):
                batch_input_device = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch_input.items()
                }
            else:
                return None  # Not a trajectory model

            pred = model(batch_input_device)

            # Check if this is a trajectory model
            if not isinstance(pred, dict) or "positions" not in pred:
                return None

            # Get times from input
            if times is None:
                t_coords = batch_input["t_coords"]  # (B, 1, nt, nx)
                times = t_coords[0, 0, :, 0].cpu().numpy()  # (nt,)

            batch_size = pred["positions"].shape[0]
            for i in range(min(num_samples - len(positions_list), batch_size)):
                positions_list.append(pred["positions"][i].cpu().numpy())
                existence_list.append(pred["existence"][i].cpu().numpy())
                disc_list.append(batch_input["discontinuities"][i].cpu().numpy())
                mask_list.append(batch_input["disc_mask"][i].cpu().numpy())

            if len(positions_list) >= num_samples:
                break

    if not positions_list:
        return None

    return {
        "positions": np.array(positions_list),
        "existence": np.array(existence_list),
        "discontinuities": np.array(disc_list),
        "masks": np.array(mask_list),
        "times": times,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    logger: WandbLogger | None = None,
) -> nn.Module:
    """Full training loop with early stopping.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        args: Training arguments.
        logger: Optional W&B logger.

    Returns:
        Trained model (best checkpoint).
    """
    loss_fn = get_loss(args.loss)
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
        train_loss, train_metrics = train_epoch(model, train_loader, loss_fn, optimizer)

        # Validate
        val_loss, val_metrics = validate_epoch(model, val_loader, loss_fn)

        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Log metrics
        if logger is not None:
            log_epoch_metrics(logger, epoch + 1, train_loss, val_loss, current_lr)
            logger.log_metrics(
                {f"train/{k}": v for k, v in train_metrics.items()},
                step=epoch + 1,
            )
            logger.log_metrics(
                {f"val/{k}": v for k, v in val_metrics.items()},
                step=epoch + 1,
            )

            # Plot every 10 epochs
            if (epoch + 1) % 10 == 0:
                # Try trajectory predictions first (for ShockNet-like models)
                traj_data = sample_trajectory_predictions(model, val_loader, num_samples=2)
                if traj_data is not None:
                    plot_trajectory_wandb(
                        traj_data["positions"],
                        traj_data["existence"],
                        traj_data["discontinuities"],
                        traj_data["masks"],
                        traj_data["times"],
                        logger,
                        epoch + 1,
                        mode="val",
                    )
                else:
                    # Standard grid-based models
                    gts, preds = sample_predictions(model, val_loader, num_samples=2)
                    if gts is not None:
                        plot_comparison_wandb(
                            gts,
                            preds,
                            args.nx,
                            args.nt,
                            args.dx,
                            args.dt,
                            logger,
                            epoch + 1,
                            mode="val",
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
        nx=args.nx,
        nt=args.nt,
        dx=args.dx,
        dt=args.dt,
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

    # Initialize logger
    logger = None
    if not args.no_wandb:
        logger = init_logger(args)
        logger.log_metrics({"model/parameters": num_params})

    # Train
    model = train_model(model, train_loader, val_loader, args, logger)

    # Final test
    print("\nRunning final evaluation on test set...")
    loss_fn = get_loss(args.loss)
    test_model(
        model=model,
        test_loader=test_loader,
        device=device,
        logger=logger,
        loss_fn=loss_fn,
        nx=args.nx,
        nt=args.nt,
        dx=args.dx,
        dt=args.dt,
    )

    # Log final model
    if logger is not None:
        logger.log_model(model, args.model, metadata=vars(args))
        logger.finish()

    print(f"\nModel saved to: {args.save_path}")


if __name__ == "__main__":
    main()
