"""Training loop primitives: step, epoch, validation, and the shared training loop.

Extracted from train.py to keep orchestration separate from loop mechanics.
"""

import numpy as np
import torch
import torch.nn as nn
from logger import WandbLogger, log_values
from metrics import compute_metrics, extract_grid_prediction
from model import save_model
from plotter import plot
from testing import collect_samples
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


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

    # Inject target for models that need it during training (CVAE, teacher forcing)
    needs_target = getattr(model, "needs_target_input", False) or getattr(
        model, "teacher_forcing_ratio", 0.0
    ) > 0
    if needs_target and isinstance(batch_input, dict):
        batch_input["target_grid"] = batch_target

    # Forward pass
    pred = model(
        batch_input,
    )

    # Compute loss (input_dict, output_dict, target)
    loss, components = loss_fn(batch_input, pred, batch_target)

    # Guard against NaN/Inf loss — skip batch to prevent corrupting weights
    if torch.isnan(loss) or torch.isinf(loss):
        optimizer.zero_grad()
        return loss.item(), detach_output(pred), components

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


def _run_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    num_epochs: int,
    save_path: str,
    model_config: dict,
    logger: WandbLogger,
    grid_config: dict,
    plot_preset: str | None,
    description: str = "Training",
    epoch_offset: int = 0,
    patience: int = 15,
    epoch_callback=None,
    extra_log: dict | None = None,
) -> float:
    """Shared epoch loop: train -> validate -> log -> plot -> early stop -> save.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        loss_fn: Loss function.
        optimizer: Optimizer.
        scheduler: LR scheduler (ReduceLROnPlateau).
        num_epochs: Number of epochs to train.
        save_path: Path to save best model checkpoint.
        model_config: Model config dict (passed to save_model).
        logger: W&B logger.
        grid_config: Dict with {nx, nt, dx, dt} for plotting.
        plot_preset: Plot preset name.
        description: Progress bar description.
        epoch_offset: Offset added to epoch index for logging (multi-phase training).
        patience: Early stopping patience (number of epochs without improvement).
        epoch_callback: Called with (epoch_index) before each epoch.
        extra_log: Extra metrics dict logged every epoch (e.g. {"phase": 1}).

    Returns:
        Best validation loss achieved.
    """
    best_val_loss = float("inf")
    patience_counter = 0

    progress_bar = tqdm(range(num_epochs), desc=description)

    for epoch in progress_bar:
        if epoch_callback is not None:
            epoch_callback(epoch)

        train_loss, train_lc, train_metrics, train_samples = train_epoch(
            model, train_loader, loss_fn, optimizer
        )
        val_loss, val_lc, val_metrics, val_samples = validate_epoch(
            model, val_loader, loss_fn
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        ep = epoch_offset + epoch + 1
        log_dict = {"train/lr": current_lr}
        if extra_log:
            log_dict.update(extra_log)
        logger.log_metrics(log_dict, epoch=ep)
        log_values(logger, train_lc, ep, "train", "loss")
        log_values(logger, val_lc, ep, "val", "loss")
        log_values(logger, train_metrics, ep, "train", "metrics")
        log_values(logger, val_metrics, ep, "val", "metrics")

        if (epoch + 1) % 5 == 0:
            plot(train_samples, grid_config, logger, ep, "train", preset=plot_preset)
            plot(val_samples, grid_config, logger, ep, "val", preset=plot_preset)

        if val_loss < best_val_loss * 0.99:
            best_val_loss = val_loss
            patience_counter = 0
            save_model(model, save_path, model_config, ep, logger=logger)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {ep}")
            break

        progress_bar.set_postfix(
            {
                "train": f"{train_loss:.4f}",
                "val": f"{val_loss:.4f}",
                "best": f"{best_val_loss:.4f}",
                "lr": f"{current_lr:.2e}",
            }
        )

    return best_val_loss
