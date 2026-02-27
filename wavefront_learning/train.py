"""Main training script for wavefront learning.

This is a standalone script that can be run with:
    python train.py --model fno --epochs 100

Minimal arguments are required - sensible defaults are provided for all parameters.
"""

import torch
import torch.nn as nn
from configs.cli import parse_args
from configs.presets import MODEL_LOSS_PRESET, MODEL_PLOT_PRESET
from configs.training_defaults import TRAINING_DEFAULTS
from data import collate_wavefront_batch, get_wavefront_datasets
from logger import WandbLogger, init_logger
from loss import create_loss_from_args
from losses.flow_matching import FlowMatchingLoss
from losses.vae_reconstruction import VAEReconstructionLoss
from model import MODELS, get_model, load_model
from testing import (
    run_profiler,
    run_sanity_check,
    test_model,
)
from torch.utils.data import DataLoader
from training_loop import _run_training_loop

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args,
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
    if grid_config is None:
        grid_config = {"nx": args.nx, "nt": args.nt, "dx": args.dx, "dt": args.dt}

    loss_fn = create_loss_from_args(args)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=TRAINING_DEFAULTS.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=TRAINING_DEFAULTS.scheduler_factor,
        patience=TRAINING_DEFAULTS.scheduler_patience,
        threshold=TRAINING_DEFAULTS.scheduler_threshold,
    )

    # Build list of per-epoch callbacks
    callbacks = []

    # KL annealing callback for CVAE models
    if hasattr(loss_fn, "set_kl_beta"):
        kl_warmup_epochs = max(
            1, int(TRAINING_DEFAULTS.kl_warmup_fraction * args.epochs)
        )
        kl_beta_target = getattr(args, "kl_beta", 1.0)

        def _kl_callback(epoch):
            progress = min(1.0, (epoch + 1) / kl_warmup_epochs)
            loss_fn.set_kl_beta(kl_beta_target * progress)

        callbacks.append(_kl_callback)

    # Teacher forcing decay for autoregressive models
    tf_initial = getattr(args, "teacher_forcing", 0.0)
    if tf_initial > 0 and hasattr(model, "teacher_forcing_ratio"):
        decay_epochs = max(1, int(args.tf_decay_fraction * args.epochs))

        def _tf_callback(epoch):
            ratio = tf_initial * max(0.0, 1.0 - epoch / decay_epochs)
            model.teacher_forcing_ratio = ratio
            logger.log_metrics({"train/teacher_forcing_ratio": ratio})

        callbacks.append(_tf_callback)

    # Curriculum + noise decay for NeuralFVSolver
    if args.model == "NeuralFVSolver" and hasattr(model, "max_rollout_steps"):
        total_rollout = args.nt - 1
        curriculum_epochs = max(1, int(args.curriculum_fraction * args.epochs))
        noise_decay_epochs = max(1, int(args.noise_decay_fraction * args.epochs))
        initial_noise = args.initial_noise_std

        def _curriculum_callback(epoch):
            progress = min(1.0, (epoch + 1) / curriculum_epochs)
            model.max_rollout_steps = max(1, int(progress * total_rollout))
            noise_progress = min(1.0, (epoch + 1) / noise_decay_epochs)
            model.noise_std = initial_noise * max(0.0, 1.0 - noise_progress)
            logger.log_metrics(
                {
                    "train/max_rollout_steps": model.max_rollout_steps,
                    "train/noise_std": model.noise_std,
                }
            )

        callbacks.append(_curriculum_callback)

    # Compose callbacks into a single function
    epoch_callback = None
    if callbacks:

        def epoch_callback(epoch):
            for cb in callbacks:
                cb(epoch)

    best_val_loss = _run_training_loop(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        scheduler,
        args.epochs,
        args.save_path,
        vars(args),
        logger,
        grid_config,
        args.plot,
        epoch_callback=epoch_callback,
    )

    model = load_model(args.save_path, device, vars(args))
    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    return model


def train_model_two_phase(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args,
    logger: WandbLogger,
    grid_config: dict | None = None,
) -> nn.Module:
    """Two-phase training for Latent Diffusion DeepONet.

    Phase 1: VAE training (encoder + decoder) with VAEReconstructionLoss.
    Phase 2: Flow matching (frozen encoder/decoder) with FlowMatchingLoss.

    Args:
        model: LatentDiffusionDeepONet instance.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        args: Training arguments.
        logger: W&B logger.
        grid_config: Dict with {nx, nt, dx, dt} for plotting.

    Returns:
        Trained model (best checkpoint from phase 2).
    """
    if grid_config is None:
        grid_config = {"nx": args.nx, "nt": args.nt, "dx": args.dx, "dt": args.dt}

    phase1_save = args.save_path.replace(".pth", "_phase1.pth")
    config = vars(args)

    # ── Phase 1: VAE training ──
    print(f"\n{'=' * 60}")
    print(f"Phase 1: VAE training ({args.ld_phase1_epochs} epochs)")
    print(f"{'=' * 60}")

    model.set_phase(1)
    vae_loss = VAEReconstructionLoss(
        beta=args.ld_beta, beta_warmup_epochs=args.ld_beta_warmup
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=TRAINING_DEFAULTS.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=TRAINING_DEFAULTS.scheduler_factor,
        patience=TRAINING_DEFAULTS.scheduler_patience,
        threshold=TRAINING_DEFAULTS.scheduler_threshold,
    )

    best = _run_training_loop(
        model,
        train_loader,
        val_loader,
        vae_loss,
        optimizer,
        scheduler,
        args.ld_phase1_epochs,
        phase1_save,
        config,
        logger,
        grid_config,
        args.plot,
        description="Phase 1 (VAE)",
        epoch_callback=vae_loss.set_epoch,
        extra_log={"phase": 1},
    )

    model = load_model(phase1_save, device, config)
    print(f"\nPhase 1 complete. Best VAE val loss: {best:.6f}")

    # ── Phase 2: Flow matching training ──
    print(f"\n{'=' * 60}")
    print(f"Phase 2: Flow matching ({args.ld_phase2_epochs} epochs)")
    print(f"{'=' * 60}")

    model.train()
    model.set_phase(2)
    fm_loss = FlowMatchingLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=TRAINING_DEFAULTS.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=TRAINING_DEFAULTS.scheduler_factor,
        patience=TRAINING_DEFAULTS.scheduler_patience,
        threshold=TRAINING_DEFAULTS.scheduler_threshold,
    )

    best = _run_training_loop(
        model,
        train_loader,
        val_loader,
        fm_loss,
        optimizer,
        scheduler,
        args.ld_phase2_epochs,
        args.save_path,
        config,
        logger,
        grid_config,
        args.plot,
        description="Phase 2 (Flow)",
        epoch_offset=args.ld_phase1_epochs,
        extra_log={"phase": 2},
    )

    model = load_model(args.save_path, device, config)
    model.set_phase(0)
    print(f"\nPhase 2 complete. Best flow matching val loss: {best:.6f}")
    return model


def main():
    """Main entry point for training."""
    args = parse_args()

    # Create grid_config dict for plotting functions
    grid_config = {"nx": args.nx, "nt": args.nt, "dx": args.dx, "dt": args.dt}

    # Auto-select loss and plot presets based on model
    if args.loss == "mse":  # default — auto-select per model
        args.loss = MODEL_LOSS_PRESET.get(args.model, "mse")
    if args.plot is None:
        args.plot = MODEL_PLOT_PRESET.get(args.model, "grid_residual")

    print(f"Using device: {device}")
    print(f"Equation: {args.equation}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")

    # Warn about unsupported ARZ + only_shocks combination
    if args.equation == "ARZ" and args.only_shocks:
        print("Warning: --only_shocks is not supported for ARZ; ignoring.")
        args.only_shocks = False

    # Build equation-specific kwargs
    equation_kwargs = None
    if args.equation == "ARZ":
        equation_kwargs = {
            "gamma": args.gamma,
            "flux_type": args.flux_type,
            "reconstruction": args.reconstruction,
            "bc_type": args.bc_type,
        }

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
        max_steps=args.max_steps,
        equation=args.equation,
        equation_kwargs=equation_kwargs,
        cell_sampling_k=args.cell_sampling_k,
        transform_override=args.transform,
        proximity_sigma=args.proximity_sigma,
        min_component_size=args.min_component_size,
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

    # Create or load model
    if args.model_path:
        print(f"\nResuming training from: {args.model_path}")
        model = load_model(args.model_path, device, vars(args))
        model.train()
    else:
        model = get_model(args.model, vars(args))
        model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Run sanity check before logging
    run_sanity_check(model, train_loader, val_loader, args, device)

    # Initialize logger (no-ops when args.no_wandb is True)
    wandb_project = "arz_learning" if args.equation == "ARZ" else "wavefront-learning"
    logger = init_logger(args, project=wandb_project)
    logger.log_metrics({"model/parameters": num_params})
    logger.watch_model(model)

    # Run profiler if requested
    if args.profile:
        run_profiler(model, train_loader, args, device, logger)

    # Train
    if args.epochs > 0:
        if args.model == "LDDeepONet":
            model = train_model_two_phase(
                model, train_loader, val_loader, args, logger, grid_config
            )
        else:
            model = train_model(
                model, train_loader, val_loader, args, logger, grid_config
            )

    # Final test (standard + high-res)
    print("\nRunning final evaluation on test set...")
    loss_fn = create_loss_from_args(args)
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
