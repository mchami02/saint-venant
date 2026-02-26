"""Main training script for wavefront learning.

This is a standalone script that can be run with:
    python train.py --model fno --epochs 100

Minimal arguments are required - sensible defaults are provided for all parameters.
"""

import argparse

import torch
import torch.nn as nn
from data import collate_wavefront_batch, get_wavefront_datasets
from data.transforms import TRANSFORMS
from logger import WandbLogger, init_logger, log_values
from loss import LOSS_PRESETS, LOSSES, create_loss_from_args
from losses.flow_matching import FlowMatchingLoss
from losses.vae_reconstruction import VAEReconstructionLoss
from metrics import cell_average_prediction, compute_metrics, extract_grid_prediction
from model import MODELS, get_model, load_model, save_model
from plotter import PLOT_PRESETS, plot
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

    # Equation selection
    parser.add_argument(
        "--equation",
        type=str,
        default="LWR",
        choices=["LWR", "ARZ"],
        help="Equation system (LWR = scalar traffic, ARZ = 2-component density+velocity)",
    )
    parser.add_argument(
        "--gamma", type=float, default=1.0, help="ARZ pressure exponent (default: 1.0)"
    )
    parser.add_argument(
        "--flux_type",
        type=str,
        default="hll",
        choices=["hll", "rusanov"],
        help="ARZ numerical flux type (default: hll)",
    )
    parser.add_argument(
        "--reconstruction",
        type=str,
        default="weno5",
        choices=["constant", "weno5"],
        help="ARZ reconstruction scheme (default: weno5)",
    )
    parser.add_argument(
        "--bc_type",
        type=str,
        default="zero_gradient",
        help="ARZ boundary condition type (default: zero_gradient)",
    )

    # Loss selection
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        choices=list(LOSSES.keys()) + list(LOSS_PRESETS.keys()),
        help="Loss function or preset (default: mse, auto-selected per model)",
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
        "--only_shocks",
        action="store_true",
        help="Generate only shock waves (no rarefactions)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=4,
        help="Max pieces in piecewise constant IC; samples drawn uniformly from {2,...,max_steps} (default: 3)",
    )
    parser.add_argument(
        "--max_test_steps",
        type=int,
        default=10,
        help="Max steps for step-count generalization test (default: same as max_steps)",
    )
    parser.add_argument(
        "--max_high_res",
        type=int,
        default=5,
        help="Max resolution multiplier for high-res generalization test (default: 5)",
    )

    # Latent Diffusion DeepONet parameters
    parser.add_argument(
        "--ld_latent_dim", type=int, default=32, help="Latent space dimension for LDDeepONet"
    )
    parser.add_argument(
        "--ld_num_basis", type=int, default=64, help="Number of DeepONet basis functions"
    )
    parser.add_argument(
        "--ld_beta", type=float, default=0.01, help="KL weight for VAE loss"
    )
    parser.add_argument(
        "--ld_beta_warmup", type=int, default=10, help="Epochs for beta warmup"
    )
    parser.add_argument(
        "--ld_phase1_epochs",
        type=int,
        default=None,
        help="Phase 1 (VAE) epochs (default: 2/3 of --epochs)",
    )
    parser.add_argument(
        "--ld_phase2_epochs",
        type=int,
        default=None,
        help="Phase 2 (flow matching) epochs (default: 1/3 of --epochs)",
    )
    parser.add_argument(
        "--ld_num_ode_steps", type=int, default=100, help="Heun ODE steps at inference"
    )
    parser.add_argument(
        "--ld_condition_dim", type=int, default=64, help="Condition embedding dimension"
    )

    # CVAE DeepONet parameters
    parser.add_argument(
        "--latent_dim", type=int, default=32, help="Latent space dimension for CVAEDeepONet"
    )
    parser.add_argument(
        "--condition_dim", type=int, default=64, help="Condition embedding dimension for CVAEDeepONet"
    )
    parser.add_argument(
        "--kl_beta", type=float, default=1.0, help="Target KL beta for CVAEDeepONet"
    )
    parser.add_argument(
        "--n_cvae_samples", type=int, default=10, help="Number of z-samples for CVAE evaluation"
    )

    # Model-specific parameters
    parser.add_argument(
        "--initial_damping_sharpness",
        type=float,
        default=5.0,
        help="Initial sharpness for collision-time bias damping in WaveNO (default: 5.0)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.05,
        help="Dropout probability for WaveNO (default: 0.05)",
    )

    # Teacher forcing for autoregressive models
    parser.add_argument(
        "--teacher_forcing",
        type=float,
        default=0.0,
        help="Initial teacher forcing ratio for autoregressive models (default: 0.0)",
    )
    parser.add_argument(
        "--tf_decay_fraction",
        type=float,
        default=0.25,
        help="Fraction of total epochs over which teacher forcing decays to 0 (default: 0.25)",
    )

    # ShockAwareDeepONet parameters
    parser.add_argument(
        "--proximity_sigma",
        type=float,
        default=0.05,
        help="Length scale for shock proximity decay (default: 0.05)",
    )

    # Cell sampling
    parser.add_argument(
        "--cell_sampling_k",
        type=int,
        default=0,
        help="Number of random query points per FV cell (0 = disabled)",
    )

    # Transform override
    parser.add_argument(
        "--transform",
        type=str,
        default=None,
        choices=list(TRANSFORMS.keys()),
        help="Override the model's default transform (default: use MODEL_TRANSFORM)",
    )

    # Resume training
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to a saved model checkpoint to resume training from",
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

    args = parser.parse_args()

    if args.max_test_steps is None:
        args.max_test_steps = args.max_steps

    # Default phase epoch splits for LDDeepONet
    if args.ld_phase1_epochs is None:
        args.ld_phase1_epochs = max(1, args.epochs * 2 // 3)
    if args.ld_phase2_epochs is None:
        args.ld_phase2_epochs = max(1, args.epochs - args.ld_phase1_epochs)

    return args


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
    if grid_config is None:
        grid_config = {"nx": args.nx, "nt": args.nt, "dx": args.dx, "dt": args.dt}

    loss_fn = create_loss_from_args(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, threshold=0.01
    )

    # Build list of per-epoch callbacks
    callbacks = []

    # KL annealing callback for CVAE models
    if hasattr(loss_fn, "set_kl_beta"):
        kl_warmup_epochs = max(1, int(0.2 * args.epochs))
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

    # Compose callbacks into a single function
    epoch_callback = None
    if callbacks:

        def epoch_callback(epoch):
            for cb in callbacks:
                cb(epoch)

    best_val_loss = _run_training_loop(
        model, train_loader, val_loader,
        loss_fn, optimizer, scheduler,
        args.epochs, args.save_path, vars(args),
        logger, grid_config, args.plot,
        epoch_callback=epoch_callback,
    )

    model = load_model(args.save_path, device, vars(args))
    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    return model


def train_model_two_phase(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
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
    print(f"\n{'='*60}")
    print(f"Phase 1: VAE training ({args.ld_phase1_epochs} epochs)")
    print(f"{'='*60}")

    model.set_phase(1)
    vae_loss = VAEReconstructionLoss(
        beta=args.ld_beta, beta_warmup_epochs=args.ld_beta_warmup
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, threshold=0.01
    )

    best = _run_training_loop(
        model, train_loader, val_loader,
        vae_loss, optimizer, scheduler,
        args.ld_phase1_epochs, phase1_save, config,
        logger, grid_config, args.plot,
        description="Phase 1 (VAE)",
        epoch_callback=vae_loss.set_epoch,
        extra_log={"phase": 1},
    )

    model = load_model(phase1_save, device, config)
    print(f"\nPhase 1 complete. Best VAE val loss: {best:.6f}")

    # ── Phase 2: Flow matching training ──
    print(f"\n{'='*60}")
    print(f"Phase 2: Flow matching ({args.ld_phase2_epochs} epochs)")
    print(f"{'='*60}")

    model.train()
    model.set_phase(2)
    fm_loss = FlowMatchingLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, threshold=0.01
    )

    best = _run_training_loop(
        model, train_loader, val_loader,
        fm_loss, optimizer, scheduler,
        args.ld_phase2_epochs, args.save_path, config,
        logger, grid_config, args.plot,
        description="Phase 2 (Flow)",
        epoch_offset=args.ld_phase1_epochs,
        extra_log={"phase": 2},
    )

    model = load_model(args.save_path, device, config)
    model.set_phase(0)
    print(f"\nPhase 2 complete. Best flow matching val loss: {best:.6f}")
    return model


MODEL_LOSS_PRESET: dict[str, str] = {
    "CVAEDeepONet": "cvae",
    "TrajTransformer": "traj_regularized",
    "ShockAwareDeepONet": "shock_proximity",
    "ShockAwareWaveNO": "shock_proximity",
}

MODEL_PLOT_PRESET: dict[str, str] = {
    "TrajDeepONet": "traj_residual",
    "TrajTransformer": "traj_residual",
    "WaveNO": "traj_residual",
    "WaveNOLocal": "traj_residual",
    "WaveNOIndepTraj": "traj_residual",
    "WaveNODisc": "traj_residual",
    "ClassifierTrajTransformer": "traj_existence",
    "ClassifierAllTrajTransformer": "traj_existence",
    "BiasedClassifierTrajTransformer": "traj_existence",
    "WaveNOCls": "traj_existence",
    "CTTBiased": "traj_existence",
    "CTTSegPhysics": "traj_existence",
    "CTTFiLM": "traj_existence",
    "CTTSeg": "traj_existence",
    "CharNO": "charno",
    "TransformerSeg": "grid_minimal",
    "WaveFrontModel": "wavefront",
    "CVAEDeepONet": "cvae",
    "ECARZ": "ecarz",
    "ShockAwareDeepONet": "shock_proximity",
    "ShockAwareWaveNO": "traj_residual",
}


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
            model = train_model(model, train_loader, val_loader, args, logger, grid_config)

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
