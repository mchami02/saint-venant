"""Command-line argument parser for wavefront learning.

Moved from train.py to keep the training script focused on orchestration.
Uses lazy imports inside the function body to avoid circular dependencies.
"""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Minimal set of arguments - most have sensible defaults.
    """
    # Lazy imports to avoid circular dependencies
    from data.transforms import TRANSFORMS
    from loss import LOSSES
    from model import MODELS

    from configs.presets import LOSS_PRESETS, PLOT_PRESETS

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
        "--ld_latent_dim",
        type=int,
        default=32,
        help="Latent space dimension for LDDeepONet",
    )
    parser.add_argument(
        "--ld_num_basis",
        type=int,
        default=64,
        help="Number of DeepONet basis functions",
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
        "--latent_dim",
        type=int,
        default=32,
        help="Latent space dimension for CVAEDeepONet",
    )
    parser.add_argument(
        "--condition_dim",
        type=int,
        default=64,
        help="Condition embedding dimension for CVAEDeepONet",
    )
    parser.add_argument(
        "--kl_beta", type=float, default=1.0, help="Target KL beta for CVAEDeepONet"
    )
    parser.add_argument(
        "--n_cvae_samples",
        type=int,
        default=10,
        help="Number of z-samples for CVAE evaluation",
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
    parser.add_argument(
        "--min_component_size",
        type=int,
        default=5,
        help="Min connected component size for shock detection (0 to disable)",
    )

    # NeuralFVSolver parameters
    parser.add_argument(
        "--stencil_k",
        type=int,
        default=3,
        help="Stencil half-width for NeuralFVSolver (default: 3)",
    )
    parser.add_argument(
        "--flux_hidden_dim",
        type=int,
        default=64,
        help="Hidden dimension for NeuralFVSolver flux network (default: 64)",
    )
    parser.add_argument(
        "--flux_n_layers",
        type=int,
        default=3,
        help="Number of layers in NeuralFVSolver flux network (default: 3)",
    )
    parser.add_argument(
        "--curriculum_fraction",
        type=float,
        default=0.5,
        help="Fraction of epochs to reach full rollout for NeuralFVSolver (default: 0.5)",
    )
    parser.add_argument(
        "--initial_noise_std",
        type=float,
        default=0.01,
        help="Initial pushforward noise std for NeuralFVSolver (default: 0.01)",
    )
    parser.add_argument(
        "--noise_decay_fraction",
        type=float,
        default=0.75,
        help="Fraction of epochs for noise decay in NeuralFVSolver (default: 0.75)",
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
