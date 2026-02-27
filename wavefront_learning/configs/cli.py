"""Command-line argument parser for wavefront learning.

Moved from train.py to keep the training script focused on orchestration.
Uses lazy imports inside the function body to avoid circular dependencies.

Default values come from configs/training.yaml (``cli_defaults`` section).
Priority: CLI flags > YAML defaults > hardcoded argparse defaults.

Unknown ``--key value`` arguments are injected into the namespace so that
arbitrary parameters can be passed on the command line for experimentation
without modifying any Python code.
"""

import argparse


def _coerce_type(value: str):
    """Auto-coerce a string CLI value to bool / int / float / str."""
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    for converter in (int, float):
        try:
            return converter(value)
        except ValueError:
            pass
    return value


def _inject_unknown_args(
    unknown: list[str], namespace: argparse.Namespace
) -> None:
    """Parse ``--key value`` pairs from unknown args into *namespace*.

    Boolean flags (``--flag`` without a following value) are set to ``True``.
    Values are auto-coerced via :func:`_coerce_type`.
    """
    i = 0
    while i < len(unknown):
        if unknown[i].startswith("--"):
            key = unknown[i][2:].replace("-", "_")
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                setattr(namespace, key, _coerce_type(unknown[i + 1]))
                i += 2
            else:
                setattr(namespace, key, True)
                i += 1
        else:
            i += 1


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Minimal set of arguments — most have sensible defaults sourced from
    ``configs/training.yaml``.  Any ``--key value`` pair not defined below
    is captured and injected into the returned namespace.
    """
    # Lazy imports to avoid circular dependencies
    from data.transforms import TRANSFORMS
    from loss import LOSSES
    from model import MODELS

    from configs.loader import load_cli_defaults
    from configs.presets import LOSS_PRESETS, PLOT_PRESETS

    parser = argparse.ArgumentParser(description="Train wavefront prediction model")

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODELS.keys()) if MODELS else ["fno"],
        help="Model architecture",
    )

    # Equation selection
    parser.add_argument(
        "--equation",
        type=str,
        choices=["LWR", "ARZ"],
        help="Equation system (LWR = scalar traffic, ARZ = 2-component density+velocity)",
    )
    parser.add_argument(
        "--gamma", type=float, help="ARZ pressure exponent (default: 1.0)"
    )
    parser.add_argument(
        "--flux_type",
        type=str,
        choices=["hll", "rusanov"],
        help="ARZ numerical flux type (default: hll)",
    )
    parser.add_argument(
        "--reconstruction",
        type=str,
        choices=["constant", "weno5"],
        help="ARZ reconstruction scheme (default: weno5)",
    )
    parser.add_argument(
        "--bc_type",
        type=str,
        help="ARZ boundary condition type (default: zero_gradient)",
    )

    # Loss selection
    parser.add_argument(
        "--loss",
        type=str,
        choices=list(LOSSES.keys()) + list(LOSS_PRESETS.keys()),
        help="Loss function or preset (default: mse, auto-selected per model)",
    )

    # Training parameters
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")

    # Data parameters
    parser.add_argument("--n_samples", type=int, help="Number of samples")
    parser.add_argument("--nx", type=int, help="Spatial grid points")
    parser.add_argument("--nt", type=int, help="Time steps")
    parser.add_argument("--dx", type=float, help="Spatial step size")
    parser.add_argument("--dt", type=float, help="Time step size")
    parser.add_argument(
        "--only_shocks",
        action="store_true",
        default=None,
        help="Generate only shock waves (no rarefactions)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        help="Max pieces in piecewise constant IC; samples drawn uniformly from {2,...,max_steps}",
    )
    parser.add_argument(
        "--max_test_steps",
        type=int,
        help="Max steps for step-count generalization test (default: same as max_steps)",
    )
    parser.add_argument(
        "--max_high_res",
        type=int,
        help="Max resolution multiplier for high-res generalization test",
    )

    # Latent Diffusion DeepONet parameters
    parser.add_argument(
        "--ld_latent_dim",
        type=int,
        help="Latent space dimension for LDDeepONet",
    )
    parser.add_argument(
        "--ld_num_basis",
        type=int,
        help="Number of DeepONet basis functions",
    )
    parser.add_argument(
        "--ld_beta", type=float, help="KL weight for VAE loss"
    )
    parser.add_argument(
        "--ld_beta_warmup", type=int, help="Epochs for beta warmup"
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
        "--ld_num_ode_steps", type=int, help="Heun ODE steps at inference"
    )
    parser.add_argument(
        "--ld_condition_dim", type=int, help="Condition embedding dimension"
    )

    # CVAE DeepONet parameters
    parser.add_argument(
        "--latent_dim",
        type=int,
        help="Latent space dimension for CVAEDeepONet",
    )
    parser.add_argument(
        "--condition_dim",
        type=int,
        help="Condition embedding dimension for CVAEDeepONet",
    )
    parser.add_argument(
        "--kl_beta", type=float, help="Target KL beta for CVAEDeepONet"
    )
    parser.add_argument(
        "--n_cvae_samples",
        type=int,
        help="Number of z-samples for CVAE evaluation",
    )

    # Model-specific parameters
    parser.add_argument(
        "--initial_damping_sharpness",
        type=float,
        help="Initial sharpness for collision-time bias damping in WaveNO",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        help="Dropout probability for WaveNO",
    )

    # Teacher forcing for autoregressive models (2-phase)
    parser.add_argument(
        "--teacher_forcing",
        type=float,
        help="(Deprecated, ignored) 2-phase TF is now automatic for AR models",
    )
    parser.add_argument(
        "--tf_decay_fraction",
        type=float,
        help="(Deprecated, ignored) 2-phase TF replaces linear decay",
    )
    parser.add_argument(
        "--tf_phase1_epochs",
        type=int,
        default=None,
        help="Phase 1 (TF=1.0) epochs for AR models (default: --epochs)",
    )
    parser.add_argument(
        "--tf_phase2_epochs",
        type=int,
        default=None,
        help="Phase 2 (TF=phase2_ratio) epochs for AR models (default: --epochs)",
    )
    parser.add_argument(
        "--tf_phase2_ratio",
        type=float,
        help="Teacher forcing ratio during phase 2 (default: 0.3)",
    )
    parser.add_argument(
        "--ar_noise_std",
        type=float,
        help="Pushforward noise std for all AR models (default: 0.01)",
    )

    # ShockAwareDeepONet parameters
    parser.add_argument(
        "--proximity_sigma",
        type=float,
        help="Length scale for shock proximity decay",
    )
    parser.add_argument(
        "--min_component_size",
        type=int,
        help="Min connected component size for shock detection (0 to disable)",
    )

    # NeuralFVSolver parameters
    parser.add_argument(
        "--stencil_k",
        type=int,
        help="Stencil half-width for NeuralFVSolver",
    )
    parser.add_argument(
        "--flux_hidden_dim",
        type=int,
        help="Hidden dimension for NeuralFVSolver flux network",
    )
    parser.add_argument(
        "--flux_n_layers",
        type=int,
        help="Number of layers in NeuralFVSolver flux network",
    )

    # Cell sampling
    parser.add_argument(
        "--cell_sampling_k",
        type=int,
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
        "--save_path", type=str, help="Model save path"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        default=None,
        help="Disable W&B logging",
    )
    parser.add_argument("--run_name", type=str, default=None, help="W&B run name")

    # Debugging/profiling
    parser.add_argument(
        "--profile",
        action="store_true",
        default=None,
        help="Run profiler before training",
    )

    # Plot preset
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        choices=list(PLOT_PRESETS.keys()),
        help="Plot preset (default: auto-detect based on model)",
    )

    # ── Apply YAML defaults (overrides argparse defaults, overridden by CLI) ──
    yaml_defaults = load_cli_defaults()
    parser.set_defaults(**yaml_defaults)

    # ── Parse known + unknown args ──
    args, unknown = parser.parse_known_args()
    _inject_unknown_args(unknown, args)

    # ── Post-parse fixups ──
    if args.max_test_steps is None:
        args.max_test_steps = args.max_steps

    # Default phase epoch splits for LDDeepONet
    if args.ld_phase1_epochs is None:
        args.ld_phase1_epochs = max(1, args.epochs * 2 // 3)
    if args.ld_phase2_epochs is None:
        args.ld_phase2_epochs = max(1, args.epochs - args.ld_phase1_epochs)

    # Default phase epoch splits for 2-phase TF (AR models)
    if args.tf_phase1_epochs is None:
        args.tf_phase1_epochs = args.epochs
    if args.tf_phase2_epochs is None:
        args.tf_phase2_epochs = args.epochs

    # Deprecation warnings
    import warnings

    if args.teacher_forcing and args.teacher_forcing > 0:
        warnings.warn(
            "--teacher_forcing is deprecated; 2-phase TF is now automatic for AR models.",
            DeprecationWarning,
            stacklevel=2,
        )
    if hasattr(args, "tf_decay_fraction") and args.tf_decay_fraction != 0.25:
        warnings.warn(
            "--tf_decay_fraction is deprecated; 2-phase TF replaces linear decay.",
            DeprecationWarning,
            stacklevel=2,
        )

    return args
