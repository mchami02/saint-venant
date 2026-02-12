"""Testing utilities for wavefront learning models.

This is a standalone script that can be run with:
    python test.py --model_path path/to/model.pth

Functions are organized into submodules:
- testing.test_running: sanity checks, profiling, inference
- testing.test_results: evaluation metrics and sample collection
"""

import argparse

import torch
from data import collate_wavefront_batch, get_wavefront_datasets
from logger import init_logger
from model import load_model
from plotter import PLOT_PRESETS
from testing import test_model
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "mps")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test wavefront prediction model")

    # Required argument
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model checkpoint"
    )

    # Data arguments (with defaults matching training)
    parser.add_argument(
        "--n_samples", type=int, default=200, help="Number of test samples"
    )
    parser.add_argument("--nx", type=int, default=50, help="Number of spatial points")
    parser.add_argument("--nt", type=int, default=250, help="Number of time steps")
    parser.add_argument("--dx", type=float, default=0.02, help="Spatial step size")
    parser.add_argument("--dt", type=float, default=0.004, help="Time step size")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=3,
        help="Maximum number of pieces in piecewise constant IC (default: 3)",
    )
    parser.add_argument(
        "--max_test_steps",
        type=int,
        default=None,
        help="Max steps for step-count generalization test (default: same as max_steps)",
    )
    parser.add_argument(
        "--max_high_res",
        type=int,
        default=5,
        help="Max resolution multiplier for high-res generalization test (default: 5)",
    )

    # Output arguments
    parser.add_argument(
        "--num_plots", type=int, default=3, help="Number of samples to plot"
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")

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

    return args


def main():
    """Main entry point for standalone testing."""
    args = parse_args()

    # Create grid_config dict for plotting functions
    grid_config = {"nx": args.nx, "nt": args.nt, "dx": args.dx, "dt": args.dt}

    print(f"Using device: {device}")
    print(f"Loading model from: {args.model_path}")

    # Load model from checkpoint (uses config stored in checkpoint)
    model = load_model(args.model_path, device, {"model": "ClassifierTrajDeepONet"})

    # Extract model registry name from checkpoint config
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model_name = checkpoint.get("config", {}).get("model", type(model).__name__)
    args.model = model_name

    print(f"Model loaded: {model_name}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Generate test data
    print("\nGenerating test data...")
    _, _, test_dataset = get_wavefront_datasets(
        n_samples=args.n_samples,
        grid_config=grid_config,
        model_name=model_name,
        train_ratio=0.0,  # All data for testing
        val_ratio=0.0,
        max_steps=args.max_steps,
        only_shocks=False,
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
        logger = init_logger(args, project="wavefront-learning-test")

    # Run full evaluation (standard + high-res)
    print("\nRunning final evaluation on test set...")
    metrics = test_model(
        model=model,
        test_loader=test_loader,
        args=args,
        device=device,
        logger=logger,
        grid_config=grid_config,
        num_plots=args.num_plots,
        plot_preset=args.plot,
    )

    # Cleanup
    if logger is not None:
        logger.finish()

    print("\nTesting complete!")
    return metrics


if __name__ == "__main__":
    main()
