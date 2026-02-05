"""
Modal script to run training on Modal's serverless infrastructure.

Usage:
    # Run with default parameters
    modal run operator_learning/modal_train.py
    
    # Run with custom arguments
    modal run operator_learning/modal_train.py --n-samples 2000 --epochs 200 --batch-size 16
    
    # Example with different model and solver
    modal run operator_learning/modal_train.py --model FNO --solver Godunov --flux Triangular
    
Note: Modal will automatically provision GPU resources and handle all dependencies.
"""

from pathlib import Path

import modal

# Get the directory where this script is located
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent

# Define the Modal app
app = modal.App("saint-venant-training")

# Define the container image with all required dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "matplotlib>=3.7.5",
        "numpy>=1.20.0",
        "numba>=0.57.0",
        "tqdm>=4.65.0",
        "h5py>=3.8.0",
        "torch>=2.5.1",
        "joblib>=1.4.2",
        "neuraloperator>=0.3.0",
        "deepxde>=1.10.1",
    )
    # Copy local code directories into the image
    .add_local_dir(
        str(project_root / "numerical_methods"),
        "/root/numerical_methods"
    )
    .add_local_dir(
        str(script_dir),
        "/root/operator_learning"
    )
)

@app.function(
    image=image,
    gpu="any",  # Request GPU
    timeout=3600 * 4,  # 4 hour timeout
    secrets=[],  # Add modal.Secret.from_name("your-secret") if needed
)
def train(args_list=None):
    """
    Run the training script on Modal infrastructure.
    
    Args:
        args_list: List of command-line arguments to pass to train.py
    """
    import sys
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/operator_learning")
    
    # Override sys.argv if custom arguments are provided
    if args_list:
        sys.argv = ["train.py"] + args_list
    
    # Import and run the main function
    from train import main
    main()

@app.local_entrypoint()
def main(
    n_samples: int = 1000,
    nx: int = 50,
    nt: int = 250,
    dx: float = 0.25,
    dt: float = 0.05,
    bc: str = "GhostCell",
    solver: str = "Godunov",
    flux: str = "Greenshields",
    batch_size: int = 8,
    epochs: int = 100,
    lr: float = 0.001,
    model: str = "FNO",
    n_modes: int = 128,
    hidden_channels: int = 64,
    in_channels: int = 3,
    out_channels: int = 1,
    n_layers: int = 4,
    step_size: int = 5,
    gamma: float = 0.5,
    save_path: str = "operator.pth",
    n_datasets: int = 1,
):
    """
    Entry point when running with `modal run modal_train.py`
    
    All arguments mirror those in train.py's parse_args()
    """
    # Build argument list
    args_list = [
        "--n_samples", str(n_samples),
        "--nx", str(nx),
        "--nt", str(nt),
        "--dx", str(dx),
        "--dt", str(dt),
        "--bc", bc,
        "--solver", solver,
        "--flux", flux,
        "--batch_size", str(batch_size),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--model", model,
        "--n_modes", str(n_modes),
        "--hidden_channels", str(hidden_channels),
        "--in_channels", str(in_channels),
        "--out_channels", str(out_channels),
        "--n_layers", str(n_layers),
        "--step_size", str(step_size),
        "--gamma", str(gamma),
        "--save_path", save_path,
        "--n_datasets", str(n_datasets),
    ]
    
    train.remote(args_list)

