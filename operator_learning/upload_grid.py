import os
import json
import tempfile
import numpy as np
import h5py
from huggingface_hub import login, upload_folder, HfApi
import argparse
from dotenv import load_dotenv
from operator_data_pipeline import get_nfv_dataset

# Load environment variables from .env file
load_dotenv()


def upload_grids_to_hf(
        array: np.ndarray,
        repo_id: str,
        path_in_repo: str,
        filename: str,
        flux_function: str,
        generation_method: str,
        nx: int,
        nt: int,
        dx: float,
        dt: float,
        max_steps: int,
        token: str | None = None,
    ):
    """
    Saves a NumPy grid dataset + metadata into a temporary folder and uploads to HuggingFace.

    Args:
        array (np.ndarray): Shape (N, nt, nx) dataset.
        repo_id (str): Hugging Face repo id, e.g. "mchami/lwr-grids".
        path_in_repo (str): Path inside the repo, e.g. "greenshield/train".
        filename (str): HDF5 filename, e.g. "grids.h5".
        flux_function (str): Which flux function generated the data.
        generation_method (str): Data generation method description.
        nx (int): Number of spatial cells.
        nt (int): Number of time steps.
        dx (float): Spatial discretization.
        dt (float): Temporal discretization.
        max_steps (int): Max piecewise steps in initial condition.
        token (str | None): HuggingFace token. If None, uses HF_TOKEN env var or interactive login.
    """
    
    # Handle authentication
    hf_token = token or os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        # Interactive login - will prompt for token
        login()
    
    # Create temporary directory for upload
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Save array as HDF5
        h5_path = os.path.join(tmpdir, filename)
        print(f"Saving {array.shape[0]} grids to {h5_path}...")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset(
                "grids",
                data=array,
                compression="gzip",
                compression_opts=9
            )

        # 2. Create comprehensive metadata.json
        metadata = {
            "flux_function": flux_function,
            "generation_method": generation_method,
            "n_samples": int(array.shape[0]),
            "nt": int(array.shape[1]),
            "nx": int(array.shape[2]),
            "dx": float(dx),
            "dt": float(dt),
            "max_steps": int(max_steps),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }

        metadata_path = os.path.join(tmpdir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        
        print(f"Metadata: {json.dumps(metadata, indent=2)}")

        # 3. Create dataset repo if it doesn't exist
        api = HfApi()
        try:
            api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        except Exception as e:
            print(f"Repo creation note: {e}")

        # 4. Upload the folder
        print(f"Uploading to {repo_id}/{path_in_repo}...")
        upload_folder(
            folder_path=tmpdir,
            repo_id=repo_id,
            path_in_repo=path_in_repo,
            repo_type="dataset",
        )

        print(f"✓ Successfully uploaded to HuggingFace: {repo_id}/{path_in_repo}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate LWR traffic flow grids and upload to HuggingFace"
    )
    
    # Grid generation parameters
    parser.add_argument("--n_samples", type=int, required=True, help="Number of samples to generate")
    parser.add_argument("--nx", type=int, required=True, help="Number of spatial cells")
    parser.add_argument("--nt", type=int, required=True, help="Number of time steps")
    parser.add_argument("--dx", type=float, required=True, help="Spatial discretization")
    parser.add_argument("--dt", type=float, required=True, help="Temporal discretization")
    parser.add_argument("--max_steps", type=int, default=3, help="Max piecewise steps in IC (default: 3)")
    
    # HuggingFace upload parameters
    parser.add_argument("--repo_id", type=str, default="mchami/grids", help="HuggingFace repo ID")
    parser.add_argument("--path_in_repo", type=str, default="lwr", help="Path inside repo")
    parser.add_argument("--filename", type=str, default="grid.h5", help="HDF5 filename")
    parser.add_argument("--flux_function", type=str, default="greenshield", help="Flux function name")
    parser.add_argument("--generation_method", type=str, default="lax-hopf", help="Solver used to generate data")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token (or set HF_TOKEN env var)")
    
    args = parser.parse_args()
    
    print(f"Generating {args.n_samples} grids with nx={args.nx}, nt={args.nt}, dx={args.dx}, dt={args.dt}...")
    grids = get_nfv_dataset(
        n_samples=args.n_samples,
        nx=args.nx,
        nt=args.nt,
        dx=args.dx,
        dt=args.dt,
        max_steps=args.max_steps,
    )
    print(f"Generated grids with shape: {grids.shape}")
    
    upload_grids_to_hf(
        array=grids,
        repo_id=args.repo_id,
        path_in_repo=args.path_in_repo,
        filename=args.filename,
        flux_function=args.flux_function,
        generation_method=args.generation_method,
        nx=args.nx,
        nt=args.nt,
        dx=args.dx,
        dt=args.dt,
        max_steps=args.max_steps,
        token=args.token,
    )
    
    print("Done!")