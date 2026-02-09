"""HuggingFace integration for wavefront learning data.

Standalone upload/download that includes only_shocks in the config key,
so shock-only and mixed datasets are cached separately.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

# Load HF_TOKEN from root-level .env
load_dotenv(Path(__file__).parent.parent / ".env")

api = HfApi(token=os.environ.get("HF_TOKEN"))

# Wavefront uses LaxHopf solver with Greenshield flux
DEFAULT_SOLVER = "LaxHopf"
DEFAULT_FLUX = "Greenshields"
DEFAULT_REPO = "mchami/grids"


def _make_config_id(
    solver: str, flux: str, nx: int, nt: int, dx: float, dt: float,
    max_steps: int, only_shocks: bool,
) -> str:
    return (
        f"{solver}_{flux}_nx{nx}_nt{nt}_dx{dx}_dt{dt}"
        f"_steps{max_steps}_onlyshocks{only_shocks}"
    )


def upload_grids(
    grids: np.ndarray,
    nx: int,
    nt: int,
    dx: float,
    dt: float,
    max_steps: int,
    only_shocks: bool = True,
    solver: str = DEFAULT_SOLVER,
    flux: str = DEFAULT_FLUX,
    repo_id: str = DEFAULT_REPO,
) -> None:
    """Upload grids to the shared mchami/grids repository.

    Args:
        grids: Grid data of shape (n_samples, nt, nx).
        nx: Number of spatial grid points.
        nt: Number of time steps.
        dx: Spatial step size.
        dt: Time step size.
        max_steps: Maximum number of pieces in piecewise constant IC.
        only_shocks: Whether data contains only shock waves.
        solver: Solver name (default: LaxHopf).
        flux: Flux name (default: Greenshields).
        repo_id: HuggingFace repo ID.
    """
    assert grids.shape[1:] == (nt, nx)
    n_samples = grids.shape[0]

    config_id = _make_config_id(solver, flux, nx, nt, dx, dt, max_steps, only_shocks)
    data_path = f"lwr/{config_id}.npz"

    # 1) Save grids locally
    np.savez_compressed("tmp_grids.npz", grids=grids)

    # 2) Upload/overwrite only this file
    api.upload_file(
        path_or_fileobj="tmp_grids.npz",
        path_in_repo=data_path,
        repo_id=repo_id,
        repo_type="dataset",
    )

    # 3) Update index
    try:
        idx_path = hf_hub_download(repo_id, "index.parquet", repo_type="dataset")
        df = pd.read_parquet(idx_path)
    except Exception:
        df = pd.DataFrame(columns=[
            "config_id", "solver", "flux", "nx", "nt", "dx", "dt",
            "max_steps", "only_shocks", "n_samples", "file"
        ])

    df = df[df["config_id"] != config_id]
    df = pd.concat([
        df,
        pd.DataFrame([{
            "config_id": config_id,
            "solver": solver,
            "flux": flux,
            "nx": nx,
            "nt": nt,
            "dx": dx,
            "dt": dt,
            "max_steps": max_steps,
            "only_shocks": only_shocks,
            "n_samples": n_samples,
            "file": data_path,
        }])
    ], ignore_index=True)

    df.to_parquet("index.parquet")
    api.upload_file(
        path_or_fileobj="index.parquet",
        path_in_repo="index.parquet",
        repo_id=repo_id,
        repo_type="dataset",
    )

    # 4) Cleanup
    os.remove("tmp_grids.npz")
    os.remove("index.parquet")


def download_grids(
    nx: int,
    nt: int,
    dx: float,
    dt: float,
    max_steps: int,
    only_shocks: bool = True,
    solver: str = DEFAULT_SOLVER,
    flux: str = DEFAULT_FLUX,
    repo_id: str = DEFAULT_REPO,
) -> np.ndarray | None:
    """Download grids from the shared mchami/grids repository.

    Args:
        nx: Number of spatial grid points.
        nt: Number of time steps.
        dx: Spatial step size.
        dt: Time step size.
        max_steps: Maximum number of pieces in piecewise constant IC.
        only_shocks: Whether data contains only shock waves.
        solver: Solver name (default: LaxHopf).
        flux: Flux name (default: Greenshields).
        repo_id: HuggingFace repo ID.

    Returns:
        Grid data of shape (n_samples, nt, nx), or None if not found.
    """
    config_id = _make_config_id(solver, flux, nx, nt, dx, dt, max_steps, only_shocks)

    try:
        idx_path = hf_hub_download(repo_id, "index.parquet", repo_type="dataset")
        df = pd.read_parquet(idx_path)
    except Exception:
        return None

    row = df[df["config_id"] == config_id]
    if row.empty:
        return None

    file_path = row.iloc[0]["file"]
    npz_local = hf_hub_download(repo_id, file_path, repo_type="dataset")
    grids = np.load(npz_local)["grids"]

    if grids is None:
        raise ValueError(f"No grids found for config {config_id}")
    return grids


if __name__ == "__main__":
    print("Testing wavefront data loading from shared repo...")
    downloaded = download_grids(
        nx=50, nt=250, dx=0.25, dt=0.05, max_steps=3
    )
    if downloaded is not None:
        print(f"Downloaded grids shape: {downloaded.shape}")
    else:
        print("No grids found for this config")
