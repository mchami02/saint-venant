"""HuggingFace integration for wavefront learning data.

Wraps operator_learning/hf_grids.py to use the shared mchami/grids repository.
All wavefront-specific preprocessing (discontinuity extraction) happens after downloading.
"""

import sys
from pathlib import Path

import numpy as np

# Add parent to path for operator_learning import
sys.path.insert(0, str(Path(__file__).parent.parent))

from operator_learning.hf_grids import download_grids as _download_grids
from operator_learning.hf_grids import upload_grids as _upload_grids

# Wavefront uses LaxHopf solver with Greenshield flux
DEFAULT_SOLVER = "LaxHopf"
DEFAULT_FLUX = "Greenshields"


def upload_grids(
    grids: np.ndarray,
    nx: int,
    nt: int,
    dx: float,
    dt: float,
    max_steps: int,
    solver: str = DEFAULT_SOLVER,
    flux: str = DEFAULT_FLUX,
) -> None:
    """Upload grids to the shared mchami/grids repository.

    Args:
        grids: Grid data of shape (n_samples, nt, nx).
        nx: Number of spatial grid points.
        nt: Number of time steps.
        dx: Spatial step size.
        dt: Time step size.
        max_steps: Maximum number of pieces in piecewise constant IC.
        solver: Solver name (default: LaxHopf).
        flux: Flux name (default: Greenshields).
    """
    _upload_grids(
        grids=grids,
        solver=solver,
        flux=flux,
        nx=nx,
        nt=nt,
        dx=dx,
        dt=dt,
        max_steps=max_steps,
    )


def download_grids(
    nx: int,
    nt: int,
    dx: float,
    dt: float,
    max_steps: int,
    solver: str = DEFAULT_SOLVER,
    flux: str = DEFAULT_FLUX,
) -> np.ndarray | None:
    """Download grids from the shared mchami/grids repository.

    Args:
        nx: Number of spatial grid points.
        nt: Number of time steps.
        dx: Spatial step size.
        dt: Time step size.
        max_steps: Maximum number of pieces in piecewise constant IC.
        solver: Solver name (default: LaxHopf).
        flux: Flux name (default: Greenshields).

    Returns:
        Grid data of shape (n_samples, nt, nx), or None if not found.
    """
    return _download_grids(
        solver=solver,
        flux=flux,
        nx=nx,
        nt=nt,
        dx=dx,
        dt=dt,
        max_steps=max_steps,
    )


if __name__ == "__main__":
    # Test download with parameters that exist in the repo
    print("Testing wavefront data loading from shared repo...")
    downloaded = download_grids(
        nx=50, nt=250, dx=0.25, dt=0.05, max_steps=3
    )
    if downloaded is not None:
        print(f"Downloaded grids shape: {downloaded.shape}")
    else:
        print("No grids found for this config")
