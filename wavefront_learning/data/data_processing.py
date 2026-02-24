"""Data generation and preprocessing for wavefront learning.

This module handles:
- Grid generation using the nfv Lax-Hopf solver
- Discontinuity extraction from discretized ICs
- Data preprocessing for neural network training
"""

import numpy as np
import torch

from data.data_loading import download_grids, upload_grids
from numerical_solvers.arz import generate_n as arz_generate_n
from numerical_solvers.lwr import generate_n as lwr_generate_n


def get_nfv_dataset(
    n_samples: int,
    nx: int,
    nt: int,
    dx: float,
    dt: float,
    max_steps: int = 3,
    only_shocks: bool = True,
) -> np.ndarray:
    """Generate grids using the Lax-Hopf solver.

    Args:
        n_samples: Number of samples to generate.
        nx: Number of spatial grid points.
        nt: Number of time steps.
        dx: Spatial step size.
        dt: Time step size.
        max_steps: Maximum number of pieces in piecewise constant IC.
        only_shocks: If True, sort ks to ensure only shock waves (no rarefactions).

    Returns:
        Grid data of shape (n_samples, nt, nx).
    """
    result = lwr_generate_n(
        n=n_samples,
        k=max_steps,
        nx=nx,
        nt=nt,
        dx=dx,
        dt=dt,
        only_shocks=only_shocks,
        show_progress=True,
        batch_size=4,
    )
    return result["rho"].cpu().numpy()


def get_arz_dataset(
    n_samples: int,
    nx: int,
    nt: int,
    dx: float,
    dt: float,
    max_steps: int = 3,
    **kwargs,
) -> np.ndarray:
    """Generate grids using the ARZ solver.

    Args:
        n_samples: Number of samples to generate.
        nx: Number of spatial grid points.
        nt: Number of time steps.
        dx: Spatial step size.
        dt: Time step size.
        max_steps: Number of pieces in piecewise constant IC.
        **kwargs: Additional ARZ solver arguments (gamma, flux_type,
            reconstruction, bc_type).

    Returns:
        Grid data of shape (n_samples, 2, nt, nx) with channels [rho, v].
    """
    # ARZ solver returns nt+1 timesteps; request nt-1 so output has nt total
    result = arz_generate_n(
        n=n_samples,
        k=max_steps,
        nx=nx,
        nt=nt - 1,
        dx=dx,
        dt=dt,
        show_progress=True,
        **kwargs,
    )
    rho = result["rho"].cpu().numpy()  # (n, nt, nx)
    v = result["v"].cpu().numpy()  # (n, nt, nx)

    # Last-resort guard: replace any remaining NaN/Inf with 0
    for name, arr in [("rho", rho), ("v", v)]:
        bad = ~np.isfinite(arr)
        n_bad = bad.sum()
        if n_bad > 0:
            print(f"WARNING: {n_bad} NaN/Inf values found in ARZ {name}, replacing with 0.")
            arr[bad] = 0.0

    return np.stack([rho, v], axis=1)  # (n, 2, nt, nx)


def _group_consecutive_diff_indices(
    ic_grid: np.ndarray,
    dx: float,
    threshold: float = 0.01,
) -> list[dict]:
    """Detect discontinuities from a finite-volume IC grid, handling mid-cell cases.

    When the Lax-Hopf solver discretizes a piecewise constant IC, a cell
    containing a discontinuity gets a cell-averaged intermediate value.
    This function recovers the exact discontinuity position by interpolating.

    Groups consecutive np.diff indices:
    - Single index ``a``: clean boundary. Position = (a + 0.5) * dx.
    - Two consecutive indices ``[a, a+1]``: mid-cell discontinuity.
      The intermediate cell is at ``a+1``. We invert the cell-average
      formula to recover the sub-cell position.
    - 3+ consecutive (shouldn't happen for piecewise constant ICs):
      fallback to midpoint of the group.

    Args:
        ic_grid: 1D array of shape (nx,).
        dx: Spatial step size.
        threshold: Minimum |diff| to consider as part of a discontinuity.

    Returns:
        List of dicts with keys ``x_pos``, ``left_val``, ``right_val``.
    """
    grad = np.abs(np.diff(ic_grid))
    indices = np.where(grad > threshold)[0]

    if len(indices) == 0:
        return []

    # Group consecutive indices
    groups: list[list[int]] = []
    current_group = [indices[0]]
    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:
            current_group.append(indices[i])
        else:
            groups.append(current_group)
            current_group = [indices[i]]
    groups.append(current_group)

    nx = len(ic_grid)
    results: list[dict] = []

    for group in groups:
        if len(group) == 1:
            # Clean boundary: discontinuity falls exactly between cells
            a = group[0]
            x_pos = (a + 0.5) * dx
            left_val = float(ic_grid[a])
            right_val = float(ic_grid[min(a + 1, nx - 1)])
        elif len(group) == 2:
            # Mid-cell discontinuity: intermediate cell at group[1] (= a+1)
            a = group[0]
            left_val = float(ic_grid[a])
            right_val = float(ic_grid[min(a + 2, nx - 1)])
            mid_val = float(ic_grid[a + 1])

            denom = left_val - right_val
            if abs(denom) > 1e-12:
                alpha = np.clip((mid_val - right_val) / denom, 0.0, 1.0)
            else:
                alpha = 0.5
            x_pos = (a + 0.5 + alpha) * dx
        else:
            # Fallback for 3+ consecutive indices (unexpected)
            a = group[0]
            b = group[-1]
            x_pos = ((a + b) / 2.0 + 0.5) * dx
            left_val = float(ic_grid[a])
            right_val = float(ic_grid[min(b + 1, nx - 1)])

        results.append({"x_pos": x_pos, "left_val": left_val, "right_val": right_val})

    return results


def extract_discontinuities_from_grid(
    ic_grid: np.ndarray,
    dx: float,
    max_discontinuities: int = 10,
    threshold: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract discontinuity points from a discretized initial condition.

    Detects jumps in the IC by looking at the gradient and extracts
    the position, left value, and right value for each discontinuity.
    For mid-cell discontinuities (where a cell contains an intermediate
    value), the exact position is recovered by inverting the cell-average.

    Args:
        ic_grid: Discretized IC of shape (nx,).
        dx: Spatial step size.
        max_discontinuities: Maximum number of discontinuities to detect.
        threshold: Minimum gradient magnitude to consider as a discontinuity.

    Returns:
        Tuple of (discontinuities, mask) where:
            - discontinuities: tensor of shape (max_discontinuities, 3)
              containing [x, left_val, right_val] for each discontinuity
            - mask: tensor of shape (max_discontinuities,) where 1 indicates
              valid discontinuity, 0 indicates padding
    """
    discs = _group_consecutive_diff_indices(ic_grid, dx, threshold)

    discontinuities = torch.zeros(max_discontinuities, 3, dtype=torch.float32)
    mask = torch.zeros(max_discontinuities, dtype=torch.float32)

    n_found = min(len(discs), max_discontinuities)
    for i in range(n_found):
        d = discs[i]
        discontinuities[i, 0] = d["x_pos"]
        discontinuities[i, 1] = d["left_val"]
        discontinuities[i, 2] = d["right_val"]
        mask[i] = 1.0

    return discontinuities, mask


def extract_ic_representation_from_grid(
    ic_grid: np.ndarray,
    dx: float,
    max_pieces: int = 10,
    threshold: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract piecewise constant representation from a discretized IC.

    Uses the same mid-cell interpolation as ``extract_discontinuities_from_grid``
    to recover exact breakpoint positions and pure (non-intermediate) piece values.

    Args:
        ic_grid: Discretized IC of shape (nx,).
        dx: Spatial step size.
        max_pieces: Maximum number of pieces to support.
        threshold: Minimum gradient magnitude to consider as a discontinuity.

    Returns:
        Tuple of (xs, ks, mask) where:
            - xs: tensor of shape (max_pieces + 1,) with breakpoint positions
            - ks: tensor of shape (max_pieces,) with piece values
            - mask: tensor of shape (max_pieces,) where 1 indicates valid piece
    """
    discs = _group_consecutive_diff_indices(ic_grid, dx, threshold)

    xs = torch.zeros(max_pieces + 1, dtype=torch.float32)
    ks = torch.zeros(max_pieces, dtype=torch.float32)
    mask = torch.zeros(max_pieces, dtype=torch.float32)

    xs[0] = 0.0

    n_disc = min(len(discs), max_pieces - 1)

    for i in range(n_disc):
        d = discs[i]
        xs[i + 1] = d["x_pos"]
        ks[i] = d["left_val"]
        mask[i] = 1.0

    # Last breakpoint at 1.0
    if n_disc < max_pieces:
        xs[n_disc + 1] = 1.0
        if n_disc > 0:
            ks[n_disc] = discs[n_disc - 1]["right_val"]
        else:
            ks[0] = float(ic_grid[0])
            mask[0] = 1.0
        mask[n_disc] = 1.0

    return xs, ks, mask


def preprocess_wavefront_data(
    grids: np.ndarray,
    nx: int,
    nt: int,
    dx: float,
    dt: float,
    max_discontinuities: int = 10,
    equation: str = "LWR",
) -> list[tuple[dict, torch.Tensor]]:
    """Preprocess grids for wavefront learning.

    Extracts discontinuities from the initial condition and creates
    input dictionaries with coordinate grids.

    Args:
        grids: Grid data of shape (n_samples, nt, nx) for LWR
            or (n_samples, 2, nt, nx) for ARZ.
        nx, nt: Grid dimensions.
        dx, dt: Grid spacing.
        max_discontinuities: Maximum number of discontinuities to support.
        equation: Equation system ("LWR" or "ARZ").

    Returns:
        List of tuples (input_data, target_grid) where:
            - input_data: dict containing 'discontinuities', 'mask', 't_coords', 'x_coords'
            - target_grid: tensor of shape (1, nt, nx) for LWR or (2, nt, nx) for ARZ
    """
    processed = []

    for idx in range(len(grids)):
        if equation == "ARZ":
            # ARZ grids: (n, 2, nt, nx) — already has channel dim
            target_grid = torch.from_numpy(grids[idx]).to(torch.float32)
            # Extract IC from rho (channel 0) at t=0
            ic_grid = grids[idx, 0, 0, :].copy()
        else:
            # LWR grids: (n, nt, nx) — add channel dim
            target_grid = torch.from_numpy(grids[idx]).to(torch.float32).unsqueeze(0)
            # Extract IC from first time step
            ic_grid = grids[idx, 0, :].copy()

        # Extract discontinuity representation
        discontinuities, disc_mask = extract_discontinuities_from_grid(
            ic_grid, dx, max_discontinuities=max_discontinuities
        )

        # Extract full IC representation (xs and ks)
        xs, ks, pieces_mask = extract_ic_representation_from_grid(
            ic_grid, dx, max_pieces=max_discontinuities
        )

        # Create coordinate grids for the output
        t_coords = (
            (torch.arange(nt).float() * dt)[:, None].expand(nt, nx).unsqueeze(0)
        )  # (1, nt, nx)
        x_coords = (
            (torch.arange(nx).float() * dx)[None, :].expand(nt, nx).unsqueeze(0)
        )  # (1, nt, nx)

        input_data = {
            "discontinuities": discontinuities,  # (max_disc, 3): [x, left_val, right_val]
            "disc_mask": disc_mask,  # (max_disc,)
            "xs": xs,  # (max_pieces + 1,): breakpoint positions
            "ks": ks,  # (max_pieces,): piece values
            "pieces_mask": pieces_mask,  # (max_pieces,)
            "t_coords": t_coords,  # (1, nt, nx)
            "x_coords": x_coords,  # (1, nt, nx)
            "dx": torch.tensor(dx, dtype=torch.float32),
            "dt": torch.tensor(dt, dtype=torch.float32),
        }

        processed.append((input_data, target_grid))

    return processed


def get_wavefront_data(
    n_samples: int,
    nx: int,
    nt: int,
    dx: float,
    dt: float,
    max_steps: int = 3,
    min_steps: int = 2,
    only_shocks: bool = True,
    random_seed: int = 42,
    max_discontinuities: int = 10,
    upload_to_hf: bool = True,
    equation: str = "LWR",
    equation_kwargs: dict | None = None,
) -> list[tuple[dict, torch.Tensor]]:
    """Get wavefront data, downloading from HuggingFace or generating locally.

    This is the main entry point for data loading. It distributes samples
    uniformly across step counts {min_steps, ..., max_steps}, downloading or
    generating grids for each step count independently. Each step count
    is cached separately on HuggingFace.

    Args:
        n_samples: Total number of samples needed (distributed across step counts).
        nx: Number of spatial grid points.
        nt: Number of time steps.
        dx: Spatial step size.
        dt: Time step size.
        max_steps: Maximum number of pieces in piecewise constant IC.
            Each sample's piece count is drawn uniformly from {min_steps, ..., max_steps}.
        min_steps: Minimum number of pieces in piecewise constant IC.
        only_shocks: If True, generate only shock waves (no rarefactions).
        random_seed: Random seed for reproducibility.
        max_discontinuities: Maximum number of discontinuities to support.
        upload_to_hf: If True, upload generated data to HuggingFace.
        equation: Equation system ("LWR" or "ARZ").
        equation_kwargs: Extra keyword arguments for the ARZ solver.

    Returns:
        List of tuples (input_data, target_grid).
    """
    np.random.seed(random_seed)

    is_arz = equation == "ARZ"
    solver = "ARZ" if is_arz else "LaxHopf"
    arz_kw = equation_kwargs or {}

    # Distribute samples uniformly across step counts {min_steps, ..., max_steps}
    step_counts = list(range(min_steps, max_steps + 1))
    n_per_step = n_samples // len(step_counts)
    remainder = n_samples % len(step_counts)

    all_grids = []
    for i, n_steps in enumerate(step_counts):
        n = n_per_step + (1 if i < remainder else 0)
        if n == 0:
            continue

        # Try to download from HuggingFace (each step count cached independently)
        grids = download_grids(
            nx, nt, dx, dt, n_steps, only_shocks, solver=solver, equation=equation
        )

        if grids is not None and len(grids) >= n:
            print(f"  steps={n_steps}: using {n} cached samples")
            grids = grids[:n]
        else:
            print(f"  steps={n_steps}: generating {n} samples locally...")
            if is_arz:
                grids = get_arz_dataset(n, nx, nt, dx, dt, n_steps, **arz_kw)
            else:
                grids = get_nfv_dataset(n, nx, nt, dx, dt, n_steps, only_shocks)

            if upload_to_hf:
                try:
                    upload_grids(
                        grids,
                        nx,
                        nt,
                        dx,
                        dt,
                        n_steps,
                        only_shocks,
                        solver=solver,
                        equation=equation,
                    )
                except Exception as e:
                    print(f"  Failed to upload steps={n_steps}: {e}")

        all_grids.append(grids)

    grids = np.concatenate(all_grids, axis=0)
    np.random.shuffle(grids)

    # Preprocess
    processed = preprocess_wavefront_data(
        grids, nx, nt, dx, dt, max_discontinuities, equation=equation
    )

    return processed


if __name__ == "__main__":
    # --- Unit tests for mid-cell discontinuity extraction ---

    def _assert_close(actual, expected, tol, label):
        assert abs(actual - expected) < tol, (
            f"{label}: expected {expected}, got {actual}"
        )

    passed = 0
    total = 0

    # Test 1: Mid-cell discontinuity
    # Discontinuity at x_d = 0.33, cell 3 centered at 0.3 spans [0.25, 0.35]
    # alpha = (0.33 - 0.25) / 0.1 = 0.8
    # mid_val = 0.8 * 1.0 + 0.2 * 0.0 = 0.8
    total += 1
    ic1 = np.array([1.0, 1.0, 1.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    discs1 = _group_consecutive_diff_indices(ic1, dx=0.1, threshold=0.01)
    assert len(discs1) == 1, f"Test 1: expected 1 disc, got {len(discs1)}"
    _assert_close(discs1[0]["x_pos"], 0.33, 1e-6, "Test 1 x_pos")
    _assert_close(discs1[0]["left_val"], 1.0, 1e-6, "Test 1 left_val")
    _assert_close(discs1[0]["right_val"], 0.0, 1e-6, "Test 1 right_val")
    # Also test via extract_discontinuities_from_grid
    disc_t, mask_t = extract_discontinuities_from_grid(ic1, dx=0.1, threshold=0.01)
    _assert_close(disc_t[0, 0].item(), 0.33, 1e-6, "Test 1 tensor x_pos")
    _assert_close(disc_t[0, 1].item(), 1.0, 1e-6, "Test 1 tensor left_val")
    _assert_close(disc_t[0, 2].item(), 0.0, 1e-6, "Test 1 tensor right_val")
    assert mask_t[0].item() == 1.0
    print("Test 1 PASSED: mid-cell discontinuity at x=0.33")
    passed += 1

    # Test 2: Clean boundary discontinuity
    # Discontinuity exactly between cells 3 and 4 -> position = (3 + 0.5) * 0.1 = 0.35
    total += 1
    ic2 = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    discs2 = _group_consecutive_diff_indices(ic2, dx=0.1, threshold=0.01)
    assert len(discs2) == 1, f"Test 2: expected 1 disc, got {len(discs2)}"
    _assert_close(discs2[0]["x_pos"], 0.35, 1e-6, "Test 2 x_pos")
    _assert_close(discs2[0]["left_val"], 1.0, 1e-6, "Test 2 left_val")
    _assert_close(discs2[0]["right_val"], 0.0, 1e-6, "Test 2 right_val")
    print("Test 2 PASSED: clean boundary discontinuity at x=0.35")
    passed += 1

    # Test 3: Multiple discontinuities, one mid-cell
    # Disc 1: mid-cell, left=1.0, right=0.0, mid_val=0.6 at cell 2
    #   alpha = (0.6 - 0.0) / (1.0 - 0.0) = 0.6
    #   x_pos = (1 + 0.5 + 0.6) * 0.1 = 0.21
    # Disc 2: mid-cell, left=0.0, right=1.0, mid_val=0.7 at cell 5
    #   alpha = (0.7 - 1.0) / (0.0 - 1.0) = 0.3
    #   x_pos = (4 + 0.5 + 0.3) * 0.1 = 0.48
    total += 1
    ic3 = np.array([1.0, 1.0, 0.6, 0.0, 0.0, 0.7, 1.0, 1.0, 1.0, 1.0])
    discs3 = _group_consecutive_diff_indices(ic3, dx=0.1, threshold=0.01)
    assert len(discs3) == 2, f"Test 3: expected 2 discs, got {len(discs3)}"
    _assert_close(discs3[0]["x_pos"], 0.21, 1e-6, "Test 3 disc1 x_pos")
    _assert_close(discs3[0]["left_val"], 1.0, 1e-6, "Test 3 disc1 left_val")
    _assert_close(discs3[0]["right_val"], 0.0, 1e-6, "Test 3 disc1 right_val")
    _assert_close(discs3[1]["x_pos"], 0.48, 1e-6, "Test 3 disc2 x_pos")
    _assert_close(discs3[1]["left_val"], 0.0, 1e-6, "Test 3 disc2 left_val")
    _assert_close(discs3[1]["right_val"], 1.0, 1e-6, "Test 3 disc2 right_val")
    # Also test extract_ic_representation_from_grid
    xs3, ks3, mask3 = extract_ic_representation_from_grid(ic3, dx=0.1, threshold=0.01)
    _assert_close(xs3[0].item(), 0.0, 1e-6, "Test 3 xs[0]")
    _assert_close(xs3[1].item(), 0.21, 1e-6, "Test 3 xs[1]")
    _assert_close(xs3[2].item(), 0.48, 1e-6, "Test 3 xs[2]")
    _assert_close(ks3[0].item(), 1.0, 1e-6, "Test 3 ks[0]")
    _assert_close(ks3[1].item(), 0.0, 1e-6, "Test 3 ks[1]")
    _assert_close(ks3[2].item(), 1.0, 1e-6, "Test 3 ks[2]")
    print("Test 3 PASSED: multiple discontinuities with mid-cell")
    passed += 1

    print(f"\nAll {passed}/{total} tests passed!")
