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


def _filter_arz_samples(grids, max_value=10.0, label=""):
    """Remove broken ARZ samples (NaN/Inf, all-zeros, extreme values).

    Args:
        grids: Array of shape (n, 2, nt, nx).
        max_value: Maximum allowed absolute value.
        label: Label for log messages (e.g. "steps=3").

    Returns:
        Tuple of (filtered_grids, good_mask) where good_mask is a boolean
        array that can be used to filter companion arrays (e.g. IC params).
    """
    n = len(grids)
    # Reshape to (n, -1) for per-sample checks
    flat = grids.reshape(n, -1)

    nan_inf = ~np.isfinite(flat).all(axis=1)
    all_zero = (flat == 0).all(axis=1)
    extreme = (np.abs(flat) > max_value).any(axis=1)

    bad = nan_inf | all_zero | extreme
    n_bad = bad.sum()

    if n_bad > 0:
        print(
            f"  {label}: removed {n_bad}/{n} ARZ samples "
            f"(NaN/Inf: {nan_inf.sum()}, all-zero: {all_zero.sum()}, "
            f"extreme (>{max_value}): {(extreme & ~nan_inf).sum()})"
        )

    good_mask = ~bad
    return grids[good_mask], good_mask


def get_nfv_dataset(
    n_samples: int,
    nx: int,
    nt: int,
    dx: float,
    dt: float,
    max_steps: int = 3,
    only_shocks: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        Tuple of (grids, ic_xs, ic_ks) where:
            - grids: shape (n_samples, nt, nx)
            - ic_xs: shape (n_samples, max_steps+1) breakpoint positions
            - ic_ks: shape (n_samples, max_steps) piece values
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
    return result["rho"].cpu().numpy(), result["ic_xs"], result["ic_ks"]


def get_arz_dataset(
    n_samples: int,
    nx: int,
    nt: int,
    dx: float,
    dt: float,
    max_steps: int = 3,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        Tuple of (grids, ic_xs, ic_rho_ks, ic_v_ks) where:
            - grids: shape (n_samples, 2, nt, nx) with channels [rho, v]
            - ic_xs: shape (n_samples, max_steps+1) breakpoint positions
            - ic_rho_ks: shape (n_samples, max_steps) density piece values
            - ic_v_ks: shape (n_samples, max_steps) velocity piece values
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
    grids = np.stack([rho, v], axis=1)  # (n, 2, nt, nx)

    return grids, result["ic_xs"], result["ic_rho_ks"], result["ic_v_ks"]


def compute_shock_proximity(
    gt_grid: np.ndarray,
    dx: float,
    sigma: float,
    min_component_size: int = 5,
) -> torch.Tensor:
    """Compute shock proximity field from a ground truth density grid.

    For each cell, computes the distance to the nearest entropy-violating
    shock interface, then maps to a proximity value via exp(-dist / sigma).

    Args:
        gt_grid: Ground truth density grid of shape (nt, nx).
        dx: Spatial step size.
        sigma: Length scale for the exponential proximity decay.

    Returns:
        Proximity field of shape (1, nt, nx) with values in [0, 1].
    """
    nt, nx = gt_grid.shape

    # Detect shocks via Lax entropy condition at each interface
    rho_L = gt_grid[:, :-1]  # (nt, nx-1)
    rho_R = gt_grid[:, 1:]  # (nt, nx-1)

    # Greenshields: characteristic speed = 1 - 2*rho
    char_L = 1.0 - 2.0 * rho_L
    char_R = 1.0 - 2.0 * rho_R
    # Rankine-Hugoniot shock speed = 1 - rho_L - rho_R
    s = 1.0 - rho_L - rho_R

    # Lax entropy condition: char_L > s > char_R
    is_shock = (char_L > s) & (s > char_R)  # (nt, nx-1)

    # Remove small isolated components (noise)
    from losses.shock_utils import filter_small_components

    is_shock = filter_small_components(is_shock, min_component_size)

    # Interface midpoints: interface j is between cell j and j+1
    x_interfaces = np.arange(1, nx) * dx  # (nx-1,)

    # Cell centers
    x_cells = (np.arange(nx) + 0.5) * dx  # (nx,)

    proximity = np.zeros((nt, nx), dtype=np.float32)

    for t in range(nt):
        shock_mask = is_shock[t]  # (nx-1,)
        if not shock_mask.any():
            continue
        shock_positions = x_interfaces[shock_mask]  # (n_shocks,)
        # Min distance from each cell center to any shock interface
        dists = np.abs(x_cells[:, None] - shock_positions[None, :])  # (nx, n_shocks)
        min_dist = dists.min(axis=1)  # (nx,)
        proximity[t] = np.exp(-min_dist / sigma)

    return torch.from_numpy(proximity).unsqueeze(0)  # (1, nt, nx)


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
    - Single index ``a``: clean boundary between cells ``a`` and ``a+1``.
      Position = (a + 1) * dx (the shared cell boundary).
    - Two consecutive indices ``[a, a+1]``: mid-cell discontinuity.
      The intermediate cell is at ``a+1``. We invert the cell-average
      formula to recover the sub-cell position within cell ``a+1``,
      giving position = (a + 1 + alpha) * dx.
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
            x_pos = (a + 1) * dx
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
            x_pos = (a + 1 + alpha) * dx
        else:
            # Fallback for 3+ consecutive indices (unexpected)
            a = group[0]
            b = group[-1]
            x_pos = ((a + b) / 2.0 + 1.0) * dx
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


def _build_ic_tensors_from_params(
    raw_xs: np.ndarray,
    raw_ks: np.ndarray,
    k: int,
    max_pieces: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build padded xs, ks, pieces_mask, discontinuities, disc_mask from raw IC params.

    Args:
        raw_xs: Breakpoint positions, at least k+1 values.
        raw_ks: Piece values, at least k values.
        k: Actual number of pieces.
        max_pieces: Padding size. If k > max_pieces, truncates to max_pieces.

    Returns:
        (xs, ks, pieces_mask, discontinuities, disc_mask)
    """
    k_eff = min(k, max_pieces)

    xs = torch.zeros(max_pieces + 1, dtype=torch.float32)
    ks = torch.zeros(max_pieces, dtype=torch.float32)
    pieces_mask = torch.zeros(max_pieces, dtype=torch.float32)

    xs[:k_eff + 1] = torch.from_numpy(raw_xs[:k_eff + 1].copy()).float()
    ks[:k_eff] = torch.from_numpy(raw_ks[:k_eff].copy()).float()
    pieces_mask[:k_eff] = 1.0

    # Build discontinuities: k-1 breakpoints between adjacent pieces
    n_disc = min(k - 1, max_pieces)
    discontinuities = torch.zeros(max_pieces, 3, dtype=torch.float32)
    disc_mask = torch.zeros(max_pieces, dtype=torch.float32)
    for j in range(n_disc):
        discontinuities[j, 0] = raw_xs[j + 1]  # breakpoint position
        discontinuities[j, 1] = raw_ks[j]  # left piece value
        discontinuities[j, 2] = raw_ks[j + 1]  # right piece value
        disc_mask[j] = 1.0

    return xs, ks, pieces_mask, discontinuities, disc_mask


def preprocess_wavefront_data(
    grids: np.ndarray,
    nx: int,
    nt: int,
    dx: float,
    dt: float,
    max_discontinuities: int = 10,
    equation: str = "LWR",
    ic_xs_all: np.ndarray | None = None,
    ic_ks_all: np.ndarray | None = None,
    ic_n_pieces_all: np.ndarray | None = None,
    ic_v_ks_all: np.ndarray | None = None,
    proximity_sigma: float | None = None,
    min_component_size: int = 5,
) -> list[tuple[dict, torch.Tensor]]:
    """Preprocess grids for wavefront learning.

    When IC params are available (ic_xs_all, ic_ks_all, ic_n_pieces_all),
    uses them directly to build input tensors. Falls back to grid-based
    extraction for samples without IC params (ic_n_pieces_all[idx] == 0)
    or when IC param arrays are not provided at all.

    Args:
        grids: Grid data of shape (n_samples, nt, nx) for LWR
            or (n_samples, 2, nt, nx) for ARZ.
        nx, nt: Grid dimensions.
        dx, dt: Grid spacing.
        max_discontinuities: Maximum number of discontinuities to support.
        equation: Equation system ("LWR" or "ARZ").
        ic_xs_all: Breakpoint positions, shape (n_samples, max_steps+1), padded.
        ic_ks_all: Piece values, shape (n_samples, max_steps), padded.
        ic_n_pieces_all: Actual piece count per sample, shape (n_samples,).
            0 means IC params not available (fall back to extraction).
        ic_v_ks_all: Velocity piece values for ARZ, shape (n_samples, max_steps).

    Returns:
        List of tuples (input_data, target_grid) where:
            - input_data: dict containing 'discontinuities', 'mask', 't_coords', 'x_coords'
            - target_grid: tensor of shape (1, nt, nx) for LWR or (2, nt, nx) for ARZ
    """
    processed = []

    for idx in range(len(grids)):
        if equation == "ARZ":
            target_grid = torch.from_numpy(grids[idx]).to(torch.float32)
        else:
            target_grid = torch.from_numpy(grids[idx]).to(torch.float32).unsqueeze(0)

        # Determine whether to use IC params or fall back to extraction
        has_ic_params = (
            ic_n_pieces_all is not None and ic_n_pieces_all[idx] > 0
        )

        if has_ic_params:
            k = int(ic_n_pieces_all[idx])
            xs, ks, pieces_mask, discontinuities, disc_mask = (
                _build_ic_tensors_from_params(
                    ic_xs_all[idx], ic_ks_all[idx], k, max_discontinuities
                )
            )
        else:
            # Fall back to grid extraction (old cached data without IC params)
            if equation == "ARZ":
                ic_grid = grids[idx, 0, 0, :].copy()
            else:
                ic_grid = grids[idx, 0, :].copy()

            discontinuities, disc_mask = extract_discontinuities_from_grid(
                ic_grid, dx, max_discontinuities=max_discontinuities
            )
            xs, ks, pieces_mask = extract_ic_representation_from_grid(
                ic_grid, dx, max_pieces=max_discontinuities
            )

        # Create coordinate grids for the output
        t_coords = (
            (torch.arange(nt).float() * dt)[:, None].expand(nt, nx).unsqueeze(0)
        )  # (1, nt, nx)
        x_coords = (
            ((torch.arange(nx).float() + 0.5) * dx)[None, :].expand(nt, nx).unsqueeze(0)
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

        # Compute shock proximity ground truth if requested
        if proximity_sigma is not None and equation == "LWR":
            input_data["shock_proximity"] = compute_shock_proximity(
                grids[idx], dx, proximity_sigma, min_component_size
            )

        # For ARZ, also handle velocity IC
        if equation == "ARZ":
            if has_ic_params and ic_v_ks_all is not None:
                k = int(ic_n_pieces_all[idx])
                k_eff = min(k, max_discontinuities)
                xs_v = xs.clone()  # same breakpoints
                ks_v = torch.zeros(max_discontinuities, dtype=torch.float32)
                pieces_mask_v = pieces_mask.clone()
                ks_v[:k_eff] = torch.from_numpy(
                    ic_v_ks_all[idx, :k_eff].copy()
                ).float()
            else:
                ic_grid_v = grids[idx, 1, 0, :].copy()
                xs_v, ks_v, pieces_mask_v = extract_ic_representation_from_grid(
                    ic_grid_v, dx, max_pieces=max_discontinuities
                )
            input_data["xs_v"] = xs_v
            input_data["ks_v"] = ks_v
            input_data["pieces_mask_v"] = pieces_mask_v

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
    proximity_sigma: float | None = None,
    min_component_size: int = 5,
) -> list[tuple[dict, torch.Tensor]]:
    """Get wavefront data, downloading from HuggingFace or generating locally.

    This is the main entry point for data loading. It distributes samples
    uniformly across step counts {min_steps, ..., max_steps}, downloading or
    generating grids for each step count independently. Each step count
    is cached separately on HuggingFace.

    IC parameters (xs, ks) are passed through the pipeline from the generators
    to avoid lossy extraction from discretized grids. Old cached data without
    IC params falls back to grid-based extraction.

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
    all_ic_xs = []
    all_ic_ks = []
    all_ic_v_ks = []  # ARZ only
    all_n_pieces = []

    for i, n_steps in enumerate(step_counts):
        n = n_per_step + (1 if i < remainder else 0)
        if n == 0:
            continue

        # Per-batch tracking: grids and IC params (None = not available)
        valid_grids = []
        valid_ic_xs = []  # list of arrays or None
        valid_ic_ks = []
        valid_ic_v_ks = []
        still_needed = n
        need_upload = False

        # Try to download from HuggingFace (each step count cached independently)
        cached_data = download_grids(
            nx, nt, dx, dt, n_steps, only_shocks, solver=solver, equation=equation
        )

        if cached_data is not None:
            cached_grids = cached_data["grids"]
            cached_ic_xs = cached_data["ic_xs"]
            cached_ic_ks = cached_data["ic_ks"]
            cached_ic_v_ks = cached_data["ic_v_ks"]

            if len(cached_grids) > 0:
                if is_arz:
                    original_len = len(cached_grids)
                    cached_grids, good_mask = _filter_arz_samples(
                        cached_grids, label=f"steps={n_steps} (cached)"
                    )
                    if len(cached_grids) < original_len:
                        need_upload = True
                    # Apply same mask to IC params
                    if cached_ic_xs is not None:
                        cached_ic_xs = cached_ic_xs[good_mask]
                        cached_ic_ks = cached_ic_ks[good_mask]
                    if cached_ic_v_ks is not None:
                        cached_ic_v_ks = cached_ic_v_ks[good_mask]

                available = cached_grids[:n]
                if len(available) > 0:
                    valid_grids.append(available)
                    n_avail = len(available)
                    valid_ic_xs.append(
                        cached_ic_xs[:n_avail] if cached_ic_xs is not None else None
                    )
                    valid_ic_ks.append(
                        cached_ic_ks[:n_avail] if cached_ic_ks is not None else None
                    )
                    valid_ic_v_ks.append(
                        cached_ic_v_ks[:n_avail]
                        if cached_ic_v_ks is not None
                        else None
                    )
                    still_needed = n - n_avail
                if still_needed <= 0:
                    print(f"  steps={n_steps}: using {n} cached samples")

        # Regeneration loop for missing samples
        max_regen = 5
        regen_round = 0
        while still_needed > 0 and regen_round < max_regen:
            regen_round += 1
            batch_size = still_needed
            print(
                f"  steps={n_steps}: generating {batch_size} samples"
                f" (round {regen_round})..."
            )
            if is_arz:
                raw_grids, raw_ic_xs, raw_ic_ks, raw_ic_v_ks = get_arz_dataset(
                    batch_size, nx, nt, dx, dt, n_steps, **arz_kw
                )
                raw_grids, good_mask = _filter_arz_samples(
                    raw_grids, label=f"steps={n_steps}"
                )
                raw_ic_xs = raw_ic_xs[good_mask]
                raw_ic_ks = raw_ic_ks[good_mask]
                raw_ic_v_ks = raw_ic_v_ks[good_mask]
            else:
                raw_grids, raw_ic_xs, raw_ic_ks = get_nfv_dataset(
                    batch_size, nx, nt, dx, dt, n_steps, only_shocks
                )
                raw_ic_v_ks = None
            if len(raw_grids) > 0:
                valid_grids.append(raw_grids)
                valid_ic_xs.append(raw_ic_xs)
                valid_ic_ks.append(raw_ic_ks)
                valid_ic_v_ks.append(raw_ic_v_ks)
                still_needed -= len(raw_grids)
            need_upload = True

        if still_needed > 0:
            print(
                f"  WARNING steps={n_steps}: only got {n - still_needed}/{n}"
                f" after {max_regen} regeneration rounds"
            )

        if len(valid_grids) == 0:
            print(f"  WARNING: steps={n_steps}: no valid samples after filtering!")
            continue

        step_grids = np.concatenate(valid_grids, axis=0)

        # Build padded IC param arrays for this step count
        padded_xs = np.zeros((len(step_grids), max_steps + 1))
        padded_ks = np.zeros((len(step_grids), max_steps))
        padded_v_ks = np.zeros((len(step_grids), max_steps))
        step_n_pieces = np.zeros(len(step_grids), dtype=int)

        offset = 0
        for j, g in enumerate(valid_grids):
            blen = len(g)
            if valid_ic_xs[j] is not None:
                padded_xs[offset : offset + blen, : valid_ic_xs[j].shape[1]] = (
                    valid_ic_xs[j]
                )
                padded_ks[offset : offset + blen, : valid_ic_ks[j].shape[1]] = (
                    valid_ic_ks[j]
                )
                step_n_pieces[offset : offset + blen] = n_steps
                if valid_ic_v_ks[j] is not None:
                    padded_v_ks[
                        offset : offset + blen, : valid_ic_v_ks[j].shape[1]
                    ] = valid_ic_v_ks[j]
            offset += blen

        # Upload with IC params when all samples have them
        if need_upload and upload_to_hf:
            has_all_ic = (step_n_pieces > 0).all()
            upload_kwargs = {}
            if has_all_ic:
                upload_kwargs["ic_xs"] = padded_xs[:, : n_steps + 1]
                upload_kwargs["ic_ks"] = padded_ks[:, :n_steps]
                if is_arz:
                    upload_kwargs["ic_v_ks"] = padded_v_ks[:, :n_steps]
            try:
                upload_grids(
                    step_grids,
                    nx,
                    nt,
                    dx,
                    dt,
                    n_steps,
                    only_shocks,
                    solver=solver,
                    equation=equation,
                    **upload_kwargs,
                )
            except Exception as e:
                print(f"  Failed to upload steps={n_steps}: {e}")

        all_grids.append(step_grids)
        all_ic_xs.append(padded_xs)
        all_ic_ks.append(padded_ks)
        all_n_pieces.append(step_n_pieces)
        if is_arz:
            all_ic_v_ks.append(padded_v_ks)

    if len(all_grids) == 0:
        raise RuntimeError("No valid samples remain after filtering!")

    grids = np.concatenate(all_grids, axis=0)
    ic_xs_all = np.concatenate(all_ic_xs, axis=0)
    ic_ks_all = np.concatenate(all_ic_ks, axis=0)
    n_pieces_all = np.concatenate(all_n_pieces, axis=0)
    ic_v_ks_all = np.concatenate(all_ic_v_ks, axis=0) if all_ic_v_ks else None

    # Shuffle with index-based permutation (not in-place) to keep arrays aligned
    perm = np.random.permutation(len(grids))
    grids = grids[perm]
    ic_xs_all = ic_xs_all[perm]
    ic_ks_all = ic_ks_all[perm]
    n_pieces_all = n_pieces_all[perm]
    if ic_v_ks_all is not None:
        ic_v_ks_all = ic_v_ks_all[perm]

    # Preprocess
    processed = preprocess_wavefront_data(
        grids,
        nx,
        nt,
        dx,
        dt,
        max_discontinuities,
        equation=equation,
        ic_xs_all=ic_xs_all,
        ic_ks_all=ic_ks_all,
        ic_n_pieces_all=n_pieces_all,
        ic_v_ks_all=ic_v_ks_all,
        proximity_sigma=proximity_sigma,
        min_component_size=min_component_size,
    )

    return processed


if __name__ == "__main__":
    # --- Unit tests for discontinuity extraction using nfv discretization ---
    # These tests use PiecewiseConstant.discretize() from nfv to generate
    # cell-averaged grids, validating extraction against the actual solver output.

    def _assert_close(actual, expected, tol, label):
        assert abs(actual - expected) < tol, (
            f"{label}: expected {expected}, got {actual}"
        )

    passed = 0
    total = 0
    nx = 10
    dx = 1.0 / nx

    # Test 1: Mid-cell discontinuity via nfv discretization
    # Discontinuity at x_d = 0.33, cell 3 spans [0.3, 0.4]
    # alpha = (0.33 - 0.3) / 0.1 = 0.3 of cell is left_val
    # mid_val = 0.3 * 1.0 + 0.7 * 0.0 = 0.3
    total += 1
    ic1 = PiecewiseConstant(ks=[1.0, 0.0])
    ic1.xs = np.array([0.0, 0.33, 1.0])
    ic1_grid = ic1.discretize(nx)
    discs1 = _group_consecutive_diff_indices(ic1_grid, dx=dx, threshold=0.01)
    assert len(discs1) == 1, f"Test 1: expected 1 disc, got {len(discs1)}"
    _assert_close(discs1[0]["x_pos"], 0.33, 1e-6, "Test 1 x_pos")
    _assert_close(discs1[0]["left_val"], 1.0, 1e-6, "Test 1 left_val")
    _assert_close(discs1[0]["right_val"], 0.0, 1e-6, "Test 1 right_val")
    # Also test via extract_discontinuities_from_grid
    disc_t, mask_t = extract_discontinuities_from_grid(ic1_grid, dx=dx, threshold=0.01)
    _assert_close(disc_t[0, 0].item(), 0.33, 1e-6, "Test 1 tensor x_pos")
    _assert_close(disc_t[0, 1].item(), 1.0, 1e-6, "Test 1 tensor left_val")
    _assert_close(disc_t[0, 2].item(), 0.0, 1e-6, "Test 1 tensor right_val")
    assert mask_t[0].item() == 1.0
    print("Test 1 PASSED: mid-cell discontinuity at x=0.33")
    passed += 1

    # Test 2: Clean boundary discontinuity
    # Discontinuity exactly at cell boundary x=0.4 (between cells 3 and 4)
    # position = (3 + 1) * 0.1 = 0.4
    total += 1
    ic2 = PiecewiseConstant(ks=[1.0, 0.0])
    ic2.xs = np.array([0.0, 0.4, 1.0])
    ic2_grid = ic2.discretize(nx)
    discs2 = _group_consecutive_diff_indices(ic2_grid, dx=dx, threshold=0.01)
    assert len(discs2) == 1, f"Test 2: expected 1 disc, got {len(discs2)}"
    _assert_close(discs2[0]["x_pos"], 0.4, 1e-6, "Test 2 x_pos")
    _assert_close(discs2[0]["left_val"], 1.0, 1e-6, "Test 2 left_val")
    _assert_close(discs2[0]["right_val"], 0.0, 1e-6, "Test 2 right_val")
    print("Test 2 PASSED: clean boundary discontinuity at x=0.4")
    passed += 1

    # Test 3: Multiple discontinuities, both mid-cell
    # Disc 1: x_d=0.16, cell 1 spans [0.1, 0.2]
    #   alpha = (0.16 - 0.1) / 0.1 = 0.6, mid_val = 0.6*1.0 + 0.4*0.0 = 0.6
    #   x_pos = (0 + 1 + 0.6) * 0.1 = 0.16
    # Disc 2: x_d=0.53, cell 5 spans [0.5, 0.6]
    #   alpha = (0.53 - 0.5) / 0.1 = 0.3, mid_val = 0.3*0.0 + 0.7*1.0 = 0.7
    #   x_pos = (4 + 1 + 0.3) * 0.1 = 0.53
    total += 1
    ic3 = PiecewiseConstant(ks=[1.0, 0.0, 1.0])
    ic3.xs = np.array([0.0, 0.16, 0.53, 1.0])
    ic3_grid = ic3.discretize(nx)
    discs3 = _group_consecutive_diff_indices(ic3_grid, dx=dx, threshold=0.01)
    assert len(discs3) == 2, f"Test 3: expected 2 discs, got {len(discs3)}"
    _assert_close(discs3[0]["x_pos"], 0.16, 1e-6, "Test 3 disc1 x_pos")
    _assert_close(discs3[0]["left_val"], 1.0, 1e-6, "Test 3 disc1 left_val")
    _assert_close(discs3[0]["right_val"], 0.0, 1e-6, "Test 3 disc1 right_val")
    _assert_close(discs3[1]["x_pos"], 0.53, 1e-6, "Test 3 disc2 x_pos")
    _assert_close(discs3[1]["left_val"], 0.0, 1e-6, "Test 3 disc2 left_val")
    _assert_close(discs3[1]["right_val"], 1.0, 1e-6, "Test 3 disc2 right_val")
    # Also test extract_ic_representation_from_grid
    xs3, ks3, mask3 = extract_ic_representation_from_grid(ic3_grid, dx=dx, threshold=0.01)
    _assert_close(xs3[0].item(), 0.0, 1e-6, "Test 3 xs[0]")
    _assert_close(xs3[1].item(), 0.16, 1e-6, "Test 3 xs[1]")
    _assert_close(xs3[2].item(), 0.53, 1e-6, "Test 3 xs[2]")
    _assert_close(ks3[0].item(), 1.0, 1e-6, "Test 3 ks[0]")
    _assert_close(ks3[1].item(), 0.0, 1e-6, "Test 3 ks[1]")
    _assert_close(ks3[2].item(), 1.0, 1e-6, "Test 3 ks[2]")
    print("Test 3 PASSED: multiple discontinuities with mid-cell")
    passed += 1

    # Test 4: Discontinuity at domain boundary (x=0.1, first cell boundary)
    total += 1
    ic4 = PiecewiseConstant(ks=[0.8, 0.2])
    ic4.xs = np.array([0.0, 0.1, 1.0])
    ic4_grid = ic4.discretize(nx)
    discs4 = _group_consecutive_diff_indices(ic4_grid, dx=dx, threshold=0.01)
    assert len(discs4) == 1, f"Test 4: expected 1 disc, got {len(discs4)}"
    _assert_close(discs4[0]["x_pos"], 0.1, 1e-6, "Test 4 x_pos")
    print("Test 4 PASSED: discontinuity at x=0.1 (first interior boundary)")
    passed += 1

    print(f"\nAll {passed}/{total} tests passed!")
