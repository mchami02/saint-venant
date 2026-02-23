"""Transforms for wavefront learning input representations.

Each transform takes (input_data, target_grid) and returns a
(possibly modified) (input_data, target_grid) pair.
"""

from functools import partial

import torch


class FlattenDiscontinuitiesTransform:
    """Transform that flattens discontinuity data into a single tensor.

    This is useful for models that expect a simple tensor input rather than
    a dictionary. The output tensor has shape (n_features,) where n_features
    includes all discontinuity information.
    """

    def __call__(self, input_data: dict, target_grid: torch.Tensor):
        # Flatten: [xs (max_pieces+1), ks (max_pieces), pieces_mask (max_pieces)]
        xs = input_data["xs"]
        ks = input_data["ks"]
        mask = input_data["pieces_mask"]

        flat_input = torch.cat([xs, ks, mask], dim=0)

        return flat_input, target_grid


class ToGridInputTransform:
    """Transform that converts discontinuity data to a grid-like input.

    This reconstructs the initial condition on the grid from discontinuities,
    similar to how operator_learning represents inputs, but also includes
    the raw discontinuity information as additional channels.
    """

    def __init__(self, nx: int, nt: int, include_coords: bool = True, **kwargs):
        self.nx = nx
        self.nt = nt
        self.include_coords = include_coords

    def __call__(self, input_data: dict, target_grid: torch.Tensor):
        xs = input_data["xs"]
        ks = input_data["ks"]
        mask = input_data["pieces_mask"]
        t_coords = input_data["t_coords"]
        x_coords = input_data["x_coords"]

        # Reconstruct IC on grid from piecewise constant representation
        ic_grid = torch.zeros(self.nx, dtype=torch.float32)
        x_positions = torch.linspace(0, 1, self.nx)

        n_pieces = int(mask.sum().item())
        for i in range(n_pieces):
            x_left = xs[i]
            x_right = xs[i + 1]
            val = ks[i]
            ic_grid[(x_positions >= x_left) & (x_positions < x_right)] = val

        # Handle the rightmost piece (include the boundary)
        if n_pieces > 0:
            ic_grid[x_positions >= xs[n_pieces - 1]] = ks[n_pieces - 1]

        # Expand IC to full grid (repeat across time)
        ic_expanded = (
            ic_grid[None, :].expand(self.nt, self.nx).unsqueeze(0)
        )  # (1, nt, nx)

        # Mask everything except initial condition (like GridMaskAllButInitial)
        ic_masked = ic_expanded.clone()
        ic_masked[:, 1:, :] = -1

        # Stack channels
        if self.include_coords:
            # [ic_masked, t_coords, x_coords] -> (3, nt, nx)
            full_input = torch.cat([ic_masked, t_coords, x_coords], dim=0)
        else:
            # ic_masked only -> (1, nt, nx)
            full_input = ic_masked

        # Return dict: grid tensor + passthrough of original keys
        result = dict(input_data)
        result["grid_input"] = full_input
        return result, target_grid


class DiscretizeICTransform:
    """Transform that discretizes the IC at evenly-spaced points.

    Creates a uniform discretization of the initial condition from the
    piecewise constant representation (xs, ks), storing the result as
    "discontinuities" in the input dict for compatibility with models
    that read from that field.

    Each point is stored as [x_position, ic_value, ic_value] in a
    (discretization, 3) tensor. Since every point is always valid,
    disc_mask is all ones.

    Args:
        discretization: Number of evenly-spaced evaluation points.
        **kwargs: Ignored (allows passing grid_config dict directly).
    """

    def __init__(self, discretization: int, **kwargs):
        self.discretization = discretization

    def __call__(self, input_data: dict, target_grid: torch.Tensor):
        xs = input_data["xs"]
        ks = input_data["ks"]
        mask = input_data["pieces_mask"]

        n_pieces = int(mask.sum().item())
        n_pts = self.discretization

        # Evenly spaced positions in [0, 1]
        x_positions = torch.linspace(0, 1, n_pts)

        # Evaluate piecewise constant IC at each position
        ic_values = torch.zeros(n_pts, dtype=torch.float32)
        for i in range(n_pieces):
            x_left = xs[i]
            x_right = xs[i + 1]
            val = ks[i]
            ic_values[(x_positions >= x_left) & (x_positions < x_right)] = val

        # Handle the rightmost piece (include the boundary)
        if n_pieces > 0:
            ic_values[x_positions >= xs[n_pieces - 1]] = ks[n_pieces - 1]

        # Pack as (discretization, 3): [x_position, ic_value, ic_value]
        discontinuities = torch.zeros(n_pts, 3, dtype=torch.float32)
        discontinuities[:, 0] = x_positions
        discontinuities[:, 1] = ic_values
        discontinuities[:, 2] = ic_values

        disc_mask = torch.ones(n_pts, dtype=torch.float32)

        result = dict(input_data)
        result["discontinuities"] = discontinuities
        result["disc_mask"] = disc_mask
        return result, target_grid


ToGridNoCoords = partial(ToGridInputTransform, include_coords=False)


class CellSamplingTransform:
    """Transform that samples k random query points per FV cell.

    Replaces the single-point x_coords (1, nt, nx) with k uniformly
    sampled points per cell, yielding (1, nt, nx*k). Expands t_coords
    to match. Stores metadata for downstream cell-averaging.

    Because sampling is stochastic (fresh offsets every call), this acts
    like data augmentation when used inside __getitem__.

    Args:
        k: Number of random query points per cell.
        **kwargs: Absorbs grid_config keys (nx, nt, dx, dt).
    """

    def __init__(self, k: int = 10, **kwargs):
        self.k = k

    def __call__(self, input_data: dict, target_grid: torch.Tensor):
        x_coords = input_data["x_coords"]  # (1, nt, nx)
        t_coords = input_data["t_coords"]  # (1, nt, nx)
        dx = input_data["dx"]  # scalar tensor

        _, nt, nx = x_coords.shape
        dx_val = dx.item()

        # Cell left edges: (nx,)
        cell_starts = torch.arange(nx, dtype=torch.float32) * dx_val

        # Random offsets within each cell: (nt, nx, k) in [0, dx)
        offsets = torch.rand(nt, nx, self.k) * dx_val

        # New x coordinates: (nt, nx, k) → (1, nt, nx*k)
        new_x = (cell_starts[None, :, None] + offsets).reshape(nt, nx * self.k)
        new_x = new_x.unsqueeze(0)  # (1, nt, nx*k)

        # Expand t_coords: repeat each time row nx*k times
        # t_coords[:, t, :] is constant across spatial dim, so just take
        # the time values and expand to nx*k columns
        t_values = t_coords[:, :, 0:1]  # (1, nt, 1)
        new_t = t_values.expand(1, nt, nx * self.k)  # (1, nt, nx*k)

        result = dict(input_data)
        result["x_coords"] = new_x
        result["t_coords"] = new_t
        result["cell_sampling_k"] = torch.tensor(self.k, dtype=torch.long)
        result["original_nx"] = torch.tensor(nx, dtype=torch.long)
        return result, target_grid


# Registry of available transforms (string name -> class)
TRANSFORMS = {
    "FlattenDiscontinuities": FlattenDiscontinuitiesTransform,
    "ToGridInput": ToGridInputTransform,
    "ToGridNoCoords": ToGridNoCoords,
    "DiscretizeIC": DiscretizeICTransform,
    "CellSampling": CellSamplingTransform,
}
