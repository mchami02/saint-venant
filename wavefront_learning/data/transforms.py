"""Transforms for wavefront learning input representations.

Each transform takes (input_data, target_grid) and returns a
(possibly modified) (input_data, target_grid) pair.
"""

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

    def __init__(self, nx: int, nt: int, **kwargs):
        self.nx = nx
        self.nt = nt

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

        # Stack: [ic_masked, t_coords, x_coords] -> (3, nt, nx)
        full_input = torch.cat([ic_masked, t_coords, x_coords], dim=0)

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


# Registry of available transforms (string name -> class)
TRANSFORMS = {
    "FlattenDiscontinuities": FlattenDiscontinuitiesTransform,
    "ToGridInput": ToGridInputTransform,
    "DiscretizeIC": DiscretizeICTransform,
}
