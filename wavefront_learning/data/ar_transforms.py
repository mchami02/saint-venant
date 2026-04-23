"""Transform for autoregressive block prediction.

``ARBlockTransform`` pulls a single grid row at timestep ``t_start`` as
input and the next ``k`` rows as target, rebuilding ``t_coords`` as
*relative* times ``[1..k] * dt`` so the downstream model sees the same
time axis regardless of where in the rollout it is.

Modes:
    - ``random``: sample ``t_start ~ Uniform[0, nt-k-1]`` each call.
        Used during training and standard (block) testing.
    - ``fixed0``: always ``t_start = 0``. Deterministic sanity path.
    - ``rollout``: same as ``fixed0`` but also injects ``full_target_grid``
        into the input dict. ``eval_ar_rollout`` drives the stitching.
"""

import torch


class ARBlockTransform:
    """Sample an autoregressive-block (state_row → next k rows) window.

    Args:
        **kwargs: Absorbs ``grid_config`` keys — reads ``ar_block_k``
            (default 1), ``ar_t_start_mode`` (default "random"), ``nt``,
            ``dt``.
    """

    def __init__(self, **kwargs):
        self.k = int(kwargs.get("ar_block_k", 1))
        self.mode = kwargs.get("ar_t_start_mode", "random")
        self.nt = int(kwargs["nt"])
        self.dt = float(kwargs["dt"])

    def __call__(self, input_data: dict, target_grid: torch.Tensor):
        # target_grid: (C, nt, nx)
        C, nt_actual, nx = target_grid.shape
        assert nt_actual == self.nt, (
            f"ARBlockTransform expected nt={self.nt}, got {nt_actual}"
        )
        assert self.k < self.nt, (
            f"ARBlockTransform needs k < nt (got k={self.k}, nt={self.nt})"
        )

        # Pick starting index. Keep one row of slack so target stays
        # within bounds: t_start + k <= nt - 1 ⇒ t_start <= nt - 1 - k.
        upper = nt_actual - self.k - 1
        if self.mode == "random":
            t_start = int(torch.randint(0, upper + 1, (1,)).item())
        else:  # "fixed0" or "rollout"
            t_start = 0

        state_row = target_grid[:, t_start, :].clone()  # (C, nx)
        block_target = target_grid[
            :, t_start + 1 : t_start + 1 + self.k, :
        ].clone()  # (C, k, nx)

        # Build relative time coords: dt * [1..k], replicated across nx.
        dt_v = float(self.dt)
        t_rel = torch.arange(1, self.k + 1, dtype=torch.float32) * dt_v  # (k,)
        t_coords = (
            t_rel.unsqueeze(-1).expand(self.k, nx).unsqueeze(0).clone()
        )  # (1, k, nx)

        # Spatial coords: reuse row 0 of input's x_coords → replicate.
        x_coords_base = input_data["x_coords"][0, 0, :]  # (nx,)
        x_coords = (
            x_coords_base.unsqueeze(0).expand(self.k, nx).unsqueeze(0).clone()
        )  # (1, k, nx)

        result = dict(input_data)
        result["state_row"] = state_row
        result["t_start"] = torch.tensor(t_start, dtype=torch.long)
        result["t_coords"] = t_coords
        result["x_coords"] = x_coords

        if self.mode == "rollout":
            result["full_target_grid"] = target_grid.clone()

        return result, block_target
