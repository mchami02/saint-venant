"""Transform for autoregressive block prediction with k-step history.

``ARBlockTransform`` pulls the **last ``k_in`` grid rows ending at
``t_start``** as input and the next ``k_out`` rows as target, rebuilding
``t_coords`` as *relative* times ``[1..k_out] * dt`` (future) and
``hist_t_coords`` as ``[-(k_in-1)..0] * dt`` (past-ending-at-now) so the
downstream model sees the same time axis regardless of where in the
rollout it is.

Modes:
    - ``random``: sample ``t_start ~ Uniform[k_in-1, nt-k_out-1]`` each
        call. Used during training and standard (block) testing.
    - ``fixed0``: ``t_start = k_in - 1`` (earliest valid t_start,
        so the first ``k_in`` rows form the full history block).
    - ``rollout``: ``t_start = 0`` and the history is IC replicated
        ``k_in`` times. ``full_target_grid`` is injected for
        ``eval_ar_rollout`` to stitch.
"""

import torch


class ARBlockTransform:
    """Sample a (state_hist → next k_out rows) window.

    Args:
        **kwargs: Absorbs ``grid_config`` keys — reads ``ar_block_k``
            (output block, default 1), ``ar_hist_k`` (input history
            length, default equal to ``ar_block_k``), ``ar_t_start_mode``
            (default "random"), ``nt``, ``dt``.
    """

    def __init__(self, **kwargs):
        self.k_out = int(kwargs.get("ar_block_k", 1))
        k_in = int(kwargs.get("ar_hist_k", -1))
        self.k_in = self.k_out if k_in <= 0 else k_in
        self.mode = kwargs.get("ar_t_start_mode", "random")
        self.nt = int(kwargs["nt"])
        self.dt = float(kwargs["dt"])

    def __call__(self, input_data: dict, target_grid: torch.Tensor):
        # target_grid: (C, nt, nx)
        C, nt_actual, nx = target_grid.shape
        assert nt_actual == self.nt, (
            f"ARBlockTransform expected nt={self.nt}, got {nt_actual}"
        )
        assert self.k_out < self.nt, (
            f"ARBlockTransform needs k_out < nt "
            f"(got k_out={self.k_out}, nt={self.nt})"
        )
        assert self.k_in >= 1, (
            f"ARBlockTransform needs k_in >= 1, got {self.k_in}"
        )

        # Valid t_start range: [0, nt - k_out - 1]. When t_start < k_in-1
        # we need to pad the earliest history rows with the IC (matches
        # how `eval_ar_rollout` seeds the first block at rollout start).
        # This exposes the model to the rollout distribution during
        # training and prevents the "collapse to IC" failure at rollout.
        hi = nt_actual - self.k_out - 1

        if self.mode == "random":
            t_start = int(torch.randint(0, hi + 1, (1,)).item())
        elif self.mode == "fixed0":
            t_start = 0
        else:  # rollout: start at IC and pad history
            t_start = 0

        # Build state_hist (C, k_in, nx) with IC-padding at the start
        # when t_start < k_in - 1.
        n_real = min(self.k_in, t_start + 1)  # number of real rows
        n_pad = self.k_in - n_real  # number of IC-padded rows (>= 0)
        ic_row = target_grid[:, 0, :]  # (C, nx)
        if n_pad > 0:
            pad_block = ic_row.unsqueeze(1).expand(C, n_pad, nx).clone()
            real_block = target_grid[
                :, t_start - n_real + 1 : t_start + 1, :
            ].clone()
            state_hist = torch.cat([pad_block, real_block], dim=1)
        else:
            state_hist = target_grid[
                :, t_start - self.k_in + 1 : t_start + 1, :
            ].clone()

        block_target = target_grid[
            :, t_start + 1 : t_start + 1 + self.k_out, :
        ].clone()  # (C, k_out, nx)

        # Relative future times: dt * [1..k_out] replicated across nx.
        dt_v = float(self.dt)
        t_rel_future = torch.arange(
            1, self.k_out + 1, dtype=torch.float32
        ) * dt_v
        t_coords = (
            t_rel_future.unsqueeze(-1).expand(self.k_out, nx).unsqueeze(0).clone()
        )  # (1, k_out, nx)

        # Relative past times for history: dt * [-(k_in-1)..0] replicated
        # across nx. t=0 is the current state (t_start).
        hist_t_rel = torch.arange(
            -(self.k_in - 1), 1, dtype=torch.float32
        ) * dt_v
        hist_t_coords = (
            hist_t_rel.unsqueeze(-1).expand(self.k_in, nx).unsqueeze(0).clone()
        )  # (1, k_in, nx)

        # Spatial coords: reuse row 0 of input's x_coords → replicate.
        x_coords_base = input_data["x_coords"][0, 0, :]  # (nx,)
        x_coords = (
            x_coords_base.unsqueeze(0).expand(self.k_out, nx).unsqueeze(0).clone()
        )  # (1, k_out, nx)
        hist_x_coords = (
            x_coords_base.unsqueeze(0).expand(self.k_in, nx).unsqueeze(0).clone()
        )  # (1, k_in, nx)

        result = dict(input_data)
        result["state_hist"] = state_hist
        result["t_start"] = torch.tensor(t_start, dtype=torch.long)
        result["t_coords"] = t_coords
        result["x_coords"] = x_coords
        result["hist_t_coords"] = hist_t_coords
        result["hist_x_coords"] = hist_x_coords

        if self.mode == "rollout":
            result["full_target_grid"] = target_grid.clone()

        return result, block_target
