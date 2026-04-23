"""Full-grid rollout evaluation for ARWaveNO-family models.

Builds a dedicated test loader whose ``ARBlockTransform`` is in
``rollout`` mode (puts the full ground-truth grid under
``full_target_grid``), then stitches ``k``-row block predictions into a
complete ``(nt, nx)`` grid, computes MSE against the truth, and reports
per-sample rollout wall-clock.
"""

import time

import numpy as np
import torch
import torch.nn as nn
from data import collate_wavefront_batch, get_wavefront_datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

from testing.test_results import _get_equation_kwargs, _sync_device


def _to_device(batch_input, device):
    if isinstance(batch_input, dict):
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch_input.items()
        }
    return batch_input.to(device)


def eval_ar_rollout(
    model: nn.Module,
    args,
    device: torch.device,
    grid_config: dict,
) -> dict[str, float]:
    """Run a full-grid autoregressive rollout and report MSE + timing.

    Args:
        model: Trained ARWaveNO-family model with ``.k`` attribute.
        args: Training arguments (nx/nt/dx/dt, equation, batch_size, ...).
        device: Computation device.
        grid_config: Grid configuration. Augmented here with
            ``ar_t_start_mode="rollout"`` so the transform attaches the
            full grid under ``full_target_grid``.

    Returns:
        ``{"rollout_mse": ..., "rollout_time_ms": ...}`` (batch-average
        wall-clock, per-sample MSE averaged over the test set).
    """
    rollout_config = {
        **grid_config,
        "ar_block_k": int(getattr(args, "ar_block_k", model.k)),
        "ar_t_start_mode": "rollout",
    }

    _, _, rollout_dataset = get_wavefront_datasets(
        n_samples=args.n_samples,
        grid_config=rollout_config,
        model_name=args.model,
        train_ratio=0.0,
        val_ratio=0.0,
        only_shocks=args.only_shocks,
        max_steps=args.max_steps,
        min_steps=getattr(args, "min_steps", 2),
        equation=args.equation,
        equation_kwargs=_get_equation_kwargs(args),
        cell_sampling_k=getattr(args, "cell_sampling_k", 0),
        transform_override="ARBlock",
        proximity_sigma=getattr(args, "proximity_sigma", None),
        min_component_size=getattr(args, "min_component_size", 5),
        random_seed=args.seed,
    )

    rollout_loader = DataLoader(
        rollout_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_wavefront_batch,
        num_workers=0,
    )

    model.eval()
    k = model.k
    nt = int(grid_config["nt"])

    all_mse: list[float] = []
    all_times_per_sample: list[float] = []

    with torch.no_grad():
        for batch_input, _ in tqdm(rollout_loader, desc="Rollout"):
            batch_input = _to_device(batch_input, device)

            full_target = batch_input["full_target_grid"]  # (B, C, nt, nx)
            B = full_target.shape[0]

            pred = torch.zeros_like(full_target)
            pred[:, :, 0, :] = full_target[:, :, 0, :]
            state = pred[:, :, 0, :].clone()  # (B, C, nx)

            block_input = dict(batch_input)

            _sync_device(device)
            start_time = time.perf_counter()

            t0 = 0
            while t0 < nt - 1:
                t_end = min(t0 + k, nt - 1)
                k_actual = t_end - t0
                block_input["state_row"] = state
                out = model(block_input)["output_grid"]  # (B, C, k, nx)
                pred[:, :, t0 + 1 : t_end + 1, :] = out[:, :, :k_actual, :]
                state = pred[:, :, t_end, :].clone()
                t0 = t_end

            _sync_device(device)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            sq_err = (pred - full_target).pow(2).mean(dim=(1, 2, 3))  # (B,)
            all_mse.extend(sq_err.detach().cpu().tolist())
            all_times_per_sample.append(elapsed_ms / B)

    return {
        "rollout_mse": float(np.mean(all_mse)),
        "rollout_time_ms": float(np.mean(all_times_per_sample)),
    }
