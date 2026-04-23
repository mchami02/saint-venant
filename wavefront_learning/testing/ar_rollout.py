"""Full-grid rollout evaluation for ARWaveNO-family models.

Maintains a running history buffer of the last ``k_in`` rows (starting
with the IC replicated) and feeds it to the model each block, stitching
``k``-row predictions into a complete ``(nt, nx)`` grid. Reports
per-sample rollout MSE + wall-clock plus plot-ready samples from the
last batch.
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
        model: Trained ARWaveNO-family model with ``.k`` and ``.k_in``.
        args: Training arguments.
        device: Computation device.
        grid_config: Grid configuration. Augmented here with
            ``ar_t_start_mode="rollout"`` so the transform attaches the
            full grid under ``full_target_grid``.

    Returns:
        ``{"rollout_mse", "rollout_time_ms", "samples"}``.
    """
    k = model.k
    k_in = getattr(model, "k_in", k)

    rollout_config = {
        **grid_config,
        "ar_block_k": int(getattr(args, "ar_block_k", k)),
        "ar_hist_k": int(getattr(args, "ar_hist_k", k_in)),
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
    nt = int(grid_config["nt"])

    all_mse: list[float] = []
    all_times_per_sample: list[float] = []
    last_pred: torch.Tensor | None = None
    last_target: torch.Tensor | None = None

    with torch.no_grad():
        for batch_input, _ in tqdm(rollout_loader, desc="Rollout"):
            batch_input = _to_device(batch_input, device)

            full_target = batch_input["full_target_grid"]  # (B, C, nt, nx)
            B, C, _, nx = full_target.shape

            pred = torch.zeros_like(full_target)
            pred[:, :, 0, :] = full_target[:, :, 0, :]

            # Seed history buffer: IC replicated k_in times along time.
            ic_row = pred[:, :, 0, :]  # (B, C, nx)
            history = ic_row.unsqueeze(2).expand(B, C, k_in, nx).clone()

            block_input = dict(batch_input)

            _sync_device(device)
            start_time = time.perf_counter()

            t0 = 0
            while t0 < nt - 1:
                t_end = min(t0 + k, nt - 1)
                k_actual = t_end - t0
                block_input["state_hist"] = history
                out = model(block_input)["output_grid"]  # (B, C, k, nx)
                pred[:, :, t0 + 1 : t_end + 1, :] = out[:, :, :k_actual, :]

                # Update history: last k_in rows of predictions stitched
                # up to and including t_end.
                if t_end + 1 >= k_in:
                    history = pred[
                        :, :, t_end + 1 - k_in : t_end + 1, :
                    ].clone()
                else:
                    # Not enough predicted rows yet -- pad remaining with IC.
                    have = t_end + 1
                    pad = k_in - have
                    ic_pad = ic_row.unsqueeze(2).expand(B, C, pad, nx).clone()
                    tail = pred[:, :, :t_end + 1, :].clone()
                    history = torch.cat([ic_pad, tail], dim=2)

                t0 = t_end

            _sync_device(device)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            sq_err = (pred - full_target).pow(2).mean(dim=(1, 2, 3))  # (B,)
            all_mse.extend(sq_err.detach().cpu().tolist())
            all_times_per_sample.append(elapsed_ms / B)

            last_pred = pred.detach()
            last_target = full_target.detach()

    result: dict = {
        "rollout_mse": float(np.mean(all_mse)),
        "rollout_time_ms": float(np.mean(all_times_per_sample)),
    }

    if last_pred is not None and last_target is not None:
        num_plots = 3
        pred_np = last_pred[:num_plots].cpu().numpy()
        tgt_np = last_target[:num_plots].cpu().numpy()
        if pred_np.ndim == 4 and pred_np.shape[1] == 1:
            pred_np = pred_np.squeeze(1)
        if tgt_np.ndim == 4 and tgt_np.shape[1] == 1:
            tgt_np = tgt_np.squeeze(1)
        result["samples"] = {
            "output_grid": pred_np,
            "grids": tgt_np,
        }
    return result
