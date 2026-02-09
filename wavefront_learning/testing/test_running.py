"""Functions for verifying that models run correctly.

Includes sanity checks (forward/backward pass verification),
profiling, and basic inference.
"""

import argparse
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from logger import WandbLogger
from loss import get_loss
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.data import DataLoader


def run_inference(
    model: nn.Module,
    input_data: dict | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Run inference on input data.

    Args:
        model: Trained model.
        input_data: Input tensor or dict.
        device: Computation device.

    Returns:
        Model prediction tensor.
    """
    model.eval()

    with torch.no_grad():
        if isinstance(input_data, dict):
            input_data = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in input_data.items()
            }
        else:
            input_data = input_data.to(device)

        pred = model(input_data)

    return pred


def run_sanity_check(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> bool:
    """Run sanity checks to verify all code paths before training.

    Performs 4 checks:
    1. Forward pass on training batch
    2. Forward pass on validation batch
    3. Loss computation
    4. Backward pass (gradient check)

    Args:
        model: Model to check.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        args: Training arguments (needs loss, dt, dx, smooth_loss_type).
        device: Computation device.

    Returns:
        True if all checks pass.

    Raises:
        RuntimeError: If any check fails.
    """
    print("\n" + "=" * 50)
    print("Running sanity checks...")
    print("=" * 50)

    model.train()

    # Get loss function - build kwargs for losses that need dt/dx
    loss_kwargs = {
        "pde_residual": {"dt": args.dt, "dx": args.dx},
        "rh_residual": {"dt": args.dt},
    }
    loss_fn = get_loss(args.loss, loss_kwargs=loss_kwargs)

    # [1/4] Forward pass on training batch
    print("\n[1/4] Testing forward pass on training batch...")
    train_batch = next(iter(train_loader))
    batch_input, batch_target = train_batch

    # Move to device
    if isinstance(batch_input, dict):
        batch_input = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch_input.items()
        }
    else:
        batch_input = batch_input.to(device)
    batch_target = batch_target.to(device)

    try:
        pred = model(batch_input)
        if isinstance(pred, dict):
            print(f"  Output type: dict with keys {list(pred.keys())}")
            for k, v in pred.items():
                if isinstance(v, torch.Tensor):
                    print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  Output shape: {pred.shape}")
        print("  [PASS] Forward pass on training batch")
    except Exception as e:
        raise RuntimeError(f"[FAIL] Forward pass on training batch: {e}") from e

    # [2/4] Forward pass on validation batch
    print("\n[2/4] Testing forward pass on validation batch...")
    val_batch = next(iter(val_loader))
    val_input, val_target = val_batch

    # Move to device
    if isinstance(val_input, dict):
        val_input = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in val_input.items()
        }
    else:
        val_input = val_input.to(device)
    val_target = val_target.to(device)

    try:
        model.eval()
        with torch.no_grad():
            val_pred = model(val_input)
        if isinstance(val_pred, dict):
            print(f"  Output type: dict with keys {list(val_pred.keys())}")
        else:
            print(f"  Output shape: {val_pred.shape}")
        print("  [PASS] Forward pass on validation batch")
    except Exception as e:
        raise RuntimeError(f"[FAIL] Forward pass on validation batch: {e}") from e

    # [3/4] Loss computation
    print("\n[3/4] Testing loss computation...")
    model.train()
    try:
        # Re-run forward pass for fresh computation graph
        pred = model(batch_input)
        # Use new signature: (input_dict, output_dict, target)
        loss, components = loss_fn(batch_input, pred, batch_target)
        print(f"  Loss value: {loss.item():.6f}")
        print(f"  Loss components: {list(components.keys())}")
        for k, v in components.items():
            print(f"    {k}: {v:.6f}")
        print("  [PASS] Loss computation")
    except Exception as e:
        raise RuntimeError(f"[FAIL] Loss computation: {e}") from e

    # [4/4] Backward pass
    print("\n[4/4] Testing backward pass...")
    try:
        loss.backward()
        grad_count = 0
        total_params = 0
        for _name, param in model.named_parameters():
            total_params += 1
            if param.grad is not None:
                grad_count += 1
        print(f"  Parameters with gradients: {grad_count}/{total_params}")
        if grad_count == 0:
            raise RuntimeError("No gradients computed!")
        print("  [PASS] Backward pass")
    except Exception as e:
        raise RuntimeError(f"[FAIL] Backward pass: {e}") from e

    # Clean up gradients
    model.zero_grad()

    print("\n" + "=" * 50)
    print("Sanity check passed!")
    print("=" * 50 + "\n")

    return True


def run_profiler(
    model: nn.Module,
    train_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    logger: WandbLogger | None = None,
    num_steps: int = 10,
    warmup_steps: int = 2,
) -> None:
    """Profile training steps and optionally upload trace to wandb.

    Args:
        model: Model to profile.
        train_loader: Training data loader.
        args: Training arguments (needs loss, dt, dx, smooth_loss_type).
        device: Computation device.
        logger: Optional WandbLogger for uploading artifacts.
        num_steps: Number of active profiling steps.
        warmup_steps: Number of warmup steps before profiling.
    """
    import wandb

    print("\n" + "=" * 50)
    print("Running profiler...")
    print("=" * 50)

    # Get loss function - build kwargs for losses that need dt/dx
    loss_kwargs = {
        "pde_residual": {"dt": args.dt, "dx": args.dx},
        "rh_residual": {"dt": args.dt},
    }
    loss_fn = get_loss(args.loss, loss_kwargs=loss_kwargs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Determine profiler activities
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    # Create profiler schedule
    prof_schedule = schedule(
        wait=0,
        warmup=warmup_steps,
        active=num_steps,
        repeat=1,
    )

    model.train()
    data_iter = iter(train_loader)

    total_steps = warmup_steps + num_steps
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Active steps: {num_steps}")
    print(f"  Total steps: {total_steps}")

    with profile(
        activities=activities,
        schedule=prof_schedule,
        on_trace_ready=lambda p: None,  # We'll export manually
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step in range(total_steps):
            try:
                batch_input, batch_target = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch_input, batch_target = next(data_iter)

            # Move to device
            if isinstance(batch_input, dict):
                batch_input = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch_input.items()
                }
            else:
                batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)

            # Training step
            optimizer.zero_grad()
            pred = model(batch_input)
            # Use new signature: (input_dict, output_dict, target)
            loss, _ = loss_fn(batch_input, pred, batch_target)
            loss.backward()
            optimizer.step()

            prof.step()

            if step < warmup_steps:
                print(f"  Step {step + 1}/{total_steps} (warmup)")
            else:
                print(f"  Step {step + 1}/{total_steps} (profiling)")

    # Generate summary table
    sort_by = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
    summary_table = prof.key_averages().table(sort_by=sort_by, row_limit=30)
    print("\nProfiler Summary:")
    print(summary_table)

    # Export trace and upload to wandb
    if logger is not None and logger.enabled and logger.run is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.json"
            summary_path = Path(tmpdir) / "summary.txt"

            # Export Chrome trace
            prof.export_chrome_trace(str(trace_path))

            # Save summary
            with open(summary_path, "w") as f:
                f.write(summary_table)

            # Create and upload artifact
            run_id = logger.run.id
            artifact = wandb.Artifact(
                f"profiler-trace-{run_id}",
                type="profile",
                description="PyTorch profiler trace and summary",
            )
            artifact.add_file(str(trace_path), name="trace.json")
            artifact.add_file(str(summary_path), name="summary.txt")
            logger.run.log_artifact(artifact)

            print(f"\n  Uploaded profiler artifact: profiler-trace-{run_id}")

    print("\n" + "=" * 50)
    print("Profiling complete!")
    print("=" * 50 + "\n")
