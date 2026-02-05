"""
Test module for evaluating trained operator learning models.

This module provides different testing scenarios:
- test_same_distri: Test on the same distribution as training
- test_high_res: Test on higher resolution grids (same nx, nt but dx/2, dt/2)
- test_different_dims: Test on different grid dimensions (same dx, dt but different nx, nt)
"""

import numpy as np
import torch
from loss.lwr_loss import LWRLoss
from operator_data_pipeline import get_dataset
from plot_data import plot_comparison_comet
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "mps")


def _unpack_model_output(output):
    """
    Unpack model output which can be either:
    - A single tensor (pred)
    - A tuple of (pred, delta_u, gate_values)
    
    Returns:
        pred: Prediction tensor
        delta_u: Delta u tensor or None
        gate_values: Gate values list or empty list
    """
    if isinstance(output, tuple) and len(output) == 3:
        return output
    else:
        # Model returns only prediction
        return output, None, []


def test_same_distri(model, test_loader, args, experiment=None, mode="test"):
    """
    Test the model on the same distribution as training.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        args: Command line arguments containing nt, nx, dt, dx, loss, num_plots
        experiment: Optional Comet experiment for logging (None disables logging)
        mode: Mode string for logging prefix
    
    Returns:
        Test loss value
    """
    model.eval()
    test_pde_loss = LWRLoss(args.nt, args.nx, args.dt, args.dx, loss_type=args.loss, pinn_weight=args.pinn_weight)
    gts = []
    preds = []
    
    for full_input, targets in tqdm(test_loader, desc=f"Testing ({mode})", leave=False):
        full_input = full_input.to(device)
        targets = targets.to(device)
        
        if args.pinn_weight > 0:
            full_input.requires_grad_(True)
        
        if args.autoregressive:
            from train import pred_autoregressive
            pred = pred_autoregressive(model, targets, 0.0, args)
        else:
            output = model(full_input)
            pred, _, _ = _unpack_model_output(output)
        
        test_pde_loss(pred, targets, full_input)
        for i in range(targets.shape[0]):
            gt = targets[i].squeeze(0).detach().cpu().numpy()
            p = pred[i].squeeze(0).detach().cpu().numpy()
            gts.append(gt)
            preds.append(p)

    test_loss = test_pde_loss.get_loss_value()

    # Log metrics and plots only if experiment is provided
    if experiment is not None:
        test_pde_loss.log_loss_values(experiment, mode)
        print(f"Test Loss ({mode}): {test_loss:.6f}")

        gts_np = np.array(gts)[:args.num_plots]
        preds_np = np.array(preds)[:args.num_plots]
        plot_comparison_comet(gts_np, preds_np, args.nx, args.nt, args.dx, args.dt, experiment, epoch=args.epochs, mode=mode)

    return test_loss


def test_high_res(model, args, experiment=None, mode="test_high_res"):
    """
    Test the model on higher resolution grids.
    Uses same nx, nt as training but with dx/2, dt/2 (2x finer resolution).
    
    Args:
        model: Trained model
        args: Command line arguments
        experiment: Optional Comet experiment for logging (None disables logging)
        mode: Mode string for logging prefix
    
    Returns:
        Test loss value
    """
    # Create high resolution dataset with same nx, nt but finer dx, dt
    high_res_dx = args.dx / 2
    high_res_dt = args.dt / 2
    
    print(f"Testing high resolution: nx={args.nx}, nt={args.nt}, dx={high_res_dx}, dt={high_res_dt}")
    
    high_res_dataset = get_dataset(
        args.solver, args.flux, 
        n_samples=max(100, args.n_samples // 10),  # Use fewer samples for testing
        nx=args.nx, 
        nt=args.nt, 
        dx=high_res_dx, 
        dt=high_res_dt
    )
    high_res_loader = DataLoader(
        high_res_dataset, 
        batch_size=max(1, args.batch_size // 2), 
        shuffle=False, 
        num_workers=4
    )
    
    model.eval()
    test_pde_loss = LWRLoss(args.nt, args.nx, high_res_dt, high_res_dx, loss_type=args.loss, pinn_weight=args.pinn_weight)
    gts = []
    preds = []
    
    for full_input, targets in tqdm(high_res_loader, desc=f"Testing ({mode})", leave=False):
        full_input = full_input.to(device)
        targets = targets.to(device)
        
        if args.pinn_weight > 0:
            full_input.requires_grad_(True)
        
        if args.autoregressive:
            from train import pred_autoregressive
            pred = pred_autoregressive(model, targets, 0.0, args)
        else:
            output = model(full_input)
            pred, _, _ = _unpack_model_output(output)
        
        test_pde_loss(pred, targets, full_input)
        for i in range(targets.shape[0]):
            gt = targets[i].squeeze(0).detach().cpu().numpy()
            p = pred[i].squeeze(0).detach().cpu().numpy()
            gts.append(gt)
            preds.append(p)

    test_loss = test_pde_loss.get_loss_value()

    # Log metrics and plots only if experiment is provided
    if experiment is not None:
        test_pde_loss.log_loss_values(experiment, mode)
        print(f"Test Loss ({mode}): {test_loss:.6f}")

        gts_np = np.array(gts)[:args.num_plots]
        preds_np = np.array(preds)[:args.num_plots]
        plot_comparison_comet(gts_np, preds_np, args.nx, args.nt, high_res_dx, high_res_dt, experiment, epoch=args.epochs, mode=mode)
    
    return test_loss


def test_different_dims(model, args, experiment=None, mode="test_diff_dims"):
    """
    Test the model on grids with different dimensions.
    Uses same dx, dt as training but with 2x nx, 2x nt.
    
    Args:
        model: Trained model
        args: Command line arguments
        experiment: Optional Comet experiment for logging (None disables logging)
        mode: Mode string for logging prefix
    
    Returns:
        Test loss value
    """
    # Create dataset with different dimensions (2x nx, 2x nt) but same dx, dt
    diff_dims_nx = args.nx * 2
    diff_dims_nt = args.nt * 2
    
    print(f"Testing different dimensions: nx={diff_dims_nx}, nt={diff_dims_nt}, dx={args.dx}, dt={args.dt}")
    
    diff_dims_dataset = get_dataset(
        args.solver, args.flux,
        n_samples=max(100, args.n_samples // 10),  # Use fewer samples for testing
        nx=diff_dims_nx,
        nt=diff_dims_nt,
        dx=args.dx,
        dt=args.dt
    )
    diff_dims_loader = DataLoader(
        diff_dims_dataset,
        batch_size=max(1, args.batch_size // 4),  # Smaller batch size for larger grids
        shuffle=False,
        num_workers=4
    )
    
    model.eval()
    test_pde_loss = LWRLoss(diff_dims_nt, diff_dims_nx, args.dt, args.dx, loss_type=args.loss, pinn_weight=args.pinn_weight)
    gts = []
    preds = []
    
    for full_input, targets in tqdm(diff_dims_loader, desc=f"Testing ({mode})", leave=False):
        full_input = full_input.to(device)
        targets = targets.to(device)
        
        if args.pinn_weight > 0:
            full_input.requires_grad_(True)
        
        if args.autoregressive:
            # For autoregressive, we need to use the different nt
            from train import pred_autoregressive
            # Create a modified args for the different nt
            class TempArgs:
                def __init__(self, orig_args, new_nt):
                    self.__dict__.update(orig_args.__dict__)
                    self.nt = new_nt
            temp_args = TempArgs(args, diff_dims_nt)
            pred = pred_autoregressive(model, targets, 0.0, temp_args)
        else:
            output = model(full_input)
            pred, _, _ = _unpack_model_output(output)
        
        test_pde_loss(pred, targets, full_input)
        for i in range(targets.shape[0]):
            gt = targets[i].squeeze(0).detach().cpu().numpy()
            p = pred[i].squeeze(0).detach().cpu().numpy()
            gts.append(gt)
            preds.append(p)

    test_loss = test_pde_loss.get_loss_value()

    # Log metrics and plots only if experiment is provided
    if experiment is not None:
        test_pde_loss.log_loss_values(experiment, mode)
        print(f"Test Loss ({mode}): {test_loss:.6f}")

        gts_np = np.array(gts)[:args.num_plots]
        preds_np = np.array(preds)[:args.num_plots]
        plot_comparison_comet(gts_np, preds_np, diff_dims_nx, diff_dims_nt, args.dx, args.dt, experiment, epoch=args.epochs, mode=mode)
    
    return test_loss


def test_model(model, args, experiment, test_loader, test_high_res_flag=False, test_dims_grid=False):
    """
    Main test function that calls different test functions based on arguments.
    
    Args:
        model: Trained model
        args: Command line arguments
        experiment: Comet experiment for logging (can be None for dry runs)
        test_loader: DataLoader for standard test data
        test_high_res_flag: If True, run high resolution test
        test_dims_grid: If True, run different dimensions test
    
    Returns:
        Dictionary with test results
    """
    results = {}
    
    # Always run standard test on same distribution
    print("\n" + "="*60)
    print("Testing on same distribution...")
    print("="*60)
    results["test_same_distri"] = test_same_distri(model, test_loader, args, experiment, mode="test")
    
    # Run high resolution test if requested
    if test_high_res_flag:
        print("\n" + "="*60)
        print("Testing on high resolution grids...")
        print("="*60)
        results["test_high_res"] = test_high_res(model, args, experiment, mode="test_high_res")
    
    # Run different dimensions test if requested
    if test_dims_grid:
        print("\n" + "="*60)
        print("Testing on different grid dimensions...")
        print("="*60)
        results["test_diff_dims"] = test_different_dims(model, args, experiment, mode="test_diff_dims")
    
    return results


def run_sanity_check(model, train_loader, val_loader, args):
    """
    Run a quick sanity check to ensure all code paths work before training.
    No logging is performed.
    
    Args:
        model: Model to test
        train_loader: Training data loader (uses first batch)
        val_loader: Validation data loader (uses first batch)
        args: Command line arguments
    
    Returns:
        True if sanity check passes, raises exception otherwise
    """
    print("\n" + "="*60)
    print("Running sanity check (no logging)...")
    print("="*60)
    
    model.eval()
    
    # Test forward pass with training data
    print("  [1/4] Testing forward pass on training batch...")
    for full_input, targets in train_loader:
        full_input = full_input.to(device)
        targets = targets.to(device)
        output = model(full_input)
        pred, delta_u, gate_values = _unpack_model_output(output)
        print(f"        Input shape: {full_input.shape}, Output shape: {pred.shape}")
        break
    
    # Test forward pass with validation data
    print("  [2/4] Testing forward pass on validation batch...")
    for full_input, targets in val_loader:
        full_input = full_input.to(device)
        targets = targets.to(device)
        output = model(full_input)
        pred, delta_u, gate_values = _unpack_model_output(output)
        break
    
    # Test loss computation
    print("  [3/4] Testing loss computation...")
    if args.pinn_weight > 0:
        full_input.requires_grad_(True)
        output = model(full_input)  # Recompute with grad enabled
        pred, _, _ = _unpack_model_output(output)
    test_pde_loss = LWRLoss(args.nt, args.nx, args.dt, args.dx, loss_type=args.loss, pinn_weight=args.pinn_weight)
    loss = test_pde_loss(pred, targets, full_input)
    loss_value = test_pde_loss.get_loss_value()
    print(f"        Loss value: {loss_value:.6f}")
    test_pde_loss.show_loss_values()
    # Test backward pass (single step)
    print("  [4/4] Testing backward pass...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer.zero_grad()
    
    for full_input, targets in train_loader:
        full_input = full_input.to(device)
        targets = targets.to(device)
        if args.pinn_weight > 0:
            full_input.requires_grad_(True)
            
        output = model(full_input)
        pred, delta_u, gate_values = _unpack_model_output(output)
        
        train_pde_loss = LWRLoss(args.nt, args.nx, args.dt, args.dx, loss_type=args.loss, pinn_weight=args.pinn_weight, subset=0.5)
        loss = train_pde_loss(pred, targets, full_input)
        loss.backward()
        
        # Check gradients exist
        has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
        if not has_grads:
            raise RuntimeError("No gradients computed during backward pass!")
        
        print("        Backward pass successful, gradients computed")
        break
    
    model.eval()
    print("\n✓ Sanity check passed! All code paths work correctly.\n")
    return True
