from comet_ml import start
from comet_ml.integration.pytorch import log_model
import argparse
import random
from typing import Any
from concurrent.futures import ThreadPoolExecutor
from numerical_methods import Godunov, Greenshields, Triangular, LWRRiemannSolver, SVERiemannSolver
from operator_data_pipeline import get_datasets
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import os
from model import create_model
from torchinfo import summary
import matplotlib.pyplot as plt
from loss.lwr_loss import LWRLoss
from loss.pde_loss import PDELoss
from plot_data import plot_comparison_comet

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# Global thread pool for async plotting (1 worker to avoid overwhelming resources)
_plot_executor = ThreadPoolExecutor(max_workers=1)


def async_plot(ground_truth, prediction, nx, nt, dx, dt, experiment, epoch):
    """Submit plotting task to background thread. Copies data to avoid race conditions."""
    # Copy data since plotting happens asynchronously
    gt_copy = np.copy(ground_truth)
    pred_copy = np.copy(prediction)
    _plot_executor.submit(plot_comparison_comet, gt_copy, pred_copy, nx, nt, dx, dt, experiment, epoch)

def get_solver(args):
    if args.solver == "Godunov":
        if args.flux == "Greenshields":
            flux = Greenshields(vmax=1.0, rho_max=1.0)
            return Godunov(riemann_solver=LWRRiemannSolver(flux))
        elif args.flux == "Triangular":
            flux = Triangular(vmax=1.0, rho_max=1.0)
            return Godunov(riemann_solver=LWRRiemannSolver(flux))
        else:
            raise ValueError(f"Flux {args.flux} not supported")
    elif args.solver == "SVESolver":
        return Godunov(riemann_solver=SVERiemannSolver())
    else:
        raise ValueError(f"Solver {args.solver} not supported")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--nx", type=int, default=50)
    parser.add_argument("--nt", type=int, default=250)
    parser.add_argument("--dx", type=float, default=0.25)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--bc", type=str, default="GhostCell")
    parser.add_argument("--solver", type=str, default="LaxHopf")
    parser.add_argument("--flux", type=str, default="Greenshields")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--model", type=str, default="FNO")
    parser.add_argument("--n_modes", type=int, default=128)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--out_channels", type=int, default=1)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--step_size", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--save_path", type=str, default="operator.pth")
    parser.add_argument("--n_datasets", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr_decay", type=float, default=0.1)
    parser.add_argument("--num_plots", type=int, default=5)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--autoregressive", action="store_true", help="Train autoregressively with scheduled sampling")
    parser.add_argument("--residuals", action="store_true", help="Predict residuals instead of full solution")
    parser.add_argument("--gamma_decay", type=float, default=1.0, help="Decay factor for gamma in decaying loss")
    parser.add_argument("--pinn_loss", action="store_true", help="Use PINN loss")
    parser.add_argument("--decaying_loss", action="store_true", help="Use decaying loss")
    parser.add_argument("--plot_every", type=int, default=5, help="Plot comparison every N epochs (0 = only at end)")
    return parser.parse_args()


def plot_training_history(train_losses, val_losses, save_path="results/training_history.png"):
    """
    Plot training and validation losses.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save the plot
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    
    # Plot losses
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {save_path}")




def pred_autoregressive(model, targets, teacher_forcing_ratio, args):
    targets = targets.to(device)
    prediction = targets.clone()
    model_input = targets[:, :, 0]
    for t in range(1, args.nt):
        if random.random() < teacher_forcing_ratio:
            model_input = targets[:, :, t]
        prediction[:, :, t, 1:-1] = model(model_input)[:, :, 1:-1]
        if args.residuals:
            prediction[:, :, t, 1:-1] += model_input[:, :, 1:-1]
        model_input = prediction[:, :, t]
    return prediction

def train_epoch(model, train_loader, val_loader, optimizer, criterion, epoch,args, experiment):
    """
    Training loop for one-shot FNO prediction.
    Model predicts entire spatiotemporal solution in one forward pass.
    """
    model.train()
    train_pde_loss = LWRLoss(args.nt, args.nx, args.dt, args.dx, decaying_loss=args.decaying_loss, pinn_loss=args.pinn_loss)
    for full_input, targets in tqdm(train_loader, desc="Train epoch", leave=False):
        # full_input: (B, nt, nx, 3)
        # targets: (B, nt, nx)
        
        full_input = full_input.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass - single prediction for entire grid
        if args.autoregressive:
            # Prevent division by zero by ensuring denominator is at least 1
            if epoch < args.epochs // 4:
                teacher_forcing_ratio = 1.0 - (epoch / (args.epochs // 4))
            else:
                teacher_forcing_ratio = 0.0
            pred = pred_autoregressive(model, targets, teacher_forcing_ratio, args)
        else:
            pred = model(full_input)  # (B, n_vals, nt, nx)
        
        # Compute loss
        loss = train_pde_loss(pred, targets)
        loss.backward()
        optimizer.step()
    
    train_loss = train_pde_loss.get_loss_value()
    train_pde_loss.log_loss_values(experiment, "train")
    train_pde_loss.show_loss_values()
    model.eval()
    val_pde_loss = LWRLoss(args.nt, args.nx, args.dt, args.dx, decaying_loss=args.decaying_loss)
    with torch.no_grad():
        for full_input, targets in tqdm(val_loader, desc="Val epoch", leave=False):
            full_input = full_input.to(device)
            targets = targets.to(device)

            if args.autoregressive:
                teacher_forcing_ratio = 0.0
                pred = pred_autoregressive(model, targets, teacher_forcing_ratio, args)
            else:
                pred = model(full_input)  # (B, n_vals, nt, nx)
            
            loss = val_pde_loss(pred, targets)

    val_loss = val_pde_loss.get_loss_value()
    val_pde_loss.log_loss_values(experiment, "val")
    return train_loss, val_loss


def sample_predictions(model, val_loader, args, num_samples=3):
    """Sample a few predictions from validation set for visualization."""
    model.eval()
    gts, preds = [], []
    with torch.no_grad():
        for full_input, targets in val_loader:
            full_input = full_input.to(device)
            targets = targets.to(device)
            
            if args.autoregressive:
                pred = pred_autoregressive(model, targets, 0.0, args)
            else:
                pred = model(full_input)
            
            for i in range(min(num_samples - len(gts), targets.shape[0])):
                gts.append(targets[i].squeeze(0).cpu().numpy())
                preds.append(pred[i].squeeze(0).cpu().numpy())
            
            if len(gts) >= num_samples:
                break
    
    return np.array(gts), np.array(preds)


def train_model(model, train_loader, val_loader, args, experiment):
    """
    Train FNO for multiple epochs with learning rate scheduling and early stopping.
    
    If args.autoregressive is True, uses scheduled sampling where teacher forcing
    ratio decays from 1.0 to 0.0 over epochs.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    epochs_without_improvement = 0
    current_lr = optimizer.param_groups[0]['lr']
    tolerance = 1e-2
    # Track training history
    train_losses = []
    val_losses = []
    
    desc = "Training (Autoregressive)" if args.autoregressive else "Training"
    bar = tqdm(range(args.epochs), desc=desc)
    for epoch in bar:
        experiment.set_epoch(epoch+1)
        train_loss, val_loss = train_epoch(model, train_loader, val_loader, optimizer, criterion, epoch, args, experiment)
        
        # Record history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        experiment.log_metric("train/lr", current_lr)
        # Check for improvement
        if val_loss < best_loss * (1 - tolerance):
            best_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), args.save_path)
        else:
            epochs_without_improvement += 1
            
        # Reduce learning rate if no improvement for patience/2 epochs
        if epochs_without_improvement == args.patience // 2:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_decay

        if epochs_without_improvement == args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs (patience={args.patience})")
            break

        # Async plotting during training (non-blocking)
        if args.plot_every > 0 and (epoch + 1) % args.plot_every == 0:
            gts, preds = sample_predictions(model, val_loader, args, num_samples=2)
            async_plot(gts, preds, args.nx, args.nt, args.dx, args.dt, experiment, epoch)
            model.train()  # Restore training mode

        postfix = {"Train": f"{train_loss:.2e}", "Val": f"{val_loss:.2e}", "LR": f"{current_lr:.2e}"}
        bar.set_postfix(postfix)
    
    print(f"\nTraining completed! Final best loss: {best_loss:.6f}")
    
    # Plot training history
    # plot_training_history(train_losses, val_losses)
    state_dict = torch.load(args.save_path, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    return model

def test_model(model, test_loader, args, experiment):
    model.eval()
    test_pde_loss = LWRLoss(args.nt, args.nx, args.dt, args.dx, decaying_loss=args.decaying_loss, pinn_loss=False)
    gts = []
    preds = []
    
    with torch.no_grad():
        for full_input, targets in test_loader:
            full_input = full_input.to(device)
            targets = targets.to(device)
            
            if args.autoregressive:
                pred = pred_autoregressive(model, targets, 0.0, args)
            else:
                pred = model(full_input)
            
            loss = test_pde_loss(pred, targets)
            for i in range(targets.shape[0]):
                gt = targets[i].squeeze(0).detach().cpu().numpy()
                p = pred[i].squeeze(0).detach().cpu().numpy()
                gts.append(gt)
                preds.append(p)
    
    test_loss = test_pde_loss.get_loss_value()
    test_pde_loss.log_loss_values(experiment, "test")
    mode = "Autoregressive" if args.autoregressive else "One-shot"
    print(f"Test Loss ({mode}): {test_loss:.6f}")
    
    gts = np.array(gts)[:args.num_plots]
    preds = np.array(preds)[:args.num_plots]
    plot_comparison_comet(gts, preds, args.nx, args.nt, args.dx, args.dt, experiment, epoch=args.epochs, test=True)

def main():
    args = parse_args()
    print(f"Using device: {device}")

    print("Using CometML for logging")
    experiment = start(api_key=os.getenv("COMET_API_KEY"), project_name="operator-learning-pde", workspace="pde-thesis")
    experiment.log_parameters(vars(args))
    experiment.log_code(folder="loss")
    experiment.log_code(folder="models")
    experiment.log_code(file_name="operator_data_pipeline.py")
    experiment.log_code(file_name="model.py")
    experiment.log_code(file_name="train.py")
    experiment.log_code(file_name="train.py")
    train_dataset, val_dataset, test_dataset = get_datasets(args.solver, args.flux, args.n_samples, args.nx, args.nt, args.dx, args.dt)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model = create_model(args).to(device)
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path, weights_only=False))
    summary(model)
    
    model = train_model(model, train_loader, val_loader, args, experiment)
    log_model(
        experiment=experiment, 
        model=model,
        model_name=args.model,
        metadata=model.metadata
    )

    test_model(model, test_loader, args, experiment)



if __name__ == "__main__":
    main()
