import argparse
import random
from typing import Any
from numerical_methods import Godunov, Greenshields, Triangular, LWRRiemannSolver, SVERiemannSolver, plot_comparison
from operator_data_pipeline import GridDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import os
from model import create_model
from torchinfo import summary
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

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
    parser.add_argument("--solver", type=str, default="Godunov")
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


def train_epoch(model, train_loader, val_loader, optimizer, criterion):
    """
    Training loop for one-shot FNO prediction.
    Model predicts entire spatiotemporal solution in one forward pass.
    """
    model.train()
    running_loss = 0.0
    n_batches = 0
    
    for full_input, targets in tqdm(train_loader, desc="Train epoch", leave=False):
        # full_input: (B, nt, nx, 3)
        # targets: (B, nt, nx)
        
        full_input = full_input.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass - single prediction for entire grid
        pred = model(full_input)  # (B, n_vals, nt, nx)
        
        # Compute loss
        loss = criterion(pred, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        n_batches += 1
    
    train_loss = running_loss / max(1, n_batches)
    
    model.eval()
    running_loss = 0.0
    n_batches = 0
    for full_input, targets in tqdm(val_loader, desc="Val epoch", leave=False):
        full_input = full_input.to(device)
        targets = targets.to(device)

        pred = model(full_input)  # (B, n_vals, nt, nx)
        
        loss = criterion(pred, targets)
        running_loss += loss.item()
        n_batches += 1

    val_loss = running_loss / max(1, n_batches)
    
    return train_loss, val_loss


def train_autoregressive_epoch(model, train_loader, val_loader, optimizer, criterion, teacher_forcing_ratio):
    """
    Autoregressive training with scheduled sampling.
    
    Instead of predicting the whole grid in one shot, this trains the model to predict
    timestep by timestep. Uses scheduled sampling where:
    - teacher_forcing_ratio = 1.0: always use ground truth as input
    - teacher_forcing_ratio = 0.0: always use model predictions as input
    
    The teacher forcing ratio should decay from 1.0 to 0.0 over training epochs.
    
    Args:
        model: The neural operator model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        criterion: Loss function
        teacher_forcing_ratio: Probability of using ground truth (1.0 = all GT, 0.0 = all predictions)
    """
    model.train()
    running_loss = 0.0
    n_batches = 0
    
    for full_input, targets in tqdm(train_loader, desc=f"Train AR (tf={teacher_forcing_ratio:.2f})", leave=False):
        # full_input: (B, n_vals, nt, nx) - has IC and boundaries, rest masked with -1
        # targets: (B, n_vals, nt, nx) - full ground truth
        
        full_input = full_input.to(device)
        targets = targets.to(device)
        
        B, n_vals, nt, nx = targets.shape
        
        optimizer.zero_grad()
        
        # First pass: get initial predictions from the masked input (no gradient)
        with torch.no_grad():
            initial_pred = model(full_input)
        
        # Build mixed input using scheduled sampling
        # For each timestep, decide whether to use ground truth or prediction
        mixed_input = full_input.clone()
        
        for t in range(1, nt):
            if random.random() < teacher_forcing_ratio:
                # Use ground truth for interior points (boundaries stay from full_input)
                mixed_input[:, :, t, 1:-1] = targets[:, :, t, 1:-1]
            else:
                # Use predictions for interior points
                mixed_input[:, :, t, 1:-1] = initial_pred[:, :, t, 1:-1]
        
        # Second pass: train with mixed input
        pred = model(mixed_input)
        loss = criterion(pred, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        n_batches += 1
    
    train_loss = running_loss / max(1, n_batches)
    
    # Validation: use pure autoregressive (no teacher forcing)
    model.eval()
    running_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for full_input, targets in tqdm(val_loader, desc="Val AR", leave=False):
            full_input = full_input.to(device)
            targets = targets.to(device)
            
            B, n_vals, nt, nx = targets.shape
            
            # True autoregressive: predict one timestep at a time using previous predictions
            current_input = full_input.clone()
            
            for t in range(1, nt):
                # Run model with current known inputs
                pred = model(current_input)
                # Use prediction for timestep t as input for next iteration
                current_input[:, :, t, 1:-1] = pred[:, :, t, 1:-1]
            
            # Final prediction after filling all timesteps
            final_pred = model(current_input)
            loss = criterion(final_pred, targets)
            running_loss += loss.item()
            n_batches += 1
    
    val_loss = running_loss / max(1, n_batches)
    
    return train_loss, val_loss


def train_model(model, train_loader, val_loader, args):
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
        if args.autoregressive:
            # Teacher forcing ratio: linearly decay from 1.0 to 0.0
            teacher_forcing_ratio = 1.0 - (epoch / max(1, args.epochs - 1))
            train_loss, val_loss = train_autoregressive_epoch(
                model, train_loader, val_loader, optimizer, criterion, teacher_forcing_ratio
            )
        else:
            teacher_forcing_ratio = None
            train_loss, val_loss = train_epoch(model, train_loader, val_loader, optimizer, criterion)
        
        # Record history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
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

        postfix = {"Train": f"{train_loss:.4f}", "Val": f"{val_loss:.4f}", "LR": f"{current_lr:.2e}"}
        if teacher_forcing_ratio is not None:
            postfix["TF"] = f"{teacher_forcing_ratio:.2f}"
        bar.set_postfix(postfix)
    
    print(f"\nTraining completed! Final best loss: {best_loss:.6f}")
    
    # Plot training history
    plot_training_history(train_losses, val_losses)
    
    model.load_state_dict(torch.load(args.save_path, weights_only=False))
    return model

def test_model(model, test_loader, args):
    model.eval()
    running_loss = 0.0
    n_batches = 0
    criterion = nn.MSELoss()
    gts = []
    preds = []
    
    with torch.no_grad():
        for full_input, targets in test_loader:
            full_input = full_input.to(device)
            targets = targets.to(device)
            
            if args.autoregressive:
                # True autoregressive inference: predict one timestep at a time
                # using the previous prediction as input for the next step
                B, n_vals, nt, nx = targets.shape
                current_input = full_input.clone()
                
                for t in range(1, nt):
                    # Run model with current known inputs (t=0 to t-1)
                    pred = model(current_input)
                    # Use prediction for timestep t as input for next iteration
                    current_input[:, :, t, 1:-1] = pred[:, :, t, 1:-1]
                
                # Final prediction after filling all timesteps autoregressively
                pred = model(current_input)
            else:
                # Standard one-shot prediction
                pred = model(full_input)
            
            loss = criterion(pred, targets)
            running_loss += loss.item()
            n_batches += 1
            
            for i in range(targets.shape[0]):
                gt = targets[i].squeeze(0).detach().cpu().numpy()
                p = pred[i].squeeze(0).detach().cpu().numpy()
                gts.append(gt)
                preds.append(p)
    
    test_loss = running_loss / max(1, n_batches)
    mode = "Autoregressive" if args.autoregressive else "One-shot"
    print(f"Test Loss ({mode}): {test_loss:.6f}")
    
    gts = np.array(gts)[:args.num_plots]
    preds = np.array(preds)[:args.num_plots]
    plot_comparison(gts, preds, args.nx, args.nt, args.dx, args.dt, save_as=f"results/test_comparison.png")

def main():
    args = parse_args()
    solver = get_solver(args)
    train_samples = int(args.n_samples * 0.8)
    val_samples = int(args.n_samples * 0.15)
    test_samples = args.n_samples - train_samples - val_samples

    train_dataset = GridDataset(solver, train_samples, args.nx, args.nt, args.dx, args.dt)
    val_dataset = GridDataset(solver, val_samples, args.nx, args.nt, args.dx, args.dt)
    test_dataset = GridDataset(solver, test_samples, args.nx, args.nt, args.dx, args.dt)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model = create_model(args).to(device)
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path, weights_only=False))
    summary(model)
    
    model = train_model(model, train_loader, val_loader, args)
    test_model(model, test_loader, args)



if __name__ == "__main__":
    main()
