import argparse
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
    
    for full_input, targets in train_loader:
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
    for full_input, targets in val_loader:
        full_input = full_input.to(device)
        targets = targets.to(device)

        pred = model(full_input)  # (B, n_vals, nt, nx)
        
        loss = criterion(pred, targets)
        running_loss += loss.item()
        n_batches += 1

    val_loss = running_loss / max(1, n_batches)
    
    return train_loss, val_loss

def train_model(model, train_loader, val_loader, args):
    """
    Train one-shot FNO for multiple epochs with learning rate scheduling and early stopping.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    epochs_without_improvement = 0
    current_lr = optimizer.param_groups[0]['lr']
    
    # Track training history
    train_losses = []
    val_losses = []
    
    bar = tqdm(range(args.epochs), desc="Training")
    for epoch in bar:
        train_loss, val_loss = train_epoch(model, train_loader, val_loader, optimizer, criterion)
        
        # Record history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Check for improvement
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), args.save_path)
        else:
            epochs_without_improvement += 1
            
        # Reduce learning rate if no improvement for 10 epochs
        if epochs_without_improvement >= args.patience // 2:
            epochs_without_improvement = 0
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_decay

        if epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs (patience={args.patience})")
            break

        bar.set_postfix({"Train Loss": train_loss, "Val Loss": val_loss, "LR": current_lr})
    
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
    for full_input, targets in test_loader:
        full_input = full_input.to(device)
        targets = targets.to(device)
        model_input = full_input  # (B, 3, nt, nx)
        pred = model(model_input)  # (B, 1, nt, nx)
        loss = criterion(pred, targets)
        running_loss += loss.item()
        n_batches += 1
        for i in range(targets.shape[0]):
            gt = targets[i].squeeze(0).detach().cpu().numpy()
            p = pred[i].squeeze(0).detach().cpu().numpy()
            gts.append(gt)
            preds.append(p)
    test_loss = running_loss / max(1, n_batches)
    print(f"Test Loss: {test_loss:.6f}")
    i = 0
    for gt, pred in zip(gts, preds):
        if i > 10:
            break
        i += 1
        os.makedirs("results", exist_ok=True)
        plot_comparison(gt, pred, args.nx, args.nt, args.dx, args.dt, save_as=f"results/test_comparison_{i}.png")
    print(f"Saved {i} comparisons")

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
    summary(model)
    model = train_model(model, train_loader, val_loader, args)
    test_model(model, test_loader, args)



if __name__ == "__main__":
    main()
