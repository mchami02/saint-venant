import argparse
from godunov_solver.solve_class import *
from godunov_solver.flux import *
from operator_data_pipeline import GridDataset
from torch.utils.data import DataLoader
from neuralop.models import FNO
import torch
from tqdm import tqdm
import torch.nn as nn
from godunov_solver.plotter import plot_comparison

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

def get_solver(args):
    if args.solver == "Godunov":
        if args.flux == "Greenshields":
            return Godunov(flux=Greenshields(vmax=1.0, rho_max=1.0))
        elif args.flux == "Triangular":
            return Godunov(flux=Triangular(vmax=1.0, rho_max=1.0))
        else:
            raise ValueError(f"Flux {args.flux} not supported")
    elif args.solver == "SVESolver":
        return SVESolver()
    else:
        raise ValueError(f"Solver {args.solver} not supported")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--nt", type=int, default=100)
    parser.add_argument("--dx", type=float, default=0.25)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--bc", type=str, default="GhostCell")
    parser.add_argument("--solver", type=str, default="Godunov")
    parser.add_argument("--flux", type=str, default="Greenshields")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--model", type=str, default="FNO")
    parser.add_argument("--n_modes", type=int, default=64)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--out_channels", type=int, default=1)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--step_size", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--save_path", type=str, default="operator.pth")
    return parser.parse_args()

def create_model(args):
    if args.model == "FNO":
        model = FNO(
        n_modes=(args.n_modes, 16),        # modes in (time, space) dimensions
        hidden_channels=args.hidden_channels,       # network width
        in_channels=args.in_channels,           # density + time + space
        out_channels=args.out_channels,          # predicted density
        n_layers=args.n_layers               # number of FNO layers
        )
    else:
        raise ValueError(f"Model {args.model} not supported")
    return model

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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    epochs_without_improvement = 0
    current_lr = optimizer.param_groups[0]['lr']
    bar = tqdm(range(args.epochs), desc="Training")
    for epoch in bar:
        train_loss, val_loss = train_epoch(model, train_loader, val_loader, optimizer, criterion)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        bar.set_postfix({"Train Loss": train_loss, "Val Loss": val_loss, "LR": current_lr})

        # Save best model and check for improvement
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), args.save_path)
    
    print(f"\nTraining completed! Final best loss: {best_loss:.6f}")
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
        plot_comparison(gt, pred, args.nx, args.nt, args.dx, args.dt, save_as=f"results/test_comparison_{i}.png")
    print(f"Saved {i} comparisons")


def main():
    args = parse_args()
    solver = get_solver(args)
    train_samples = int(args.n_samples * 0.7)
    val_samples = int(args.n_samples * 0.2)
    test_samples = args.n_samples - train_samples - val_samples
    train_dataset = GridDataset(solver, train_samples, args.nx, args.nt, args.dx, args.dt)
    val_dataset = GridDataset(solver, val_samples, args.nx, args.nt, args.dx, args.dt)
    test_dataset = GridDataset(solver, test_samples, args.nx, args.nt, args.dx, args.dt)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model = create_model(args).to(device)

    model = train_model(model, train_loader, val_loader, args)
    test_model(model, test_loader, args)



if __name__ == "__main__":
    main()
