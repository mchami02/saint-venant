import sys
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from learner.model import SVEModel
from learner.data_pipeline import train_val_test_split, SVEDataset, AutoRegressiveDataset
from learner.test_pipeline import test_model_on_samples, plot_prediction_comparison


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch_x, batch_y in tqdm(dataloader, desc="Training", leave=False):
        # batch_x and batch_y are tuples of (h, u)
        h_x, u_x = batch_x
        h_y, u_y = batch_y
        
        # Move to device
        h_x = h_x.to(device)
        u_x = u_x.to(device)
        h_y = h_y.to(device)
        u_y = u_y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        h_pred, u_pred = model((h_x, u_x))
        
        # Compute loss (MSE for both h and u)
        loss_h = criterion(h_pred, h_y)
        loss_u = criterion(u_pred, u_y)
        loss = loss_h + loss_u
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            # batch_x and batch_y are tuples of (h, u)
            h_x, u_x = batch_x
            h_y, u_y = batch_y
            
            # Move to device
            h_x = h_x.to(device)
            u_x = u_x.to(device)
            h_y = h_y.to(device)
            u_y = u_y.to(device)
            
            # Forward pass
            h_pred, u_pred = model((h_x, u_x))
            
            # Compute loss
            loss_h = criterion(h_pred, h_y)
            loss_u = criterion(u_pred, u_y)
            loss = loss_h + loss_u
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def test_autoregressive(model, dataloader, criterion, device, return_predictions=False):
    """Test the model using autoregressive prediction over full trajectories."""
    model.eval()
    total_loss = 0.0
    total_h_loss = 0.0
    total_u_loss = 0.0
    
    all_h_predictions = []
    all_u_predictions = []
    all_h_ground_truth = []
    all_u_ground_truth = []
    
    with torch.no_grad():
        for batch_initial, batch_trajectory in dataloader:
            # batch_initial: (h[0], u[0])
            # batch_trajectory: (h[all times], u[all times])
            h_initial, u_initial = batch_initial
            h_trajectory, u_trajectory = batch_trajectory
            
            # Move to device
            h_initial = h_initial.to(device)
            u_initial = u_initial.to(device)
            h_trajectory = h_trajectory.to(device)
            u_trajectory = u_trajectory.to(device)
            
            batch_size, time_steps, spatial_size = h_trajectory.shape
            
            # Storage for predictions
            h_predictions = torch.zeros_like(h_trajectory)
            u_predictions = torch.zeros_like(u_trajectory)
            
            # Initialize predictions with ground truth at t=0 to ensure exact match
            h_predictions[:, 0, :] = h_trajectory[:, 0, :].clone()
            u_predictions[:, 0, :] = u_trajectory[:, 0, :].clone()
            
            # Initialize current state from the ground truth trajectory
            h_current = h_trajectory[:, 0, :].clone()
            u_current = u_trajectory[:, 0, :].clone()
            
            # Autoregressive prediction
            for t in range(1, time_steps):
                h_next, u_next = model((h_current, u_current))
                h_predictions[:, t, :] = h_next
                u_predictions[:, t, :] = u_next
                h_current = h_next
                u_current = u_next
            
            # Compute loss over predicted time steps only (excluding t=0 which is just initial condition)
            loss_h = criterion(h_predictions[:, 1:, :], h_trajectory[:, 1:, :])
            loss_u = criterion(u_predictions[:, 1:, :], u_trajectory[:, 1:, :])
            loss = loss_h + loss_u
            
            total_loss += loss.item()
            total_h_loss += loss_h.item()
            total_u_loss += loss_u.item()
            
            # Store predictions if requested
            if return_predictions:
                all_h_predictions.append(h_predictions.cpu().numpy())
                all_u_predictions.append(u_predictions.cpu().numpy())
                all_h_ground_truth.append(h_trajectory.cpu().numpy())
                all_u_ground_truth.append(u_trajectory.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    avg_h_loss = total_h_loss / len(dataloader)
    avg_u_loss = total_u_loss / len(dataloader)
    
    if return_predictions:
        import numpy as np
        # Concatenate all batches
        all_h_predictions = np.concatenate(all_h_predictions, axis=0)
        all_u_predictions = np.concatenate(all_u_predictions, axis=0)
        all_h_ground_truth = np.concatenate(all_h_ground_truth, axis=0)
        all_u_ground_truth = np.concatenate(all_u_ground_truth, axis=0)
        return avg_loss, avg_h_loss, avg_u_loss, all_h_predictions, all_u_predictions, all_h_ground_truth, all_u_ground_truth
    
    return avg_loss, avg_h_loss, avg_u_loss

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train SVE model')
    parser.add_argument('--N_x', type=int, default=100, help='Number of spatial grid points')
    parser.add_argument('--d_x', type=float, default=0.01, help='Spatial step size')
    parser.add_argument('--d_t', type=float, default=0.001, help='Time step size')
    parser.add_argument('--T', type=float, default=1.0, help='Total simulation time')
    parser.add_argument('--n_samples', type=int, default=10, help='Number of samples to generate')
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden layer size')
    parser.add_argument('--latent_size', type=int, default=1024, help='Latent representation size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for regularization')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu, cuda, mps, or auto)')
    parser.add_argument('--save_path', type=str, default='sve_model.pt', help='Path to save the trained model')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Generate data
    print(f"\nGenerating data with N_x={args.N_x}, d_x={args.d_x}, d_t={args.d_t}, T={args.T}")
    print(f"Total samples: {args.n_samples}")
    
    train_h, train_u, val_h, val_u, test_h, test_u = train_val_test_split(
        args.N_x, args.d_x, args.d_t, args.T, args.n_samples
    )
    
    print(f"Train samples: {len(train_h)}, Val samples: {len(val_h)}, Test samples: {len(test_h)}")
    
    # Create datasets
    # Training and validation use pair dataset (one-step prediction)
    train_dataset = SVEDataset(train_h, train_u)
    val_dataset = SVEDataset(val_h, val_u)
    # Testing uses autoregressive dataset (full trajectory prediction)
    test_dataset = AutoRegressiveDataset(test_h, test_u)
    
    print(f"Train pairs: {len(train_dataset)}, Val pairs: {len(val_dataset)}, Test trajectories: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    # Note: N_x + 2 because the solver adds boundary cells
    model = SVEModel(N_x=args.N_x + 2, hidden_size=args.hidden_size, latent_size=args.latent_size, dropout_rate=args.dropout)
    model = model.to(device)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs (patience={args.patience})...")
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model and check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': vars(args)
            }, args.save_path)
            print(f"  -> Saved best model with val loss: {val_loss:.6f}")
        else:
            epochs_without_improvement += 1
            print(f"  -> No improvement for {epochs_without_improvement} epoch(s)")
            
            # Early stopping
            if epochs_without_improvement >= args.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs (patience={args.patience})")
                break
    
    # Load best model for testing
    print(f"\nLoading best model from {args.save_path}...")
    checkpoint = torch.load(args.save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test the model (autoregressive prediction using dataset)
    print("\nTesting model (autoregressive prediction)...")
    test_loss, test_h_loss, test_u_loss, all_h_pred, all_u_pred, all_h_gt, all_u_gt = test_autoregressive(
        model, test_loader, criterion, device, return_predictions=True
    )
    
    print(f"\nAutoregressive Test Results:")
    print(f"  Total Loss: {test_loss:.6f}")
    print(f"  Height (h) Loss: {test_h_loss:.6f}")
    print(f"  Velocity (u) Loss: {test_u_loss:.6f}")
    
    # Plot one sample comparison
    if len(all_h_gt) > 0:
        print("\nPlotting prediction comparison for first test sample...")
        sample_idx = 0
        
        # Verify that time step 0 matches exactly
        h_error_t0 = np.abs(all_h_gt[sample_idx][0] - all_h_pred[sample_idx][0]).max()
        u_error_t0 = np.abs(all_u_gt[sample_idx][0] - all_u_pred[sample_idx][0]).max()
        print(f"Verification - Max error at t=0:")
        print(f"  Height (h): {h_error_t0:.2e}")
        print(f"  Velocity (u): {u_error_t0:.2e}")
        
        plot_prediction_comparison(
            all_h_gt[sample_idx], 
            all_u_gt[sample_idx], 
            all_h_pred[sample_idx], 
            all_u_pred[sample_idx],
            args.N_x, 
            args.d_x, 
            args.d_t, 
            args.T,
            output_prefix='autoregressive_test_sample'
        )
    
    print(f"\nTraining complete! Model saved to {args.save_path}")


if __name__ == "__main__":
    main()

