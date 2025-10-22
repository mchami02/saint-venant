import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import godunov_solver
sys.path.append(str(Path(__file__).parent.parent))
from godunov_solver.plotter import Plotter


def test_model(model, initial_h, initial_u, ground_truth_h, ground_truth_u, time_steps, device):
    """
    Test the model by predicting autoregressively for a given number of time steps.
    
    Args:
        model: The trained SVE model
        initial_h: Initial height condition (numpy array or tensor of shape [N_x+2])
        initial_u: Initial velocity condition (numpy array or tensor of shape [N_x+2])
        ground_truth_h: Ground truth height trajectory (numpy array of shape [time_steps, N_x+2])
        ground_truth_u: Ground truth velocity trajectory (numpy array of shape [time_steps, N_x+2])
        time_steps: Number of time steps to predict
        device: Device to run the model on
    
    Returns:
        mse_h: Mean squared error for height predictions
        mse_u: Mean squared error for velocity predictions
        predictions_h: Predicted height trajectory (numpy array)
        predictions_u: Predicted velocity trajectory (numpy array)
    """
    model.eval()
    
    # Convert to tensors if needed
    if isinstance(initial_h, np.ndarray):
        current_h = torch.from_numpy(initial_h).float().to(device)
    else:
        current_h = initial_h.float().to(device)
    
    if isinstance(initial_u, np.ndarray):
        current_u = torch.from_numpy(initial_u).float().to(device)
    else:
        current_u = initial_u.float().to(device)
    
    # Storage for predictions
    predictions_h = []
    predictions_u = []
    
    with torch.no_grad():
        # Store initial condition
        predictions_h.append(current_h.cpu().numpy())
        predictions_u.append(current_u.cpu().numpy())
        
        # Autoregressive prediction
        for t in range(time_steps - 1):
            # Add batch dimension if needed
            if current_h.dim() == 1:
                h_input = current_h.unsqueeze(0)
                u_input = current_u.unsqueeze(0)
            else:
                h_input = current_h
                u_input = current_u
            
            # Predict next step
            h_next, u_next = model((h_input, u_input))
            
            # Remove batch dimension if it was added
            if current_h.dim() == 1:
                h_next = h_next.squeeze(0)
                u_next = u_next.squeeze(0)
            
            # Update current state
            current_h = h_next
            current_u = u_next
            
            # Store predictions
            predictions_h.append(current_h.cpu().numpy())
            predictions_u.append(current_u.cpu().numpy())
    
    # Convert predictions to numpy arrays
    predictions_h = np.array(predictions_h)
    predictions_u = np.array(predictions_u)
    
    # Compute MSE
    mse_h = np.mean((predictions_h - ground_truth_h[:time_steps]) ** 2)
    mse_u = np.mean((predictions_u - ground_truth_u[:time_steps]) ** 2)
    
    return mse_h, mse_u, predictions_h, predictions_u


def test_model_on_samples(model, h_samples, u_samples, device):
    """
    Test the model on multiple samples by predicting autoregressively.
    
    Args:
        model: The trained SVE model
        h_samples: Height samples (numpy array of shape [n_samples, time_steps, N_x+2])
        u_samples: Velocity samples (numpy array of shape [n_samples, time_steps, N_x+2])
        device: Device to run the model on
    
    Returns:
        avg_mse_h: Average MSE for height predictions across all samples
        avg_mse_u: Average MSE for velocity predictions across all samples
        all_predictions_h: List of predicted trajectories for height
        all_predictions_u: List of predicted trajectories for velocity
    """
    n_samples = len(h_samples)
    time_steps = h_samples.shape[1]
    
    mse_h_list = []
    mse_u_list = []
    all_predictions_h = []
    all_predictions_u = []
    
    for i in range(n_samples):
        # Get initial condition and ground truth
        initial_h = h_samples[i, 0]
        initial_u = u_samples[i, 0]
        ground_truth_h = h_samples[i]
        ground_truth_u = u_samples[i]
        
        # Test model
        mse_h, mse_u, pred_h, pred_u = test_model(
            model, initial_h, initial_u, 
            ground_truth_h, ground_truth_u, 
            time_steps, device
        )
        
        mse_h_list.append(mse_h)
        mse_u_list.append(mse_u)
        all_predictions_h.append(pred_h)
        all_predictions_u.append(pred_u)
    
    avg_mse_h = np.mean(mse_h_list)
    avg_mse_u = np.mean(mse_u_list)
    
    return avg_mse_h, avg_mse_u, all_predictions_h, all_predictions_u


def plot_prediction_comparison(ground_truth_h, ground_truth_u, predicted_h, predicted_u, 
                                N_x, d_x, d_t, T, output_prefix=''):
    """
    Plot comparison between ground truth and predicted trajectories.
    
    Args:
        ground_truth_h: Ground truth height trajectory
        ground_truth_u: Ground truth velocity trajectory
        predicted_h: Predicted height trajectory
        predicted_u: Predicted velocity trajectory
        N_x: Number of spatial grid points
        d_x: Spatial step size
        d_t: Time step size
        T: Total simulation time
        output_prefix: Prefix for output files. If empty, only show plots.
    """
    plotter = Plotter(N_x, d_x, d_t, T)
    
    # Plot height comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Calculate shared color limits for height
    h_min = min(ground_truth_h.min(), predicted_h.min())
    h_max = max(ground_truth_h.max(), predicted_h.max())
    
    # Ground truth height
    im1 = axes[0].imshow(ground_truth_h, cmap='viridis', aspect='auto', origin='lower', 
                         vmin=h_min, vmax=h_max)
    axes[0].set_title('Ground Truth - Height (h)')
    axes[0].set_xlabel('Cell index')
    axes[0].set_ylabel('Time step')
    plt.colorbar(im1, ax=axes[0], label='h')
    
    # Predicted height
    im2 = axes[1].imshow(predicted_h, cmap='viridis', aspect='auto', origin='lower',
                         vmin=h_min, vmax=h_max)
    axes[1].set_title('Predicted - Height (h)')
    axes[1].set_xlabel('Cell index')
    axes[1].set_ylabel('Time step')
    plt.colorbar(im2, ax=axes[1], label='h')
    
    # Difference (error)
    diff_h = np.abs(ground_truth_h - predicted_h)
    im3 = axes[2].imshow(diff_h, cmap='Reds', aspect='auto', origin='lower')
    axes[2].set_title('Absolute Error - Height (h)')
    axes[2].set_xlabel('Cell index')
    axes[2].set_ylabel('Time step')
    plt.colorbar(im3, ax=axes[2], label='|error|')
    
    plt.tight_layout()
    if output_prefix:
        plt.savefig(f'{output_prefix}_height_comparison.png', dpi=150, bbox_inches='tight')
        print(f"Saved height comparison to {output_prefix}_height_comparison.png")
        plt.close()
    else:
        plt.show()
    
    # Plot velocity comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Calculate shared color limits for velocity
    u_min = min(ground_truth_u.min(), predicted_u.min())
    u_max = max(ground_truth_u.max(), predicted_u.max())
    
    # Ground truth velocity
    im1 = axes[0].imshow(ground_truth_u, cmap='coolwarm', aspect='auto', origin='lower',
                         vmin=u_min, vmax=u_max)
    axes[0].set_title('Ground Truth - Velocity (u)')
    axes[0].set_xlabel('Cell index')
    axes[0].set_ylabel('Time step')
    plt.colorbar(im1, ax=axes[0], label='u')
    
    # Predicted velocity
    im2 = axes[1].imshow(predicted_u, cmap='coolwarm', aspect='auto', origin='lower',
                         vmin=u_min, vmax=u_max)
    axes[1].set_title('Predicted - Velocity (u)')
    axes[1].set_xlabel('Cell index')
    axes[1].set_ylabel('Time step')
    plt.colorbar(im2, ax=axes[1], label='u')
    
    # Difference (error)
    diff_u = np.abs(ground_truth_u - predicted_u)
    im3 = axes[2].imshow(diff_u, cmap='Reds', aspect='auto', origin='lower')
    axes[2].set_title('Absolute Error - Velocity (u)')
    axes[2].set_xlabel('Cell index')
    axes[2].set_ylabel('Time step')
    plt.colorbar(im3, ax=axes[2], label='|error|')
    
    plt.tight_layout()
    if output_prefix:
        plt.savefig(f'{output_prefix}_velocity_comparison.png', dpi=150, bbox_inches='tight')
        print(f"Saved velocity comparison to {output_prefix}_velocity_comparison.png")
        plt.close()
    else:
        plt.show()
    
    # Create animations if output_prefix is provided
    if output_prefix:
        # Animate ground truth height
        plotter.animate_density(ground_truth_h, f'{output_prefix}_height_groundtruth.gif')
        
        # Animate predicted height
        plotter.animate_density(predicted_h, f'{output_prefix}_height_predicted.gif')

