import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

try:
    from IPython.display import HTML
    get_ipython()  # pylint: disable=undefined-variable
    IN_NOTEBOOK = True
except (ImportError, NameError):    
    IN_NOTEBOOK = False

def plot_grid_density(grid, nx, nt, dx, dt, save_as=''):
    plt.figure(figsize=(10, 6))

    plt.imshow(
        grid, 
        extent=[0, nx * dx, 0, nt * dt],
        aspect='auto',
        origin='lower',
        cmap='jet',  # 'jet' goes from blue (low) through green/yellow/orange to red (high),
        vmin=0,
        vmax=1
    )

    plt.colorbar(label='Density ρ(x,t)')
    plt.xlabel('Space x')
    plt.ylabel('Time t')
    plt.title('Space–Time Density Heatmap')
    
    if save_as:
        # Add .png extension if not already present
        if not save_as.endswith('.png'):
            save_as = save_as + '.png'
        plt.savefig(save_as)
        print(f"Saved plot to {save_as}")
        plt.close()
    else:
        plt.show()

def animate_density(arr, nx, nt, dx, dt, output_name='', skip_frames=10, fps=30):
    ''' Create an animation showing h(x) evolving through time.
        Args:
            arr: Array to animate
            output_name: If empty string, only show. Otherwise, save to this filename (e.g., 'animation.gif' or 'animation.mp4').
    '''
    fig, ax = plt.subplots(figsize=(10, 6))
    # Set up the spatial grid
    
    x = np.linspace(0, nx * dx, nx)
    # Find global min/max for consistent y-axis
    h_min, h_max = arr.min(), arr.max()
    y_margin = (h_max - h_min) * 0.1
    # Initialize the line
    line, = ax.plot([], [], 'b-', linewidth=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    ax.set_xlim(0, nx * dx)
    ax.set_ylim(h_min - y_margin, h_max + y_margin)
    ax.set_xlabel('Position x')
    ax.set_ylabel('Water height h')
    ax.set_title('Water Height Evolution')
    ax.grid(True, alpha=0.3)
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text
    def update(frame):
        i = frame * skip_frames
        if i >= nt:
            i = nt - 1
        line.set_data(x, arr[i])
        time_text.set_text(f'Time: {i * dt:.3f} s (step {i}/{nt})')
        return line, time_text
    n_frames = min(nt // skip_frames, nt)
    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init, 
                         blit=True, interval=1000/fps)
    if output_name:
        # Save animation
        if output_name.endswith('.gif'):
            anim.save(output_name, writer='pillow', fps=fps)
        elif output_name.endswith('.mp4'):
            anim.save(output_name, writer='ffmpeg', fps=fps)
        else:
            # Default to gif
            anim.save(output_name + '.gif', writer='pillow', fps=fps)
        print(f"Saved animation to {output_name}")
        plt.close()
        return anim
    elif IN_NOTEBOOK:
        plt.close()
        return HTML(anim.to_jshtml())
    else:
        plt.show()
        return anim

def plot_comparison(ground_truth, prediction, nx, nt, dx, dt, save_as=''):
    '''
    Create a three-panel comparison plot showing ground truth, prediction, and their difference.
    
    Args:
        ground_truth: Ground truth grid (2D array (nx, nt) or 3D array (B, nx, nt))
        prediction: Prediction grid (2D array (nx, nt) or 3D array (B, nx, nt))
        nx: Number of spatial points
        nt: Number of time steps
        dx: Spatial step size
        dt: Time step size
        save_as: If not empty, save the plot to this filename (automatically adds .png extension)
    '''
    # Check input shape
    ground_truth = np.asarray(ground_truth)
    prediction = np.asarray(prediction)
    
    if ground_truth.ndim == 2 and prediction.ndim == 2:
        # Original behavior for (nx, nt) inputs
        _plot_single_comparison(ground_truth, prediction, nx, nt, dx, dt, save_as)
    elif ground_truth.ndim == 3 and prediction.ndim == 3:
        # New behavior for (B, nx, nt) inputs - plot grid
        B = ground_truth.shape[0]
        _plot_batch_comparison(ground_truth, prediction, B, nx, nt, dx, dt, save_as)
    else:
        raise ValueError(f"Unexpected input shapes: ground_truth {ground_truth.shape}, prediction {prediction.shape}. "
                        "Expected (nx, nt) or (B, nx, nt)")

def _plot_single_comparison(ground_truth, prediction, nx, nt, dx, dt, save_as=''):
    '''Helper function to plot a single comparison (original behavior)'''
    # Calculate the difference
    difference = np.abs(prediction - ground_truth)
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Common imshow parameters
    extent = [0, nx * dx, 0, nt * dt]
    imshow_kwargs = {
        'extent': extent,
        'aspect': 'auto',
        'origin': 'lower',
        'cmap': 'jet',
        'vmin': 0,  # Fixed minimum value
        'vmax': 1   # Fixed maximum value
    }
    
    # Plot 1: Ground Truth
    im1 = axes[0].imshow(ground_truth, **imshow_kwargs)
    axes[0].set_xlabel('Space x')
    axes[0].set_ylabel('Time t')
    axes[0].set_title('Ground Truth')
    plt.colorbar(im1, ax=axes[0], label='Value')
    
    # Plot 2: Prediction
    im2 = axes[1].imshow(prediction, **imshow_kwargs)
    axes[1].set_xlabel('Space x')
    axes[1].set_ylabel('Time t')
    axes[1].set_title('Prediction')
    plt.colorbar(im2, ax=axes[1], label='Value')
    
    # Plot 3: Difference (use diverging colormap for difference)
    imshow_kwargs_diff = imshow_kwargs.copy()
    imshow_kwargs_diff['cmap'] = 'RdBu_r'  # Red-Blue diverging colormap
    
    # Center colormap on zero for difference plot
    im3 = axes[2].imshow(difference, **imshow_kwargs_diff)
    axes[2].set_xlabel('Space x')
    axes[2].set_ylabel('Time t')
    axes[2].set_title('Difference (Prediction - Ground Truth)')
    plt.colorbar(im3, ax=axes[2], label='Error')
    
    plt.tight_layout()
    
    if save_as:
        # Add .png extension if not already present
        if not save_as.endswith('.png'):
            save_as = save_as + '.png'
        plt.savefig(save_as, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_as}")
        plt.close()
    else:
        plt.show()

def _plot_batch_comparison(ground_truth, prediction, B, nx, nt, dx, dt, save_as=''):
    '''Helper function to plot batch comparisons in a grid'''
    # Calculate the difference for all samples
    difference = np.abs(prediction - ground_truth)
    
    # Create figure with B rows and 3 columns
    fig, axes = plt.subplots(B, 3, figsize=(18, 5 * B))
    
    # Handle the case when B=1 (axes won't be 2D)
    if B == 1:
        axes = axes.reshape(1, -1)
    
    # Common imshow parameters
    extent = [0, nx * dx, 0, nt * dt]
    imshow_kwargs = {
        'extent': extent,
        'aspect': 'auto',
        'origin': 'lower',
        'cmap': 'jet',
        'vmin': 0,  # Fixed minimum value
        'vmax': 1   # Fixed maximum value
    }
    
    # Plot each sample in the batch
    for b in range(B):
        # Plot 1: Ground Truth
        im1 = axes[b, 0].imshow(ground_truth[b], **imshow_kwargs)
        axes[b, 0].set_xlabel('Space x')
        axes[b, 0].set_ylabel('Time t')
        axes[b, 0].set_title(f'Ground Truth (Sample {b+1})')
        plt.colorbar(im1, ax=axes[b, 0], label='Value')
        
        # Plot 2: Prediction
        im2 = axes[b, 1].imshow(prediction[b], **imshow_kwargs)
        axes[b, 1].set_xlabel('Space x')
        axes[b, 1].set_ylabel('Time t')
        axes[b, 1].set_title(f'Prediction (Sample {b+1})')
        plt.colorbar(im2, ax=axes[b, 1], label='Value')
        
        # Plot 3: Difference
        imshow_kwargs_diff = imshow_kwargs.copy()
        imshow_kwargs_diff['cmap'] = 'RdBu_r'  # Red-Blue diverging colormap
        
        im3 = axes[b, 2].imshow(difference[b], **imshow_kwargs_diff)
        axes[b, 2].set_xlabel('Space x')
        axes[b, 2].set_ylabel('Time t')
        axes[b, 2].set_title(f'Difference (Sample {b+1})')
        plt.colorbar(im3, ax=axes[b, 2], label='Error')
    
    plt.tight_layout()
    
    if save_as:
        # Add .png extension if not already present
        if not save_as.endswith('.png'):
            save_as = save_as + '.png'
        plt.savefig(save_as, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_as}")
        plt.close()
    else:
        plt.show()