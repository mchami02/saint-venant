"""Generate and plot data from the GridDataset."""

import argparse
import tempfile
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (thread-safe, no GUI)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from operator_data_pipeline import GridDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and plot GridDataset samples")
    parser.add_argument("--n_plots", type=int, default=4, help="Number of samples to plot")
    parser.add_argument("--nx", type=int, default=50, help="Number of spatial grid points")
    parser.add_argument("--nt", type=int, default=250, help="Number of time steps")
    parser.add_argument("--dx", type=float, default=0.25, help="Spatial step size")
    parser.add_argument("--dt", type=float, default=0.05, help="Time step size")
    parser.add_argument("--output", type=str, default="data_samples.png", help="Output file name")
    return parser.parse_args()

def _create_comparison_animation(gt, pred, nx, nt, dx, dt, sample_idx, skip_frames=5, fps=20):
    """
    Create an animation showing ground truth vs prediction side by side through time.
    
    Args:
        gt: Ground truth array (nt, nx) - each row is spatial values at a time step
        pred: Prediction array (nt, nx) - each row is spatial values at a time step
        nx, nt: Grid dimensions
        dx, dt: Grid spacing
        sample_idx: Sample index for title
        skip_frames: Number of frames to skip
        fps: Frames per second
    
    Returns:
        FuncAnimation object and the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.linspace(0, nx * dx, gt.shape[1])  # Use actual data shape
    
    # Set consistent y-axis limits
    y_min = min(gt.min(), pred.min())
    y_max = max(gt.max(), pred.max())
    y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
    
    # Initialize lines with initial data
    line_gt, = axes[0].plot(x, gt[0], 'b-', linewidth=2)
    line_pred, = axes[1].plot(x, pred[0], 'r-', linewidth=2)
    
    for ax, title in zip(axes, [f'Ground Truth (Sample {sample_idx+1})', f'Prediction (Sample {sample_idx+1})']):
        ax.set_xlim(0, nx * dx)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_xlabel('Position x')
        ax.set_ylabel('Density ρ')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    time_text = axes[0].text(0.02, 0.95, '', transform=axes[0].transAxes, fontsize=10)
    plt.tight_layout()
    
    def update(frame):
        t = frame * skip_frames
        if t >= nt:
            t = nt - 1
        line_gt.set_ydata(gt[t])
        line_pred.set_ydata(pred[t])
        time_text.set_text(f'Time: {t * dt:.3f} s (step {t}/{nt})')
        return line_gt, line_pred, time_text
    
    n_frames = max(1, nt // skip_frames)
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps)
    
    return anim, fig


def plot_comparison_comet(ground_truth, prediction, nx, nt, dx, dt, experiment, epoch, test=False):
    """
    Create comparison plots (ground truth, prediction, difference) and upload to Comet.
    Also creates animated GIFs showing the evolution through time.
    
    Args:
        ground_truth: (B, nt, nx) or (nt, nx) array
        prediction: (B, nt, nx) or (nt, nx) array
        nx, nt: Grid dimensions
        dx, dt: Grid spacing
        experiment: Comet experiment object
        name: Name prefix for the logged figure
    """
    ground_truth = np.asarray(ground_truth)
    prediction = np.asarray(prediction)
    
    # Handle 2D input by adding batch dimension
    if ground_truth.ndim == 2:
        ground_truth = ground_truth[np.newaxis, ...]
        prediction = prediction[np.newaxis, ...]
    
    B = ground_truth.shape[0]
    difference = np.abs(prediction - ground_truth)
    
    # Create static comparison plots
    fig, axes = plt.subplots(B, 3, figsize=(18, 5 * B))
    if B == 1:
        axes = axes.reshape(1, -1)
    
    extent = [0, nx * dx, 0, nt * dt]
    
    for b in range(B):
        # Ground Truth
        im1 = axes[b, 0].imshow(ground_truth[b], extent=extent, aspect='auto', 
                                 origin='lower', cmap='jet', vmin=0, vmax=1)
        axes[b, 0].set_xlabel('Space x')
        axes[b, 0].set_ylabel('Time t')
        axes[b, 0].set_title(f'Ground Truth (Sample {b+1})')
        plt.colorbar(im1, ax=axes[b, 0], label='Value')
        
        # Prediction
        im2 = axes[b, 1].imshow(prediction[b], extent=extent, aspect='auto',
                                 origin='lower', cmap='jet', vmin=0, vmax=1)
        axes[b, 1].set_xlabel('Space x')
        axes[b, 1].set_ylabel('Time t')
        axes[b, 1].set_title(f'Prediction (Sample {b+1})')
        plt.colorbar(im2, ax=axes[b, 1], label='Value')
        
        # Difference
        im3 = axes[b, 2].imshow(difference[b], extent=extent, aspect='auto',
                                 origin='lower', cmap='RdBu_r', vmin=0, vmax=1)
        axes[b, 2].set_xlabel('Space x')
        axes[b, 2].set_ylabel('Time t')
        axes[b, 2].set_title(f'Difference (Sample {b+1})')
        plt.colorbar(im3, ax=axes[b, 2], label='Error')
    
    plt.tight_layout()
    if not test:
        experiment.log_figure(figure_name="comparison_plot", figure=fig, step=epoch)
    else:
        experiment.log_figure(figure_name="test_comparison_plot", figure=fig, step=epoch)

    plt.close(fig)
    
    # Create and upload animated GIFs for each sample
    for b in range(B):
        # Data is in (nt, nx) format - each row is spatial values at a time step
        anim, anim_fig = _create_comparison_animation(
            ground_truth[b], prediction[b], nx, nt, dx, dt, sample_idx=b
        )
        
        # Save to temporary file and upload to Comet
        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp:
            anim.save(tmp.name, writer='pillow', fps=20)
            if not test:
                experiment.log_video(tmp.name, name=f"animation_sample_{b+1}", step=epoch)
            else:
                experiment.log_video(tmp.name, name=f"test_animation_sample_{b+1}", step=epoch)
        
        plt.close(anim_fig)


def plot_delta_u_comet(ground_truth, delta_u, nx, nt, dx, dt, experiment, epoch):
    """
    Create comparison plots (ground truth, delta_u) and upload to Comet.
    
    Args:
        ground_truth: (B, nt, nx) or (nt, nx) array
        delta_u: (B, nt, nx) or (nt, nx) array - the delta/correction term
        nx, nt: Grid dimensions
        dx, dt: Grid spacing
        experiment: Comet experiment object
        epoch: Current epoch for logging
    """
    ground_truth = np.asarray(ground_truth)
    delta_u = np.asarray(delta_u)
    
    # Handle 2D input by adding batch dimension
    if ground_truth.ndim == 2:
        ground_truth = ground_truth[np.newaxis, ...]
        delta_u = delta_u[np.newaxis, ...]
    
    B = ground_truth.shape[0]
    
    # Create static comparison plots
    fig, axes = plt.subplots(B, 2, figsize=(12, 5 * B))
    if B == 1:
        axes = axes.reshape(1, -1)
    
    extent = [0, nx * dx, 0, nt * dt]
    
    for b in range(B):
        # Ground Truth
        im1 = axes[b, 0].imshow(ground_truth[b], extent=extent, aspect='auto', 
                                 origin='lower', cmap='jet', vmin=0, vmax=1)
        axes[b, 0].set_xlabel('Space x')
        axes[b, 0].set_ylabel('Time t')
        axes[b, 0].set_title(f'Ground Truth (Sample {b+1})')
        plt.colorbar(im1, ax=axes[b, 0], label='Value')
        
        # Delta U
        delta_min = delta_u[b].min()
        delta_max = delta_u[b].max()
        # Use symmetric color scale centered at 0 for delta
        delta_abs_max = max(abs(delta_min), abs(delta_max))
        im2 = axes[b, 1].imshow(delta_u[b], extent=extent, aspect='auto',
                                 origin='lower', cmap='RdBu_r', 
                                 vmin=-delta_abs_max, vmax=delta_abs_max)
        axes[b, 1].set_xlabel('Space x')
        axes[b, 1].set_ylabel('Time t')
        axes[b, 1].set_title(f'Delta U (Sample {b+1})')
        plt.colorbar(im2, ax=axes[b, 1], label='Δu')
    
    plt.tight_layout()
    experiment.log_figure(figure_name="delta_u_comparison_plot", figure=fig, step=epoch)
    plt.close(fig)

def main():
    args = parse_args()
    
    # Generate dataset
    print(f"Generating {args.n_plots} samples with nx={args.nx}, nt={args.nt}, dx={args.dx}, dt={args.dt}")
    dataset = GridDataset(
        solver=None,  # Not used when using nfv backend
        n_samples=args.n_plots,
        nx=args.nx,
        nt=args.nt,
        dx=args.dx,
        dt=args.dt
    )
    
    # Create subplots
    n_cols = min(args.n_plots, 4)
    n_rows = (args.n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if args.n_plots == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Compute extent for proper axis labels
    x_extent = args.nx * args.dx
    t_extent = args.nt * args.dt
    
    for i in range(args.n_plots):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        
        # Get the target grid (full solution)
        _, target = dataset[i]
        # target shape is (n_vals, nt, nx) - take the first channel (density)
        density = target[0].numpy()  # (nt, nx)
        
        im = ax.imshow(
            density,
            aspect='auto',
            origin='lower',
            extent=[0, x_extent, 0, t_extent],
            cmap='viridis'
        )
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_title(f'Sample {i + 1}')
        plt.colorbar(im, ax=ax, label='Density')
    
    # Hide unused subplots
    for i in range(args.n_plots, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].axis('off')
    
    plt.suptitle('LWR Traffic Flow - Density Evolution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()

