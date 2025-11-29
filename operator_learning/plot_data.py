"""Generate and plot data from the GridDataset."""

import argparse
import matplotlib.pyplot as plt
import numpy as np
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

