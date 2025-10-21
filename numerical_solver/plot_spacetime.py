"""
Space-time diagram plotting for shallow water solver results.

Creates beautiful visualizations showing the evolution of water height over space and time.

Usage:
    python plot_spacetime.py results_test/solution_LF.h5
"""

import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def create_blue_white_colormap():
    """Create a custom colormap: white (low) to dark blue (high)."""
    # Reversed order: white -> light blue -> dark blue
    colors = ['#ffffff', '#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#3182bd', '#08519c']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('white_blue', colors, N=n_bins)
    return cmap


def plot_spacetime_diagram(h5_file: str, variable: str = 'h', save_fig: bool = False, output_file: str = None):
    """Plot space-time diagram from HDF5 results.
    
    Args:
        h5_file: Path to HDF5 results file
        variable: Which variable to plot ('h' for height, 'q' for discharge)
        save_fig: Whether to save figure to file (default: False, shows interactively)
        output_file: Optional output filename (if None, auto-generated from h5_file)
                    Always saved in plots/ directory
    """
    import os
    
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    print(f"Loading data from {h5_file}...")
    
    with h5py.File(h5_file, 'r') as f:
        # Load data
        x = f['mesh/x'][:]
        topography = f['mesh/topography'][:]
        time = f['solution/time'][:]
        
        if variable == 'h':
            data = f['solution/h'][:]
            var_label = 'Water Height h [m]'
            title = 'Water Height Evolution'
        elif variable == 'q':
            data = f['solution/q'][:]
            var_label = 'Discharge q [m²/s]'
            title = 'Discharge Evolution'
        else:
            raise ValueError(f"Unknown variable: {variable}. Use 'h' or 'q'.")
        
        # Load metadata
        dx = f.attrs.get('dx', 0.0)
        dt = f.attrs.get('time_step', 0.0)
        flux_scheme = f.attrs.get('flux_scheme', 'unknown')
        time_scheme = f.attrs.get('time_scheme', 'unknown')
        
        # Load exact solution if available
        has_exact = 'exact' in f
        if has_exact and variable == 'h':
            exact_h = f['exact/h'][:]
            L2_error = f.attrs.get('L2_error_h', None)
            L1_error = f.attrs.get('L1_error_h', None)
    
    print(f"Data loaded: {data.shape[0]} time steps, {data.shape[1]} spatial points")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 1], width_ratios=[20, 1], 
                          hspace=0.3, wspace=0.05)
    
    # Main space-time plot
    ax_main = fig.add_subplot(gs[0, 0])
    ax_cbar = fig.add_subplot(gs[0, 1])
    
    # Create custom colormap (blue to white)
    cmap = create_blue_white_colormap()
    
    # Plot space-time diagram
    X, T = np.meshgrid(x, time)
    im = ax_main.pcolormesh(X, T, data, cmap=cmap, shading='auto')
    
    ax_main.set_xlabel('Position x [m]', fontsize=12)
    ax_main.set_ylabel('Time t [s]', fontsize=12)
    ax_main.set_title(f'{title} - Space-Time Diagram', fontsize=14, fontweight='bold')
    ax_main.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Colorbar
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label(var_label, fontsize=11)
    
    # Plot: Initial and final profiles
    ax_profiles = fig.add_subplot(gs[1, 0])
    ax_profiles.plot(x, data[0, :], 'b-', linewidth=2, label=f't = {time[0]:.1f} s (initial)')
    ax_profiles.plot(x, data[-1, :], 'r-', linewidth=2, label=f't = {time[-1]:.1f} s (final)')
    
    # Add topography
    ax_profiles.fill_between(x, topography, -1, alpha=0.3, color='brown', label='Topography')
    
    # Add exact solution if available
    if has_exact and variable == 'h':
        ax_profiles.plot(x, exact_h, 'g--', linewidth=1.5, label='Exact solution', alpha=0.7)
    
    ax_profiles.set_xlabel('Position x [m]', fontsize=11)
    ax_profiles.set_ylabel(var_label, fontsize=11)
    ax_profiles.set_title('Initial and Final Profiles', fontsize=12, fontweight='bold')
    ax_profiles.legend(loc='best', fontsize=9)
    ax_profiles.grid(True, alpha=0.3)
    
    # Plot: Time evolution at specific locations
    ax_time_series = fig.add_subplot(gs[2, 0])
    
    # Select 4 representative locations
    n_locs = 4
    indices = np.linspace(0, len(x) - 1, n_locs, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, n_locs))
    
    for idx, color in zip(indices, colors):
        ax_time_series.plot(time, data[:, idx], linewidth=1.5, 
                           label=f'x = {x[idx]:.2f} m', color=color)
    
    ax_time_series.set_xlabel('Time t [s]', fontsize=11)
    ax_time_series.set_ylabel(var_label, fontsize=11)
    ax_time_series.set_title('Time Evolution at Selected Locations', fontsize=12, fontweight='bold')
    ax_time_series.legend(loc='best', fontsize=9, ncol=2)
    ax_time_series.grid(True, alpha=0.3)
    
    # Add metadata text
    metadata_text = f'Scheme: {flux_scheme} + {time_scheme}\n'
    metadata_text += f'Grid: {len(x)} cells (dx={dx:.4f}m), dt={dt:.6f}s\n'
    metadata_text += f'Domain: [{x[0]:.2f}, {x[-1]:.2f}] m, Time: [{time[0]:.2f}, {time[-1]:.2f}] s'
    
    if has_exact and L2_error is not None:
        metadata_text += f'\nL2 error = {L2_error:.6e}, L1 error = {L1_error:.6e}'
    
    fig.text(0.02, 0.02, metadata_text, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle(f'1D Saint-Venant Equations - {title}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save or show figure
    if save_fig:
        if output_file is None:
            # Auto-generate filename from h5_file
            basename = os.path.basename(h5_file).replace('.h5', f'_spacetime_{variable}.png')
            output_file = f"plots/{basename}"
        else:
            # Use provided filename but always in plots/ directory
            output_file = f"plots/{os.path.basename(output_file)}"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {output_file}")
    else:
        print("Displaying plot (close window to continue)...")
        plt.show()


def plot_both_variables(h5_file: str):
    """Plot space-time diagrams for both h and q.
    
    Args:
        h5_file: Path to HDF5 results file
    """
    plot_spacetime_diagram(h5_file, variable='h', save_fig=True)
    plot_spacetime_diagram(h5_file, variable='q', save_fig=True)


def compare_results(h5_files: list):
    """Compare multiple simulation results side by side.
    
    Args:
        h5_files: List of HDF5 file paths to compare
    """
    n_files = len(h5_files)
    fig, axes = plt.subplots(1, n_files, figsize=(8*n_files, 6))
    
    if n_files == 1:
        axes = [axes]
    
    cmap = create_blue_white_colormap()
    
    for ax, h5_file in zip(axes, h5_files):
        with h5py.File(h5_file, 'r') as f:
            x = f['mesh/x'][:]
            time = f['solution/time'][:]
            h = f['solution/h'][:]
            flux_scheme = f.attrs.get('flux_scheme', 'unknown')
            
            X, T = np.meshgrid(x, time)
            im = ax.pcolormesh(X, T, h, cmap=cmap, shading='auto')
            
            ax.set_xlabel('Position x [m]')
            ax.set_ylabel('Time t [s]')
            ax.set_title(f'{flux_scheme} scheme')
            plt.colorbar(im, ax=ax, label='Water Height h [m]')
    
    plt.suptitle('Comparison of Different Schemes', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = 'plots/comparison_spacetime.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved comparison to: {output_path}")
    plt.show()


def main():
    """Main function for command-line usage."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Display plot:        python plot_spacetime.py results_test/solution_LF.h5")
        print("  Save plot:           python plot_spacetime.py results_test/solution_LF.h5 --save")
        print("  Plot discharge:      python plot_spacetime.py results_test/solution_LF.h5 --variable q")
        print("  Both variables:      python plot_spacetime.py results_test/solution_LF.h5 --both")
        print("  Compare schemes:     python plot_spacetime.py file1.h5 file2.h5 --compare")
        print("\nOptions:")
        print("  --save               Save figure instead of showing")
        print("  --output FILE        Custom output filename")
        print("  --variable h|q       Variable to plot ('h' or 'q')")
        print("  --both               Plot both h and q (saves both)")
        print("  --compare            Compare multiple schemes")
        sys.exit(1)
    
    # Parse arguments
    save_fig = '--save' in sys.argv
    variable = 'h'
    output_file = None
    
    if '--variable' in sys.argv:
        idx = sys.argv.index('--variable')
        if idx + 1 < len(sys.argv):
            variable = sys.argv[idx + 1]
    
    if '--output' in sys.argv:
        idx = sys.argv.index('--output')
        if idx + 1 < len(sys.argv):
            output_file = sys.argv[idx + 1]
            save_fig = True  # Implicitly save if output specified
    
    if '--compare' in sys.argv:
        h5_files = [arg for arg in sys.argv[1:] if arg.endswith('.h5')]
        compare_results(h5_files)
    elif '--both' in sys.argv:
        plot_both_variables(sys.argv[1])
    else:
        plot_spacetime_diagram(sys.argv[1], variable=variable, save_fig=save_fig, output_file=output_file)


if __name__ == "__main__":
    main()

