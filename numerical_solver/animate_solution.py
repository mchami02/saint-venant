"""
Animate the evolution of water height over time.

Creates video-like animations showing how the water surface evolves.

Usage:
    python animate_solution.py results_test/solution_LF.h5
    python animate_solution.py results_test/solution_LF.h5 --fps 30 --output animation.mp4
"""

import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon


def animate_surface_evolution(h5_file: str, output_file: str = None, fps: int = 30, 
                               interval: int = 1, max_frames: int = None):
    """Create animation of water surface evolution.
    
    Args:
        h5_file: Path to HDF5 results file
        output_file: Output video filename (e.g., 'animation.mp4', 'animation.gif')
                    Always saved in plots/ directory
        fps: Frames per second for video
        interval: Use every Nth timestep (1 = all frames, 2 = every other frame, etc.)
        max_frames: Maximum number of frames to use (None = all)
    """
    import os
    
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    print(f"Loading data from {h5_file}...")
    
    with h5py.File(h5_file, 'r') as f:
        # Load data
        x = f['mesh/x'][:]
        topography = f['mesh/topography'][:]
        time = f['solution/time'][::interval]
        h = f['solution/h'][::interval, :]
        q = f['solution/q'][::interval, :]
        
        # Load metadata
        flux_scheme = f.attrs.get('flux_scheme', 'unknown')
        time_scheme = f.attrs.get('time_scheme', 'unknown')
        g = f.attrs.get('gravity', 9.81)
    
    # Limit frames if requested
    if max_frames is not None and len(time) > max_frames:
        indices = np.linspace(0, len(time)-1, max_frames, dtype=int)
        time = time[indices]
        h = h[indices, :]
        q = q[indices, :]
    
    # Compute derived quantities
    H = h + topography  # Free surface elevation
    u = np.divide(q, h, out=np.zeros_like(q), where=h>1e-10)  # Velocity
    Fr = np.abs(u) / np.sqrt(g * h + 1e-10)  # Froude number
    
    print(f"Creating animation with {len(time)} frames...")
    
    # Setup figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.patch.set_facecolor('white')
    
    # Initialize plots
    ax1, ax2, ax3 = axes
    
    # Plot 1: Water surface and topography
    ax1.set_xlim(x[0], x[-1])
    y_min = min(topography.min(), H.min()) - 0.5
    y_max = max(topography.max(), H.max()) + 0.5
    ax1.set_ylim(y_min, y_max)
    ax1.set_xlabel('Position x [m]', fontsize=11)
    ax1.set_ylabel('Elevation [m]', fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Fill topography
    ax1.fill_between(x, topography, y_min, alpha=0.3, color='saddlebrown', 
                     label='Topography', zorder=1)
    
    # Water surface line
    line_surface, = ax1.plot([], [], 'b-', linewidth=2.5, label='Water surface', zorder=3)
    
    # Water fill
    water_fill = ax1.fill_between(x, topography, topography, alpha=0.4, 
                                   color='dodgerblue', zorder=2)
    
    # Time text
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, 
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_title('Water Surface Evolution', fontsize=13, fontweight='bold')
    
    # Plot 2: Velocity
    ax2.set_xlim(x[0], x[-1])
    u_max = max(abs(u.min()), abs(u.max())) + 0.5
    ax2.set_ylim(-u_max, u_max)
    ax2.set_xlabel('Position x [m]', fontsize=11)
    ax2.set_ylabel('Velocity u [m/s]', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.8, alpha=0.5)
    
    line_velocity, = ax2.plot([], [], 'r-', linewidth=2, label='Velocity')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_title('Velocity Profile', fontsize=13, fontweight='bold')
    
    # Plot 3: Froude number
    ax3.set_xlim(x[0], x[-1])
    ax3.set_ylim(0, min(Fr.max() + 0.2, 3.0))
    ax3.set_xlabel('Position x [m]', fontsize=11)
    ax3.set_ylabel('Froude Number [-]', fontsize=11)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.axhline(y=1, color='r', linestyle='--', linewidth=1.5, 
                label='Critical (Fr=1)', alpha=0.7)
    
    line_froude, = ax3.plot([], [], 'g-', linewidth=2, label='Froude number')
    
    # Shade subcritical/supercritical regions
    ax3.fill_between([x[0], x[-1]], [0, 0], [1, 1], alpha=0.1, 
                     color='blue', label='Subcritical (Fr<1)')
    ax3.fill_between([x[0], x[-1]], [1, 1], [3, 3], alpha=0.1, 
                     color='red', label='Supercritical (Fr>1)')
    
    ax3.legend(loc='upper right', fontsize=9, ncol=2)
    ax3.set_title('Froude Number', fontsize=13, fontweight='bold')
    
    # Add metadata
    metadata_text = f'Scheme: {flux_scheme} + {time_scheme}\n'
    metadata_text += f'Grid: {len(x)} cells, Domain: [{x[0]:.1f}, {x[-1]:.1f}] m'
    fig.text(0.02, 0.01, metadata_text, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('1D Saint-Venant Equations - Surface Evolution', 
                 fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    
    # Store the water fill collection for updating
    water_collection = None
    
    # Animation update function
    def update(frame):
        nonlocal water_collection
        
        # Update surface
        H_frame = H[frame, :]
        line_surface.set_data(x, H_frame)
        
        # Remove old water fill and create new one
        if water_collection is not None:
            water_collection.remove()
        water_collection = ax1.fill_between(x, topography, H_frame, alpha=0.4, 
                                            color='dodgerblue', zorder=2)
        
        # Update velocity
        line_velocity.set_data(x, u[frame, :])
        
        # Update Froude number
        line_froude.set_data(x, Fr[frame, :])
        
        # Update time text
        time_text.set_text(f't = {time[frame]:.2f} s')
        
        return line_surface, line_velocity, line_froude, time_text
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=len(time), 
                                   interval=1000/fps, blit=False, repeat=True)
    
    # Save or show
    if output_file:
        # Always save in plots/ directory
        output_path = f"plots/{os.path.basename(output_file)}"
        print(f"Saving animation to {output_path}...")
        
        if output_file.endswith('.mp4'):
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='Saint-Venant Solver'), 
                          bitrate=1800)
            anim.save(output_path, writer=writer)
        elif output_file.endswith('.gif'):
            anim.save(output_path, writer='pillow', fps=fps)
        else:
            print("Unsupported format. Use .mp4 or .gif")
            return
        
        print(f"Animation saved to: {output_path}")
    else:
        print("Displaying animation (close window to exit)...")
        plt.show()


def create_comparison_animation(h5_files: list, output_file: str = None, fps: int = 30):
    """Create side-by-side comparison animation of multiple schemes.
    
    Args:
        h5_files: List of HDF5 file paths
        output_file: Output video filename
        fps: Frames per second
    """
    n_schemes = len(h5_files)
    
    # Load all data
    data_sets = []
    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as f:
            data = {
                'x': f['mesh/x'][:],
                'topo': f['mesh/topography'][:],
                'time': f['solution/time'][:],
                'h': f['solution/h'][:],
                'scheme': f.attrs.get('flux_scheme', 'unknown')
            }
            data_sets.append(data)
    
    # Find common time points
    min_frames = min(len(d['time']) for d in data_sets)
    
    # Setup figure
    fig, axes = plt.subplots(1, n_schemes, figsize=(7*n_schemes, 5))
    if n_schemes == 1:
        axes = [axes]
    
    lines = []
    fills = []
    time_texts = []
    
    for ax, data in zip(axes, data_sets):
        x = data['x']
        topo = data['topo']
        H = data['h'][0, :] + topo
        
        ax.set_xlim(x[0], x[-1])
        y_min = topo.min() - 0.5
        y_max = (data['h'] + topo[np.newaxis, :]).max() + 0.5
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Position x [m]')
        ax.set_ylabel('Elevation [m]')
        ax.set_title(f'{data["scheme"]} Scheme', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Topography
        ax.fill_between(x, topo, y_min, alpha=0.3, color='saddlebrown')
        
        # Water surface
        line, = ax.plot(x, H, 'b-', linewidth=2)
        lines.append(line)
        
        fill = ax.fill_between(x, topo, H, alpha=0.4, color='dodgerblue')
        fills.append((ax, fill))
        
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        time_texts.append(time_text)
    
    plt.tight_layout()
    
    def update(frame):
        for i, data in enumerate(data_sets):
            H = data['h'][frame, :] + data['topo']
            lines[i].set_ydata(H)
            
            ax, old_fill = fills[i]
            old_fill.remove()
            new_fill = ax.fill_between(data['x'], data['topo'], H, 
                                       alpha=0.4, color='dodgerblue')
            fills[i] = (ax, new_fill)
            
            time_texts[i].set_text(f't = {data["time"][frame]:.2f} s')
        
        return lines + time_texts
    
    anim = animation.FuncAnimation(fig, update, frames=min_frames,
                                   interval=1000/fps, blit=False, repeat=True)
    
    if output_file:
        # Always save in plots/ directory
        output_path = f"plots/{os.path.basename(output_file)}"
        print(f"Saving comparison animation to {output_path}...")
        if output_file.endswith('.mp4'):
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='Saint-Venant Solver'), bitrate=1800)
            anim.save(output_path, writer=writer)
        elif output_file.endswith('.gif'):
            anim.save(output_path, writer='pillow', fps=fps)
        print(f"Saved to: {output_path}")
    else:
        plt.show()


def main():
    """Main function for command-line usage."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Display:      python animate_solution.py results_test/solution_LF.h5")
        print("  Save GIF:     python animate_solution.py results_test/solution_LF.h5 --save animation.gif")
        print("  Save video:   python animate_solution.py results_test/solution_LF.h5 --save animation.mp4")
        print("\nOptions:")
        print("  --save FILE       Save animation to file (mp4 or gif)")
        print("  --fps N           Frames per second (default: 30)")
        print("  --interval N      Use every Nth timestep (default: 1)")
        print("  --max-frames N    Limit to N frames")
        print("  --compare         Compare multiple schemes side-by-side")
        print("\nExamples:")
        print("  python animate_solution.py results_test/solution_LF.h5")
        print("  python animate_solution.py results_test/solution_LF.h5 --fps 60")
        print("  python animate_solution.py results_test/solution_LF.h5 --save anim.gif --fps 20")
        print("  python animate_solution.py sol1.h5 sol2.h5 --compare --save comparison.mp4")
        sys.exit(1)
    
    # Parse arguments
    output_file = None
    fps = 30
    interval = 1
    max_frames = None
    compare = '--compare' in sys.argv
    
    if '--save' in sys.argv:
        idx = sys.argv.index('--save')
        if idx + 1 < len(sys.argv):
            output_file = sys.argv[idx + 1]
    
    if '--fps' in sys.argv:
        idx = sys.argv.index('--fps')
        fps = int(sys.argv[idx + 1])
    
    if '--interval' in sys.argv:
        idx = sys.argv.index('--interval')
        interval = int(sys.argv[idx + 1])
    
    if '--max-frames' in sys.argv:
        idx = sys.argv.index('--max-frames')
        max_frames = int(sys.argv[idx + 1])
    
    # Get HDF5 files
    h5_files = [arg for arg in sys.argv[1:] if arg.endswith('.h5')]
    
    if compare and len(h5_files) > 1:
        create_comparison_animation(h5_files, output_file, fps)
    else:
        animate_surface_evolution(h5_files[0], output_file, fps, interval, max_frames)


if __name__ == "__main__":
    main()

