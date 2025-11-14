import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

try:
    from IPython.display import HTML
    get_ipython()  # pylint: disable=undefined-variable
    IN_NOTEBOOK = True
except (ImportError, NameError):    
    IN_NOTEBOOK = False

def plot_grid_density(grid, nx, nt, dx, dt):
    plt.figure(figsize=(10, 6))

    plt.imshow(
        grid, 
        extent=[0, nx * dx, 0, nt * dt],
        aspect='auto',
        origin='lower',
        cmap='jet'  # 'jet' goes from blue (low) through green/yellow/orange to red (high)
    )

    plt.colorbar(label='Density ρ(x,t)')
    plt.xlabel('Space x')
    plt.ylabel('Time t')
    plt.title('Space–Time Density Heatmap')
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