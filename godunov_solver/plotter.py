import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

try:
    from IPython.display import HTML
    get_ipython()  # pylint: disable=undefined-variable
    IN_NOTEBOOK = True
except (ImportError, NameError):    
    IN_NOTEBOOK = False

class Plotter():
    def __init__(self, N_x, d_x, d_t, T, skip_frames=10, fps=30) -> None:
        self.N_x = N_x
        self.d_x = d_x
        self.d_t = d_t
        self.T = T
        self.time_steps = int(T // d_t)
        self.skip_frames = skip_frames
        self.fps = fps
        
    def plot_density(self, arr, output_name=''):
        ''' Plot arr as a grid with x on the x-axis and t on the y-axis, and the values of h as the color.
            Time increases upward, with t=0 at the bottom. 
            
            Args:
                arr: Array to plot
                output_name: If empty string, only show. Otherwise, save to this filename.
        '''
        plt.figure(figsize=(10, 6))
        plt.imshow(arr, cmap='viridis', aspect='auto', origin='lower')
        plt.colorbar(label='Water height h')
        plt.xlabel('Cell index')
        plt.ylabel('Time step')
        
        if output_name:
            plt.savefig(output_name, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {output_name}")
            plt.close()
        else:
            plt.show()

    def animate_density(self, arr, output_name=''):
        ''' Create an animation showing h(x) evolving through time.

            Args:
                arr: Array to animate
                output_name: If empty string, only show. Otherwise, save to this filename (e.g., 'animation.gif' or 'animation.mp4').
        '''
        fig, ax = plt.subplots(figsize=(10, 6))

        # Set up the spatial grid
        x = np.linspace(0, self.N_x * self.d_x, self.N_x + 2)

        # Find global min/max for consistent y-axis
        h_min, h_max = arr.min(), arr.max()
        y_margin = (h_max - h_min) * 0.1

        # Initialize the line
        line, = ax.plot([], [], 'b-', linewidth=2)
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

        ax.set_xlim(0, self.N_x * self.d_x)
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
            i = frame * self.skip_frames
            if i >= self.time_steps:
                i = self.time_steps - 1
            line.set_data(x, arr[i])
            time_text.set_text(f'Time: {i * self.d_t:.3f} s (step {i}/{self.time_steps})')
            return line, time_text

        n_frames = min(self.time_steps // self.skip_frames, self.time_steps)
        anim = FuncAnimation(fig, update, frames=n_frames, init_func=init, 
                             blit=True, interval=1000/self.fps)

        if output_name:
            # Save animation
            if output_name.endswith('.gif'):
                anim.save(output_name, writer='pillow', fps=self.fps)
            elif output_name.endswith('.mp4'):
                anim.save(output_name, writer='ffmpeg', fps=self.fps)
            else:
                # Default to gif
                anim.save(output_name + '.gif', writer='pillow', fps=self.fps)
            print(f"Saved animation to {output_name}")
            plt.close()
            return anim
        elif IN_NOTEBOOK:
            plt.close()
            return HTML(anim.to_jshtml())
        else:
            plt.show()
            return anim
