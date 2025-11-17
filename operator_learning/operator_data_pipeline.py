from godunov_solver.grid_generator import GridGenerator
from godunov_solver.solve_class import *
from godunov_solver.flux import *
from joblib import Memory
from tqdm import tqdm
from torch.utils.data import Dataset
import torch

mem = Memory(location='.cache')



@mem.cache
def generate_data(solver, n_samples, nx, nt, dx, dt, initial_condition = None, boundary_condition = None, ic_kwargs = None, bc_kwargs = None):
    grid_generator = GridGenerator(solver, nx, nt, dx, dt)
    grids = []
    for _ in tqdm(range(n_samples), desc="Generating grids"):
        grids.append(grid_generator(initial_condition, boundary_condition, ic_kwargs, bc_kwargs))
    return grids


class GridDataset(Dataset):
    def __init__(self, solver, n_samples, nx, nt, dx, dt):
        self.n_samples = n_samples
        self.nx = nx
        self.nt = nt
        self.dx = dx
        self.dt = dt
        self.grids = generate_data(solver, n_samples, nx, nt, dx, dt)

    def __len__(self):
        return len(self.grids)

    def __getitem__(self, idx):
        input_grids = []
        for val in self.grids[idx].values():
            input_grids.append(torch.from_numpy(self.grids[idx].get_array(val)).to(torch.float32))
            
        input_grid = torch.stack(input_grids, dim=-1)  # (nt, nx, n_vals)
        target_grid = input_grid.clone()
        nt, nx, _ = input_grid.shape
        
        # Repeat initial condition for all timesteps
        input_grid[1:, 1:-1] = -1 # mask all but the initial condition and boundary
        
        # Create coordinate grids that tell the model what time/space to predict
        t_coords = (torch.arange(nt).float() * self.grids[idx].dt)[:, None].expand(nt, nx)  # (nt, nx)
        x_coords = (torch.arange(nx).float() * self.grids[idx].dx)[None, :].expand(nt, nx)  # (nt, nx)
        
        # Stack: (nt, nx, n_vals + 2) where channels are [initial_density_repeated, time, space]
        full_input = torch.cat([input_grid, t_coords.unsqueeze(-1), x_coords.unsqueeze(-1)], dim=-1)
        return full_input.permute(2, 0, 1), target_grid.permute(2, 0, 1)  # Returns: (n_vals + 2, nt, nx), (n_vals, nt, nx)
