from numerical_methods import GridGenerator, Godunov, Greenshields, Triangular, LWRRiemannSolver, Grid
from joblib import Memory
from tqdm import tqdm
from torch.utils.data import Dataset, ConcatDataset
import torch
import numpy as np
from nfv.initial_conditions import PiecewiseConstant
from nfv.flows import Greenshield
from nfv.solvers import LaxHopf
from nfv.problem import Problem
mem = Memory(location='.cache')



@mem.cache
def generate_data(solver, n_samples, nx, nt, dx, dt, initial_condition = None, boundary_condition = None, ic_kwargs = None, bc_kwargs = None):
    grid_generator = GridGenerator(solver, nx, nt, dx, dt, randomize=False)
    grids = []
    for _ in tqdm(range(n_samples), desc="Generating grids"):
        grids.append(grid_generator(initial_condition, boundary_condition, ic_kwargs, bc_kwargs))
    return grids

class PiecewiseRandom(PiecewiseConstant):
    def __init__(self, ks, x_noise=False):
        super().__init__(ks, x_noise)
        self.xs = np.random.rand(len(ks) - 1)
        self.xs = np.sort(self.xs)
        self.xs = np.concatenate([[0], self.xs, [1]])

@mem.cache
def get_nfv_dataset(n_samples, nx, nt, dx, dt, max_steps = 3):
    ics = [PiecewiseRandom(ks=[np.random.rand() for _ in range(max_steps)], x_noise=False) for _ in range(n_samples)]
    problem = Problem(nx=nx, nt=nt, dx=dx, dt=dt, ic=ics, flow=Greenshield())
    grids = problem.solve(LaxHopf, batch_size=4, dtype=torch.float64, progressbar=True).cpu().numpy()
    return grids


class GridDataset(Dataset):
    def __init__(self, solver, n_samples, nx, nt, dx, dt):
        self.n_samples = n_samples
        self.nx = nx
        self.nt = nt
        self.dx = dx
        self.dt = dt
        self.grids = get_nfv_dataset(n_samples, nx, nt, dx, dt)

    def __len__(self):
        return len(self.grids)

    def __getitem__(self, idx):
        if isinstance(self.grids, list):
            input_grids = []
            for val in self.grids[idx].values():
                input_grids.append(torch.from_numpy(self.grids[idx].get_array(val)).to(torch.float32))

            input_grid = torch.stack(input_grids, dim=-1)  # (nt, nx, n_vals)
        else:
            input_grid = torch.from_numpy(self.grids[idx]).to(torch.float32).unsqueeze(-1)
        target_grid = input_grid.clone()
        nt, nx, _ = input_grid.shape
        
        # Repeat initial condition for all timesteps
        input_grid[1:, 1:-1] = -1 # mask all but the initial condition and boundary
        
        # Create coordinate grids that tell the model what time/space to predict
        # t_coords = (torch.arange(nt).float() * self.grids[idx].dt)[:, None].expand(nt, nx)  # (nt, nx)
        # x_coords = (torch.arange(nx).float() * self.grids[idx].dx)[None, :].expand(nt, nx)  # (nt, nx)
        
        # Stack: (nt, nx, n_vals + 2) where channels are [initial_density_repeated, time, space]
        full_input = input_grid
        return full_input.permute(2, 0, 1), target_grid.permute(2, 0, 1)  # Returns: (n_vals + 2, nt, nx), (n_vals, nt, nx)
