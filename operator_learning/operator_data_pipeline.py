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
from hf_grids import download_grids, upload_grids

mem = Memory(location='.cache')

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

def get_nfv_dataset(n_samples, nx, nt, dx, dt, max_steps = 3, only_shocks = False):
    ics = [PiecewiseRandom(ks=[np.random.rand() for _ in range(max_steps)], x_noise=False) for _ in range(n_samples)]
    if only_shocks:
        for ic in ics:
            ic.ks.sort()

    problem = Problem(nx=nx, nt=nt, dx=dx, dt=dt, ic=ics, flow=Greenshield())
    grids = problem.solve(LaxHopf, batch_size=4, dtype=torch.float64, progressbar=True).cpu().numpy()
    return grids

def get_dataset(solver, flux, n_samples, nx, nt, dx, dt, max_steps=3, random_seed=42):
    '''
    Get the train, val and test datasets for the given solver, n_samples, nx, nt, dx, dt
    If available in the repository, download the grids from the repository, otherwise generate them and upload them to the repository
    '''
    # Try to download the grids from the repository
    grids = download_grids(solver, flux, nx, nt, dx, dt, max_steps)
    if grids is None or len(grids) < n_samples:
        print(f"No big enough grids available in the repository, generating {n_samples} grids")
        grids = get_nfv_dataset(n_samples, nx, nt, dx, dt, max_steps)
        upload_grids(grids, solver, flux, nx, nt, dx, dt, max_steps)

    grids_idx = np.arange(len(grids))
    np.random.seed(random_seed)
    np.random.shuffle(grids_idx)
    grids_idx = grids_idx[:n_samples]
    
    dataset = GridDataset(grids[grids_idx], nx, nt, dx, dt)

    return dataset

def get_datasets(solver, flux, n_samples, nx, nt, dx, dt, max_steps=3, train_ratio=0.8, val_ratio=0.1, random_seed=42):
    '''
    Get the train, val and test datasets for the given solver, n_samples, nx, nt, dx, dt
    If available in the repository, download the grids from the repository, otherwise generate them and upload them to the repository
    '''
    # Try to download the grids from the repository
    grids = download_grids(solver, flux, nx, nt, dx, dt, max_steps)
    if grids is None or len(grids) < n_samples:
        print(f"No big enough grids available in the repository, generating {n_samples} grids")
        grids = get_nfv_dataset(n_samples, nx, nt, dx, dt, 2, True)
        upload_grids(grids, solver, flux, nx, nt, dx, dt, max_steps)

    grids_idx = np.arange(len(grids))
    np.random.seed(random_seed)
    np.random.shuffle(grids_idx)
    grids_idx = grids_idx[:n_samples]
    train_idx = grids_idx[:int(train_ratio * len(grids_idx))]
    val_idx = grids_idx[int(train_ratio * len(grids_idx)):int((train_ratio + val_ratio) * len(grids_idx))]
    test_idx = grids_idx[int((train_ratio + val_ratio) * len(grids_idx)):]
    
    train_dataset = GridDataset(grids[train_idx], nx, nt, dx, dt)
    val_dataset = GridDataset(grids[val_idx], nx, nt, dx, dt)
    test_dataset = GridDataset(grids[test_idx], nx, nt, dx, dt)

    return train_dataset, val_dataset, test_dataset


class GridMaskInner:
    def __init__(self):
        pass

    def __call__(self, grid):
        grid[:, 1:, 1:-1] = -1
        return grid

class GridMaskRandom:
    def __init__(self, mask_ratio=0.5):
        self.mask_ratio = mask_ratio

    def __call__(self, grid):
        mask = torch.rand(grid.shape[1] - 1, grid.shape[2] - 2) < self.mask_ratio
        grid[0, 1:, 1:-1][mask] = -1.0
        return grid
    
class GridDataset(Dataset):
    def __init__(self, grids, nx, nt, dx, dt, transform=GridMaskInner()):
        self.nx = nx
        self.nt = nt
        self.dx = dx
        self.dt = dt
        self.grids = grids
        self.transform = transform

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
        
        # Create coordinate grids that tell the model what time/space to predict
        t_coords = (torch.arange(nt).float() * self.dt)[:, None].expand(nt, nx).unsqueeze(-1)  # (nt, nx, 1)
        x_coords = (torch.arange(nx).float() * self.dx)[None, :].expand(nt, nx).unsqueeze(-1)  # (nt, nx, 1)
        
        # Stack: (nt, nx, n_vals + 2) where channels are [initial_density_repeated, time, space]
        full_input = torch.cat([input_grid, t_coords, x_coords], dim=-1).permute(2, 0, 1)
        target_grid = target_grid.permute(2, 0, 1)

        full_input = self.transform(full_input)
        return full_input, target_grid  # Returns: (n_vals + 2, nt, nx), (n_vals, nt, nx)
