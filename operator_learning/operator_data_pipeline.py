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


def get_multi_res_datasets(solver, flux, n_samples, nx, nt, dx, dt, n_res=5, max_steps=3, train_ratio=0.8, val_ratio=0.1, random_seed=42):
    '''
    Get multi-resolution train and val datasets with the same nx, nt but different dx, dt.
    
    Creates n_res x n_res = 25 different resolution combinations:
    - dx values: linearly distributed between dx and dx * 2
    - dt values: linearly distributed between dt and dt * 2
    
    Note: Both dx and dt are scaled together to maintain CFL condition stability.
    The resolutions are paired: (dx*scale, dt*scale) for each scale factor.
    
    Test dataset uses the base resolution (dx, dt) only.
    
    Args:
        solver: Solver type
        flux: Flux type
        n_samples: Number of samples total (distributed across resolutions)
        nx: Number of spatial grid points (same for all resolutions)
        nt: Number of time steps (same for all resolutions)
        dx: Base spatial step size
        dt: Base time step size
        n_res: Number of resolution steps in each dimension (default 5, gives 25 total)
        max_steps: Maximum number of steps for initial conditions
        train_ratio: Ratio of samples for training
        val_ratio: Ratio of samples for validation
        random_seed: Random seed for reproducibility
    
    Returns:
        train_dataset: ConcatDataset of all training datasets at different resolutions
        val_dataset: ConcatDataset of all validation datasets at different resolutions
        test_dataset: GridDataset at base resolution
    '''
    # Create linearly spaced scale factors between 1 and 2
    # Both dx and dt scale together to maintain CFL condition
    dx_scales = np.linspace(1.0, 2.0, n_res)
    dt_scales = np.linspace(1.0, 2.0, n_res)
    
    train_datasets = []
    val_datasets = []
    
    # Samples per resolution for train/val (split from n_samples)
    samples_per_res = max(1, n_samples // (n_res * n_res))
    
    print(f"Creating multi-resolution datasets: {n_res}x{n_res} = {n_res*n_res} resolutions")
    print(f"  dx range: [{dx:.4f}, {dx*2:.4f}] (5 values)")
    print(f"  dt range: [{dt:.4f}, {dt*2:.4f}] (5 values)")
    print(f"  Samples per resolution: {samples_per_res}")
    
    successful_resolutions = 0
    failed_resolutions = []
    
    for i, dx_scale in enumerate(dx_scales):
        for j, dt_scale in enumerate(dt_scales):
            dx_i = dx * dx_scale
            dt_j = dt * dt_scale
            
            print(f"  Loading resolution ({i+1}/{n_res}, {j+1}/{n_res}): dx={dx_i:.4f}, dt={dt_j:.4f}")
            
            try:
                # Download or generate grids for this resolution
                grids = download_grids(solver, flux, nx, nt, dx_i, dt_j, max_steps)
                if grids is None or len(grids) < samples_per_res:
                    print(f"    Generating {samples_per_res} grids...")
                    grids = get_nfv_dataset(samples_per_res, nx, nt, dx_i, dt_j, 2, True)
                    upload_grids(grids, solver, flux, nx, nt, dx_i, dt_j, max_steps)
                
                # Shuffle and split
                grids_idx = np.arange(len(grids))
                np.random.seed(random_seed + i * n_res + j)  # Different seed per resolution
                np.random.shuffle(grids_idx)
                grids_idx = grids_idx[:samples_per_res]
                
                train_end = int(train_ratio * len(grids_idx))
                val_end = int((train_ratio + val_ratio) * len(grids_idx))
                
                train_idx = grids_idx[:train_end]
                val_idx = grids_idx[train_end:val_end]
                
                if len(train_idx) > 0:
                    train_datasets.append(GridDataset(grids[train_idx], nx, nt, dx_i, dt_j))
                if len(val_idx) > 0:
                    val_datasets.append(GridDataset(grids[val_idx], nx, nt, dx_i, dt_j))
                
                successful_resolutions += 1
                
            except Exception as e:
                print(f"    Warning: Failed to generate grids for dx={dx_i:.4f}, dt={dt_j:.4f}: {e}")
                failed_resolutions.append((dx_i, dt_j))
                continue
    
    if not train_datasets:
        raise RuntimeError("Failed to generate any training datasets. Check solver compatibility with the resolution parameters.")
    
    # Combine all train and val datasets
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets) if val_datasets else None
    
    # Test dataset uses base resolution only
    print(f"  Loading test dataset at base resolution: dx={dx:.4f}, dt={dt:.4f}")
    grids = download_grids(solver, flux, nx, nt, dx, dt, max_steps)
    if grids is None or len(grids) < samples_per_res:
        grids = get_nfv_dataset(samples_per_res, nx, nt, dx, dt, 2, True)
        upload_grids(grids, solver, flux, nx, nt, dx, dt, max_steps)
    
    grids_idx = np.arange(len(grids))
    np.random.seed(random_seed)
    np.random.shuffle(grids_idx)
    # Use remaining samples for test
    test_start = int((train_ratio + val_ratio) * len(grids_idx[:samples_per_res]))
    test_idx = grids_idx[test_start:samples_per_res]
    test_dataset = GridDataset(grids[test_idx], nx, nt, dx, dt)
    
    print(f"\nMulti-resolution datasets created:")
    print(f"  Successful resolutions: {successful_resolutions}/{n_res*n_res}")
    if failed_resolutions:
        print(f"  Failed resolutions: {len(failed_resolutions)} (skipped)")
    print(f"  Train: {len(train_dataset)} samples from {len(train_datasets)} resolutions")
    print(f"  Val: {len(val_dataset) if val_dataset else 0} samples from {len(val_datasets)} resolutions")
    print(f"  Test: {len(test_dataset)} samples at base resolution")
    
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
