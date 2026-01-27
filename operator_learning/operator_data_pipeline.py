
import numpy as np
import torch
from hf_grids import download_grids, upload_grids
from joblib import Memory
from nfv.flows import Greenshield
from nfv.initial_conditions import PiecewiseConstant
from nfv.problem import Problem
from nfv.solvers import LaxHopf
from torch.utils.data import Dataset
from tqdm import tqdm

from numerical_methods import GridGenerator

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


def preprocess_grids(grids, nx, nt, dx, dt):
    """
    Preprocess raw grids by adding coordinate channels.
    
    Args:
        grids: numpy array of shape (n_samples, nt, nx) or list of Grid objects
        nx, nt: Grid dimensions
        dx, dt: Grid spacing
    
    Returns:
        List of tuples (full_input, target_grid) where:
        - full_input: tensor of shape (n_vals + 2, nt, nx) with coordinates
        - target_grid: tensor of shape (n_vals, nt, nx)
    """
    processed = []
    
    for idx in range(len(grids)):
        if isinstance(grids, list):
            input_grids = []
            for val in grids[idx].values():
                input_grids.append(torch.from_numpy(grids[idx].get_array(val)).to(torch.float32))
            input_grid = torch.stack(input_grids, dim=-1)  # (nt, nx, n_vals)
        else:
            input_grid = torch.from_numpy(grids[idx]).to(torch.float32).unsqueeze(-1)
        
        target_grid = input_grid.clone()
        grid_nt, grid_nx, _ = input_grid.shape
        
        # Create coordinate grids that tell the model what time/space to predict
        t_coords = (torch.arange(grid_nt).float() * dt)[:, None].expand(grid_nt, grid_nx).unsqueeze(-1)  # (nt, nx, 1)
        x_coords = (torch.arange(grid_nx).float() * dx)[None, :].expand(grid_nt, grid_nx).unsqueeze(-1)  # (nt, nx, 1)
        
        # Stack: (nt, nx, n_vals + 2) where channels are [initial_density_repeated, time, space]
        full_input = torch.cat([input_grid, t_coords, x_coords], dim=-1).permute(2, 0, 1)
        target_grid = target_grid.permute(2, 0, 1)
        
        processed.append((full_input, target_grid))
    
    return processed


def get_dataset(solver, flux, n_samples, nx, nt, dx, dt, max_steps=3, random_seed=42):
    '''
    Get a dataset for the given solver, n_samples, nx, nt, dx, dt
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
    
    # Preprocess grids with coordinates
    processed = preprocess_grids(grids[grids_idx], nx, nt, dx, dt)
    dataset = GridDataset(processed)

    return dataset

def get_grids(solver, flux, nx, nt, dx, dt, max_steps, n_samples):
    grids = download_grids(solver, flux, nx, nt, dx, dt, max_steps)
    if grids is None or len(grids) < n_samples:
        print(f"No big enough grids available in the repository, generating {n_samples} grids")
        grids = get_nfv_dataset(n_samples, nx, nt, dx, dt, 2, True)
        upload_grids(grids, solver, flux, nx, nt, dx, dt, max_steps)
    return grids

def get_datasets(solver, flux, n_samples, nx, nt, dx, dt, max_steps=2, max_train_steps=2, train_ratio=0.8, val_ratio=0.1, random_seed=42):
    '''
    Get the train, val and test datasets for the given solver, n_samples, nx, nt, dx, dt
    If available in the repository, download the grids from the repository, otherwise generate them and upload them to the repository
    '''
    n_train = int(train_ratio * n_samples)
    n_val = int(val_ratio * n_samples)
    n_test = n_samples - n_train - n_val
    
    def shuffle_and_select(grids, n, seed):
        idx = np.arange(len(grids))
        np.random.seed(seed)
        np.random.shuffle(idx)
        return grids[idx[:n]]
    
    if max_steps == max_train_steps:
        grids = get_grids(solver, flux, nx, nt, dx, dt, max_steps, n_samples)
        grids = shuffle_and_select(grids, n_samples, random_seed)
        train_grids = grids[:n_train]
        val_grids = grids[n_train:n_train + n_val]
        test_grids = grids[n_train + n_val:]
    else:
        train_val_grids = get_grids(solver, flux, nx, nt, dx, dt, max_train_steps, n_train + n_val)
        train_val_grids = shuffle_and_select(train_val_grids, n_train + n_val, random_seed)
        train_grids = train_val_grids[:n_train]
        val_grids = train_val_grids[n_train:]
        
        test_grids = get_grids(solver, flux, nx, nt, dx, dt, max_steps, n_test)
        test_grids = shuffle_and_select(test_grids, n_test, random_seed)


    # Preprocess grids with coordinates
    train_processed = preprocess_grids(train_grids, nx, nt, dx, dt)
    val_processed = preprocess_grids(val_grids, nx, nt, dx, dt)
    test_processed = preprocess_grids(test_grids, nx, nt, dx, dt)
    
    train_dataset = GridDataset(train_processed)
    val_dataset = GridDataset(val_processed)
    test_dataset = GridDataset(test_processed)

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
        train_dataset: GridDataset with all training samples at different resolutions
        val_dataset: GridDataset with all validation samples at different resolutions
        test_dataset: GridDataset at base resolution
    '''
    # Create linearly spaced scale factors between 1 and 2
    # Both dx and dt scale together to maintain CFL condition
    dx_scales = np.linspace(1.0, 2.0, n_res)
    dt_scales = np.linspace(1.0, 2.0, n_res)
    
    all_train_processed = []
    all_val_processed = []
    
    # Samples per resolution for train/val (split from n_samples)
    # Ensure at least 10 samples per resolution for proper train/val split
    samples_per_res = max(10, n_samples // (n_res * n_res))
    
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
                
                # Preprocess and add to lists
                if len(train_idx) > 0:
                    train_processed = preprocess_grids(grids[train_idx], nx, nt, dx_i, dt_j)
                    all_train_processed.extend(train_processed)
                if len(val_idx) > 0:
                    val_processed = preprocess_grids(grids[val_idx], nx, nt, dx_i, dt_j)
                    all_val_processed.extend(val_processed)
                
                successful_resolutions += 1
                
            except Exception as e:
                print(f"    Warning: Failed to generate grids for dx={dx_i:.4f}, dt={dt_j:.4f}: {e}")
                failed_resolutions.append((dx_i, dt_j))
                continue
    
    if not all_train_processed:
        raise RuntimeError("Failed to generate any training datasets. Check solver compatibility with the resolution parameters.")
    
    # Create single datasets from all processed grids
    train_dataset = GridDataset(all_train_processed)
    val_dataset = GridDataset(all_val_processed, cleaner=None) if all_val_processed else GridDataset([], cleaner=None)
    
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
    test_processed = preprocess_grids(grids[test_idx], nx, nt, dx, dt)
    test_dataset = GridDataset(test_processed, cleaner=None)
    
    print("\nMulti-resolution datasets created:")
    print(f"  Successful resolutions: {successful_resolutions}/{n_res*n_res}")
    if failed_resolutions:
        print(f"  Failed resolutions: {len(failed_resolutions)} (skipped)")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples at base resolution")
    
    return train_dataset, val_dataset, test_dataset


class GridMaskInner:
    """Transform that masks inner cells (keeps only initial condition and boundaries)."""
    def __init__(self):
        pass

    def __call__(self, full_input, target_grid):
        """
        Apply mask to input grid.
        
        Args:
            full_input: tensor of shape (n_vals + 2, nt, nx)
            target_grid: tensor of shape (n_vals, nt, nx)
        
        Returns:
            masked full_input, target_grid
        """
        full_input = full_input.clone()
        full_input[0, 1:, 1:-1] = -1
        return full_input, target_grid

class GridMaskRandom:
    def __init__(self, mask_ratio=0.5):
        self.mask_ratio = mask_ratio

    def __call__(self, full_input, target_grid):
        full_input = full_input.clone()
        mask = torch.rand(full_input.shape[1] - 1, full_input.shape[2] - 2) < self.mask_ratio
        full_input[0, 1:, 1:-1][mask] = -1.0
        return full_input, target_grid
    
class GridMaskAllButInitial:
    def __init__(self):
        pass
    
    def __call__(self, full_input, target_grid):
        full_input = full_input.clone()
        full_input[0, 1:, :] = -1
        return full_input, target_grid

class OffsetCoordinates:
    def __init__(self, offset_range=10.0, random_seed=42):
        self.offset_range = offset_range
        self.rng = np.random.default_rng(random_seed)

    def __call__(self, full_input, target_grid):
        full_input = full_input.clone()
        offset_t = self.rng.uniform(-self.offset_range, self.offset_range)
        offset_x = self.rng.uniform(-self.offset_range, self.offset_range)
        full_input[1:, :] += offset_t
        full_input[2:, :] += offset_x
        return full_input, target_grid

class NoisyCoordinates:
    def __init__(self, noise_level=0.01, random_seed=42):
        self.noise_level = noise_level
        self.rng = np.random.default_rng(random_seed)

    def __call__(self, full_input, target_grid):
        full_input = full_input.clone()
        full_input[1:, :] += self.rng.normal(0, self.noise_level, size=(full_input.shape[1], full_input.shape[2]))
        full_input[2:, :] += self.rng.normal(0, self.noise_level, size=(full_input.shape[1], full_input.shape[2]))
        return full_input, target_grid


class ConstCleaner:
    """
    Cleaner that filters out grids with too many identical values.
    
    This removes grids where the most common value appears more than
    `max_const_ratio` fraction of the total grid cells.
    
    Args:
        max_const_ratio: Maximum allowed ratio of identical values (default 0.9).
                        Grids with more than this fraction of identical values are removed.
    """
    def __init__(self, max_const_ratio: float = 0.4):
        self.max_const_ratio = max_const_ratio
    
    def __call__(self, processed_grids):
        """
        Filter grids that have too many identical values.
        
        Args:
            processed_grids: List of tuples (full_input, target_grid) where:
                - full_input: tensor of shape (n_vals + 2, nt, nx)
                - target_grid: tensor of shape (n_vals, nt, nx)
        
        Returns:
            Filtered list of (full_input, target_grid) tuples
        """
        if not processed_grids:
            return processed_grids
        
        filtered = []
        for full_input, target_grid in processed_grids:
            # Check the target grid (the actual solution values, not coordinates)
            # target_grid shape: (n_vals, nt, nx)
            total_cells = target_grid.numel()
            
            # Find the most common value and its count
            flat = target_grid.flatten()
            # Use bincount for integer-like values or unique for floats
            unique_vals, counts = torch.unique(flat, return_counts=True)
            max_count = counts.max().item()
            
            const_ratio = max_count / total_cells
            
            if const_ratio <= self.max_const_ratio:
                filtered.append((full_input, target_grid))
        
        n_removed = len(processed_grids) - len(filtered)
        if n_removed > 0:
            print(f"ConstCleaner: Removed {n_removed}/{len(processed_grids)} grids with >{self.max_const_ratio*100:.0f}% identical values")
        
        return filtered



class ICCleaner:
    """
    Cleaner that removes isolated cells in initial conditions.
    
    A cell is considered isolated if its value differs from both its left and 
    right neighbors. Such cells are replaced with the value of the neighbor
    they are numerically closest to.
    """
    def __init__(self):
        pass

    def __call__(self, processed_grids):
        cleaned = []
        for full_input, target_grid in processed_grids:
            full_input = full_input.clone()
            ic = full_input[0, 0, :]  # Initial condition: shape (nx,)
            nx = ic.shape[0]
            
            # Check each cell (except boundary cells)
            for i in range(1, nx - 1):
                left_val = ic[i - 1]
                curr_val = ic[i]
                right_val = ic[i + 1]
                
                # Check if current cell differs from both neighbors
                if curr_val != left_val and curr_val != right_val:
                    # Replace with the value of the closest neighbor
                    left_dist = abs(curr_val - left_val)
                    right_dist = abs(curr_val - right_val)
                    
                    if left_dist <= right_dist:
                        ic[i] = left_val
                    else:
                        ic[i] = right_val
            
            cleaned.append((full_input, target_grid))
        return cleaned

class GridDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Dataset for preprocessed grids.
    
    Args:
        processed_grids: List of tuples (full_input, target_grid) where:
            - full_input: tensor of shape (n_vals + 2, nt, nx) with coordinates already added
            - target_grid: tensor of shape (n_vals, nt, nx)
        transform: Optional transform to apply (e.g., GridMaskInner)
        cleaner: Optional cleaner to filter grids (e.g., ConstCleaner). Set to None to disable.
    """
    def __init__(self, processed_grids, transform=[GridMaskAllButInitial()], cleaner=ICCleaner()):
        if cleaner is not None:
            processed_grids = cleaner(processed_grids)
        self.processed_grids = processed_grids
        if transform is not None:
            self.transform = transform

    def __len__(self):
        return len(self.processed_grids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        full_input, target_grid = self.processed_grids[idx]
        
        for transform in self.transform:
            full_input, target_grid = transform(full_input, target_grid)
        
        return full_input, target_grid  # Returns: (n_vals + 2, nt, nx), (n_vals, nt, nx)
