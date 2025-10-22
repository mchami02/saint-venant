import sys
from pathlib import Path

# Add parent directory to path to import godunov_solver
sys.path.append(str(Path(__file__).parent.parent))

from godunov_solver.solver import Solver
from godunov_solver.plotter import Plotter

from torch.utils.data import Dataset
import torch
import numpy as np
from tqdm import tqdm
from joblib import Memory

memory = Memory(location='./data_cache', compress=True)

def boundary_conditions(time_steps, n_conditions):
    left_boundary = [] # Each element is a float or an array of length time_steps
    right_boundary = [] # Each element is a float or an array of length time_steps
    
    # Generate random floats
    for _ in range(n_conditions):
        left_boundary.append(np.random.uniform(0, 1))
        right_boundary.append(np.random.uniform(0, 1))
    return left_boundary, right_boundary

def generate_data(N_x, d_x, d_t, T, n_samples):
    solver = Solver(N_x, T, d_x, d_t)
    left_boundary, right_boundary = boundary_conditions(solver.time_steps, n_samples)

    h = np.zeros((n_samples, solver.time_steps, N_x + 2))
    u = np.zeros((n_samples, solver.time_steps, N_x + 2))
    for i in tqdm(range(n_samples), desc="Generating data"):
        h[i], u[i] = solver.solve(left_boundary[i], right_boundary[i])

    return h, u

@memory.cache
def train_val_test_split(N_x, d_x, d_t, T, n_samples):
    h, u = generate_data(N_x, d_x, d_t, T, n_samples)
    
    # Generate random indices for splitting
    indices = np.random.permutation(n_samples)
    
    # Calculate split sizes
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    # Split indices into train/val/test
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    # Use indices to split data randomly
    train_h = h[train_idx]
    train_u = u[train_idx]
    val_h = h[val_idx] 
    val_u = u[val_idx]
    test_h = h[test_idx]
    test_u = u[test_idx]
    return train_h, train_u, val_h, val_u, test_h, test_u

class PairDataset(Dataset):
    def __init__(self, h, u):
        self.x, self.y = self.init_pairs(h, u)

    def init_pairs(self, h, u):
        x = []
        y = []
        for i in range(len(h)):
            for j in range(len(h[i]) - 1):
                x.append((h[i][j], u[i][j]))
                y.append((h[i][j+1], u[i][j+1]))
        return x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_h, x_u = self.x[idx]
        y_h, y_u = self.y[idx]
        # Convert to float32 tensors to avoid MPS float64 issues
        x_h = torch.from_numpy(x_h).float()
        x_u = torch.from_numpy(x_u).float()
        y_h = torch.from_numpy(y_h).float()
        y_u = torch.from_numpy(y_u).float()
        return (x_h, x_u), (y_h, y_u)

class AutoRegressiveDataset(Dataset):
    """Dataset that returns full trajectories for autoregressive testing."""
    def __init__(self, h, u):
        self.h = h
        self.u = u

    def __len__(self):
        return len(self.h)

    def __getitem__(self, idx):
        # Return initial condition and full trajectory
        # Input: (h[0], u[0])
        # Output: (h[all times], u[all times])
        initial_h = torch.from_numpy(self.h[idx, 0]).float()
        initial_u = torch.from_numpy(self.u[idx, 0]).float()
        trajectory_h = torch.from_numpy(self.h[idx]).float()
        trajectory_u = torch.from_numpy(self.u[idx]).float()
        return (initial_h, initial_u), (trajectory_h, trajectory_u)

# Alias for backward compatibility
SVEDataset = PairDataset