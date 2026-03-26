"""Shared fixtures for numerical_solvers tests."""

import pytest
import torch

# Tolerance constants
ATOL_CONSERVATION = 1e-5  # for conservation checks (float32)
ATOL_CONSERVATION_F64 = 1e-10  # for conservation checks (float64)
ATOL_STEADY = 1e-7  # for steady-state preservation


@pytest.fixture
def small_grid_1d():
    """Small 1D grid parameters for fast tests."""
    nx = 32
    dx = 0.05
    dt = 0.005
    nt = 20
    x_f32 = torch.arange(nx, dtype=torch.float32) * dx
    x_f64 = torch.arange(nx, dtype=torch.float64) * dx
    return {"nx": nx, "dx": dx, "dt": dt, "nt": nt, "x_f32": x_f32, "x_f64": x_f64}


@pytest.fixture
def torch_rng():
    """Seeded PyTorch generator for reproducible tests."""
    rng = torch.Generator()
    rng.manual_seed(42)
    return rng
