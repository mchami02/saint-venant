"""Uniform-state preservation for 2D Euler."""

import pytest
import torch

from numerical_solvers.src.euler2d import generate_one


@pytest.mark.parametrize("flux_type", ["hllc", "hll", "rusanov"])
@pytest.mark.parametrize("bc_type", ["extrap", "periodic"])
def test_uniform_state_preserved(flux_type, bc_type):
    nx = ny = 16
    dx = dy = 1.0 / nx
    dt = 0.001; nt = 20
    rho0 = torch.full((ny, nx), 1.2, dtype=torch.float64)
    u0 = torch.full((ny, nx), 0.3, dtype=torch.float64)
    v0 = torch.full((ny, nx), -0.4, dtype=torch.float64)
    p0 = torch.full((ny, nx), 0.8, dtype=torch.float64)

    result = generate_one(
        rho0, u0, v0, p0,
        dx=dx, dy=dy, dt=dt, nt=nt,
        flux_type=flux_type, bc_type=bc_type, reconstruction="constant",
    )
    assert result["valid"]
    torch.testing.assert_close(result["rho"], torch.full_like(result["rho"], 1.2), atol=1e-10, rtol=0)
    torch.testing.assert_close(result["u"], torch.full_like(result["u"], 0.3), atol=1e-10, rtol=0)
    torch.testing.assert_close(result["v"], torch.full_like(result["v"], -0.4), atol=1e-10, rtol=0)
    torch.testing.assert_close(result["p"], torch.full_like(result["p"], 0.8), atol=1e-10, rtol=0)
