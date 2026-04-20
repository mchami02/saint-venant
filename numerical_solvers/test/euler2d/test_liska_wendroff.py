"""Smoke test for Liska-Wendroff 2D Riemann configurations."""

import pytest
import torch

from numerical_solvers.src.euler2d import generate_one, liska_wendroff


@pytest.mark.parametrize("config", [3, 4, 6])
def test_config_runs_without_nan(config):
    nx = ny = 40
    dx = dy = 1.0 / nx
    dt = 0.01; nt = 20
    x = torch.arange(nx, dtype=torch.float64) * dx
    y = torch.arange(ny, dtype=torch.float64) * dy
    rho0, u0, v0, p0 = liska_wendroff(x, y, config=config)

    result = generate_one(
        rho0, u0, v0, p0,
        dx=dx, dy=dy, dt=dt, nt=nt,
        flux_type="hllc", reconstruction="constant", bc_type="extrap",
    )
    assert result["valid"]
    assert torch.isfinite(result["rho"]).all()
    assert (result["rho"] > 0).all()
    assert (result["p"] > 0).all()
