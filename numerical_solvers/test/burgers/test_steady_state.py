"""Steady-state (constant IC) preservation for Burgers."""

import pytest
import torch

from numerical_solvers.src.burgers import generate_one


@pytest.mark.parametrize("flux_type", ["godunov", "rusanov"])
@pytest.mark.parametrize("reconstruction", ["constant", "weno5"])
def test_constant_preserved(flux_type, reconstruction):
    nx, dx, dt, nt = 40, 0.025, 0.005, 30
    u0 = torch.full((nx,), 0.7, dtype=torch.float64)
    res = generate_one(
        u0, dx=dx, dt=dt, nt=nt,
        bc_type="periodic", flux_type=flux_type, reconstruction=reconstruction,
    )
    assert res["valid"]
    torch.testing.assert_close(
        res["u"], torch.full_like(res["u"], 0.7), atol=1e-10, rtol=0,
    )
