"""Stability tests for Burgers: max_value gating, NaN containment."""

import torch

from numerical_solvers.src.burgers import generate_one


def test_max_value_gating():
    """An IC that the gate flags should yield valid=False."""
    nx, dx, dt, nt = 40, 0.025, 0.01, 20
    # Provide an IC that already exceeds max_value → triggers immediate bailout
    u0 = torch.full((nx,), 100.0, dtype=torch.float64)
    res = generate_one(
        u0, dx=dx, dt=dt, nt=nt,
        bc_type="periodic", reconstruction="constant", max_value=10.0,
    )
    assert res["valid"] is False


def test_no_nan_leak_on_valid_run():
    """A valid run must have zero NaNs in u_hist."""
    nx, dx, dt, nt = 80, 0.0125, 0.002, 50
    x = torch.arange(nx, dtype=torch.float64) * dx
    u0 = torch.sin(2 * torch.pi * x)
    res = generate_one(
        u0, dx=dx, dt=dt, nt=nt,
        bc_type="periodic", reconstruction="weno5",
    )
    assert res["valid"]
    assert torch.isfinite(res["u"]).all()
