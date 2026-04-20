"""Mass-like conservation tests for Burgers under periodic BCs."""

import torch

from numerical_solvers.src.burgers import generate_one, riemann


def test_integral_conserved_periodic():
    nx, dx, dt, nt = 100, 0.01, 0.002, 100
    x = torch.arange(nx, dtype=torch.float64) * dx
    u0 = riemann(x, u_left=1.0, u_right=-1.0, x_split=0.5)  # shocked + rarefy

    result = generate_one(
        u0, dx=dx, dt=dt, nt=nt,
        flux_type="godunov", reconstruction="constant", bc_type="periodic",
    )
    assert result["valid"]

    integrals = (result["u"] * dx).sum(dim=-1)
    # Under periodic BCs, ∫u dx is conserved exactly (to FP precision)
    torch.testing.assert_close(integrals, torch.full_like(integrals, integrals[0].item()), atol=1e-10, rtol=0)


def test_integral_conserved_weno5_periodic():
    nx, dx, dt, nt = 100, 0.01, 0.002, 100
    x = torch.arange(nx, dtype=torch.float64) * dx
    # Smooth IC that steepens into a shock
    u0 = torch.sin(2 * torch.pi * x)

    result = generate_one(
        u0, dx=dx, dt=dt, nt=nt,
        flux_type="godunov", reconstruction="weno5", bc_type="periodic",
    )
    assert result["valid"]

    integrals = (result["u"] * dx).sum(dim=-1)
    max_drift = (integrals - integrals[0]).abs().max().item()
    assert max_drift < 1e-10, f"conservation drift {max_drift:.3e} > 1e-10"
