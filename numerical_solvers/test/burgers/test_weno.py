"""WENO-5 smoke test for Burgers: smooth sinusoid stays bounded."""

import torch

from numerical_solvers.src.burgers import generate_one


def test_weno5_smooth_sinusoid_stable():
    nx, dx, dt, nt = 100, 0.01, 0.002, 50
    x = torch.arange(nx, dtype=torch.float64) * dx
    u0 = 0.5 * torch.sin(2 * torch.pi * x)

    result = generate_one(
        u0, dx=dx, dt=dt, nt=nt,
        flux_type="godunov", reconstruction="weno5", bc_type="periodic",
    )
    assert result["valid"]
    assert torch.isfinite(result["u"]).all()
    # Max |u| cannot grow above the initial amplitude for Burgers
    assert result["u"].abs().max().item() <= 0.51


def test_weno5_vs_constant_shock():
    """Both schemes converge on the shock speed but WENO5 is sharper."""
    nx, dx, dt, nt = 200, 0.005, 0.002, 100
    x = torch.arange(nx, dtype=torch.float64) * dx
    from numerical_solvers.src.burgers.initial_conditions import riemann

    u0 = riemann(x, u_left=1.0, u_right=0.0, x_split=0.3)
    r_c = generate_one(u0, dx=dx, dt=dt, nt=nt, reconstruction="constant")
    r_w = generate_one(u0, dx=dx, dt=dt, nt=nt, reconstruction="weno5")

    # Transition width: count cells where 0.05 < u < 0.95
    mid_c = ((r_c["u"][-1] > 0.05) & (r_c["u"][-1] < 0.95)).sum().item()
    mid_w = ((r_w["u"][-1] > 0.05) & (r_w["u"][-1] < 0.95)).sum().item()
    # WENO5 should resolve the shock at least as sharply as constant
    assert mid_w <= mid_c
