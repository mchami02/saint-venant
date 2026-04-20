"""Analytical-solution checks for Burgers: rarefaction fan u = x/t."""

import torch

from numerical_solvers.src.burgers import generate_one, riemann


def test_centered_rarefaction_fan():
    """uL=-1, uR=1 → centered fan with u = (x - x0)/t for -t < x-x0 < t.

    We verify that inside the fan the numerical solution matches u = x/t
    (measured relative to the initial discontinuity) to within a few dx.
    """
    nx = 400
    dx = 1.0 / nx
    dt = 0.001
    nt = 300
    x = torch.arange(nx, dtype=torch.float64) * dx
    x_split = 0.5
    u0 = riemann(x, u_left=-1.0, u_right=1.0, x_split=x_split)

    result = generate_one(
        u0, dx=dx, dt=dt, nt=nt,
        flux_type="godunov", reconstruction="weno5", bc_type="extrap",
    )
    assert result["valid"]

    t_final = nt * dt
    u_final = result["u"][-1]

    # Interior of the fan: [-t + x_split + eps, t + x_split - eps]
    eps = 5 * dx
    mask = ((x - x_split) > -t_final + eps) & ((x - x_split) < t_final - eps)

    expected = (x - x_split) / t_final
    err = (u_final[mask] - expected[mask]).abs().max().item()
    assert err < 5e-2, f"rarefaction fan error {err:.3e} > 5e-2"


def test_transonic_rarefaction_no_spurious_shock():
    """uL = -0.5, uR = 1.0 is transonic (sonic point at u=0).

    The entropy-admissible solution is a centered fan through 0.
    A non-entropy-fixed Godunov flux would produce a stationary expansion
    shock at x = x_split; ensure no such artefact exists.
    """
    nx = 200
    dx = 1.0 / nx
    dt = 0.001
    nt = 200
    x = torch.arange(nx, dtype=torch.float64) * dx
    x_split = 0.5
    u0 = riemann(x, u_left=-0.5, u_right=1.0, x_split=x_split)

    result = generate_one(
        u0, dx=dx, dt=dt, nt=nt,
        flux_type="godunov", reconstruction="constant", bc_type="extrap",
    )
    assert result["valid"]

    # At x = x_split, t = nt*dt, the exact solution is u = 0.
    u_final = result["u"][-1]
    idx_split = int(x_split / dx)
    u_at_split = u_final[idx_split].item()
    assert abs(u_at_split) < 5e-2, (
        f"u at sonic point = {u_at_split:.4e}, expected ~0 (fan through zero)"
    )
