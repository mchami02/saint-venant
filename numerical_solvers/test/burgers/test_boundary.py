"""Boundary condition tests for Burgers."""

import torch

from numerical_solvers.src.burgers import generate_one
from numerical_solvers.src.burgers.boundary import apply_ghost_cells


class TestApplyGhostCells:
    def test_periodic(self):
        u = torch.arange(6, dtype=torch.float64)
        g = apply_ghost_cells(u, "periodic", n_ghost=2)
        # Left ghosts should be last 2 cells, right ghosts first 2
        torch.testing.assert_close(g[:2], u[-2:])
        torch.testing.assert_close(g[-2:], u[:2])
        torch.testing.assert_close(g[2:-2], u)

    def test_extrap(self):
        u = torch.tensor([3.0, 1.0, 2.0, -4.0], dtype=torch.float64)
        g = apply_ghost_cells(u, "extrap", n_ghost=2)
        torch.testing.assert_close(g[:2], torch.tensor([3.0, 3.0], dtype=torch.float64))
        torch.testing.assert_close(g[-2:], torch.tensor([-4.0, -4.0], dtype=torch.float64))

    def test_wall(self):
        u = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        g = apply_ghost_cells(u, "wall", n_ghost=2)
        # Left ghosts: mirror of u[:2] with sign flip → [-2, -1]
        torch.testing.assert_close(g[:2], torch.tensor([-2.0, -1.0], dtype=torch.float64))
        # Right ghosts: mirror of u[-2:] with sign flip → [-4, -3]
        torch.testing.assert_close(g[-2:], torch.tensor([-4.0, -3.0], dtype=torch.float64))


def test_periodic_run_is_translation_invariant():
    """Periodic run should be invariant under initial shift."""
    nx, dx, dt, nt = 40, 0.025, 0.005, 20
    x = torch.arange(nx, dtype=torch.float64) * dx
    u0 = torch.sin(2 * torch.pi * x)
    u0_shift = torch.roll(u0, shifts=5)

    r1 = generate_one(u0, dx=dx, dt=dt, nt=nt, bc_type="periodic", reconstruction="constant")
    r2 = generate_one(u0_shift, dx=dx, dt=dt, nt=nt, bc_type="periodic", reconstruction="constant")

    torch.testing.assert_close(torch.roll(r1["u"], shifts=5, dims=-1), r2["u"], atol=1e-12, rtol=0)


def test_wall_keeps_positive_pulse_bounded():
    """A positive pulse travelling into a wall should remain valid and bounded
    (wall BC prevents the characteristic from escaping by flipping the sign
    in the ghost layer, which acts as a barrier for Burgers)."""
    nx, dx, dt, nt = 200, 0.005, 0.001, 800
    x = torch.arange(nx, dtype=torch.float64) * dx
    # Pulse near right boundary so it reaches the wall within nt*dt
    u0 = 0.5 * torch.exp(-((x - 0.9) ** 2) / 0.02**2)

    result = generate_one(
        u0, dx=dx, dt=dt, nt=nt,
        flux_type="godunov", reconstruction="constant", bc_type="wall",
    )
    assert result["valid"]
    assert torch.isfinite(result["u"]).all()
    # Never exceeds initial max (Burgers is max-principle-preserving + wall
    # does not inject energy)
    assert result["u"].max().item() <= u0.max().item() + 1e-10
