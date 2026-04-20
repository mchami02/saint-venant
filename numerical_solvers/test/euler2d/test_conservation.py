"""Conservation tests under periodic BCs for 2D Euler."""

import torch

from numerical_solvers.src.euler2d import generate_one, liska_wendroff


def test_mass_momentum_energy_conserved_periodic():
    nx = ny = 40
    dx = dy = 1.0 / nx
    dt = 0.001; nt = 40
    x = torch.arange(nx, dtype=torch.float64) * dx
    y = torch.arange(ny, dtype=torch.float64) * dy

    # Smooth-ish IC (still discontinuous but small enough jumps to stay valid)
    rho0, u0, v0, p0 = liska_wendroff(x, y, config=3)

    result = generate_one(
        rho0, u0, v0, p0,
        dx=dx, dy=dy, dt=dt, nt=nt,
        flux_type="hllc", reconstruction="constant", bc_type="periodic",
    )
    assert result["valid"]

    # Conserved quantities: mass, rho_u, rho_v, total E
    rho_hist = result["rho"]
    u_hist = result["u"]
    v_hist = result["v"]
    p_hist = result["p"]
    gamma = 1.4
    ke = 0.5 * rho_hist * (u_hist * u_hist + v_hist * v_hist)
    E_hist = p_hist / (gamma - 1.0) + ke

    mass = (rho_hist * dx * dy).sum(dim=(-2, -1))
    mom_x = (rho_hist * u_hist * dx * dy).sum(dim=(-2, -1))
    mom_y = (rho_hist * v_hist * dx * dy).sum(dim=(-2, -1))
    energy = (E_hist * dx * dy).sum(dim=(-2, -1))

    for name, q in (("mass", mass), ("mom_x", mom_x), ("mom_y", mom_y), ("energy", energy)):
        drift = (q - q[0]).abs().max().item()
        assert drift < 1e-10, f"{name} drift {drift:.3e}"
