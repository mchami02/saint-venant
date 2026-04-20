"""Boundary-condition tests for 2D Euler."""

import torch

from numerical_solvers.src.euler2d.boundary import apply_ghost_cells


def _const_state(ny=3, nx=4):
    rho = torch.full((ny, nx), 1.0, dtype=torch.float64)
    ru = torch.full((ny, nx), 0.5, dtype=torch.float64)
    rv = torch.full((ny, nx), -0.3, dtype=torch.float64)
    E = torch.full((ny, nx), 2.5, dtype=torch.float64)
    return rho, ru, rv, E


def test_extrap_padding_matches_edge_values():
    rho, ru, rv, E = _const_state()
    rho_g, ru_g, rv_g, E_g = apply_ghost_cells(rho, ru, rv, E, "extrap", n_ghost=1)
    assert rho_g.shape == (5, 6)
    assert (rho_g == 1.0).all()
    assert (ru_g == 0.5).all()
    assert (rv_g == -0.3).all()


def test_periodic_padding_wraps():
    rho = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    _, ru, rv, E = _const_state()
    rho_g, _, _, _ = apply_ghost_cells(rho, ru, rv, E, "periodic", n_ghost=1)
    # Left ghost column = rightmost column
    torch.testing.assert_close(rho_g[1:-1, 0], rho[:, -1])
    # Bottom ghost row = top row
    torch.testing.assert_close(rho_g[0, 1:-1], rho[-1])


def test_wall_flips_normal_momentum():
    rho = torch.full((2, 2), 1.0, dtype=torch.float64)
    ru = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    rv = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float64)
    E = torch.full((2, 2), 10.0, dtype=torch.float64)
    rho_g, ru_g, rv_g, E_g = apply_ghost_cells(rho, ru, rv, E, "wall", n_ghost=1)

    # Left ghost column of ru reflects interior[:, 0] and flips sign
    torch.testing.assert_close(ru_g[1:-1, 0], -ru[:, 0])
    # Right ghost column of ru reflects interior[:, -1] and flips sign
    torch.testing.assert_close(ru_g[1:-1, -1], -ru[:, -1])
    # rv should NOT have sign flipped on x-walls: left/right ghost rv matches interior
    torch.testing.assert_close(rv_g[1:-1, 0], rv[:, 0])
    torch.testing.assert_close(rv_g[1:-1, -1], rv[:, -1])

    # Bottom ghost row of rv reflects interior[0] with sign flip
    torch.testing.assert_close(rv_g[0, 1:-1], -rv[0])
    # ru on y-walls is preserved (no sign flip)
    torch.testing.assert_close(ru_g[0, 1:-1], ru[0])
