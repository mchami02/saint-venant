"""Physics roundtrip + EOS tests for 2D Euler."""

import torch

from numerical_solvers.src.euler2d.physics import (
    conservative_to_primitive,
    pressure_from_conservative,
    primitive_to_conservative,
    sound_speed,
)


def test_prim_cons_roundtrip():
    rho = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
    u = torch.tensor([0.3, -0.4, 1.0], dtype=torch.float64)
    v = torch.tensor([-0.1, 0.8, -0.5], dtype=torch.float64)
    p = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
    gamma = 1.4
    _, ru, rv, E = primitive_to_conservative(rho, u, v, p, gamma)
    rho2, u2, v2, p2 = conservative_to_primitive(rho, ru, rv, E, gamma)
    torch.testing.assert_close(rho2, rho)
    torch.testing.assert_close(u2, u)
    torch.testing.assert_close(v2, v)
    torch.testing.assert_close(p2, p)


def test_pressure_from_conservative_matches_eos():
    rho = torch.tensor([1.0], dtype=torch.float64)
    u = torch.tensor([0.5], dtype=torch.float64)
    v = torch.tensor([0.2], dtype=torch.float64)
    p = torch.tensor([1.5], dtype=torch.float64)
    gamma = 1.4
    _, ru, rv, E = primitive_to_conservative(rho, u, v, p, gamma)
    p_back = pressure_from_conservative(rho, ru, rv, E, gamma)
    torch.testing.assert_close(p_back, p)


def test_sound_speed_positive():
    rho = torch.tensor([0.1, 1.0, 5.0], dtype=torch.float64)
    p = torch.tensor([0.1, 1.0, 3.0], dtype=torch.float64)
    c = sound_speed(rho, p, gamma=1.4)
    assert (c > 0).all()
    expected = torch.sqrt(1.4 * p / rho)
    torch.testing.assert_close(c, expected)
