"""Tests for Euler physics: EOS, conversions, sound speed."""

import math

import torch

from numerical_solvers.euler.physics import (
    conservative_to_primitive,
    pressure_from_conservative,
    primitive_to_conservative,
    sound_speed,
)


class TestPrimitiveToConservative:
    def test_roundtrip(self):
        """primitive -> conservative -> primitive should be identity."""
        rho = torch.tensor([1.0, 0.5, 2.0], dtype=torch.float64)
        u = torch.tensor([0.3, -1.0, 2.0], dtype=torch.float64)
        p = torch.tensor([1.0, 0.5, 3.0], dtype=torch.float64)
        gamma = 1.4

        rho_c, rho_u, E = primitive_to_conservative(rho, u, p, gamma)
        rho2, u2, p2 = conservative_to_primitive(rho_c, rho_u, E, gamma)

        torch.testing.assert_close(rho2, rho, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(u2, u, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(p2, p, atol=1e-12, rtol=1e-12)

    def test_stationary(self):
        """u=0: rho_u=0 and E = p/(gamma-1)."""
        rho = torch.tensor([1.0], dtype=torch.float64)
        u = torch.tensor([0.0], dtype=torch.float64)
        p = torch.tensor([2.5], dtype=torch.float64)
        gamma = 1.4

        _, rho_u, E = primitive_to_conservative(rho, u, p, gamma)
        assert rho_u.item() == 0.0
        expected_E = p.item() / (gamma - 1.0)
        assert abs(E.item() - expected_E) < 1e-12

    def test_energy_formula(self):
        """E = p/(gamma-1) + 0.5*rho*u^2."""
        rho = torch.tensor([2.0], dtype=torch.float64)
        u = torch.tensor([3.0], dtype=torch.float64)
        p = torch.tensor([1.0], dtype=torch.float64)
        gamma = 1.4

        _, _, E = primitive_to_conservative(rho, u, p, gamma)
        expected = p / (gamma - 1.0) + 0.5 * rho * u**2
        torch.testing.assert_close(E, expected)


class TestSoundSpeed:
    def test_known_value(self):
        """gamma=1.4, rho=1, p=1: c = sqrt(1.4)."""
        rho = torch.tensor([1.0], dtype=torch.float64)
        p = torch.tensor([1.0], dtype=torch.float64)
        c = sound_speed(rho, p, 1.4)
        assert abs(c.item() - math.sqrt(1.4)) < 1e-10

    def test_sod_right_state(self):
        """gamma=1.4, rho=0.125, p=0.1: c = sqrt(1.4*0.1/0.125)."""
        rho = torch.tensor([0.125], dtype=torch.float64)
        p = torch.tensor([0.1], dtype=torch.float64)
        c = sound_speed(rho, p, 1.4)
        expected = math.sqrt(1.4 * 0.1 / 0.125)
        assert abs(c.item() - expected) < 1e-10


class TestPressureFromConservative:
    def test_matches_roundtrip(self):
        rho = torch.tensor([1.0, 2.0], dtype=torch.float64)
        u = torch.tensor([0.5, -1.0], dtype=torch.float64)
        p = torch.tensor([1.0, 3.0], dtype=torch.float64)
        gamma = 1.4

        _, rho_u, E = primitive_to_conservative(rho, u, p, gamma)
        p_recon = pressure_from_conservative(rho, rho_u, E, gamma)
        torch.testing.assert_close(p_recon, p, atol=1e-12, rtol=1e-12)

    def test_stationary_pressure(self):
        """u=0: p = (gamma-1)*E."""
        gamma = 1.4
        rho = torch.tensor([1.0], dtype=torch.float64)
        rho_u = torch.tensor([0.0], dtype=torch.float64)
        E = torch.tensor([2.5], dtype=torch.float64)
        p = pressure_from_conservative(rho, rho_u, E, gamma)
        expected = (gamma - 1.0) * E
        torch.testing.assert_close(p, expected)
