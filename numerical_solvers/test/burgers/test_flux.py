"""Tests for Burgers numerical fluxes: Godunov + entropy fix, Rusanov."""

import torch

from numerical_solvers.src.burgers.flux import godunov, rusanov
from numerical_solvers.src.burgers.physics import flux as phys_flux


class TestGodunov:
    def test_consistency(self):
        # F(u, u) = f(u) for any u
        u = torch.linspace(-2.0, 2.0, 9, dtype=torch.float64)
        F = godunov(u, u)
        torch.testing.assert_close(F, phys_flux(u))

    def test_shock_right_state_smaller_positive(self):
        # uL > uR >= 0: shock moves right, upwind flux is from the left
        uL = torch.tensor([2.0], dtype=torch.float64)
        uR = torch.tensor([1.0], dtype=torch.float64)
        F = godunov(uL, uR)
        # Since f is monotonically increasing on u >= 0 and uL > uR >= 0,
        # max(f(uL), f(uR)) = f(uL).
        torch.testing.assert_close(F, phys_flux(uL))

    def test_shock_mixed_signs(self):
        # uL=1, uR=-1: shock, f(uL)=f(uR)=0.5 → F=0.5
        uL = torch.tensor([1.0], dtype=torch.float64)
        uR = torch.tensor([-1.0], dtype=torch.float64)
        F = godunov(uL, uR)
        torch.testing.assert_close(F, torch.tensor([0.5], dtype=torch.float64))

    def test_rarefaction_non_transonic(self):
        # 0 <= uL < uR: all characteristics move right, F = f(uL)
        uL = torch.tensor([0.5], dtype=torch.float64)
        uR = torch.tensor([1.5], dtype=torch.float64)
        F = godunov(uL, uR)
        torch.testing.assert_close(F, phys_flux(uL))

    def test_transonic_entropy_fix(self):
        # uL < 0 < uR: sonic point in the fan, F = f(0) = 0
        uL = torch.tensor([-1.0, -2.0, -0.5], dtype=torch.float64)
        uR = torch.tensor([1.0, 0.5, 2.0], dtype=torch.float64)
        F = godunov(uL, uR)
        torch.testing.assert_close(F, torch.zeros(3, dtype=torch.float64))

    def test_both_negative_rarefaction(self):
        # uL < uR <= 0: all characteristics move left, F = f(uR)
        uL = torch.tensor([-2.0], dtype=torch.float64)
        uR = torch.tensor([-0.5], dtype=torch.float64)
        F = godunov(uL, uR)
        torch.testing.assert_close(F, phys_flux(uR))


class TestRusanov:
    def test_consistency(self):
        u = torch.linspace(-2.0, 2.0, 9, dtype=torch.float64)
        F = rusanov(u, u)
        torch.testing.assert_close(F, phys_flux(u))

    def test_monotone_dissipation(self):
        # Rusanov ≤ average of physical fluxes when states differ
        uL = torch.tensor([1.0], dtype=torch.float64)
        uR = torch.tensor([-1.0], dtype=torch.float64)
        F = rusanov(uL, uR)
        # At (1, -1), smax = 1, avg = 0.5, jump = -2, so F = 0.5 - 0.5*1*(-2) = 1.5
        torch.testing.assert_close(F, torch.tensor([1.5], dtype=torch.float64))
