"""Tests for ARZ physics: pressure, dp_drho, eigenvalues."""

import torch

from numerical_solvers.arz.physics import dp_drho, eigenvalues, pressure


class TestPressure:
    def test_gamma_1(self):
        rho = torch.tensor([0.0, 0.3, 0.5, 1.0])
        torch.testing.assert_close(pressure(rho, 1.0), rho)

    def test_gamma_2(self):
        rho = torch.tensor([0.0, 0.5, 1.0, 2.0])
        torch.testing.assert_close(pressure(rho, 2.0), rho**2)

    def test_zero_density(self):
        rho = torch.tensor([0.0])
        for gamma in [0.5, 1.0, 1.5, 2.0]:
            assert pressure(rho, gamma).item() == 0.0


class TestDpDrho:
    def test_gamma_1(self):
        rho = torch.tensor([0.1, 0.5, 1.0, 2.0])
        torch.testing.assert_close(dp_drho(rho, 1.0), torch.ones_like(rho))

    def test_gamma_2(self):
        rho = torch.tensor([0.1, 0.5, 1.0, 2.0])
        torch.testing.assert_close(dp_drho(rho, 2.0), 2.0 * rho)

    def test_gamma_1_5(self):
        rho = torch.tensor([0.25, 1.0, 4.0])
        expected = 1.5 * rho.pow(0.5)
        torch.testing.assert_close(dp_drho(rho, 1.5), expected)


class TestEigenvalues:
    def test_gamma_1(self):
        rho = torch.tensor([0.5, 1.0])
        v = torch.tensor([0.3, 0.6])
        lam1, lam2 = eigenvalues(rho, v, 1.0)
        # lam1 = v, lam2 = v - rho * 1 = v - rho
        torch.testing.assert_close(lam1, v)
        torch.testing.assert_close(lam2, v - rho)

    def test_ordering(self):
        """For rho > 0 and gamma > 0, lam2 < lam1."""
        rho = torch.tensor([0.1, 0.5, 1.0, 2.0])
        v = torch.tensor([0.5, 0.5, 0.5, 0.5])
        lam1, lam2 = eigenvalues(rho, v, 1.0)
        assert (lam2 < lam1).all()

    def test_zero_density_collapse(self):
        """At rho=0, both eigenvalues equal v."""
        rho = torch.tensor([0.0])
        v = torch.tensor([0.7])
        lam1, lam2 = eigenvalues(rho, v, 1.0)
        torch.testing.assert_close(lam1, lam2)
