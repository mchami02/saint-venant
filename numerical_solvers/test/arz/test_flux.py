"""Tests for ARZ numerical fluxes: rusanov and hll."""

import pytest
import torch

from numerical_solvers.arz.flux import hll, rusanov
from numerical_solvers.arz.physics import pressure


@pytest.fixture(params=["rusanov", "hll"], ids=["rusanov", "hll"])
def flux_fn(request):
    return {"rusanov": rusanov, "hll": hll}[request.param]


class TestFluxConsistency:
    """Properties that hold for any consistent numerical flux."""

    def test_identical_states_physical_flux(self, flux_fn):
        """When left == right, flux must equal the physical flux."""
        gamma = 1.0
        rho = torch.tensor([0.5, 0.8])
        v = torch.tensor([0.3, 0.1])
        w = v + pressure(rho, gamma)
        rho_w = rho * w

        f_rho, f_rw = flux_fn(rho, rho_w, rho, rho_w, gamma)
        expected_f_rho = rho * v
        expected_f_rw = rho_w * v
        torch.testing.assert_close(f_rho, expected_f_rho, atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(f_rw, expected_f_rw, atol=1e-7, rtol=1e-7)

    def test_zero_density_zero_flux(self, flux_fn):
        gamma = 1.0
        zeros = torch.zeros(3)
        f_rho, f_rw = flux_fn(zeros, zeros, zeros, zeros, gamma)
        torch.testing.assert_close(f_rho, zeros, atol=1e-10, rtol=0)
        torch.testing.assert_close(f_rw, zeros, atol=1e-10, rtol=0)

    def test_output_shape(self, flux_fn):
        n = 10
        rhoL = torch.rand(n)
        rho_wL = torch.rand(n)
        rhoR = torch.rand(n)
        rho_wR = torch.rand(n)
        f_rho, f_rw = flux_fn(rhoL, rho_wL, rhoR, rho_wR, 1.0)
        assert f_rho.shape == (n,)
        assert f_rw.shape == (n,)


class TestHLLSupersonic:
    def test_all_positive_waves_left_flux(self):
        """When all wave speeds > 0, HLL returns the left physical flux."""
        gamma = 1.0
        # Large positive velocity, small density => all eigenvalues positive
        rhoL = torch.tensor([0.1])
        vL = torch.tensor([5.0])
        wL = vL + pressure(rhoL, gamma)
        rho_wL = rhoL * wL

        rhoR = torch.tensor([0.2])
        vR = torch.tensor([4.0])
        wR = vR + pressure(rhoR, gamma)
        rho_wR = rhoR * wR

        f_rho, f_rw = hll(rhoL, rho_wL, rhoR, rho_wR, gamma)
        expected_f_rho = rhoL * vL
        expected_f_rw = rho_wL * vL
        torch.testing.assert_close(f_rho, expected_f_rho, atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(f_rw, expected_f_rw, atol=1e-7, rtol=1e-7)
