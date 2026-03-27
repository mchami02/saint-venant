"""Tests for Euler numerical fluxes: hllc, hll, rusanov."""

import pytest
import torch

from numerical_solvers.src.euler.flux import hll, hllc, rusanov
from numerical_solvers.src.euler.physics import primitive_to_conservative


@pytest.fixture(params=["hllc", "hll", "rusanov"], ids=["hllc", "hll", "rusanov"])
def flux_fn(request):
    return {"hllc": hllc, "hll": hll, "rusanov": rusanov}[request.param]


GAMMA = 1.4


class TestFluxConsistency:
    """Properties that hold for any consistent numerical flux."""

    def test_identical_states_physical_flux(self, flux_fn):
        """When left == right, flux = physical Euler flux."""
        rho = torch.tensor([1.0, 0.5], dtype=torch.float64)
        u = torch.tensor([0.3, -0.2], dtype=torch.float64)
        p = torch.tensor([1.0, 0.5], dtype=torch.float64)
        _, rho_u, E = primitive_to_conservative(rho, u, p, GAMMA)

        f_rho, f_rho_u, f_E = flux_fn(rho, rho_u, E, rho, rho_u, E, GAMMA)

        expected_f_rho = rho_u
        expected_f_rho_u = rho_u * u + p
        expected_f_E = u * (E + p)

        torch.testing.assert_close(f_rho, expected_f_rho, atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(f_rho_u, expected_f_rho_u, atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(f_E, expected_f_E, atol=1e-8, rtol=1e-8)

    def test_output_shapes(self, flux_fn):
        n = 10
        rho = torch.rand(n, dtype=torch.float64) + 0.1
        u = torch.randn(n, dtype=torch.float64)
        p = torch.rand(n, dtype=torch.float64) + 0.1
        _, rho_u, E = primitive_to_conservative(rho, u, p, GAMMA)
        rhoR = torch.rand(n, dtype=torch.float64) + 0.1
        uR = torch.randn(n, dtype=torch.float64)
        pR = torch.rand(n, dtype=torch.float64) + 0.1
        _, rho_uR, ER = primitive_to_conservative(rhoR, uR, pR, GAMMA)

        f_rho, f_rho_u, f_E = flux_fn(rho, rho_u, E, rhoR, rho_uR, ER, GAMMA)
        assert f_rho.shape == (n,)
        assert f_rho_u.shape == (n,)
        assert f_E.shape == (n,)


class TestRusanovDissipation:
    def test_dissipation_nonzero(self):
        """For different left/right states, Rusanov flux differs from avg of physical fluxes."""
        rhoL = torch.tensor([1.0], dtype=torch.float64)
        uL = torch.tensor([0.0], dtype=torch.float64)
        pL = torch.tensor([1.0], dtype=torch.float64)
        _, rho_uL, EL = primitive_to_conservative(rhoL, uL, pL, GAMMA)

        rhoR = torch.tensor([0.125], dtype=torch.float64)
        uR = torch.tensor([0.0], dtype=torch.float64)
        pR = torch.tensor([0.1], dtype=torch.float64)
        _, rho_uR, ER = primitive_to_conservative(rhoR, uR, pR, GAMMA)

        f_rho, _, _ = rusanov(rhoL, rho_uL, EL, rhoR, rho_uR, ER, GAMMA)
        avg_f_rho = 0.5 * (rho_uL + rho_uR)
        # Rusanov adds dissipation, so it should differ from the average
        assert not torch.allclose(f_rho, avg_f_rho)
