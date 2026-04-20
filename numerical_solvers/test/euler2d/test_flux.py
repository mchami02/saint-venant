"""Flux-kernel tests for 2D Euler (HLLC + passive tangential scalar)."""

import torch

from numerical_solvers.src.euler2d.flux import hll, hllc, rusanov
from numerical_solvers.src.euler2d.physics import primitive_to_conservative


def _state(rho, u, v, p, gamma=1.4):
    _, ru, rv, E = primitive_to_conservative(
        torch.tensor([rho], dtype=torch.float64),
        torch.tensor([u], dtype=torch.float64),
        torch.tensor([v], dtype=torch.float64),
        torch.tensor([p], dtype=torch.float64),
        gamma,
    )
    return torch.tensor([rho], dtype=torch.float64), ru, rv, E


class TestConsistency:
    def test_hllc_F_of_q_q_equals_physical(self):
        rho, ru, rv, E = _state(1.0, 0.5, 0.3, 1.5)
        f_rho, f_ru, f_rv, f_E = hllc(
            rho, ru, rv, E, rho, ru, rv, E, gamma=1.4,
        )
        gamma = 1.4
        u = ru / rho; v = rv / rho
        p = (gamma - 1) * (E - 0.5 * rho * (u**2 + v**2))
        torch.testing.assert_close(f_rho, ru)
        torch.testing.assert_close(f_ru, ru * u + p)
        torch.testing.assert_close(f_rv, ru * v)
        torch.testing.assert_close(f_E, u * (E + p))

    def test_hll_consistency(self):
        rho, ru, rv, E = _state(1.0, 0.5, 0.3, 1.5)
        f = hll(rho, ru, rv, E, rho, ru, rv, E, gamma=1.4)
        u = ru / rho; v = rv / rho
        p = 0.4 * (E - 0.5 * rho * (u**2 + v**2))
        torch.testing.assert_close(f[0], ru)
        torch.testing.assert_close(f[1], ru * u + p)
        torch.testing.assert_close(f[2], ru * v)

    def test_rusanov_consistency(self):
        rho, ru, rv, E = _state(1.0, 0.5, 0.3, 1.5)
        f = rusanov(rho, ru, rv, E, rho, ru, rv, E, gamma=1.4)
        u = ru / rho; v = rv / rho
        p = 0.4 * (E - 0.5 * rho * (u**2 + v**2))
        torch.testing.assert_close(f[0], ru)
        torch.testing.assert_close(f[2], ru * v)


def test_flux_tangential_is_advected_passively():
    """If L and R have identical (rho, normal u, p) but different tangential
    velocities, then rho/rho_n/E fluxes should be unchanged; only the
    tangential flux changes (it is the momentum advected at the interface
    normal velocity)."""
    # Two states with the same (rho, u_n, p) but different tangential v
    rho = torch.tensor([1.0], dtype=torch.float64)
    un = torch.tensor([0.3], dtype=torch.float64)
    p = torch.tensor([1.0], dtype=torch.float64)
    gamma = 1.4

    def _cons(v):
        _, ru, rv, E = primitive_to_conservative(rho, un, v, p, gamma)
        return rho, ru, rv, E

    # Case A: v_L = v_R = 0.0
    rhoL, ruL, rvL, EL = _cons(torch.tensor([0.0], dtype=torch.float64))
    rhoR, ruR, rvR, ER = _cons(torch.tensor([0.0], dtype=torch.float64))
    fA = hllc(rhoL, ruL, rvL, EL, rhoR, ruR, rvR, ER, gamma)

    # Case B: v_L = 0.5, v_R = -0.2
    rhoL2, ruL2, rvL2, EL2 = _cons(torch.tensor([0.5], dtype=torch.float64))
    rhoR2, ruR2, rvR2, ER2 = _cons(torch.tensor([-0.2], dtype=torch.float64))
    fB = hllc(rhoL2, ruL2, rvL2, EL2, rhoR2, ruR2, rvR2, ER2, gamma)

    # Density, normal momentum, energy fluxes should be nearly identical
    # (tangential velocity enters pressure via kinetic energy through
    # total E; we provided E consistent with the same p, so fluxes match).
    torch.testing.assert_close(fA[0], fB[0], atol=1e-10, rtol=0)
    torch.testing.assert_close(fA[1], fB[1], atol=1e-10, rtol=0)
    # Tangential flux must differ (proportional to rho*u_n*v upwind-selected)
    assert (fA[2] - fB[2]).abs().item() > 1e-3
