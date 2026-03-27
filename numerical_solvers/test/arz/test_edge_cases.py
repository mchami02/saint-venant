"""Comprehensive edge case tests for the ARZ solver.

Covers extreme values (density, velocity, gamma), grid resolution limits,
discontinuity placement, boundary condition edge cases, the w=v+p(rho)
invariant, many-piece ICs, and generate_n corner cases.
"""

import math

import pytest
import torch

from numerical_solvers.src.arz import generate_n, generate_one
from numerical_solvers.src.arz.boundary import apply_ghost_cells
from numerical_solvers.src.arz.flux import hll, rusanov
from numerical_solvers.src.arz.initial_conditions import (
    from_steps,
    random_piecewise,
    riemann,
    three_region,
)
from numerical_solvers.src.arz.physics import dp_drho, eigenvalues, pressure
from numerical_solvers.src.arz.weno import weno5_reconstruct


# ====================================================================
# Fixtures
# ====================================================================

@pytest.fixture(
    params=[
        ("rusanov", "constant"),
        ("hll", "constant"),
        ("rusanov", "weno5"),
        ("hll", "weno5"),
    ],
    ids=["rusanov-const", "hll-const", "rusanov-weno5", "hll-weno5"],
)
def flux_recon(request):
    return request.param


@pytest.fixture(params=["rusanov", "hll"], ids=["rusanov", "hll"])
def flux_type(request):
    return request.param


@pytest.fixture(params=["constant", "weno5"], ids=["const", "weno5"])
def reconstruction(request):
    return request.param


# ====================================================================
# 1. Extreme density values
# ====================================================================

class TestExtremeDensityValues:
    """Edge cases in density values."""

    def test_near_zero_density(self):
        nx = 32
        rho0 = torch.full((nx,), 1e-8)
        v0 = torch.full((nx,), 0.1)
        result = generate_one(rho0, v0, dx=0.05, dt=0.005, nt=20, reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_very_high_density(self):
        nx = 32
        rho0 = torch.full((nx,), 5.0)
        v0 = torch.full((nx,), 0.1)
        result = generate_one(rho0, v0, dx=0.05, dt=0.005, nt=20, reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_mixed_near_zero_and_high(self):
        nx = 32
        x = torch.arange(nx, dtype=torch.float32) * 0.05
        rho0, v0 = riemann(x, rho_left=1e-6, rho_right=2.0, v0=0.1)
        result = generate_one(rho0, v0, dx=0.05, dt=0.002, nt=20, reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_density_exactly_zero_everywhere(self):
        """Zero density everywhere -- vacuum state, should stay zero."""
        nx = 32
        rho0 = torch.zeros(nx)
        v0 = torch.zeros(nx)
        result = generate_one(
            rho0, v0, dx=0.05, dt=0.005, nt=20,
            bc_type="periodic", reconstruction="constant",
        )
        assert result["valid"]
        torch.testing.assert_close(result["rho"], torch.zeros(21, nx), atol=1e-10, rtol=0)

    def test_density_epsilon_above_zero(self):
        """Machine-epsilon density should not blow up."""
        nx = 32
        rho0 = torch.full((nx,), torch.finfo(torch.float32).tiny)
        v0 = torch.full((nx,), 0.1)
        result = generate_one(
            rho0, v0, dx=0.05, dt=0.005, nt=10,
            reconstruction="constant", flux_type="rusanov",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()
        assert torch.isfinite(result["v"]).all()

    def test_density_large_ratio_discontinuity(self):
        """1000:1 density ratio across a jump."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float32) * 0.025
        rho0, v0 = riemann(x, rho_left=1.0, rho_right=0.001, v0=0.05)
        result = generate_one(
            rho0, v0, dx=0.025, dt=0.001, nt=30,
            reconstruction="constant", flux_type="hll",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_density_one_cell_spike(self):
        """A single-cell spike in density amid uniform background."""
        nx = 40
        rho0 = torch.full((nx,), 0.2)
        rho0[nx // 2] = 3.0
        v0 = torch.full((nx,), 0.2)
        result = generate_one(
            rho0, v0, dx=0.025, dt=0.002, nt=30,
            reconstruction="constant",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_density_one_cell_dip(self):
        """A single-cell dip in density amid uniform background."""
        nx = 40
        rho0 = torch.full((nx,), 0.8)
        rho0[nx // 2] = 1e-6
        v0 = torch.full((nx,), 0.2)
        result = generate_one(
            rho0, v0, dx=0.025, dt=0.002, nt=30,
            reconstruction="constant",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_alternating_zero_nonzero_density(self):
        """Checkerboard: cells alternate between 0 and 0.5 density."""
        nx = 40
        rho0 = torch.zeros(nx)
        rho0[::2] = 0.5
        v0 = torch.full((nx,), 0.1)
        result = generate_one(
            rho0, v0, dx=0.025, dt=0.002, nt=20,
            reconstruction="constant",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()


# ====================================================================
# 2. Extreme velocity values
# ====================================================================

class TestExtremeVelocityValues:
    """Edge cases in velocity values."""

    def test_zero_velocity(self):
        nx = 32
        rho0 = torch.full((nx,), 0.5)
        v0 = torch.zeros(nx)
        result = generate_one(rho0, v0, dx=0.05, dt=0.005, nt=20, reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_negative_velocity(self):
        """Negative velocity (backward-moving traffic)."""
        nx = 32
        rho0 = torch.full((nx,), 0.5)
        v0 = torch.full((nx,), -0.5)
        result = generate_one(rho0, v0, dx=0.05, dt=0.002, nt=20, reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_large_velocity(self):
        nx = 32
        rho0 = torch.full((nx,), 0.3)
        v0 = torch.full((nx,), 5.0)
        result = generate_one(rho0, v0, dx=0.05, dt=0.001, nt=20, reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_velocity_discontinuity_uniform_density(self):
        """Jump in velocity only, uniform density."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float32) * 0.05
        rho0, v0 = from_steps(
            x,
            rho_steps=[(2.0, 0.5)],
            v_steps=[(1.0, 0.8), (2.0, 0.1)],
        )
        result = generate_one(rho0, v0, dx=0.05, dt=0.002, nt=30, reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()
        assert torch.isfinite(result["v"]).all()

    def test_large_negative_velocity(self):
        """Large negative velocity everywhere."""
        nx = 32
        rho0 = torch.full((nx,), 0.3)
        v0 = torch.full((nx,), -5.0)
        result = generate_one(
            rho0, v0, dx=0.05, dt=0.001, nt=20, reconstruction="constant",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_mixed_positive_negative_velocity(self):
        """Half domain positive, half domain negative velocity."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float32) * 0.05
        rho0, v0 = from_steps(
            x,
            rho_steps=[(2.0, 0.5)],
            v_steps=[(1.0, 0.5), (2.0, -0.5)],
        )
        result = generate_one(
            rho0, v0, dx=0.05, dt=0.002, nt=30, reconstruction="constant",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_converging_velocities_compression(self):
        """Opposing velocities that push traffic inward -- strong compression."""
        nx = 60
        x = torch.arange(nx, dtype=torch.float32) * 0.025
        rho0, v0 = from_steps(
            x,
            rho_steps=[(2.0, 0.5)],
            v_steps=[(0.75, 0.8), (2.0, -0.3)],
        )
        result = generate_one(
            rho0, v0, dx=0.025, dt=0.001, nt=40,
            reconstruction="constant", flux_type="hll",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_zero_velocity_with_density_discontinuity(self):
        """Density jump but zero velocity everywhere: pure contact + backward wave."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float32) * 0.025
        rho0, v0 = riemann(x, rho_left=0.8, rho_right=0.2, v0=0.0)
        result = generate_one(
            rho0, v0, dx=0.025, dt=0.002, nt=30, reconstruction="constant",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_velocity_equal_to_max_wave_speed(self):
        """v chosen so that one eigenvalue is exactly zero."""
        # lam2 = v - rho * dp_drho = 0 => v = rho (for gamma=1)
        nx = 32
        rho0 = torch.full((nx,), 0.5)
        v0 = torch.full((nx,), 0.5)  # lam2 = 0.5 - 0.5*1 = 0
        result = generate_one(
            rho0, v0, dx=0.05, dt=0.005, nt=20,
            bc_type="periodic", reconstruction="constant",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()


# ====================================================================
# 3. Gamma (pressure exponent) variations
# ====================================================================

class TestGammaVariations:
    """Test different gamma (pressure exponent) values."""

    @pytest.mark.parametrize("gamma", [0.5, 1.0, 1.5, 2.0, 3.0])
    def test_various_gamma_stable(self, gamma):
        nx = 32
        x = torch.arange(nx, dtype=torch.float32) * 0.05
        rho0, v0 = riemann(x, rho_left=0.5, rho_right=0.3, v0=0.2)
        result = generate_one(
            rho0, v0, dx=0.05, dt=0.002, nt=20, gamma=gamma,
            reconstruction="constant", flux_type="hll",
        )
        assert result["valid"], f"Failed for gamma={gamma}"
        assert torch.isfinite(result["rho"]).all()

    @pytest.mark.parametrize("gamma", [0.5, 1.0, 1.5, 2.0])
    def test_various_gamma_conservation(self, gamma):
        """Mass conservation with periodic BC for various gamma."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float32) * 0.025
        rho0, v0 = riemann(x, rho_left=0.6, rho_right=0.3, v0=0.2)
        result = generate_one(
            rho0, v0, dx=0.025, dt=0.002, nt=30, gamma=gamma,
            bc_type="periodic", reconstruction="constant",
        )
        mass = result["rho"].sum(dim=-1) * 0.025
        torch.testing.assert_close(mass, mass[0].expand_as(mass), atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0])
    def test_steady_state_various_gamma(self, gamma):
        """Uniform state preserved for any gamma."""
        nx = 32
        rho0 = torch.full((nx,), 0.5)
        v0 = torch.full((nx,), 0.3)
        result = generate_one(
            rho0, v0, dx=0.05, dt=0.005, nt=20, gamma=gamma,
            bc_type="periodic", reconstruction="constant",
        )
        torch.testing.assert_close(result["rho"][-1], rho0, atol=1e-5, rtol=1e-5)

    def test_very_small_gamma(self):
        """gamma close to zero: pressure approaches 1 for all rho>0."""
        nx = 32
        x = torch.arange(nx, dtype=torch.float32) * 0.05
        rho0, v0 = riemann(x, rho_left=0.6, rho_right=0.3, v0=0.2)
        result = generate_one(
            rho0, v0, dx=0.05, dt=0.002, nt=20, gamma=0.1,
            reconstruction="constant", flux_type="hll",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_large_gamma_stable(self):
        """Large gamma: stronger nonlinearity but should remain stable."""
        nx = 32
        x = torch.arange(nx, dtype=torch.float32) * 0.05
        rho0, v0 = riemann(x, rho_left=0.4, rho_right=0.3, v0=0.1)
        result = generate_one(
            rho0, v0, dx=0.05, dt=0.001, nt=20, gamma=5.0,
            reconstruction="constant", flux_type="hll",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    @pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0])
    def test_gamma_with_weno5(self, gamma):
        """WENO5 reconstruction for various gamma values."""
        nx = 32
        x = torch.arange(nx, dtype=torch.float32) * 0.05
        rho0, v0 = riemann(x, rho_left=0.5, rho_right=0.3, v0=0.2)
        result = generate_one(
            rho0, v0, dx=0.05, dt=0.002, nt=20, gamma=gamma,
            reconstruction="weno5", flux_type="hll",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()


# ====================================================================
# 4. Grid resolution
# ====================================================================

class TestGridResolution:
    """Test across different grid sizes."""

    @pytest.mark.parametrize("nx", [8, 16, 32, 64])
    def test_varying_nx(self, nx):
        dx = 1.0 / nx
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = riemann(x, rho_left=0.6, rho_right=0.3)
        result = generate_one(rho0, v0, dx=dx, dt=0.002, nt=20, reconstruction="constant")
        assert result["rho"].shape == (21, nx)
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_very_few_cells(self):
        """Minimal grid that can work with constant reconstruction."""
        nx = 4
        rho0 = torch.tensor([0.8, 0.8, 0.2, 0.2])
        v0 = torch.full((nx,), 0.1)
        result = generate_one(rho0, v0, dx=0.25, dt=0.01, nt=10, reconstruction="constant")
        assert result["valid"]

    def test_weno5_minimum_cells(self):
        """WENO-5 needs at least nx > 8 (for 4 ghost cells each side)."""
        nx = 16
        x = torch.arange(nx, dtype=torch.float32) * 0.1
        rho0, v0 = riemann(x)
        result = generate_one(rho0, v0, dx=0.1, dt=0.005, nt=10, reconstruction="weno5")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_two_cells_constant(self):
        """Absolute minimum: 2 cells with constant reconstruction."""
        nx = 2
        rho0 = torch.tensor([0.8, 0.2])
        v0 = torch.full((nx,), 0.1)
        result = generate_one(
            rho0, v0, dx=0.5, dt=0.02, nt=5,
            reconstruction="constant", bc_type="periodic",
        )
        assert result["valid"]
        assert result["rho"].shape == (6, 2)

    def test_three_cells_constant(self):
        """Three cells with constant reconstruction."""
        nx = 3
        rho0 = torch.tensor([0.8, 0.5, 0.2])
        v0 = torch.full((nx,), 0.1)
        result = generate_one(
            rho0, v0, dx=0.5, dt=0.02, nt=5,
            reconstruction="constant",
        )
        assert result["valid"]
        assert result["rho"].shape == (6, 3)

    @pytest.mark.parametrize("nx", [128, 200])
    def test_large_grid(self, nx):
        """Larger grid sizes should also work."""
        dx = 2.0 / nx
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = riemann(x, rho_left=0.6, rho_right=0.3)
        result = generate_one(
            rho0, v0, dx=dx, dt=0.001, nt=10,
            reconstruction="constant",
        )
        assert result["valid"]
        assert result["rho"].shape == (11, nx)

    def test_single_time_step(self):
        """nt=1: only one step forward."""
        nx = 20
        x = torch.arange(nx, dtype=torch.float32) * 0.05
        rho0, v0 = riemann(x)
        result = generate_one(rho0, v0, dx=0.05, dt=0.005, nt=1, reconstruction="constant")
        assert result["valid"]
        assert result["rho"].shape == (2, nx)

    def test_zero_time_steps(self):
        """nt=0: should just return the initial condition."""
        nx = 20
        rho0 = torch.full((nx,), 0.5)
        v0 = torch.full((nx,), 0.3)
        result = generate_one(rho0, v0, dx=0.05, dt=0.005, nt=0, reconstruction="constant")
        assert result["valid"]
        assert result["rho"].shape == (1, nx)
        torch.testing.assert_close(result["rho"][0], rho0)

    def test_many_time_steps(self):
        """Many time steps with small dt."""
        nx = 32
        x = torch.arange(nx, dtype=torch.float32) * 0.05
        rho0, v0 = riemann(x, rho_left=0.5, rho_right=0.3, v0=0.2)
        result = generate_one(
            rho0, v0, dx=0.05, dt=0.001, nt=500,
            reconstruction="constant", bc_type="periodic",
        )
        assert result["valid"]
        assert result["rho"].shape == (501, nx)


# ====================================================================
# 5. Discontinuity positions
# ====================================================================

class TestDiscontinuityPositions:
    """Test shocks at various positions."""

    @pytest.mark.parametrize("x_split", [0.05, 0.25, 0.5, 0.75, 0.95])
    def test_split_positions(self, x_split):
        nx = 40
        x = torch.arange(nx, dtype=torch.float32) * 0.025
        rho0, v0 = riemann(x, rho_left=0.7, rho_right=0.2, x_split=x_split)
        result = generate_one(rho0, v0, dx=0.025, dt=0.002, nt=20, reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_discontinuity_at_first_cell(self):
        """Jump between cell 0 and cell 1."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float32) * 0.025
        rho0, v0 = riemann(x, rho_left=0.8, rho_right=0.2, x_split=0.025)
        result = generate_one(
            rho0, v0, dx=0.025, dt=0.002, nt=20, reconstruction="constant",
        )
        assert result["valid"]

    def test_discontinuity_at_last_cell(self):
        """Jump between second-to-last and last cell."""
        nx = 40
        dx = 0.025
        x = torch.arange(nx, dtype=torch.float32) * dx
        x_split = x[-2].item() + dx / 2
        rho0, v0 = riemann(x, rho_left=0.8, rho_right=0.2, x_split=x_split)
        result = generate_one(
            rho0, v0, dx=dx, dt=0.002, nt=20, reconstruction="constant",
        )
        assert result["valid"]

    def test_two_close_discontinuities(self):
        """Two jumps separated by only a few cells."""
        nx = 60
        x = torch.arange(nx, dtype=torch.float32) * 0.025
        rho0, v0 = three_region(
            x, rho_left=0.8, rho_mid=0.2, rho_right=0.6,
            v0=0.2, x1=0.5, x2=0.55,
        )
        result = generate_one(
            rho0, v0, dx=0.025, dt=0.002, nt=30, reconstruction="constant",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()


# ====================================================================
# 6. Boundary condition edge cases
# ====================================================================

class TestBCEdgeCases:
    """Boundary condition edge cases."""

    def test_dirichlet_mismatch_with_ic(self):
        """BC values differ from IC at boundaries -- should still work."""
        nx = 32
        rho0 = torch.full((nx,), 0.5)
        v0 = torch.full((nx,), 0.3)
        result = generate_one(
            rho0, v0, dx=0.05, dt=0.005, nt=20, gamma=1.0,
            bc_type="dirichlet", bc_left=(0.9, 0.1), bc_right=(0.1, 0.9),
            reconstruction="constant",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_inflow_outflow_with_custom_left(self):
        nx = 32
        x = torch.arange(nx, dtype=torch.float32) * 0.05
        rho0, v0 = riemann(x)
        result = generate_one(
            rho0, v0, dx=0.05, dt=0.005, nt=20, gamma=1.0,
            bc_type="inflow_outflow", bc_left=(0.7, 0.5),
            reconstruction="constant",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_time_varying_inflow_default_fallback(self):
        """Time-varying inflow without callable (uses sinusoidal fallback)."""
        nx = 32
        rho0 = torch.full((nx,), 0.5)
        v0 = torch.full((nx,), 0.3)
        result = generate_one(
            rho0, v0, dx=0.05, dt=0.005, nt=20, gamma=1.0,
            bc_type="time_varying_inflow", reconstruction="constant",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_dirichlet_zero_density_boundary(self):
        """Dirichlet BC with zero density at the boundary."""
        nx = 32
        rho0 = torch.full((nx,), 0.5)
        v0 = torch.full((nx,), 0.3)
        result = generate_one(
            rho0, v0, dx=0.05, dt=0.005, nt=20,
            bc_type="dirichlet", bc_left=(0.0, 0.0), bc_right=(0.0, 0.0),
            reconstruction="constant",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_dirichlet_high_density_boundary(self):
        """Dirichlet BC with high density at both boundaries."""
        nx = 32
        rho0 = torch.full((nx,), 0.5)
        v0 = torch.full((nx,), 0.3)
        result = generate_one(
            rho0, v0, dx=0.05, dt=0.002, nt=20,
            bc_type="dirichlet", bc_left=(3.0, 0.1), bc_right=(3.0, 0.1),
            reconstruction="constant",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_dirichlet_asymmetric(self):
        """Dirichlet BC with very different left and right values."""
        nx = 32
        rho0 = torch.full((nx,), 0.5)
        v0 = torch.full((nx,), 0.3)
        result = generate_one(
            rho0, v0, dx=0.05, dt=0.002, nt=20,
            bc_type="dirichlet", bc_left=(0.01, 0.9), bc_right=(2.0, 0.01),
            reconstruction="constant",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_time_varying_inflow_with_callable(self):
        """Time-varying inflow with a custom callable."""
        nx = 32
        rho0 = torch.full((nx,), 0.5)
        v0 = torch.full((nx,), 0.3)

        def bc_fn(t):
            return (0.5 + 0.2 * math.sin(2 * math.pi * t), 0.3)

        result = generate_one(
            rho0, v0, dx=0.05, dt=0.005, nt=20,
            bc_type="time_varying_inflow", bc_left_time=bc_fn,
            reconstruction="constant",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_time_varying_inflow_rapidly_oscillating(self):
        """High-frequency oscillating inflow."""
        nx = 32
        rho0 = torch.full((nx,), 0.5)
        v0 = torch.full((nx,), 0.3)

        def bc_fn(t):
            return (
                0.5 + 0.3 * math.sin(50.0 * math.pi * t),
                0.3 + 0.1 * math.cos(50.0 * math.pi * t),
            )

        result = generate_one(
            rho0, v0, dx=0.05, dt=0.005, nt=20,
            bc_type="time_varying_inflow", bc_left_time=bc_fn,
            reconstruction="constant",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_inflow_outflow_default_left(self):
        """Inflow-outflow with no explicit bc_left -- uses default (0.5, 1.0)."""
        nx = 32
        rho0 = torch.full((nx,), 0.5)
        v0 = torch.full((nx,), 0.3)
        result = generate_one(
            rho0, v0, dx=0.05, dt=0.005, nt=20,
            bc_type="inflow_outflow", reconstruction="constant",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    @pytest.mark.parametrize(
        "bc_type", ["periodic", "zero_gradient", "dirichlet", "inflow_outflow"],
    )
    def test_all_bc_types_run(self, bc_type):
        """Every BC type should at least produce a valid result on a simple IC."""
        nx = 32
        x = torch.arange(nx, dtype=torch.float32) * 0.05
        rho0, v0 = riemann(x, rho_left=0.5, rho_right=0.3, v0=0.2)
        kwargs = dict(
            dx=0.05, dt=0.002, nt=15,
            bc_type=bc_type, reconstruction="constant",
        )
        if bc_type == "dirichlet":
            kwargs["bc_left"] = (0.5, 0.2)
            kwargs["bc_right"] = (0.3, 0.2)
        result = generate_one(rho0, v0, **kwargs)
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_periodic_bc_symmetry(self):
        """Symmetric IC with periodic BC should yield symmetric solution."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float32) * 0.025
        mid = nx // 2
        rho0 = torch.cat([
            torch.full((mid,), 0.3),
            torch.full((nx - mid,), 0.7),
        ])
        # Make it symmetric about midpoint
        rho0_sym = torch.cat([rho0[:mid], rho0[:mid].flip(0)])
        v0 = torch.full((nx,), 0.2)
        result = generate_one(
            rho0_sym, v0, dx=0.025, dt=0.002, nt=10,
            bc_type="periodic", reconstruction="constant",
        )
        assert result["valid"]
        # At each time step, density should be approximately symmetric
        for t_idx in range(result["rho"].shape[0]):
            rho_t = result["rho"][t_idx]
            torch.testing.assert_close(
                rho_t, rho_t.flip(0), atol=1e-5, rtol=1e-5,
            )


# ====================================================================
# 7. w = v + p(rho) invariant
# ====================================================================

class TestWRelationship:
    """Verify w = v + pressure(rho, gamma) is maintained at every time step."""

    @pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("reconstruction", ["constant", "weno5"])
    def test_w_equals_v_plus_pressure_all_times(self, gamma, reconstruction):
        nx = 32
        x = torch.arange(nx, dtype=torch.float32) * 0.05
        rho0, v0 = riemann(x, rho_left=0.6, rho_right=0.3)
        result = generate_one(
            rho0, v0, dx=0.05, dt=0.002, nt=20, gamma=gamma,
            reconstruction=reconstruction,
        )
        expected_w = result["v"] + pressure(result["rho"], gamma)
        torch.testing.assert_close(result["w"], expected_w, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("bc_type", ["periodic", "zero_gradient", "dirichlet"])
    def test_w_relationship_all_bc_types(self, bc_type):
        """w = v + p(rho) should hold regardless of boundary conditions."""
        nx = 32
        x = torch.arange(nx, dtype=torch.float32) * 0.05
        rho0, v0 = riemann(x, rho_left=0.6, rho_right=0.3, v0=0.2)
        gamma = 1.0
        kwargs = dict(
            dx=0.05, dt=0.002, nt=20, gamma=gamma,
            bc_type=bc_type, reconstruction="constant",
        )
        if bc_type == "dirichlet":
            kwargs["bc_left"] = (0.6, 0.2)
            kwargs["bc_right"] = (0.3, 0.2)
        result = generate_one(rho0, v0, **kwargs)
        expected_w = result["v"] + pressure(result["rho"], gamma)
        torch.testing.assert_close(result["w"], expected_w, atol=1e-5, rtol=1e-5)

    def test_w_relationship_vacuum(self):
        """w relationship at zero density: both w and v should be zero."""
        nx = 20
        rho0 = torch.zeros(nx)
        v0 = torch.zeros(nx)
        result = generate_one(
            rho0, v0, dx=0.05, dt=0.005, nt=10, gamma=1.0,
            bc_type="periodic", reconstruction="constant",
        )
        # At zero density, v = w - p(0) = w - 0 = w, and both should be 0
        torch.testing.assert_close(result["w"], result["v"], atol=1e-10, rtol=0)


# ====================================================================
# 8. Many piecewise-constant pieces
# ====================================================================

class TestManyPieces:
    """Test with many piecewise-constant regions."""

    def test_10_pieces(self):
        nx = 80
        x = torch.arange(nx, dtype=torch.float32) * 0.02
        rng = torch.Generator().manual_seed(42)
        rho0, v0, _ = random_piecewise(x, 10, rng)
        result = generate_one(rho0, v0, dx=0.02, dt=0.002, nt=50, reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_20_pieces_weno5(self):
        nx = 100
        x = torch.arange(nx, dtype=torch.float32) * 0.015
        rng = torch.Generator().manual_seed(42)
        rho0, v0, _ = random_piecewise(x, 20, rng)
        result = generate_one(rho0, v0, dx=0.015, dt=0.001, nt=50, reconstruction="weno5")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_2_pieces_minimum(self):
        """Minimum number of pieces: 2 regions (one breakpoint)."""
        nx = 32
        x = torch.arange(nx, dtype=torch.float32) * 0.05
        rng = torch.Generator().manual_seed(0)
        rho0, v0, ic_params = random_piecewise(x, 2, rng)
        assert len(ic_params["rho_ks"]) == 2
        result = generate_one(rho0, v0, dx=0.05, dt=0.005, nt=20, reconstruction="constant")
        assert result["valid"]

    def test_1_piece_uniform(self):
        """Single piece: uniform IC."""
        nx = 32
        x = torch.arange(nx, dtype=torch.float32) * 0.05
        rng = torch.Generator().manual_seed(0)
        rho0, v0, ic_params = random_piecewise(x, 1, rng)
        assert len(ic_params["rho_ks"]) == 1
        # Should be uniform
        assert rho0.unique().shape[0] == 1
        assert v0.unique().shape[0] == 1

    def test_many_pieces_different_seeds(self):
        """Different seeds should produce different ICs."""
        nx = 60
        x = torch.arange(nx, dtype=torch.float32) * 0.025
        results = []
        for seed in range(5):
            rng = torch.Generator().manual_seed(seed)
            rho0, _, _ = random_piecewise(x, 5, rng)
            results.append(rho0)
        # At least some pairs should differ
        different_count = 0
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                if not torch.allclose(results[i], results[j]):
                    different_count += 1
        assert different_count > 0, "All seeds produced the same IC"


# ====================================================================
# 9. generate_n edge cases
# ====================================================================

class TestGenerateNEdgeCases:
    """Edge cases for batch generation."""

    def test_single_sample_k1(self):
        result = generate_n(
            1, 1, nx=20, dx=0.05, dt=0.005, nt=10, seed=42,
            show_progress=False, reconstruction="constant",
        )
        assert result["rho"].shape == (1, 11, 20)

    def test_many_samples(self):
        result = generate_n(
            10, 2, nx=20, dx=0.05, dt=0.005, nt=10, seed=42,
            show_progress=False, reconstruction="constant",
        )
        assert result["rho"].shape == (10, 11, 20)
        assert torch.isfinite(result["rho"]).all()

    def test_custom_ranges(self):
        result = generate_n(
            3, 3, nx=20, dx=0.05, dt=0.005, nt=10, seed=42,
            show_progress=False, rho_range=(0.3, 0.7), v_range=(0.1, 0.4),
            reconstruction="constant",
        )
        for i in range(3):
            for v in result["ic_rho_ks"][i]:
                assert 0.3 <= v <= 0.7
            for v in result["ic_v_ks"][i]:
                assert 0.1 <= v <= 0.4

    def test_generate_n_reproducible(self):
        """Same seed -> same results."""
        kwargs = dict(
            nx=20, dx=0.05, dt=0.005, nt=10, seed=99,
            show_progress=False, reconstruction="constant",
        )
        r1 = generate_n(3, 2, **kwargs)
        r2 = generate_n(3, 2, **kwargs)
        torch.testing.assert_close(r1["rho"], r2["rho"])
        torch.testing.assert_close(r1["v"], r2["v"])
        torch.testing.assert_close(r1["w"], r2["w"])

    def test_generate_n_different_seeds_differ(self):
        """Different seeds should produce different samples."""
        r1 = generate_n(
            2, 2, nx=20, dx=0.05, dt=0.005, nt=10, seed=1,
            show_progress=False, reconstruction="constant",
        )
        r2 = generate_n(
            2, 2, nx=20, dx=0.05, dt=0.005, nt=10, seed=2,
            show_progress=False, reconstruction="constant",
        )
        assert not torch.allclose(r1["rho"], r2["rho"])

    def test_generate_n_hll_and_rusanov(self):
        """generate_n works with both flux types."""
        for ft in ["hll", "rusanov"]:
            result = generate_n(
                2, 2, nx=20, dx=0.05, dt=0.005, nt=10, seed=42,
                show_progress=False, reconstruction="constant", flux_type=ft,
            )
            assert result["rho"].shape == (2, 11, 20)
            assert torch.isfinite(result["rho"]).all()

    def test_generate_n_weno5(self):
        """generate_n with WENO5 reconstruction."""
        result = generate_n(
            2, 2, nx=20, dx=0.05, dt=0.002, nt=10, seed=42,
            show_progress=False, reconstruction="weno5",
        )
        assert result["rho"].shape == (2, 11, 20)
        assert torch.isfinite(result["rho"]).all()

    def test_generate_n_periodic(self):
        """generate_n with periodic BCs."""
        result = generate_n(
            2, 2, nx=20, dx=0.05, dt=0.005, nt=10, seed=42,
            show_progress=False, reconstruction="constant", bc_type="periodic",
        )
        assert result["rho"].shape == (2, 11, 20)
        assert torch.isfinite(result["rho"]).all()

    def test_generate_n_return_keys(self):
        """All expected keys present."""
        result = generate_n(
            2, 2, nx=20, dx=0.05, dt=0.005, nt=10, seed=42,
            show_progress=False, reconstruction="constant",
        )
        expected = {"rho", "v", "w", "x", "t", "dx", "dt", "nt", "ic_xs", "ic_rho_ks", "ic_v_ks"}
        assert set(result.keys()) == expected

    def test_generate_n_ic_params_shapes(self):
        """IC parameter arrays have correct shapes."""
        n, k = 5, 4
        result = generate_n(
            n, k, nx=30, dx=0.05, dt=0.005, nt=10, seed=42,
            show_progress=False, reconstruction="constant",
        )
        assert result["ic_xs"].shape == (n, k + 1)
        assert result["ic_rho_ks"].shape == (n, k)
        assert result["ic_v_ks"].shape == (n, k)

    def test_generate_n_narrow_ranges(self):
        """Very narrow rho/v range: all IC values almost identical."""
        result = generate_n(
            3, 3, nx=20, dx=0.05, dt=0.005, nt=10, seed=42,
            show_progress=False, reconstruction="constant",
            rho_range=(0.5, 0.51), v_range=(0.3, 0.31),
        )
        assert torch.isfinite(result["rho"]).all()
        # All initial densities should be nearly uniform
        for i in range(3):
            rho_ic = result["rho"][i, 0]
            assert (rho_ic.max() - rho_ic.min()) < 0.05


# ====================================================================
# 10. CFL / time-stepping edge cases
# ====================================================================

class TestCFLEdgeCases:
    """Test behavior near CFL limits and with unusual dt/dx ratios."""

    def test_very_small_dt(self):
        """Very small time step: almost no change per step."""
        nx = 32
        x = torch.arange(nx, dtype=torch.float32) * 0.05
        rho0, v0 = riemann(x, rho_left=0.6, rho_right=0.3, v0=0.2)
        result = generate_one(
            rho0, v0, dx=0.05, dt=1e-6, nt=10,
            reconstruction="constant",
        )
        assert result["valid"]
        # Almost no change with tiny dt
        torch.testing.assert_close(result["rho"][-1], result["rho"][0], atol=1e-3, rtol=1e-3)

    def test_large_dt_may_blow_up(self):
        """Large dt relative to dx: expected to exceed CFL and blow up."""
        nx = 32
        x = torch.arange(nx, dtype=torch.float32) * 0.05
        rho0, v0 = riemann(x, rho_left=0.8, rho_right=0.2, v0=0.5)
        result = generate_one(
            rho0, v0, dx=0.05, dt=0.5, nt=20,
            reconstruction="constant", max_value=100.0,
        )
        # Should either blow up or remain valid -- just check finite if valid
        if result["valid"]:
            assert torch.isfinite(result["rho"]).all()

    def test_fine_dx_coarse_dt(self):
        """Fine spatial resolution with relatively large dt."""
        nx = 100
        dx = 0.01
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = riemann(x, rho_left=0.5, rho_right=0.3, v0=0.1)
        result = generate_one(
            rho0, v0, dx=dx, dt=0.001, nt=20,
            reconstruction="constant",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()


# ====================================================================
# 11. max_value termination
# ====================================================================

class TestMaxValueTermination:
    """Test early termination via max_value."""

    def test_max_value_not_triggered(self):
        """With a generous max_value, solution should remain valid."""
        nx = 32
        x = torch.arange(nx, dtype=torch.float32) * 0.05
        rho0, v0 = riemann(x, rho_left=0.6, rho_right=0.3, v0=0.2)
        result = generate_one(
            rho0, v0, dx=0.05, dt=0.005, nt=20,
            reconstruction="constant", max_value=100.0,
        )
        assert result["valid"]

    def test_max_value_triggers_on_tight_threshold(self):
        """Very tight threshold: should trigger early termination."""
        nx = 32
        x = torch.arange(nx, dtype=torch.float32) * 0.05
        rho0, v0 = riemann(x, rho_left=0.8, rho_right=0.2)
        result = generate_one(
            rho0, v0, dx=0.05, dt=0.005, nt=20,
            reconstruction="constant", max_value=0.01,
        )
        assert not result["valid"]

    def test_max_value_none_no_termination(self):
        """max_value=None means no threshold check."""
        nx = 32
        x = torch.arange(nx, dtype=torch.float32) * 0.05
        rho0, v0 = riemann(x, rho_left=0.6, rho_right=0.3, v0=0.2)
        result = generate_one(
            rho0, v0, dx=0.05, dt=0.005, nt=20,
            reconstruction="constant", max_value=None,
        )
        assert result["valid"]

    def test_max_value_fills_nan_after_failure(self):
        """After early termination, remaining time steps should be NaN."""
        nx = 32
        x = torch.arange(nx, dtype=torch.float32) * 0.05
        rho0, v0 = riemann(x, rho_left=0.8, rho_right=0.2)
        result = generate_one(
            rho0, v0, dx=0.05, dt=0.005, nt=50,
            reconstruction="constant", max_value=0.01,
        )
        assert not result["valid"]
        # There should be some NaN rows at the end
        has_nan = torch.isnan(result["rho"]).any(dim=1)
        assert has_nan.any(), "Expected NaN-filled rows after early termination"


# ====================================================================
# 12. Flux function edge cases
# ====================================================================

class TestFluxEdgeCases:
    """Edge cases for the numerical flux functions directly."""

    def test_rusanov_single_interface(self):
        """Rusanov with a single interface."""
        gamma = 1.0
        rhoL = torch.tensor([0.5])
        vL = torch.tensor([0.3])
        wL = vL + pressure(rhoL, gamma)
        rho_wL = rhoL * wL
        rhoR = torch.tensor([0.3])
        vR = torch.tensor([0.2])
        wR = vR + pressure(rhoR, gamma)
        rho_wR = rhoR * wR
        f_rho, f_rw = rusanov(rhoL, rho_wL, rhoR, rho_wR, gamma)
        assert f_rho.shape == (1,)
        assert f_rw.shape == (1,)
        assert torch.isfinite(f_rho).all()
        assert torch.isfinite(f_rw).all()

    def test_hll_single_interface(self):
        """HLL with a single interface."""
        gamma = 1.0
        rhoL = torch.tensor([0.5])
        vL = torch.tensor([0.3])
        wL = vL + pressure(rhoL, gamma)
        rho_wL = rhoL * wL
        rhoR = torch.tensor([0.3])
        vR = torch.tensor([0.2])
        wR = vR + pressure(rhoR, gamma)
        rho_wR = rhoR * wR
        f_rho, f_rw = hll(rhoL, rho_wL, rhoR, rho_wR, gamma)
        assert f_rho.shape == (1,)
        assert f_rw.shape == (1,)
        assert torch.isfinite(f_rho).all()
        assert torch.isfinite(f_rw).all()

    def test_flux_zero_density_both_sides(self, flux_type):
        """Zero density on both sides should give zero flux."""
        fn = hll if flux_type == "hll" else rusanov
        zeros = torch.zeros(5)
        f_rho, f_rw = fn(zeros, zeros, zeros, zeros, 1.0)
        torch.testing.assert_close(f_rho, zeros, atol=1e-10, rtol=0)
        torch.testing.assert_close(f_rw, zeros, atol=1e-10, rtol=0)

    def test_flux_zero_density_one_side(self, flux_type):
        """Zero density on one side, nonzero on the other."""
        fn = hll if flux_type == "hll" else rusanov
        gamma = 1.0
        rhoL = torch.tensor([0.0, 0.5])
        rhoR = torch.tensor([0.5, 0.0])
        vL = torch.tensor([0.0, 0.3])
        vR = torch.tensor([0.3, 0.0])
        wL = vL + pressure(rhoL, gamma)
        wR = vR + pressure(rhoR, gamma)
        rho_wL = rhoL * wL
        rho_wR = rhoR * wR
        f_rho, f_rw = fn(rhoL, rho_wL, rhoR, rho_wR, gamma)
        assert torch.isfinite(f_rho).all()
        assert torch.isfinite(f_rw).all()

    def test_flux_identical_states_is_physical(self, flux_type):
        """Consistency: F(U,U) = f(U) for any consistent flux."""
        fn = hll if flux_type == "hll" else rusanov
        gamma = 1.5
        rho = torch.tensor([0.3, 0.7, 1.0])
        v = torch.tensor([0.2, 0.5, -0.1])
        w = v + pressure(rho, gamma)
        rho_w = rho * w
        f_rho, f_rw = fn(rho, rho_w, rho, rho_w, gamma)
        expected_f_rho = rho * v
        expected_f_rw = rho_w * v
        torch.testing.assert_close(f_rho, expected_f_rho, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(f_rw, expected_f_rw, atol=1e-6, rtol=1e-6)

    def test_flux_many_interfaces(self, flux_type):
        """Flux computation with many interfaces at once."""
        fn = hll if flux_type == "hll" else rusanov
        n = 1000
        rhoL = torch.rand(n) * 0.9 + 0.1
        rhoR = torch.rand(n) * 0.9 + 0.1
        vL = torch.rand(n)
        vR = torch.rand(n)
        gamma = 1.0
        wL = vL + pressure(rhoL, gamma)
        wR = vR + pressure(rhoR, gamma)
        rho_wL = rhoL * wL
        rho_wR = rhoR * wR
        f_rho, f_rw = fn(rhoL, rho_wL, rhoR, rho_wR, gamma)
        assert f_rho.shape == (n,)
        assert torch.isfinite(f_rho).all()
        assert torch.isfinite(f_rw).all()


# ====================================================================
# 13. WENO edge cases
# ====================================================================

class TestWENOEdgeCases:
    """Edge cases for WENO-5 reconstruction."""

    def test_weno5_near_zero_data(self):
        """Near-zero data should not produce NaN."""
        v = torch.full((16,), 1e-10)
        v_minus, v_plus = weno5_reconstruct(v)
        assert torch.isfinite(v_minus).all()
        assert torch.isfinite(v_plus).all()

    def test_weno5_large_values(self):
        """Large values should not overflow."""
        v = torch.full((16,), 1e6, dtype=torch.float64)
        v_minus, v_plus = weno5_reconstruct(v)
        assert torch.isfinite(v_minus).all()
        assert torch.isfinite(v_plus).all()

    def test_weno5_sharp_step(self):
        """Sharp step function: WENO should not oscillate excessively."""
        v = torch.cat([torch.ones(10), torch.full((10,), 0.1)])
        v_minus, v_plus = weno5_reconstruct(v)
        assert torch.isfinite(v_minus).all()
        assert torch.isfinite(v_plus).all()
        # v_minus and v_plus should be bounded by [0.1, 1.0] approximately
        assert v_minus.max() <= 1.0 + 0.1
        assert v_minus.min() >= 0.1 - 0.1

    def test_weno5_negative_values(self):
        """WENO should handle negative values without issues."""
        v = torch.linspace(-2.0, 2.0, 20)
        v_minus, v_plus = weno5_reconstruct(v)
        assert torch.isfinite(v_minus).all()
        assert torch.isfinite(v_plus).all()

    def test_weno5_quadratic_data(self):
        """Quadratic data: WENO should reconstruct well (up to order)."""
        N = 20
        v = torch.arange(N, dtype=torch.float64) ** 2 * 0.01
        v_minus, v_plus = weno5_reconstruct(v)
        assert torch.isfinite(v_minus).all()
        assert torch.isfinite(v_plus).all()

    def test_weno5_minimum_input_length(self):
        """Minimum input length: N=8 -> ni=1 interface."""
        v = torch.rand(8)
        v_minus, v_plus = weno5_reconstruct(v)
        assert v_minus.shape == (1,)
        assert v_plus.shape == (1,)
        assert torch.isfinite(v_minus).all()
        assert torch.isfinite(v_plus).all()

    def test_weno5_alternating_values(self):
        """Alternating high-low pattern: potential oscillation source."""
        v = torch.zeros(20)
        v[::2] = 1.0
        v_minus, v_plus = weno5_reconstruct(v)
        assert torch.isfinite(v_minus).all()
        assert torch.isfinite(v_plus).all()


# ====================================================================
# 14. Boundary (ghost cell) edge cases
# ====================================================================

class TestGhostCellEdgeCases:
    """Edge cases for ghost cell application."""

    def test_periodic_preserves_sum(self):
        """Periodic ghost cells should not change the total mass."""
        nx = 16
        rho = torch.rand(nx)
        rho_w = torch.rand(nx)
        for ng in [1, 4]:
            rho_g, _ = apply_ghost_cells(
                rho, rho_w, "periodic", t=0.0, n_ghost=ng, gamma=1.0,
            )
            # Interior values should be unchanged
            torch.testing.assert_close(rho_g[ng:-ng], rho)

    def test_zero_gradient_uniform(self):
        """Zero-gradient on uniform data: extended array is all the same value."""
        nx = 16
        val = 0.42
        rho = torch.full((nx,), val)
        rho_w = torch.full((nx,), val * 0.5)
        for ng in [1, 4]:
            rho_g, _ = apply_ghost_cells(
                rho, rho_w, "zero_gradient", t=0.0, n_ghost=ng, gamma=1.0,
            )
            torch.testing.assert_close(rho_g, torch.full((nx + 2 * ng,), val))

    def test_dirichlet_negative_velocity_bc(self):
        """Dirichlet with negative velocity at boundary."""
        nx = 16
        rho = torch.rand(nx)
        rho_w = torch.rand(nx)
        rho_g, rho_w_g = apply_ghost_cells(
            rho, rho_w, "dirichlet", t=0.0, n_ghost=1, gamma=1.0,
            bc_left=(0.5, -0.3), bc_right=(0.5, -0.3),
        )
        assert rho_g.shape == (nx + 2,)
        assert torch.isfinite(rho_g).all()
        assert torch.isfinite(rho_w_g).all()

    def test_ghost_cells_do_not_modify_input(self):
        """Applying ghost cells should not modify the original tensors."""
        nx = 16
        rho = torch.rand(nx)
        rho_w = torch.rand(nx)
        rho_copy = rho.clone()
        rho_w_copy = rho_w.clone()
        apply_ghost_cells(rho, rho_w, "periodic", t=0.0, n_ghost=4, gamma=1.0)
        torch.testing.assert_close(rho, rho_copy)
        torch.testing.assert_close(rho_w, rho_w_copy)

    @pytest.mark.parametrize("bc_type", [
        "periodic", "zero_gradient", "dirichlet", "inflow_outflow", "time_varying_inflow",
    ])
    def test_all_bc_types_output_length(self, bc_type):
        """Every BC type should produce output of length nx + 2*n_ghost."""
        nx = 16
        ng = 4
        rho = torch.rand(nx)
        rho_w = torch.rand(nx)
        kwargs = dict(n_ghost=ng, gamma=1.0)
        if bc_type == "dirichlet":
            kwargs["bc_left"] = (0.5, 0.3)
            kwargs["bc_right"] = (0.3, 0.5)
        rho_g, rho_w_g = apply_ghost_cells(rho, rho_w, bc_type, t=0.0, **kwargs)
        assert rho_g.shape == (nx + 2 * ng,)
        assert rho_w_g.shape == (nx + 2 * ng,)


# ====================================================================
# 15. Physics function edge cases
# ====================================================================

class TestPhysicsEdgeCases:
    """Edge cases for physics functions."""

    def test_pressure_zero_density(self):
        """p(0) = 0 for any gamma."""
        rho = torch.tensor([0.0])
        for gamma in [0.1, 0.5, 1.0, 2.0, 5.0]:
            assert pressure(rho, gamma).item() == 0.0

    def test_pressure_one_density(self):
        """p(1) = 1 for any gamma."""
        rho = torch.tensor([1.0])
        for gamma in [0.1, 0.5, 1.0, 2.0, 5.0]:
            torch.testing.assert_close(
                pressure(rho, gamma), torch.tensor([1.0]), atol=1e-7, rtol=1e-7,
            )

    def test_dp_drho_matches_finite_difference(self):
        """dp_drho should approximate (p(rho+eps) - p(rho-eps)) / (2*eps)."""
        rho = torch.tensor([0.3, 0.5, 1.0, 2.0])
        eps = 1e-5
        for gamma in [0.5, 1.0, 1.5, 2.0]:
            dp_exact = dp_drho(rho, gamma)
            dp_fd = (pressure(rho + eps, gamma) - pressure(rho - eps, gamma)) / (2 * eps)
            torch.testing.assert_close(dp_exact, dp_fd, atol=1e-2, rtol=1e-2)

    def test_eigenvalue_ordering_many_states(self):
        """lam2 < lam1 for all states with rho > 0."""
        rho = torch.rand(100) * 2 + 0.01
        v = torch.rand(100) * 2 - 1
        for gamma in [0.5, 1.0, 2.0]:
            lam1, lam2 = eigenvalues(rho, v, gamma)
            assert (lam2 < lam1).all(), f"Eigenvalue ordering violated for gamma={gamma}"

    def test_eigenvalues_at_vacuum(self):
        """At rho=0, both eigenvalues should equal v."""
        rho = torch.zeros(5)
        v = torch.tensor([0.1, -0.2, 0.5, 1.0, -1.0])
        for gamma in [0.5, 1.0, 2.0]:
            lam1, lam2 = eigenvalues(rho, v, gamma)
            torch.testing.assert_close(lam1, v)
            # lam2 = v - 0 * dp_drho = v (but dp_drho(0, gamma<1) could be inf)
            # For gamma >= 1, dp_drho(0) is well-defined
            if gamma >= 1.0:
                torch.testing.assert_close(lam2, v)

    def test_pressure_large_density(self):
        """Pressure at large density should be finite."""
        rho = torch.tensor([100.0, 1000.0])
        for gamma in [0.5, 1.0, 2.0]:
            p = pressure(rho, gamma)
            assert torch.isfinite(p).all()

    def test_pressure_batch(self):
        """Pressure on a batch of densities."""
        rho = torch.rand(1000) * 3.0
        for gamma in [0.5, 1.0, 2.0]:
            p = pressure(rho, gamma)
            assert p.shape == (1000,)
            assert torch.isfinite(p).all()
            assert (p >= 0).all()


# ====================================================================
# 16. Initial condition generators
# ====================================================================

class TestInitialConditionEdgeCases:
    """Edge cases for IC generators."""

    def test_from_steps_single_step(self):
        """Single step: uniform value everywhere."""
        x = torch.arange(20, dtype=torch.float32) * 0.05
        rho0, v0 = from_steps(x, rho_steps=[(10.0, 0.5)])
        torch.testing.assert_close(rho0, torch.full_like(rho0, 0.5))

    def test_from_steps_many_steps(self):
        """Many step transitions."""
        x = torch.arange(100, dtype=torch.float32) * 0.01
        steps = [(0.1 * i, 0.1 * i) for i in range(1, 11)]
        rho0, v0 = from_steps(x, rho_steps=steps, default_v=0.3)
        assert rho0.shape == (100,)
        assert rho0.min() >= 0.0

    def test_riemann_entire_domain_left(self):
        """x_split beyond domain: entire domain should be uniform.

        When x_split > x.max()+1, the sentinel sorts before x_split,
        so all cells get rho_right. The result is a uniform IC.
        """
        x = torch.arange(20, dtype=torch.float32) * 0.05
        rho0, v0 = riemann(x, rho_left=0.8, rho_right=0.2, x_split=10.0)
        # All cells should be uniform (rho_right because of how from_steps sorts)
        assert rho0.unique().shape[0] == 1, "Expected uniform density"

    def test_riemann_entire_domain_right(self):
        """x_split before domain: entire domain is 'right' state."""
        x = torch.arange(20, dtype=torch.float32) * 0.05
        rho0, v0 = riemann(x, rho_left=0.8, rho_right=0.2, x_split=-1.0)
        torch.testing.assert_close(rho0, torch.full_like(rho0, 0.2))

    def test_three_region_narrow_mid(self):
        """Middle region spans only a few cells."""
        x = torch.arange(40, dtype=torch.float32) * 0.025
        rho0, v0 = three_region(
            x, rho_left=0.3, rho_mid=0.9, rho_right=0.5,
            v0=0.2, x1=0.5, x2=0.525,
        )
        mid_mask = (x >= 0.5) & (x < 0.525)
        assert mid_mask.sum() >= 1  # At least one cell in middle region
        assert (rho0[mid_mask] == 0.9).all()

    def test_three_region_x1_equals_x2(self):
        """x1 == x2: middle region vanishes, effectively two regions."""
        x = torch.arange(40, dtype=torch.float32) * 0.025
        rho0, v0 = three_region(
            x, rho_left=0.3, rho_mid=0.9, rho_right=0.5,
            v0=0.2, x1=0.5, x2=0.5,
        )
        # Should be a two-region IC: left(0.3) | right(0.5)
        assert rho0.shape == (40,)

    def test_random_piecewise_range_boundaries(self):
        """Values at the extremes of the range."""
        x = torch.arange(32, dtype=torch.float32) * 0.05
        rng = torch.Generator().manual_seed(42)
        rho0, v0, _ = random_piecewise(x, 5, rng, rho_range=(0.1, 0.1), v_range=(0.5, 0.5))
        # When min == max, all values should be exactly that value
        torch.testing.assert_close(rho0, torch.full_like(rho0, 0.1))
        torch.testing.assert_close(v0, torch.full_like(v0, 0.5))

    def test_random_piecewise_raises_too_many_breaks(self):
        """More breakpoints than cells should raise ValueError."""
        x = torch.arange(5, dtype=torch.float32) * 0.05
        rng = torch.Generator().manual_seed(42)
        with pytest.raises(ValueError, match="Cannot place"):
            random_piecewise(x, 10, rng)


# ====================================================================
# 17. Cross-combination stress tests
# ====================================================================

class TestCrossCombinations:
    """Stress tests combining multiple axes of variation."""

    @pytest.mark.parametrize("flux_type", ["rusanov", "hll"])
    @pytest.mark.parametrize("bc_type", ["periodic", "zero_gradient"])
    @pytest.mark.parametrize("reconstruction", ["constant", "weno5"])
    @pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0])
    def test_all_combos_riemann(self, flux_type, bc_type, reconstruction, gamma):
        """Every (flux, bc, reconstruction, gamma) combo on a Riemann IC."""
        nx = 32
        x = torch.arange(nx, dtype=torch.float32) * 0.05
        rho0, v0 = riemann(x, rho_left=0.5, rho_right=0.3, v0=0.2)
        result = generate_one(
            rho0, v0, dx=0.05, dt=0.002, nt=15, gamma=gamma,
            bc_type=bc_type, flux_type=flux_type, reconstruction=reconstruction,
        )
        assert result["valid"], (
            f"Failed: {flux_type}/{bc_type}/{reconstruction}/gamma={gamma}"
        )
        assert torch.isfinite(result["rho"]).all()

    @pytest.mark.parametrize("flux_type", ["rusanov", "hll"])
    @pytest.mark.parametrize("reconstruction", ["constant", "weno5"])
    def test_three_region_all_fluxes(self, flux_type, reconstruction):
        """Three-region IC with all flux/reconstruction combos."""
        nx = 60
        x = torch.arange(nx, dtype=torch.float32) * 0.025
        rho0, v0 = three_region(
            x, rho_left=0.7, rho_mid=0.2, rho_right=0.5,
            v0=0.2, x1=0.3, x2=0.7,
        )
        result = generate_one(
            rho0, v0, dx=0.025, dt=0.002, nt=30,
            flux_type=flux_type, reconstruction=reconstruction,
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()


# ====================================================================
# 18. Metadata and return-value correctness
# ====================================================================

class TestReturnValues:
    """Verify correctness of returned metadata and shapes."""

    def test_x_array_matches_dx(self):
        nx = 32
        dx = 0.05
        rho0 = torch.full((nx,), 0.5)
        v0 = torch.full((nx,), 0.3)
        result = generate_one(rho0, v0, dx=dx, dt=0.005, nt=10, reconstruction="constant")
        expected_x = torch.arange(nx, dtype=torch.float32) * dx
        torch.testing.assert_close(result["x"], expected_x)

    def test_t_array_matches_dt(self):
        nx = 20
        dt = 0.005
        nt = 10
        rho0 = torch.full((nx,), 0.5)
        v0 = torch.full((nx,), 0.3)
        result = generate_one(rho0, v0, dx=0.05, dt=dt, nt=nt, reconstruction="constant")
        expected_t = torch.arange(nt + 1, dtype=torch.float32) * dt
        torch.testing.assert_close(result["t"], expected_t)

    def test_metadata_fields(self):
        nx = 20
        dx, dt, nt = 0.05, 0.005, 10
        rho0 = torch.full((nx,), 0.5)
        v0 = torch.full((nx,), 0.3)
        result = generate_one(rho0, v0, dx=dx, dt=dt, nt=nt, reconstruction="constant")
        assert result["dx"] == dx
        assert result["dt"] == dt
        assert result["nt"] == nt
        assert isinstance(result["valid"], bool)

    def test_initial_row_matches_input(self):
        """First row of output should match the input IC exactly."""
        nx = 20
        x = torch.arange(nx, dtype=torch.float32) * 0.05
        rho0, v0 = riemann(x, rho_left=0.7, rho_right=0.3, v0=0.25)
        result = generate_one(rho0, v0, dx=0.05, dt=0.005, nt=10, reconstruction="constant")
        torch.testing.assert_close(result["rho"][0], rho0)
        torch.testing.assert_close(result["v"][0], v0)

    def test_output_dtypes_float32(self):
        """Output tensors should be float32 when input is float32."""
        nx = 20
        rho0 = torch.full((nx,), 0.5, dtype=torch.float32)
        v0 = torch.full((nx,), 0.3, dtype=torch.float32)
        result = generate_one(rho0, v0, dx=0.05, dt=0.005, nt=10, reconstruction="constant")
        assert result["rho"].dtype == torch.float32
        assert result["v"].dtype == torch.float32
        assert result["w"].dtype == torch.float32
        assert result["x"].dtype == torch.float32
        assert result["t"].dtype == torch.float32


# ====================================================================
# 19. Density non-negativity
# ====================================================================

class TestDensityNonNegativity:
    """Density should remain non-negative (it is clamped in the solver)."""

    @pytest.mark.parametrize("flux_type", ["rusanov", "hll"])
    @pytest.mark.parametrize("reconstruction", ["constant", "weno5"])
    def test_density_nonnegative_riemann(self, flux_type, reconstruction):
        nx = 40
        x = torch.arange(nx, dtype=torch.float32) * 0.025
        rho0, v0 = riemann(x, rho_left=0.9, rho_right=0.05, v0=0.2)
        result = generate_one(
            rho0, v0, dx=0.025, dt=0.002, nt=30,
            flux_type=flux_type, reconstruction=reconstruction,
        )
        if result["valid"]:
            assert (result["rho"] >= -1e-10).all(), (
                f"Negative density: min={result['rho'].min().item()}"
            )

    def test_density_nonnegative_vacuum_adjacent(self):
        """Vacuum (rho=0) next to high density: density should stay >= 0."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float32) * 0.025
        rho0, v0 = riemann(x, rho_left=0.0, rho_right=1.0, v0=0.1)
        result = generate_one(
            rho0, v0, dx=0.025, dt=0.002, nt=20,
            reconstruction="constant",
        )
        assert result["valid"]
        assert (result["rho"] >= -1e-10).all()

    def test_density_nonnegative_many_random_ics(self):
        """Check non-negativity across many random ICs."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float32) * 0.025
        for seed in range(10):
            rng = torch.Generator().manual_seed(seed)
            rho0, v0, _ = random_piecewise(x, 3, rng, rho_range=(0.05, 1.0))
            result = generate_one(
                rho0, v0, dx=0.025, dt=0.002, nt=20,
                reconstruction="constant",
            )
            if result["valid"]:
                assert (result["rho"] >= -1e-10).all(), (
                    f"Negative density with seed={seed}: min={result['rho'].min().item()}"
                )


# ====================================================================
# 20. Symmetry tests
# ====================================================================

class TestSymmetry:
    """Solutions should respect symmetry of initial conditions."""

    def test_uniform_ic_stays_uniform(self):
        """Uniform IC with periodic BC should remain perfectly uniform."""
        nx = 32
        rho0 = torch.full((nx,), 0.5)
        v0 = torch.full((nx,), 0.3)
        result = generate_one(
            rho0, v0, dx=0.05, dt=0.005, nt=30,
            bc_type="periodic", reconstruction="constant",
        )
        for t_idx in range(result["rho"].shape[0]):
            rho_t = result["rho"][t_idx]
            assert (rho_t - 0.5).abs().max() < 1e-6

    def test_left_right_mirror_periodic(self):
        """Mirrored IC with zero velocity and periodic BC should produce mirrored solution.

        With v=0 and a spatially symmetric density profile, the ARZ system
        is symmetric under spatial reflection, so the solution should
        remain symmetric at all times.
        """
        nx = 40
        rho_left = torch.linspace(0.3, 0.6, nx // 2)
        rho_right = rho_left.flip(0)
        rho0 = torch.cat([rho_left, rho_right])
        v0 = torch.zeros(nx)  # Zero velocity for true spatial symmetry
        result = generate_one(
            rho0, v0, dx=0.025, dt=0.002, nt=10,
            bc_type="periodic", reconstruction="constant",
        )
        for t_idx in range(result["rho"].shape[0]):
            rho_t = result["rho"][t_idx]
            torch.testing.assert_close(
                rho_t, rho_t.flip(0), atol=1e-5, rtol=1e-5,
            )
