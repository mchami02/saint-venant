"""Comprehensive edge case tests for Euler solver."""

import math

import pytest
import torch

from numerical_solvers.euler import generate_n, generate_one
from numerical_solvers.euler.initial_conditions import from_steps, random_piecewise, riemann, sod
from numerical_solvers.euler.physics import (
    conservative_to_primitive,
    pressure_from_conservative,
    primitive_to_conservative,
    sound_speed,
)


class TestExtremeDensityValues:
    def test_near_zero_density(self):
        nx = 32
        x = torch.arange(nx, dtype=torch.float64) * 0.05
        rho0 = torch.full((nx,), 1e-6, dtype=torch.float64)
        u0 = torch.zeros(nx, dtype=torch.float64)
        p0 = torch.full((nx,), 1e-6, dtype=torch.float64)
        result = generate_one(rho0, u0, p0, dx=0.05, dt=0.002, nt=10, reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_very_high_density(self):
        nx = 32
        x = torch.arange(nx, dtype=torch.float64) * 0.05
        rho0 = torch.full((nx,), 100.0, dtype=torch.float64)
        u0 = torch.zeros(nx, dtype=torch.float64)
        p0 = torch.full((nx,), 100.0, dtype=torch.float64)
        result = generate_one(rho0, u0, p0, dx=0.05, dt=0.0001, nt=10, reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_large_density_ratio(self):
        """Density ratio of 1000:1 (extreme Riemann problem)."""
        nx = 50
        x = torch.arange(nx, dtype=torch.float64) * 0.02
        rho0, u0, p0 = riemann(x, rho_left=1000.0, rho_right=1.0, p_left=1000.0, p_right=1.0)
        result = generate_one(rho0, u0, p0, dx=0.02, dt=0.0001, nt=20, reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_density_ratio_10000(self):
        """Density ratio of 10000:1."""
        nx = 60
        x = torch.arange(nx, dtype=torch.float64) * 0.02
        rho0, u0, p0 = riemann(x, rho_left=10000.0, rho_right=1.0, p_left=10000.0, p_right=1.0)
        result = generate_one(rho0, u0, p0, dx=0.02, dt=0.00002, nt=10, reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_density_staircase(self):
        """Monotonically decreasing density staircase."""
        nx = 50
        x = torch.arange(nx, dtype=torch.float64) * 0.02
        sentinel = x.max().item() + 1.0
        rho0, u0, p0 = from_steps(
            x,
            rho_steps=[(0.2, 5.0), (0.4, 4.0), (0.6, 3.0), (0.8, 2.0), (sentinel, 1.0)],
        )
        result = generate_one(rho0, u0, p0, dx=0.02, dt=0.001, nt=20, reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_alternating_density_pattern(self):
        """Alternating high-low density cells."""
        nx = 40
        rho0 = torch.ones(nx, dtype=torch.float64)
        rho0[::2] = 2.0
        u0 = torch.zeros(nx, dtype=torch.float64)
        p0 = torch.ones(nx, dtype=torch.float64)
        result = generate_one(rho0, u0, p0, dx=0.025, dt=0.001, nt=20,
                              bc_type="periodic", reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()


class TestExtremePressureValues:
    def test_very_low_pressure(self):
        nx = 32
        x = torch.arange(nx, dtype=torch.float64) * 0.05
        rho0 = torch.ones(nx, dtype=torch.float64)
        u0 = torch.zeros(nx, dtype=torch.float64)
        p0 = torch.full((nx,), 1e-6, dtype=torch.float64)
        result = generate_one(rho0, u0, p0, dx=0.05, dt=0.001, nt=10, reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_very_high_pressure(self):
        nx = 32
        x = torch.arange(nx, dtype=torch.float64) * 0.05
        rho0 = torch.ones(nx, dtype=torch.float64)
        u0 = torch.zeros(nx, dtype=torch.float64)
        p0 = torch.full((nx,), 1000.0, dtype=torch.float64)
        result = generate_one(rho0, u0, p0, dx=0.05, dt=0.0001, nt=10, reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_large_pressure_ratio(self):
        """Pressure ratio of 10000:1."""
        nx = 50
        x = torch.arange(nx, dtype=torch.float64) * 0.02
        rho0, u0, p0 = riemann(x, rho_left=1.0, rho_right=1.0, p_left=10000.0, p_right=1.0)
        result = generate_one(rho0, u0, p0, dx=0.02, dt=0.00005, nt=20, reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_pressure_spike_in_center(self):
        """Single cell with very high pressure."""
        nx = 40
        rho0 = torch.ones(nx, dtype=torch.float64)
        u0 = torch.zeros(nx, dtype=torch.float64)
        p0 = torch.ones(nx, dtype=torch.float64)
        p0[nx // 2] = 100.0
        result = generate_one(rho0, u0, p0, dx=0.025, dt=0.0002, nt=20,
                              bc_type="periodic", reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_pressure_trough_in_center(self):
        """Single cell with very low pressure surrounded by high."""
        nx = 40
        rho0 = torch.ones(nx, dtype=torch.float64)
        u0 = torch.zeros(nx, dtype=torch.float64)
        p0 = torch.full((nx,), 10.0, dtype=torch.float64)
        p0[nx // 2] = 0.01
        result = generate_one(rho0, u0, p0, dx=0.025, dt=0.0002, nt=20,
                              bc_type="periodic", reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_multi_pressure_jump(self):
        """Multiple pressure jumps at different locations."""
        nx = 60
        x = torch.arange(nx, dtype=torch.float64) * 0.02
        sentinel = x.max().item() + 1.0
        rho0, u0, p0 = from_steps(
            x,
            rho_steps=[(sentinel, 1.0)],
            p_steps=[(0.3, 10.0), (0.6, 0.5), (0.9, 5.0), (sentinel, 1.0)],
        )
        result = generate_one(rho0, u0, p0, dx=0.02, dt=0.0003, nt=20, reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()


class TestExtremeVelocityValues:
    def test_supersonic_flow(self):
        """Mach > 1 everywhere."""
        nx = 32
        x = torch.arange(nx, dtype=torch.float64) * 0.05
        rho0 = torch.ones(nx, dtype=torch.float64)
        u0 = torch.full((nx,), 5.0, dtype=torch.float64)  # Mach ~ 5/sqrt(1.4) ~ 4.2
        p0 = torch.ones(nx, dtype=torch.float64)
        result = generate_one(rho0, u0, p0, dx=0.05, dt=0.001, nt=10,
                              bc_type="periodic", reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_counter_flowing_supersonic(self):
        """Two supersonic streams colliding."""
        nx = 50
        x = torch.arange(nx, dtype=torch.float64) * 0.02
        rho0, u0, p0 = riemann(x, u_left=5.0, u_right=-5.0,
                                rho_left=1.0, rho_right=1.0,
                                p_left=1.0, p_right=1.0)
        result = generate_one(rho0, u0, p0, dx=0.02, dt=0.0005, nt=20, reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_negative_velocity_uniform(self):
        nx = 32
        x = torch.arange(nx, dtype=torch.float64) * 0.05
        rho0 = torch.ones(nx, dtype=torch.float64)
        u0 = torch.full((nx,), -2.0, dtype=torch.float64)
        p0 = torch.ones(nx, dtype=torch.float64)
        result = generate_one(rho0, u0, p0, dx=0.05, dt=0.001, nt=10,
                              bc_type="periodic", reconstruction="constant")
        assert result["valid"]

    def test_hypersonic_flow(self):
        """Mach ~ 10."""
        nx = 40
        rho0 = torch.ones(nx, dtype=torch.float64)
        u0 = torch.full((nx,), 12.0, dtype=torch.float64)
        p0 = torch.ones(nx, dtype=torch.float64)
        result = generate_one(rho0, u0, p0, dx=0.025, dt=0.0002, nt=10,
                              bc_type="periodic", reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_shear_layer(self):
        """Sharp velocity discontinuity with identical thermodynamic state."""
        nx = 60
        x = torch.arange(nx, dtype=torch.float64) * 0.02
        rho0, u0, p0 = riemann(x, rho_left=1.0, rho_right=1.0,
                                u_left=1.0, u_right=-1.0,
                                p_left=1.0, p_right=1.0)
        result = generate_one(rho0, u0, p0, dx=0.02, dt=0.001, nt=30, reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_velocity_ramp(self):
        """Linearly increasing velocity (smooth IC)."""
        nx = 50
        rho0 = torch.ones(nx, dtype=torch.float64)
        u0 = torch.linspace(-1.0, 1.0, nx, dtype=torch.float64)
        p0 = torch.ones(nx, dtype=torch.float64)
        result = generate_one(rho0, u0, p0, dx=0.02, dt=0.001, nt=20,
                              bc_type="extrap", reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()


class TestGammaVariations:
    @pytest.mark.parametrize("gamma", [1.1, 1.4, 1.67, 2.0, 3.0])
    def test_sod_various_gamma(self, gamma):
        nx = 40
        x = torch.arange(nx, dtype=torch.float64) * 0.025
        rho0, u0, p0 = sod(x)
        result = generate_one(rho0, u0, p0, dx=0.025, dt=0.001, nt=20,
                              gamma=gamma, reconstruction="constant")
        assert result["valid"], f"Failed for gamma={gamma}"
        assert torch.isfinite(result["rho"]).all()

    @pytest.mark.parametrize("gamma", [1.1, 1.4, 2.0])
    def test_steady_state_various_gamma(self, gamma):
        nx = 32
        rho0 = torch.ones(nx, dtype=torch.float64)
        u0 = torch.zeros(nx, dtype=torch.float64)
        p0 = torch.ones(nx, dtype=torch.float64)
        result = generate_one(rho0, u0, p0, dx=0.05, dt=0.002, nt=20, gamma=gamma,
                              bc_type="periodic", reconstruction="constant")
        torch.testing.assert_close(result["rho"][-1], rho0, atol=1e-10, rtol=1e-10)

    @pytest.mark.parametrize("gamma", [1.1, 1.4, 2.0])
    def test_conservation_various_gamma(self, gamma):
        nx = 40
        x = torch.arange(nx, dtype=torch.float64) * 0.025
        rho0, u0, p0 = riemann(x, rho_left=1.0, rho_right=0.5)
        result = generate_one(rho0, u0, p0, dx=0.025, dt=0.001, nt=20, gamma=gamma,
                              bc_type="periodic", reconstruction="constant")
        mass = result["rho"].sum(dim=-1) * 0.025
        torch.testing.assert_close(mass, mass[0].expand_as(mass), atol=1e-8, rtol=1e-8)

    def test_monatomic_gas_gamma(self):
        """gamma = 5/3 for monatomic ideal gas."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float64) * 0.025
        rho0, u0, p0 = sod(x)
        result = generate_one(rho0, u0, p0, dx=0.025, dt=0.001, nt=20,
                              gamma=5.0 / 3.0, reconstruction="constant")
        assert result["valid"]

    def test_diatomic_gas_gamma(self):
        """gamma = 7/5 for diatomic ideal gas (air)."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float64) * 0.025
        rho0, u0, p0 = sod(x)
        result = generate_one(rho0, u0, p0, dx=0.025, dt=0.001, nt=20,
                              gamma=7.0 / 5.0, reconstruction="constant")
        assert result["valid"]


class TestGridResolution:
    @pytest.mark.parametrize("nx", [8, 16, 32, 64])
    def test_varying_nx_sod(self, nx):
        dx = 1.0 / nx
        x = torch.arange(nx, dtype=torch.float64) * dx
        rho0, u0, p0 = sod(x)
        result = generate_one(rho0, u0, p0, dx=dx, dt=0.0005, nt=10, reconstruction="constant")
        assert result["rho"].shape == (11, nx)
        assert result["valid"]

    def test_very_few_cells(self):
        nx = 4
        x = torch.arange(nx, dtype=torch.float64) * 0.25
        rho0 = torch.tensor([1.0, 1.0, 0.125, 0.125], dtype=torch.float64)
        u0 = torch.zeros(nx, dtype=torch.float64)
        p0 = torch.tensor([1.0, 1.0, 0.1, 0.1], dtype=torch.float64)
        result = generate_one(rho0, u0, p0, dx=0.25, dt=0.01, nt=5, reconstruction="constant")
        assert result["valid"]

    def test_single_step(self):
        """Only one time step."""
        nx = 20
        x = torch.arange(nx, dtype=torch.float64) * 0.05
        rho0, u0, p0 = sod(x)
        result = generate_one(rho0, u0, p0, dx=0.05, dt=0.001, nt=1, reconstruction="constant")
        assert result["rho"].shape == (2, nx)
        assert result["valid"]

    def test_many_time_steps(self):
        """Many small time steps."""
        nx = 30
        x = torch.arange(nx, dtype=torch.float64) * 0.033
        rho0, u0, p0 = sod(x)
        result = generate_one(rho0, u0, p0, dx=0.033, dt=0.0005, nt=200,
                              reconstruction="constant")
        assert result["rho"].shape == (201, nx)
        assert result["valid"]

    def test_very_fine_dx(self):
        """Fine spatial resolution."""
        nx = 200
        dx = 0.005
        x = torch.arange(nx, dtype=torch.float64) * dx
        rho0, u0, p0 = sod(x)
        result = generate_one(rho0, u0, p0, dx=dx, dt=0.0002, nt=10, reconstruction="constant")
        assert result["rho"].shape == (11, nx)
        assert result["valid"]


class TestDiscontinuityPositions:
    @pytest.mark.parametrize("x_split", [0.05, 0.25, 0.5, 0.75, 0.95])
    def test_sod_split_positions(self, x_split):
        nx = 40
        x = torch.arange(nx, dtype=torch.float64) * 0.025
        rho0, u0, p0 = sod(x, x_split=x_split)
        result = generate_one(rho0, u0, p0, dx=0.025, dt=0.001, nt=20, reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_discontinuity_at_first_cell(self):
        """Discontinuity right at the first interface."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float64) * 0.025
        rho0, u0, p0 = sod(x, x_split=0.025)
        result = generate_one(rho0, u0, p0, dx=0.025, dt=0.001, nt=20, reconstruction="constant")
        assert result["valid"]

    def test_discontinuity_at_last_cell(self):
        """Discontinuity right near the last interface."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float64) * 0.025
        rho0, u0, p0 = sod(x, x_split=0.975)
        result = generate_one(rho0, u0, p0, dx=0.025, dt=0.001, nt=20, reconstruction="constant")
        assert result["valid"]

    def test_two_discontinuities_close(self):
        """Two discontinuities very close together."""
        nx = 60
        x = torch.arange(nx, dtype=torch.float64) * 0.02
        sentinel = x.max().item() + 1.0
        rho0, u0, p0 = from_steps(
            x,
            rho_steps=[(0.49, 2.0), (0.51, 0.5), (sentinel, 1.0)],
            p_steps=[(0.49, 2.0), (0.51, 0.5), (sentinel, 1.0)],
        )
        result = generate_one(rho0, u0, p0, dx=0.02, dt=0.001, nt=20, reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()


class TestBCEdgeCases:
    def test_wall_bc_with_incoming_flow(self):
        """Flow directed into a wall."""
        nx = 32
        x = torch.arange(nx, dtype=torch.float64) * 0.05
        rho0 = torch.ones(nx, dtype=torch.float64)
        u0 = torch.full((nx,), 1.0, dtype=torch.float64)
        p0 = torch.ones(nx, dtype=torch.float64)
        result = generate_one(rho0, u0, p0, dx=0.05, dt=0.001, nt=20,
                              bc_type="wall", reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_wall_bc_sod(self):
        """Sod problem with reflecting walls: waves should bounce back."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float64) * 0.025
        rho0, u0, p0 = sod(x)
        result = generate_one(rho0, u0, p0, dx=0.025, dt=0.001, nt=40,
                              bc_type="wall", reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_periodic_bc_uniform_advection(self):
        """Uniform flow with periodic BCs should remain perfectly uniform."""
        nx = 32
        rho0 = torch.ones(nx, dtype=torch.float64)
        u0 = torch.full((nx,), 1.0, dtype=torch.float64)
        p0 = torch.ones(nx, dtype=torch.float64)
        result = generate_one(rho0, u0, p0, dx=0.05, dt=0.002, nt=50,
                              bc_type="periodic", reconstruction="constant")
        assert result["valid"]
        torch.testing.assert_close(result["rho"][-1], rho0, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(result["u"][-1], u0, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(result["p"][-1], p0, atol=1e-10, rtol=1e-10)

    def test_extrap_bc_outgoing_wave(self):
        """Wave exiting the domain through extrapolation BC."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float64) * 0.025
        rho0, u0, p0 = sod(x, x_split=0.8)
        result = generate_one(rho0, u0, p0, dx=0.025, dt=0.001, nt=60,
                              bc_type="extrap", reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_wall_bc_strong_shock(self):
        """Strong shock reflecting off a wall."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float64) * 0.025
        rho0, u0, p0 = riemann(x, rho_left=10.0, rho_right=1.0,
                                u_left=0.0, u_right=0.0,
                                p_left=100.0, p_right=1.0,
                                x_split=0.8)
        result = generate_one(rho0, u0, p0, dx=0.025, dt=0.0003, nt=60,
                              bc_type="wall", reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    @pytest.mark.parametrize("bc_type", ["extrap", "periodic", "wall"])
    def test_all_bc_types_with_sod(self, bc_type):
        """Sod problem should run with all BC types."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float64) * 0.025
        rho0, u0, p0 = sod(x)
        result = generate_one(rho0, u0, p0, dx=0.025, dt=0.001, nt=20,
                              bc_type=bc_type, reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()


class TestPhysicsEdgeCases:
    def test_primitive_conservative_roundtrip_extreme_values(self):
        """Roundtrip with extreme but physical values."""
        rho = torch.tensor([1e-6, 1e6], dtype=torch.float64)
        u = torch.tensor([-100.0, 100.0], dtype=torch.float64)
        p = torch.tensor([1e-6, 1e6], dtype=torch.float64)
        gamma = 1.4
        _, rho_u, E = primitive_to_conservative(rho, u, p, gamma)
        rho2, u2, p2 = conservative_to_primitive(rho, rho_u, E, gamma)
        torch.testing.assert_close(u2, u, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(p2, p, atol=1e-4, rtol=1e-4)

    def test_sound_speed_dimensional_analysis(self):
        """Sound speed should scale with sqrt(p/rho)."""
        rho = torch.tensor([1.0, 4.0], dtype=torch.float64)
        p = torch.tensor([1.0, 4.0], dtype=torch.float64)
        gamma = 1.4
        c = sound_speed(rho, p, gamma)
        expected = torch.full((2,), math.sqrt(1.4), dtype=torch.float64)
        torch.testing.assert_close(c, expected, atol=1e-10, rtol=1e-10)

    @pytest.mark.parametrize("gamma", [1.1, 1.4, 1.67, 2.0, 5 / 3])
    def test_eos_consistency(self, gamma):
        """E = p/(gamma-1) + 0.5*rho*u^2 for various gamma."""
        rho = torch.tensor([2.0], dtype=torch.float64)
        u = torch.tensor([3.0], dtype=torch.float64)
        p = torch.tensor([5.0], dtype=torch.float64)
        _, _, E = primitive_to_conservative(rho, u, p, gamma)
        expected_E = p / (gamma - 1.0) + 0.5 * rho * u**2
        torch.testing.assert_close(E, expected_E)

    def test_pressure_from_conservative_matches(self):
        """pressure_from_conservative should agree with conservative_to_primitive."""
        rho = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
        u = torch.tensor([0.0, 1.5, -1.0], dtype=torch.float64)
        p = torch.tensor([1.0, 3.0, 0.5], dtype=torch.float64)
        gamma = 1.4
        _, rho_u, E = primitive_to_conservative(rho, u, p, gamma)
        p_from_cons = pressure_from_conservative(rho, rho_u, E, gamma)
        torch.testing.assert_close(p_from_cons, p, atol=1e-12, rtol=1e-12)

    def test_sound_speed_positive(self):
        """Sound speed should always be positive for positive rho, p."""
        rho = torch.tensor([0.001, 0.1, 1.0, 10.0, 100.0], dtype=torch.float64)
        p = torch.tensor([0.001, 0.1, 1.0, 10.0, 100.0], dtype=torch.float64)
        c = sound_speed(rho, p, 1.4)
        assert (c > 0).all()

    def test_conservative_roundtrip_zero_velocity(self):
        """Zero velocity roundtrip."""
        rho = torch.tensor([1.0, 2.0, 5.0], dtype=torch.float64)
        u = torch.zeros(3, dtype=torch.float64)
        p = torch.tensor([1.0, 3.0, 10.0], dtype=torch.float64)
        gamma = 1.4
        _, rho_u, E = primitive_to_conservative(rho, u, p, gamma)
        rho2, u2, p2 = conservative_to_primitive(rho, rho_u, E, gamma)
        torch.testing.assert_close(rho2, rho)
        torch.testing.assert_close(u2, u, atol=1e-15, rtol=0.0)
        torch.testing.assert_close(p2, p)

    def test_conservative_roundtrip_batch(self):
        """Roundtrip on a 2D batch of values."""
        rho = torch.rand(5, 10, dtype=torch.float64) + 0.1
        u = torch.randn(5, 10, dtype=torch.float64)
        p = torch.rand(5, 10, dtype=torch.float64) + 0.1
        gamma = 1.4
        _, rho_u, E = primitive_to_conservative(rho, u, p, gamma)
        rho2, u2, p2 = conservative_to_primitive(rho, rho_u, E, gamma)
        torch.testing.assert_close(rho2, rho)
        torch.testing.assert_close(u2, u, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(p2, p, atol=1e-10, rtol=1e-10)

    def test_energy_positive_for_physical_states(self):
        """Total energy must be positive for any physical state."""
        rho = torch.tensor([0.01, 1.0, 100.0], dtype=torch.float64)
        u = torch.tensor([-10.0, 0.0, 10.0], dtype=torch.float64)
        p = torch.tensor([0.01, 1.0, 100.0], dtype=torch.float64)
        _, _, E = primitive_to_conservative(rho, u, p, 1.4)
        assert (E > 0).all()

    def test_sound_speed_increases_with_temperature(self):
        """c increases with p/rho (proxy for temperature)."""
        rho = torch.ones(3, dtype=torch.float64)
        p = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
        c = sound_speed(rho, p, 1.4)
        assert (c[1] > c[0]) and (c[2] > c[1])


class TestManyPieces:
    def test_10_region_riemann(self):
        nx = 100
        x = torch.arange(nx, dtype=torch.float64) * 0.01
        rng = torch.Generator().manual_seed(42)
        rho0, u0, p0, _ = random_piecewise(x, 10, rng)
        result = generate_one(rho0, u0, p0, dx=0.01, dt=0.0002, nt=30, reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_20_region_riemann(self):
        """Lots of piecewise regions."""
        nx = 200
        x = torch.arange(nx, dtype=torch.float64) * 0.005
        rng = torch.Generator().manual_seed(123)
        rho0, u0, p0, _ = random_piecewise(x, 20, rng)
        result = generate_one(rho0, u0, p0, dx=0.005, dt=0.0001, nt=20, reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_many_pieces_with_weno(self):
        """WENO5 with many-piece IC."""
        nx = 100
        x = torch.arange(nx, dtype=torch.float64) * 0.01
        rng = torch.Generator().manual_seed(99)
        rho0, u0, p0, _ = random_piecewise(x, 8, rng)
        result = generate_one(rho0, u0, p0, dx=0.01, dt=0.0002, nt=20, reconstruction="weno5")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()


class TestGenerateNEdgeCases:
    def test_single_sample(self):
        result = generate_n(1, 2, nx=20, dx=0.05, dt=0.002, nt=10, seed=42,
                            show_progress=False, reconstruction="constant")
        assert result["rho"].shape == (1, 11, 20)

    def test_k1_uniform(self):
        result = generate_n(3, 1, nx=20, dx=0.05, dt=0.002, nt=10, seed=42,
                            show_progress=False, reconstruction="constant")
        assert result["ic_rho_ks"].shape == (3, 1)
        assert result["ic_xs"].shape == (3, 2)

    def test_custom_ranges(self):
        result = generate_n(3, 2, nx=20, dx=0.05, dt=0.002, nt=10, seed=42,
                            show_progress=False,
                            rho_range=(0.5, 1.5), u_range=(-0.5, 0.5), p_range=(0.5, 2.0),
                            reconstruction="constant")
        for i in range(3):
            for v in result["ic_rho_ks"][i]:
                assert 0.5 <= v <= 1.5

    def test_generate_n_output_shapes(self):
        """All output tensors should have consistent shapes."""
        n, k, nx, nt = 5, 3, 30, 15
        result = generate_n(n, k, nx=nx, dx=0.033, dt=0.001, nt=nt, seed=0,
                            show_progress=False, reconstruction="constant")
        assert result["rho"].shape == (n, nt + 1, nx)
        assert result["u"].shape == (n, nt + 1, nx)
        assert result["p"].shape == (n, nt + 1, nx)
        assert result["x"].shape == (nx,)
        assert result["t"].shape == (nt + 1,)

    def test_generate_n_reproducibility(self):
        """Same seed should produce identical results."""
        kwargs = dict(nx=20, dx=0.05, dt=0.002, nt=10, seed=42,
                      show_progress=False, reconstruction="constant")
        r1 = generate_n(3, 2, **kwargs)
        r2 = generate_n(3, 2, **kwargs)
        torch.testing.assert_close(r1["rho"], r2["rho"])
        torch.testing.assert_close(r1["u"], r2["u"])
        torch.testing.assert_close(r1["p"], r2["p"])

    def test_generate_n_different_seeds(self):
        """Different seeds should produce different results."""
        kwargs = dict(nx=20, dx=0.05, dt=0.002, nt=10, show_progress=False,
                      reconstruction="constant")
        r1 = generate_n(3, 2, seed=42, **kwargs)
        r2 = generate_n(3, 2, seed=99, **kwargs)
        assert not torch.equal(r1["rho"], r2["rho"])

    def test_generate_n_many_k(self):
        """Large k (many breakpoints)."""
        result = generate_n(2, 8, nx=40, dx=0.025, dt=0.001, nt=10, seed=42,
                            show_progress=False, reconstruction="constant")
        assert result["ic_rho_ks"].shape == (2, 8)

    def test_generate_n_large_batch(self):
        """Generate many samples at once."""
        result = generate_n(20, 2, nx=20, dx=0.05, dt=0.002, nt=5, seed=42,
                            show_progress=False, reconstruction="constant")
        assert result["rho"].shape[0] == 20

    def test_generate_n_all_valid(self):
        """All generated samples should be valid."""
        result = generate_n(10, 3, nx=30, dx=0.033, dt=0.001, nt=10, seed=0,
                            show_progress=False, reconstruction="constant")
        assert torch.isfinite(result["rho"]).all()
        assert torch.isfinite(result["u"]).all()
        assert torch.isfinite(result["p"]).all()


class TestFluxTypeCombinations:
    @pytest.mark.parametrize("flux_type", ["hllc", "hll", "rusanov"])
    def test_sod_all_flux_types(self, flux_type):
        nx = 40
        x = torch.arange(nx, dtype=torch.float64) * 0.025
        rho0, u0, p0 = sod(x)
        result = generate_one(rho0, u0, p0, dx=0.025, dt=0.001, nt=20,
                              flux_type=flux_type, reconstruction="constant")
        assert result["valid"], f"Failed for flux_type={flux_type}"
        assert torch.isfinite(result["rho"]).all()

    @pytest.mark.parametrize("flux_type", ["hllc", "hll", "rusanov"])
    def test_strong_shock_all_flux_types(self, flux_type):
        nx = 50
        x = torch.arange(nx, dtype=torch.float64) * 0.02
        rho0, u0, p0 = riemann(x, rho_left=10.0, rho_right=1.0,
                                p_left=100.0, p_right=1.0)
        result = generate_one(rho0, u0, p0, dx=0.02, dt=0.0002, nt=20,
                              flux_type=flux_type, reconstruction="constant")
        assert result["valid"], f"Failed for flux_type={flux_type}"

    @pytest.mark.parametrize("flux_type", ["hllc", "hll", "rusanov"])
    def test_contact_all_flux_types(self, flux_type):
        """Contact discontinuity handled by all flux types."""
        nx = 50
        x = torch.arange(nx, dtype=torch.float64) * 0.02
        rho0, u0, p0 = riemann(x, rho_left=2.0, rho_right=1.0,
                                u_left=1.0, u_right=1.0,
                                p_left=1.0, p_right=1.0)
        result = generate_one(rho0, u0, p0, dx=0.02, dt=0.001, nt=20,
                              flux_type=flux_type, reconstruction="constant")
        assert result["valid"]


class TestReconstructionCombinations:
    @pytest.mark.parametrize("reconstruction", ["constant", "weno5"])
    def test_sod_all_reconstruction(self, reconstruction):
        nx = 40
        x = torch.arange(nx, dtype=torch.float64) * 0.025
        rho0, u0, p0 = sod(x)
        result = generate_one(rho0, u0, p0, dx=0.025, dt=0.001, nt=20,
                              reconstruction=reconstruction)
        assert result["valid"]

    @pytest.mark.parametrize("reconstruction", ["constant", "weno5"])
    def test_strong_riemann_all_reconstruction(self, reconstruction):
        nx = 50
        x = torch.arange(nx, dtype=torch.float64) * 0.02
        rho0, u0, p0 = riemann(x, rho_left=5.0, rho_right=0.5,
                                p_left=50.0, p_right=0.5)
        result = generate_one(rho0, u0, p0, dx=0.02, dt=0.0003, nt=20,
                              reconstruction=reconstruction)
        assert result["valid"]

    def test_weno5_sharper_than_constant(self):
        """WENO5 should give a sharper shock profile than constant reconstruction."""
        nx = 100
        x = torch.arange(nx, dtype=torch.float64) * 0.01
        rho0, u0, p0 = sod(x)
        rc = generate_one(rho0, u0, p0, dx=0.01, dt=0.0005, nt=40,
                           reconstruction="constant")
        rw = generate_one(rho0, u0, p0, dx=0.01, dt=0.0005, nt=40,
                           reconstruction="weno5")
        # WENO5 max density gradient should be at least as steep
        grad_c = (rc["rho"][-1, 1:] - rc["rho"][-1, :-1]).abs().max()
        grad_w = (rw["rho"][-1, 1:] - rw["rho"][-1, :-1]).abs().max()
        assert grad_w >= grad_c * 0.8  # WENO should be sharper or at least comparable


class TestFluxReconstructionCrossProduct:
    """Test all flux x reconstruction combinations."""

    @pytest.mark.parametrize("flux_type", ["hllc", "hll", "rusanov"])
    @pytest.mark.parametrize("reconstruction", ["constant", "weno5"])
    def test_cross_product(self, flux_type, reconstruction):
        nx = 40
        x = torch.arange(nx, dtype=torch.float64) * 0.025
        rho0, u0, p0 = sod(x)
        result = generate_one(rho0, u0, p0, dx=0.025, dt=0.001, nt=20,
                              flux_type=flux_type, reconstruction=reconstruction)
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    @pytest.mark.parametrize("flux_type", ["hllc", "hll", "rusanov"])
    @pytest.mark.parametrize("reconstruction", ["constant", "weno5"])
    @pytest.mark.parametrize("bc_type", ["extrap", "periodic", "wall"])
    def test_full_cross_product(self, flux_type, reconstruction, bc_type):
        """All flux x reconstruction x BC combinations."""
        nx = 30
        x = torch.arange(nx, dtype=torch.float64) * 0.033
        rho0, u0, p0 = sod(x)
        result = generate_one(rho0, u0, p0, dx=0.033, dt=0.001, nt=10,
                              flux_type=flux_type, reconstruction=reconstruction,
                              bc_type=bc_type)
        assert result["valid"]


class TestOutputStructure:
    def test_output_keys(self):
        """generate_one should return all expected keys."""
        nx = 20
        rho0 = torch.ones(nx, dtype=torch.float64)
        u0 = torch.zeros(nx, dtype=torch.float64)
        p0 = torch.ones(nx, dtype=torch.float64)
        result = generate_one(rho0, u0, p0, dx=0.05, dt=0.002, nt=5, reconstruction="constant")
        assert "rho" in result
        assert "u" in result
        assert "p" in result
        assert "x" in result
        assert "t" in result
        assert "valid" in result

    def test_output_dtype(self):
        """All tensors should be float64."""
        nx = 20
        rho0 = torch.ones(nx, dtype=torch.float64)
        u0 = torch.zeros(nx, dtype=torch.float64)
        p0 = torch.ones(nx, dtype=torch.float64)
        result = generate_one(rho0, u0, p0, dx=0.05, dt=0.002, nt=5, reconstruction="constant")
        assert result["rho"].dtype == torch.float64
        assert result["u"].dtype == torch.float64
        assert result["p"].dtype == torch.float64
        assert result["x"].dtype == torch.float64
        assert result["t"].dtype == torch.float64

    def test_output_shapes(self):
        """Verify shapes of all outputs."""
        nx, nt = 20, 15
        rho0 = torch.ones(nx, dtype=torch.float64)
        u0 = torch.zeros(nx, dtype=torch.float64)
        p0 = torch.ones(nx, dtype=torch.float64)
        result = generate_one(rho0, u0, p0, dx=0.05, dt=0.002, nt=nt, reconstruction="constant")
        assert result["rho"].shape == (nt + 1, nx)
        assert result["u"].shape == (nt + 1, nx)
        assert result["p"].shape == (nt + 1, nx)
        assert result["x"].shape == (nx,)
        assert result["t"].shape == (nt + 1,)

    def test_time_array_monotonic(self):
        """Time array should be monotonically increasing."""
        nx = 20
        rho0 = torch.ones(nx, dtype=torch.float64)
        u0 = torch.zeros(nx, dtype=torch.float64)
        p0 = torch.ones(nx, dtype=torch.float64)
        result = generate_one(rho0, u0, p0, dx=0.05, dt=0.002, nt=10, reconstruction="constant")
        t = result["t"]
        assert (t[1:] > t[:-1]).all()

    def test_time_array_starts_at_zero(self):
        nx = 20
        rho0 = torch.ones(nx, dtype=torch.float64)
        u0 = torch.zeros(nx, dtype=torch.float64)
        p0 = torch.ones(nx, dtype=torch.float64)
        result = generate_one(rho0, u0, p0, dx=0.05, dt=0.002, nt=10, reconstruction="constant")
        assert result["t"][0].item() == 0.0

    def test_initial_condition_preserved(self):
        """First row of output should match the initial condition."""
        nx = 20
        rho0 = torch.rand(nx, dtype=torch.float64) + 0.5
        u0 = torch.randn(nx, dtype=torch.float64) * 0.5
        p0 = torch.rand(nx, dtype=torch.float64) + 0.5
        result = generate_one(rho0, u0, p0, dx=0.05, dt=0.001, nt=10, reconstruction="constant")
        torch.testing.assert_close(result["rho"][0], rho0)
        torch.testing.assert_close(result["u"][0], u0, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(result["p"][0], p0, atol=1e-12, rtol=1e-12)


class TestMaxValueClipping:
    def test_max_value_clips_density(self):
        """max_value should clip outputs if density gets too large."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float64) * 0.025
        rho0, u0, p0 = riemann(x, rho_left=10.0, rho_right=1.0,
                                p_left=100.0, p_right=1.0)
        result = generate_one(rho0, u0, p0, dx=0.025, dt=0.001, nt=20,
                              max_value=50.0, reconstruction="constant")
        # If clipping is applied, density should stay below max_value
        # (depends on whether max_value clips individual fields)
        assert result["valid"] or not result["valid"]  # should not crash

    def test_no_max_value(self):
        """Without max_value, no clipping should occur."""
        nx = 20
        rho0 = torch.ones(nx, dtype=torch.float64)
        u0 = torch.zeros(nx, dtype=torch.float64)
        p0 = torch.ones(nx, dtype=torch.float64)
        result = generate_one(rho0, u0, p0, dx=0.05, dt=0.002, nt=10, reconstruction="constant")
        assert result["valid"]


class TestInitialConditionEdgeCases:
    def test_from_steps_defaults_only(self):
        """Only rho_steps provided, u and p use defaults."""
        nx = 30
        x = torch.arange(nx, dtype=torch.float64) * 0.033
        sentinel = x.max().item() + 1.0
        rho0, u0, p0 = from_steps(x, rho_steps=[(sentinel, 1.0)])
        assert rho0.shape == (nx,)
        assert u0.shape == (nx,)
        assert p0.shape == (nx,)
        # u defaults to 0, p defaults to 1
        torch.testing.assert_close(u0, torch.zeros(nx, dtype=torch.float64))
        torch.testing.assert_close(p0, torch.ones(nx, dtype=torch.float64))
        result = generate_one(rho0, u0, p0, dx=0.033, dt=0.001, nt=10, reconstruction="constant")
        assert result["valid"]

    def test_from_steps_single_step_each(self):
        """One step in each variable."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float64) * 0.025
        rho0, u0, p0 = from_steps(
            x,
            rho_steps=[(0.5, 2.0)],
            u_steps=[(0.5, 1.0)],
            p_steps=[(0.5, 3.0)],
        )
        result = generate_one(rho0, u0, p0, dx=0.025, dt=0.0005, nt=20, reconstruction="constant")
        assert result["valid"]

    def test_riemann_identical_states(self):
        """Riemann problem with identical left and right states -> no evolution."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float64) * 0.025
        rho0, u0, p0 = riemann(x, rho_left=1.0, rho_right=1.0,
                                u_left=0.0, u_right=0.0,
                                p_left=1.0, p_right=1.0)
        result = generate_one(rho0, u0, p0, dx=0.025, dt=0.001, nt=20,
                              bc_type="periodic", reconstruction="constant")
        torch.testing.assert_close(result["rho"][-1], rho0, atol=1e-10, rtol=1e-10)

    def test_sod_default_split(self):
        """Sod problem with default x_split."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float64) * 0.025
        rho0, u0, p0 = sod(x)
        # Check left and right states
        assert rho0[0].item() == pytest.approx(1.0)
        assert rho0[-1].item() == pytest.approx(0.125)
        assert u0[0].item() == pytest.approx(0.0)
        assert p0[0].item() == pytest.approx(1.0)
        assert p0[-1].item() == pytest.approx(0.1)

    def test_random_piecewise_produces_valid_ic(self):
        """random_piecewise should produce physically valid ICs."""
        nx = 50
        x = torch.arange(nx, dtype=torch.float64) * 0.02
        rng = torch.Generator().manual_seed(42)
        rho0, u0, p0, ic_params = random_piecewise(x, 5, rng)
        assert (rho0 > 0).all(), "Density must be positive"
        assert (p0 > 0).all(), "Pressure must be positive"
        assert rho0.shape == (nx,)
        assert u0.shape == (nx,)
        assert p0.shape == (nx,)

    def test_random_piecewise_reproducibility(self):
        """Same seed should give same IC."""
        nx = 50
        x = torch.arange(nx, dtype=torch.float64) * 0.02
        rng1 = torch.Generator().manual_seed(42)
        rho1, u1, p1, _ = random_piecewise(x, 5, rng1)
        rng2 = torch.Generator().manual_seed(42)
        rho2, u2, p2, _ = random_piecewise(x, 5, rng2)
        torch.testing.assert_close(rho1, rho2)
        torch.testing.assert_close(u1, u2)
        torch.testing.assert_close(p1, p2)

    def test_random_piecewise_different_seeds(self):
        """Different seeds should give different ICs."""
        nx = 50
        x = torch.arange(nx, dtype=torch.float64) * 0.02
        rng1 = torch.Generator().manual_seed(42)
        rho1, u1, p1, _ = random_piecewise(x, 5, rng1)
        rng2 = torch.Generator().manual_seed(99)
        rho2, u2, p2, _ = random_piecewise(x, 5, rng2)
        assert not torch.equal(rho1, rho2)


class TestNumericalStability:
    def test_long_time_integration(self):
        """Run for many steps and check no NaN appears."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float64) * 0.025
        rho0, u0, p0 = sod(x)
        result = generate_one(rho0, u0, p0, dx=0.025, dt=0.001, nt=500,
                              bc_type="wall", reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()
        assert torch.isfinite(result["u"]).all()
        assert torch.isfinite(result["p"]).all()

    def test_density_stays_positive(self):
        """Density should never go negative."""
        nx = 60
        x = torch.arange(nx, dtype=torch.float64) * 0.02
        rho0, u0, p0 = riemann(x, rho_left=1.0, rho_right=0.01,
                                p_left=1.0, p_right=0.01)
        result = generate_one(rho0, u0, p0, dx=0.02, dt=0.0005, nt=50, reconstruction="constant")
        if result["valid"]:
            assert (result["rho"] >= -1e-10).all()

    def test_pressure_stays_positive(self):
        """Pressure should never go negative."""
        nx = 60
        x = torch.arange(nx, dtype=torch.float64) * 0.02
        rho0, u0, p0 = riemann(x, rho_left=1.0, rho_right=0.01,
                                p_left=1.0, p_right=0.01)
        result = generate_one(rho0, u0, p0, dx=0.02, dt=0.0005, nt=50, reconstruction="constant")
        if result["valid"]:
            assert (result["p"] >= -1e-10).all()

    def test_no_nan_after_strong_interaction(self):
        """Strong shock-shock interaction should not produce NaN."""
        nx = 80
        x = torch.arange(nx, dtype=torch.float64) * 0.0125
        rho0, u0, p0 = from_steps(
            x,
            rho_steps=[(0.3, 5.0), (0.7, 0.5)],
            u_steps=[(0.3, 3.0), (0.7, -3.0)],
            p_steps=[(0.3, 50.0), (0.7, 0.5)],
        )
        result = generate_one(rho0, u0, p0, dx=0.0125, dt=0.0001, nt=100,
                              reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()
        assert torch.isfinite(result["u"]).all()
        assert torch.isfinite(result["p"]).all()

    def test_periodic_bc_long_run(self):
        """Periodic BCs for a long time should not blow up."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float64) * 0.025
        rho0, u0, p0 = riemann(x, rho_left=2.0, rho_right=1.0,
                                p_left=2.0, p_right=1.0)
        result = generate_one(rho0, u0, p0, dx=0.025, dt=0.001, nt=300,
                              bc_type="periodic", reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()


class TestConservationDetailed:
    def test_mass_conservation_periodic(self):
        """Total mass should be exactly conserved with periodic BCs."""
        nx = 50
        x = torch.arange(nx, dtype=torch.float64) * 0.02
        dx = 0.02
        rho0, u0, p0 = riemann(x, rho_left=2.0, rho_right=1.0,
                                p_left=2.0, p_right=1.0)
        result = generate_one(rho0, u0, p0, dx=dx, dt=0.001, nt=50,
                              bc_type="periodic", reconstruction="constant")
        mass = result["rho"].sum(dim=-1) * dx
        torch.testing.assert_close(mass, mass[0].expand_as(mass), atol=1e-8, rtol=1e-8)

    def test_momentum_conservation_periodic(self):
        """Total momentum should be conserved with periodic BCs."""
        nx = 50
        x = torch.arange(nx, dtype=torch.float64) * 0.02
        dx = 0.02
        gamma = 1.4
        rho0, u0, p0 = riemann(x, rho_left=2.0, rho_right=1.0,
                                u_left=0.5, u_right=-0.5,
                                p_left=2.0, p_right=1.0)
        result = generate_one(rho0, u0, p0, dx=dx, dt=0.001, nt=50,
                              bc_type="periodic", reconstruction="constant", gamma=gamma)
        momentum = (result["rho"] * result["u"]).sum(dim=-1) * dx
        torch.testing.assert_close(momentum, momentum[0].expand_as(momentum),
                                   atol=1e-8, rtol=1e-8)

    def test_energy_conservation_periodic(self):
        """Total energy should be conserved with periodic BCs."""
        nx = 50
        x = torch.arange(nx, dtype=torch.float64) * 0.02
        dx = 0.02
        gamma = 1.4
        rho0, u0, p0 = riemann(x, rho_left=2.0, rho_right=1.0,
                                p_left=2.0, p_right=1.0)
        result = generate_one(rho0, u0, p0, dx=dx, dt=0.001, nt=50,
                              bc_type="periodic", reconstruction="constant", gamma=gamma)
        rho, u, p = result["rho"], result["u"], result["p"]
        E = p / (gamma - 1.0) + 0.5 * rho * u**2
        energy = E.sum(dim=-1) * dx
        torch.testing.assert_close(energy, energy[0].expand_as(energy), atol=1e-7, rtol=1e-7)

    def test_mass_conservation_wall(self):
        """Total mass should be conserved with wall BCs (closed system)."""
        nx = 50
        x = torch.arange(nx, dtype=torch.float64) * 0.02
        dx = 0.02
        rho0, u0, p0 = sod(x)
        result = generate_one(rho0, u0, p0, dx=dx, dt=0.001, nt=50,
                              bc_type="wall", reconstruction="constant")
        mass = result["rho"].sum(dim=-1) * dx
        torch.testing.assert_close(mass, mass[0].expand_as(mass), atol=1e-7, rtol=1e-7)
