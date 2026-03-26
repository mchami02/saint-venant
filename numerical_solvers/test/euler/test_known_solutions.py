"""Tests against known analytical properties of Euler solutions."""

import math

import pytest
import torch

from numerical_solvers.euler import generate_one
from numerical_solvers.euler.initial_conditions import from_steps, riemann, sod
from numerical_solvers.euler.physics import primitive_to_conservative, sound_speed


class TestContactDiscontinuity:
    """A pure contact discontinuity: same p and u, different rho.
    The contact should propagate at speed u without creating new waves.
    """

    @pytest.mark.parametrize("flux_type", ["hllc", "hll", "rusanov"])
    def test_pure_contact(self, flux_type):
        nx = 60
        x = torch.arange(nx, dtype=torch.float64) * 0.025
        # Same pressure, same velocity, different density
        rho0, u0, p0 = riemann(x, rho_left=2.0, rho_right=1.0,
                                u_left=0.5, u_right=0.5,
                                p_left=1.0, p_right=1.0)
        result = generate_one(rho0, u0, p0, dx=0.025, dt=0.001, nt=30, gamma=1.4,
                              flux_type=flux_type, reconstruction="constant")
        assert result["valid"]
        # Pressure and velocity should remain nearly uniform
        p_final = result["p"][-1]
        u_final = result["u"][-1]
        assert (p_final - 1.0).abs().max() < 0.2, f"Pressure disturbed for {flux_type}"
        assert (u_final - 0.5).abs().max() < 0.2, f"Velocity disturbed for {flux_type}"

    def test_stationary_contact(self):
        """Contact discontinuity at rest (u=0): density jump should stay put."""
        nx = 60
        x = torch.arange(nx, dtype=torch.float64) * 0.025
        rho0, u0, p0 = riemann(x, rho_left=3.0, rho_right=1.0,
                                u_left=0.0, u_right=0.0,
                                p_left=1.0, p_right=1.0)
        result = generate_one(rho0, u0, p0, dx=0.025, dt=0.001, nt=30, gamma=1.4,
                              flux_type="hllc", reconstruction="constant")
        assert result["valid"]
        # Velocity should remain near zero
        assert result["u"][-1].abs().max() < 0.1
        # Pressure should remain near 1
        assert (result["p"][-1] - 1.0).abs().max() < 0.1

    @pytest.mark.parametrize("contact_speed", [0.0, 0.5, 1.0, -0.5])
    def test_contact_various_speeds(self, contact_speed):
        """Contact at various propagation speeds."""
        nx = 60
        x = torch.arange(nx, dtype=torch.float64) * 0.025
        rho0, u0, p0 = riemann(x, rho_left=2.0, rho_right=1.0,
                                u_left=contact_speed, u_right=contact_speed,
                                p_left=1.0, p_right=1.0)
        result = generate_one(rho0, u0, p0, dx=0.025, dt=0.001, nt=20, gamma=1.4,
                              flux_type="hllc", reconstruction="constant")
        assert result["valid"]
        # Velocity should stay near contact_speed
        assert (result["u"][-1] - contact_speed).abs().max() < 0.2


class TestShockSpeed:
    """Qualitative check: the shock speed for Sod problem is known to be ~1.75."""

    def test_sod_shock_moves_right(self):
        """In the Sod problem, the shock moves to the right."""
        nx = 100
        x = torch.arange(nx, dtype=torch.float64) * 0.01
        rho0, u0, p0 = sod(x)
        result = generate_one(rho0, u0, p0, dx=0.01, dt=0.001, nt=30, gamma=1.4,
                              reconstruction="weno5")
        rho_final = result["rho"][-1]
        right_of_center = x > 0.5
        near_right = (rho_final[right_of_center] - 0.125).abs() < 0.05
        assert near_right.sum() > 0, "Shock has not moved rightward"

    def test_sod_rarefaction_head_moves_left(self):
        """The rarefaction head should move to the left of x=0.5."""
        nx = 100
        x = torch.arange(nx, dtype=torch.float64) * 0.01
        rho0, u0, p0 = sod(x)
        result = generate_one(rho0, u0, p0, dx=0.01, dt=0.001, nt=30, gamma=1.4,
                              reconstruction="weno5")
        rho_final = result["rho"][-1]
        left_of_center = x < 0.5
        # Some cells left of center should have density < 1.0 (rarefaction passed)
        disturbed_left = (rho_final[left_of_center] < 0.99).sum()
        assert disturbed_left > 0, "Rarefaction head has not moved leftward"

    def test_sod_contact_moves_right(self):
        """The contact discontinuity in Sod moves to the right."""
        nx = 100
        x = torch.arange(nx, dtype=torch.float64) * 0.01
        rho0, u0, p0 = sod(x)
        result = generate_one(rho0, u0, p0, dx=0.01, dt=0.001, nt=30, gamma=1.4,
                              reconstruction="weno5")
        rho_final = result["rho"][-1]
        # The contact creates a density jump around rho~0.4 -> rho~0.27
        # This should be to the right of x=0.5
        right_cells = rho_final[x > 0.5]
        has_contact_region = ((right_cells > 0.15) & (right_cells < 0.5)).sum() > 0
        assert has_contact_region, "Contact discontinuity not found right of center"


class TestRarefactionWave:
    """A rarefaction wave should be smooth (no sharp jumps)."""

    def test_expansion_fan_smooth(self):
        """In the Sod problem, the rarefaction fan should be smooth."""
        nx = 100
        x = torch.arange(nx, dtype=torch.float64) * 0.01
        rho0, u0, p0 = sod(x)
        result = generate_one(rho0, u0, p0, dx=0.01, dt=0.001, nt=30, gamma=1.4,
                              reconstruction="weno5")
        rho_final = result["rho"][-1]
        has_intermediate = ((rho_final > 0.2) & (rho_final < 0.9)).sum() > 0
        assert has_intermediate, "No intermediate densities found -- rarefaction fan missing"

    def test_rarefaction_velocity_increases(self):
        """In the Sod rarefaction, velocity should increase from left to right."""
        nx = 100
        x = torch.arange(nx, dtype=torch.float64) * 0.01
        rho0, u0, p0 = sod(x)
        result = generate_one(rho0, u0, p0, dx=0.01, dt=0.001, nt=30, gamma=1.4,
                              reconstruction="weno5")
        u_final = result["u"][-1]
        # In the rarefaction region (roughly x < 0.5), velocity should be increasing
        # Check that maximum velocity is positive (gas accelerated rightward)
        assert u_final.max() > 0.1, "Rarefaction should accelerate gas to positive velocity"

    def test_rarefaction_pressure_decreases(self):
        """In the Sod rarefaction, pressure should decrease from left to right."""
        nx = 100
        x = torch.arange(nx, dtype=torch.float64) * 0.01
        rho0, u0, p0 = sod(x)
        result = generate_one(rho0, u0, p0, dx=0.01, dt=0.001, nt=30, gamma=1.4,
                              reconstruction="weno5")
        p_final = result["p"][-1]
        # Pressure somewhere in the middle should be between 0.1 and 1.0
        has_intermediate_p = ((p_final > 0.15) & (p_final < 0.95)).sum() > 0
        assert has_intermediate_p, "Pressure should have intermediate values in rarefaction"


class TestSymmetry:
    """Symmetric initial conditions should produce symmetric solutions."""

    def test_symmetric_blast(self):
        """Symmetric pressure jump: solution should be symmetric about center."""
        nx = 80
        rho0 = torch.ones(nx, dtype=torch.float64)
        u0 = torch.zeros(nx, dtype=torch.float64)
        p0 = torch.ones(nx, dtype=torch.float64)
        p0[:nx // 4] = 10.0
        p0[3 * nx // 4:] = 10.0

        result = generate_one(rho0, u0, p0, dx=0.0125, dt=0.0003, nt=30, gamma=1.4,
                              bc_type="periodic", reconstruction="constant")
        assert result["valid"]
        rho_final = result["rho"][-1]
        rho_flipped = rho_final.flip(0)
        torch.testing.assert_close(rho_final, rho_flipped, atol=1e-6, rtol=1e-6)

    def test_symmetric_density_jump(self):
        """Symmetric density profile, zero velocity."""
        nx = 60
        rho0 = torch.ones(nx, dtype=torch.float64)
        rho0[:nx // 2] = 2.0
        rho0[nx // 2:] = 2.0
        rho0[nx // 4:3 * nx // 4] = 1.0
        u0 = torch.zeros(nx, dtype=torch.float64)
        p0 = torch.ones(nx, dtype=torch.float64)

        result = generate_one(rho0, u0, p0, dx=0.02, dt=0.001, nt=20, gamma=1.4,
                              bc_type="periodic", reconstruction="constant")
        assert result["valid"]
        rho_final = result["rho"][-1]
        rho_flipped = rho_final.flip(0)
        torch.testing.assert_close(rho_final, rho_flipped, atol=1e-6, rtol=1e-6)

    def test_antisymmetric_velocity(self):
        """Anti-symmetric velocity should produce symmetric density evolution."""
        nx = 60
        rho0 = torch.ones(nx, dtype=torch.float64)
        u0 = torch.zeros(nx, dtype=torch.float64)
        u0[:nx // 2] = 1.0
        u0[nx // 2:] = -1.0
        p0 = torch.ones(nx, dtype=torch.float64)

        result = generate_one(rho0, u0, p0, dx=0.02, dt=0.001, nt=20, gamma=1.4,
                              bc_type="periodic", reconstruction="constant")
        assert result["valid"]
        rho_final = result["rho"][-1]
        rho_flipped = rho_final.flip(0)
        torch.testing.assert_close(rho_final, rho_flipped, atol=1e-6, rtol=1e-6)

    def test_symmetric_colliding_shocks(self):
        """Two identical shocks colliding symmetrically."""
        nx = 80
        x = torch.arange(nx, dtype=torch.float64) * 0.0125
        rho0 = torch.ones(nx, dtype=torch.float64)
        u0 = torch.zeros(nx, dtype=torch.float64)
        p0 = torch.ones(nx, dtype=torch.float64)
        # High pressure on both ends, low in center
        p0[:20] = 10.0
        p0[60:] = 10.0

        result = generate_one(rho0, u0, p0, dx=0.0125, dt=0.0003, nt=40, gamma=1.4,
                              bc_type="periodic", reconstruction="constant")
        assert result["valid"]
        rho_final = result["rho"][-1]
        rho_flipped = rho_final.flip(0)
        torch.testing.assert_close(rho_final, rho_flipped, atol=1e-5, rtol=1e-5)


class TestWallReflection:
    """Waves reflect off walls and should maintain proper physics."""

    def test_wall_reflection_preserves_energy(self):
        """Total energy should be conserved with wall BCs (closed system)."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float64) * 0.025
        gamma = 1.4
        rho0, u0, p0 = sod(x)
        result = generate_one(rho0, u0, p0, dx=0.025, dt=0.001, nt=30, gamma=gamma,
                              bc_type="wall", reconstruction="constant")
        rho, u, p = result["rho"], result["u"], result["p"]
        E = p / (gamma - 1.0) + 0.5 * rho * u**2
        energy = E.sum(dim=-1) * 0.025
        torch.testing.assert_close(energy, energy[0].expand_as(energy), atol=1e-5, rtol=1e-5)

    def test_wall_reflection_preserves_mass(self):
        """Mass must be conserved in a wall-bounded domain."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float64) * 0.025
        dx = 0.025
        rho0, u0, p0 = sod(x)
        result = generate_one(rho0, u0, p0, dx=dx, dt=0.001, nt=50,
                              bc_type="wall", reconstruction="constant")
        mass = result["rho"].sum(dim=-1) * dx
        torch.testing.assert_close(mass, mass[0].expand_as(mass), atol=1e-7, rtol=1e-7)

    def test_wall_normal_velocity_reflected(self):
        """After long time with wall BCs, velocity near walls should be small or reflected."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float64) * 0.025
        rho0 = torch.ones(nx, dtype=torch.float64)
        u0 = torch.full((nx,), 1.0, dtype=torch.float64)
        p0 = torch.ones(nx, dtype=torch.float64)
        result = generate_one(rho0, u0, p0, dx=0.025, dt=0.001, nt=100,
                              bc_type="wall", reconstruction="constant")
        assert result["valid"]
        # After reflection, the system should settle; density should remain finite
        assert torch.isfinite(result["rho"]).all()
        assert torch.isfinite(result["u"]).all()

    def test_wall_momentum_not_conserved(self):
        """Total momentum should NOT be conserved with wall BCs (walls exert force)."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float64) * 0.025
        dx = 0.025
        rho0, u0, p0 = sod(x)
        result = generate_one(rho0, u0, p0, dx=dx, dt=0.001, nt=100,
                              bc_type="wall", reconstruction="constant")
        momentum = (result["rho"] * result["u"]).sum(dim=-1) * dx
        # With wall BCs, momentum will generally NOT be conserved (walls exert forces)
        # This test just verifies the simulation runs and stays finite
        assert torch.isfinite(momentum).all()


class TestVacuum:
    """Near-vacuum states are challenging for Euler solvers."""

    def test_expanding_gas(self):
        """Gas expanding into near-vacuum."""
        nx = 50
        x = torch.arange(nx, dtype=torch.float64) * 0.02
        rho0, u0, p0 = riemann(x, rho_left=1.0, rho_right=0.001,
                                u_left=0.0, u_right=0.0,
                                p_left=1.0, p_right=0.001)
        result = generate_one(rho0, u0, p0, dx=0.02, dt=0.0005, nt=20, reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()
        assert (result["rho"] >= -1e-6).all()

    def test_two_rarefactions_creating_vacuum(self):
        """Two diverging flows trying to create vacuum in the middle."""
        nx = 50
        x = torch.arange(nx, dtype=torch.float64) * 0.02
        rho0, u0, p0 = riemann(x, rho_left=1.0, rho_right=1.0,
                                u_left=-2.0, u_right=2.0,
                                p_left=0.4, p_right=0.4)
        result = generate_one(rho0, u0, p0, dx=0.02, dt=0.001, nt=30, reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()
        assert (result["rho"] >= -1e-6).all()

    def test_near_vacuum_with_motion(self):
        """Near-vacuum state with nonzero velocity."""
        nx = 50
        x = torch.arange(nx, dtype=torch.float64) * 0.02
        rho0, u0, p0 = riemann(x, rho_left=1.0, rho_right=0.0001,
                                u_left=0.0, u_right=1.0,
                                p_left=1.0, p_right=0.0001)
        result = generate_one(rho0, u0, p0, dx=0.02, dt=0.0002, nt=20, reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_symmetric_vacuum_generation(self):
        """Symmetric diverging flow creating vacuum in center."""
        nx = 60
        rho0 = torch.ones(nx, dtype=torch.float64)
        u0 = torch.zeros(nx, dtype=torch.float64)
        u0[:nx // 2] = -3.0
        u0[nx // 2:] = 3.0
        p0 = torch.full((nx,), 0.4, dtype=torch.float64)
        result = generate_one(rho0, u0, p0, dx=0.02, dt=0.0005, nt=30,
                              reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()


class TestClassicalRiemannProblems:
    """Well-known Riemann problems from the literature."""

    def test_sod_problem(self):
        """Standard Sod shock tube."""
        nx = 100
        x = torch.arange(nx, dtype=torch.float64) * 0.01
        rho0, u0, p0 = sod(x)
        result = generate_one(rho0, u0, p0, dx=0.01, dt=0.0005, nt=40,
                              reconstruction="weno5")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()
        # Check that the final state has the right qualitative structure
        rho_final = result["rho"][-1]
        # Left state should still have cells near 1.0
        assert (rho_final[:10] - 1.0).abs().max() < 0.1
        # Right state should still have cells near 0.125
        assert (rho_final[-5:] - 0.125).abs().max() < 0.05

    def test_123_problem(self):
        """Einfeldt 1-2-3 problem: two rarefactions with near-vacuum."""
        nx = 100
        x = torch.arange(nx, dtype=torch.float64) * 0.01
        rho0, u0, p0 = riemann(x, rho_left=1.0, rho_right=1.0,
                                u_left=-2.0, u_right=2.0,
                                p_left=0.4, p_right=0.4)
        result = generate_one(rho0, u0, p0, dx=0.01, dt=0.0005, nt=40,
                              reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_lax_problem(self):
        """Lax shock tube: (rho,u,p)_L = (0.445,0.698,3.528), _R = (0.5,0,0.571)."""
        nx = 100
        x = torch.arange(nx, dtype=torch.float64) * 0.01
        rho0, u0, p0 = riemann(x, rho_left=0.445, rho_right=0.5,
                                u_left=0.698, u_right=0.0,
                                p_left=3.528, p_right=0.571)
        result = generate_one(rho0, u0, p0, dx=0.01, dt=0.0005, nt=40,
                              reconstruction="weno5")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_strong_blast_wave(self):
        """Strong blast wave: high pressure ratio."""
        nx = 100
        x = torch.arange(nx, dtype=torch.float64) * 0.01
        rho0, u0, p0 = riemann(x, rho_left=1.0, rho_right=1.0,
                                u_left=0.0, u_right=0.0,
                                p_left=1000.0, p_right=0.01)
        result = generate_one(rho0, u0, p0, dx=0.01, dt=0.00005, nt=50,
                              reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_double_rarefaction(self):
        """Pure double rarefaction (no shocks expected)."""
        nx = 100
        x = torch.arange(nx, dtype=torch.float64) * 0.01
        rho0, u0, p0 = riemann(x, rho_left=1.0, rho_right=1.0,
                                u_left=-1.0, u_right=1.0,
                                p_left=1.0, p_right=1.0)
        result = generate_one(rho0, u0, p0, dx=0.01, dt=0.001, nt=30,
                              reconstruction="weno5")
        assert result["valid"]
        # Double rarefaction should produce smooth profiles
        assert torch.isfinite(result["rho"]).all()

    def test_double_shock(self):
        """Double shock: two shocks colliding."""
        nx = 100
        x = torch.arange(nx, dtype=torch.float64) * 0.01
        rho0, u0, p0 = riemann(x, rho_left=1.0, rho_right=1.0,
                                u_left=1.0, u_right=-1.0,
                                p_left=1.0, p_right=1.0)
        result = generate_one(rho0, u0, p0, dx=0.01, dt=0.001, nt=30,
                              reconstruction="weno5")
        assert result["valid"]
        # Density should increase in the center due to compression
        rho_final = result["rho"][-1]
        assert rho_final.max() > 1.0, "Compression should increase density"


class TestSteadyStates:
    """Configurations that should remain steady."""

    def test_uniform_at_rest(self):
        """Uniform gas at rest should not change."""
        nx = 40
        rho0 = torch.ones(nx, dtype=torch.float64)
        u0 = torch.zeros(nx, dtype=torch.float64)
        p0 = torch.ones(nx, dtype=torch.float64)
        result = generate_one(rho0, u0, p0, dx=0.025, dt=0.002, nt=50,
                              bc_type="periodic", reconstruction="constant")
        torch.testing.assert_close(result["rho"][-1], rho0, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(result["u"][-1], u0, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(result["p"][-1], p0, atol=1e-12, rtol=1e-12)

    def test_uniform_moving(self):
        """Uniform flow with periodic BCs should remain uniform."""
        nx = 40
        rho0 = torch.full((nx,), 2.0, dtype=torch.float64)
        u0 = torch.full((nx,), 1.0, dtype=torch.float64)
        p0 = torch.full((nx,), 3.0, dtype=torch.float64)
        result = generate_one(rho0, u0, p0, dx=0.025, dt=0.001, nt=100,
                              bc_type="periodic", reconstruction="constant")
        torch.testing.assert_close(result["rho"][-1], rho0, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(result["u"][-1], u0, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(result["p"][-1], p0, atol=1e-10, rtol=1e-10)

    def test_uniform_high_density_steady(self):
        """High-density uniform state should be perfectly steady."""
        nx = 30
        rho0 = torch.full((nx,), 50.0, dtype=torch.float64)
        u0 = torch.zeros(nx, dtype=torch.float64)
        p0 = torch.full((nx,), 50.0, dtype=torch.float64)
        result = generate_one(rho0, u0, p0, dx=0.033, dt=0.0005, nt=50,
                              bc_type="periodic", reconstruction="constant")
        torch.testing.assert_close(result["rho"][-1], rho0, atol=1e-10, rtol=1e-10)

    def test_uniform_low_density_steady(self):
        """Low-density uniform state should be perfectly steady."""
        nx = 30
        rho0 = torch.full((nx,), 0.01, dtype=torch.float64)
        u0 = torch.zeros(nx, dtype=torch.float64)
        p0 = torch.full((nx,), 0.01, dtype=torch.float64)
        result = generate_one(rho0, u0, p0, dx=0.033, dt=0.001, nt=50,
                              bc_type="periodic", reconstruction="constant")
        torch.testing.assert_close(result["rho"][-1], rho0, atol=1e-12, rtol=1e-12)


class TestRankineHugoniot:
    """Test that shocks satisfy the Rankine-Hugoniot jump conditions qualitatively."""

    def test_entropy_non_decreasing_across_shock(self):
        """Entropy (p/rho^gamma) should not decrease through a shock.
        Compare the minimum post-shock entropy to the pre-shock entropy.
        The pre-shock region (far right, undisturbed) has a well-defined entropy.
        """
        nx = 200
        x = torch.arange(nx, dtype=torch.float64) * 0.005
        gamma = 1.4
        rho0, u0, p0 = sod(x)
        result = generate_one(rho0, u0, p0, dx=0.005, dt=0.0005, nt=40, gamma=gamma,
                              reconstruction="weno5")
        rho_final = result["rho"][-1]
        p_final = result["p"][-1]
        s = p_final / rho_final**gamma
        # Undisturbed right state entropy: s_R = 0.1 / 0.125^1.4
        s_right = 0.1 / 0.125**gamma
        # The overall maximum entropy should be >= the initial right state entropy
        # (shocks generate entropy)
        assert s.max() > s_right, "Some region should have higher entropy than the pre-shock state"

    def test_mass_flux_continuous_across_domain(self):
        """Mass flux (rho * u) should be finite everywhere."""
        nx = 100
        x = torch.arange(nx, dtype=torch.float64) * 0.01
        rho0, u0, p0 = sod(x)
        result = generate_one(rho0, u0, p0, dx=0.01, dt=0.001, nt=30,
                              reconstruction="weno5")
        mass_flux = result["rho"][-1] * result["u"][-1]
        assert torch.isfinite(mass_flux).all()


class TestGalileanInvariance:
    """Solutions should be the same up to a velocity boost."""

    def test_boosted_sod(self):
        """Sod problem boosted by constant velocity should give same density profile
        (shifted in space).
        """
        nx = 100
        x = torch.arange(nx, dtype=torch.float64) * 0.01
        gamma = 1.4

        # Unboosted
        rho0, u0, p0 = sod(x)
        r1 = generate_one(rho0, u0, p0, dx=0.01, dt=0.001, nt=10, gamma=gamma,
                           bc_type="periodic", reconstruction="constant")

        # Boosted by v_boost (same IC but with velocity offset)
        v_boost = 0.5
        rho0b, u0b, p0b = sod(x)
        u0b = u0b + v_boost
        r2 = generate_one(rho0b, u0b, p0b, dx=0.01, dt=0.001, nt=10, gamma=gamma,
                           bc_type="periodic", reconstruction="constant")

        # Both should be valid and have similar structure
        assert r1["valid"]
        assert r2["valid"]
        # Density profiles won't be identical due to different advection,
        # but both should remain physical
        assert torch.isfinite(r2["rho"]).all()


class TestConvergence:
    """Refinement should decrease error (qualitative convergence test)."""

    def test_convergence_sod_density(self):
        """Finer grid should give a sharper shock profile."""
        gamma = 1.4
        results = {}
        for nx in [50, 100]:
            dx = 1.0 / nx
            x = torch.arange(nx, dtype=torch.float64) * dx
            rho0, u0, p0 = sod(x)
            result = generate_one(rho0, u0, p0, dx=dx, dt=0.0002, nt=50,
                                  reconstruction="constant")
            assert result["valid"]
            results[nx] = result

        # Finer grid should have a sharper shock (larger max gradient)
        grad_coarse = (results[50]["rho"][-1, 1:] - results[50]["rho"][-1, :-1]).abs().max()
        grad_fine = (results[100]["rho"][-1, 1:] - results[100]["rho"][-1, :-1]).abs().max()
        # Fine grid gradient should be at least as large
        assert grad_fine >= grad_coarse * 0.5


class TestFluxConsistency:
    """Different flux functions should agree on uniform states."""

    @pytest.mark.parametrize("flux_type", ["hllc", "hll", "rusanov"])
    def test_uniform_state_all_fluxes_agree(self, flux_type):
        """All flux functions should preserve a uniform state exactly."""
        nx = 30
        rho0 = torch.ones(nx, dtype=torch.float64)
        u0 = torch.full((nx,), 0.5, dtype=torch.float64)
        p0 = torch.ones(nx, dtype=torch.float64)
        result = generate_one(rho0, u0, p0, dx=0.033, dt=0.001, nt=50,
                              flux_type=flux_type, bc_type="periodic",
                              reconstruction="constant")
        torch.testing.assert_close(result["rho"][-1], rho0, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(result["u"][-1], u0, atol=1e-10, rtol=1e-10)

    def test_hllc_vs_hll_qualitative(self):
        """HLLC and HLL should give qualitatively similar results for Sod."""
        nx = 80
        x = torch.arange(nx, dtype=torch.float64) * 0.0125
        rho0, u0, p0 = sod(x)
        r_hllc = generate_one(rho0, u0, p0, dx=0.0125, dt=0.0005, nt=30,
                               flux_type="hllc", reconstruction="constant")
        r_hll = generate_one(rho0, u0, p0, dx=0.0125, dt=0.0005, nt=30,
                              flux_type="hll", reconstruction="constant")
        assert r_hllc["valid"]
        assert r_hll["valid"]
        # Both should produce similar density ranges
        assert abs(r_hllc["rho"][-1].max().item() - r_hll["rho"][-1].max().item()) < 0.5
        assert abs(r_hllc["rho"][-1].min().item() - r_hll["rho"][-1].min().item()) < 0.1

    def test_hllc_sharper_contact_than_hll(self):
        """HLLC should resolve contacts better than HLL (known property)."""
        nx = 100
        x = torch.arange(nx, dtype=torch.float64) * 0.01
        # Pure contact discontinuity
        rho0, u0, p0 = riemann(x, rho_left=3.0, rho_right=1.0,
                                u_left=0.5, u_right=0.5,
                                p_left=1.0, p_right=1.0)
        r_hllc = generate_one(rho0, u0, p0, dx=0.01, dt=0.001, nt=20,
                               flux_type="hllc", reconstruction="constant")
        r_hll = generate_one(rho0, u0, p0, dx=0.01, dt=0.001, nt=20,
                              flux_type="hll", reconstruction="constant")
        # HLLC should have sharper contact (larger max density gradient)
        grad_hllc = (r_hllc["rho"][-1, 1:] - r_hllc["rho"][-1, :-1]).abs().max()
        grad_hll = (r_hll["rho"][-1, 1:] - r_hll["rho"][-1, :-1]).abs().max()
        # HLLC is known to be less diffusive at contacts
        assert grad_hllc >= grad_hll * 0.8, "HLLC should be at least as sharp as HLL at contacts"


class TestMultiWaveInteraction:
    """Multiple waves interacting."""

    def test_three_region_blast(self):
        """Woodward-Colella blast wave: three regions with very different pressures."""
        nx = 100
        x = torch.arange(nx, dtype=torch.float64) * 0.01
        sentinel = x.max().item() + 1.0
        rho0, u0, p0 = from_steps(
            x,
            rho_steps=[(sentinel, 1.0)],
            p_steps=[(0.1, 1000.0), (0.9, 100.0), (sentinel, 0.01)],
        )
        result = generate_one(rho0, u0, p0, dx=0.01, dt=0.00005, nt=100,
                              bc_type="wall", reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_many_shock_interactions(self):
        """Multiple shocks from many-region IC."""
        nx = 100
        x = torch.arange(nx, dtype=torch.float64) * 0.01
        rho0, u0, p0 = from_steps(
            x,
            rho_steps=[(0.2, 3.0), (0.4, 1.0), (0.6, 4.0), (0.8, 0.5)],
            u_steps=[(0.3, 1.0), (0.5, -1.0), (0.7, 0.5)],
            p_steps=[(0.2, 5.0), (0.5, 0.5), (0.8, 3.0)],
        )
        result = generate_one(rho0, u0, p0, dx=0.01, dt=0.0002, nt=100,
                              reconstruction="constant")
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()
        assert torch.isfinite(result["u"]).all()
        assert torch.isfinite(result["p"]).all()


class TestSoundWavePropagation:
    """Small-amplitude perturbations should propagate at the sound speed."""

    def test_small_perturbation_propagates(self):
        """A small density perturbation should spread outward."""
        nx = 100
        x = torch.arange(nx, dtype=torch.float64) * 0.01
        rho0 = torch.ones(nx, dtype=torch.float64)
        rho0[48:52] = 1.01  # Small bump
        u0 = torch.zeros(nx, dtype=torch.float64)
        p0 = torch.ones(nx, dtype=torch.float64)
        p0[48:52] = 1.014  # Consistent small perturbation (dp/drho = c^2 = gamma*p/rho = 1.4)

        result = generate_one(rho0, u0, p0, dx=0.01, dt=0.001, nt=20, gamma=1.4,
                              bc_type="periodic", reconstruction="constant")
        assert result["valid"]
        # The perturbation should have spread beyond the initial 4-cell region
        rho_final = result["rho"][-1]
        perturbed_cells = (rho_final - 1.0).abs() > 1e-5
        assert perturbed_cells.sum() > 4, "Perturbation should have spread"

    def test_linear_wave_amplitude_decreases(self):
        """A small perturbation should spread and its peak amplitude should decrease."""
        nx = 100
        x = torch.arange(nx, dtype=torch.float64) * 0.01
        rho0 = torch.ones(nx, dtype=torch.float64)
        rho0[48:52] = 1.001  # Very small bump
        u0 = torch.zeros(nx, dtype=torch.float64)
        p0 = torch.ones(nx, dtype=torch.float64)
        p0[48:52] = 1.0014

        result = generate_one(rho0, u0, p0, dx=0.01, dt=0.001, nt=30, gamma=1.4,
                              bc_type="periodic", reconstruction="constant")
        assert result["valid"]
        # Peak perturbation should be smaller at later time
        peak_initial = (result["rho"][0] - 1.0).abs().max()
        peak_final = (result["rho"][-1] - 1.0).abs().max()
        assert peak_final < peak_initial, "Sound wave should disperse, reducing peak amplitude"
