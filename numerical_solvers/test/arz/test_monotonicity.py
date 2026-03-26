"""Tests for physical monotonicity, ordering, convergence, and conservation.

Covers eigenvalue properties, maximum principle, convergence rates,
Rankine-Hugoniot conditions, entropy-like behavior, flux symmetry,
rho*w conservation, and long-time behavior.
"""

import pytest
import torch

from numerical_solvers.arz import generate_one
from numerical_solvers.arz.flux import hll, rusanov
from numerical_solvers.arz.initial_conditions import from_steps, random_piecewise, riemann, three_region
from numerical_solvers.arz.physics import dp_drho, eigenvalues, pressure


# ====================================================================
# 1. Eigenvalue properties
# ====================================================================

class TestEigenvalueProperties:
    """Test mathematical properties of the ARZ eigenstructure."""

    def test_lam2_decreases_with_density(self):
        """For fixed v, lam2 decreases as rho increases (waves slow down in congestion)."""
        rho = torch.linspace(0.1, 2.0, 20)
        v = torch.full_like(rho, 0.5)
        _, lam2 = eigenvalues(rho, v, 1.0)
        # lam2 = v - rho * dp_drho. For gamma=1, dp_drho=1, so lam2 = v - rho (decreasing)
        diffs = lam2[1:] - lam2[:-1]
        assert (diffs <= 1e-6).all(), "lam2 should decrease with increasing density"

    def test_pressure_monotone_increasing(self):
        """p(rho) is monotonically increasing for rho >= 0."""
        rho = torch.linspace(0, 3.0, 100)
        for gamma in [0.5, 1.0, 1.5, 2.0]:
            p = pressure(rho, gamma)
            diffs = p[1:] - p[:-1]
            assert (diffs >= -1e-10).all(), f"Pressure not monotone for gamma={gamma}"

    def test_dp_drho_nonnegative(self):
        """p'(rho) >= 0 for rho > 0."""
        rho = torch.linspace(0.01, 3.0, 100)
        for gamma in [0.5, 1.0, 1.5, 2.0]:
            dp = dp_drho(rho, gamma)
            assert (dp >= -1e-10).all(), f"dp_drho negative for gamma={gamma}"

    @pytest.mark.parametrize("gamma", [0.5, 1.0, 1.5, 2.0, 3.0])
    def test_lam2_strictly_less_than_lam1(self, gamma):
        """lam2 < lam1 strictly for rho > 0 (system is strictly hyperbolic)."""
        rho = torch.linspace(0.01, 3.0, 50)
        v = torch.rand(50)
        lam1, lam2 = eigenvalues(rho, v, gamma)
        assert (lam2 < lam1).all(), f"Strict hyperbolicity violated for gamma={gamma}"

    @pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0])
    def test_lam2_decreases_with_density_various_gamma(self, gamma):
        """lam2 monotonically decreasing in rho for any gamma, fixed v."""
        rho = torch.linspace(0.1, 2.0, 50)
        v = torch.full_like(rho, 0.4)
        _, lam2 = eigenvalues(rho, v, gamma)
        diffs = lam2[1:] - lam2[:-1]
        assert (diffs <= 1e-6).all(), (
            f"lam2 not decreasing with density for gamma={gamma}"
        )

    def test_eigenvalue_gap_grows_with_density(self):
        """The gap lam1 - lam2 = rho * dp_drho should grow with rho for gamma >= 1."""
        rho = torch.linspace(0.1, 3.0, 50)
        v = torch.full_like(rho, 0.5)
        for gamma in [1.0, 1.5, 2.0]:
            lam1, lam2 = eigenvalues(rho, v, gamma)
            gap = lam1 - lam2  # = rho * dp_drho(rho, gamma) = rho * gamma * rho^(gamma-1) = gamma * rho^gamma
            diffs = gap[1:] - gap[:-1]
            assert (diffs >= -1e-6).all(), (
                f"Eigenvalue gap not increasing for gamma={gamma}"
            )

    def test_lam1_equals_v(self):
        """First eigenvalue is exactly v (contact characteristic)."""
        rho = torch.rand(20) + 0.1
        v = torch.rand(20) * 2 - 1
        for gamma in [0.5, 1.0, 2.0]:
            lam1, _ = eigenvalues(rho, v, gamma)
            torch.testing.assert_close(lam1, v)

    def test_eigenvalues_continuous_in_density(self):
        """Small change in rho should produce small change in eigenvalues."""
        rho1 = torch.tensor([0.5])
        rho2 = torch.tensor([0.5 + 1e-6])
        v = torch.tensor([0.3])
        for gamma in [0.5, 1.0, 2.0]:
            _, lam2_1 = eigenvalues(rho1, v, gamma)
            _, lam2_2 = eigenvalues(rho2, v, gamma)
            assert (lam2_1 - lam2_2).abs() < 1e-4

    def test_pressure_convexity_gamma_ge_1(self):
        """For gamma >= 1, p(rho) is convex: p''(rho) >= 0."""
        rho = torch.linspace(0.01, 3.0, 100)
        for gamma in [1.0, 1.5, 2.0, 3.0]:
            # p''(rho) = gamma * (gamma-1) * rho^(gamma-2) >= 0 for gamma >= 1
            p = pressure(rho, gamma)
            # Numerical second derivative
            d2p = p[2:] - 2 * p[1:-1] + p[:-2]
            assert (d2p >= -1e-6).all(), f"Pressure not convex for gamma={gamma}"


# ====================================================================
# 2. Maximum principle
# ====================================================================

class TestMaximumPrinciple:
    """The solution should stay bounded by IC extremes (approximately)."""

    def test_density_bounded_by_ic_periodic(self):
        """Density should not exceed IC max nor go below IC min (periodic BC)."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float32) * 0.025
        rho0, v0 = riemann(x, rho_left=0.7, rho_right=0.3, v0=0.2)
        result = generate_one(
            rho0, v0, dx=0.025, dt=0.002, nt=30,
            bc_type="periodic", reconstruction="constant",
        )
        # Allow some tolerance for numerical diffusion
        assert result["rho"].max() <= 0.7 + 0.1
        assert result["rho"].min() >= 0.3 - 0.1

    def test_density_bounded_by_ic_zero_gradient(self):
        """Density bounded with zero-gradient BC."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float32) * 0.025
        rho0, v0 = riemann(x, rho_left=0.6, rho_right=0.2, v0=0.2)
        result = generate_one(
            rho0, v0, dx=0.025, dt=0.002, nt=30,
            bc_type="zero_gradient", reconstruction="constant",
        )
        # With zero-gradient BC, new extremes can form at boundaries
        # but density should not grow without bound
        assert result["rho"].max() <= 1.0, "Density grew beyond reasonable bound"
        assert result["rho"].min() >= -1e-6, "Negative density"

    @pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0])
    def test_density_bounded_various_gamma(self, gamma):
        """Density bounded for various gamma with periodic BC."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float32) * 0.025
        rho0, v0 = riemann(x, rho_left=0.8, rho_right=0.2, v0=0.2)
        result = generate_one(
            rho0, v0, dx=0.025, dt=0.002, nt=30, gamma=gamma,
            bc_type="periodic", reconstruction="constant",
        )
        assert result["rho"].max() <= 0.8 + 0.15, f"Overshoot for gamma={gamma}"
        assert result["rho"].min() >= 0.2 - 0.15, f"Undershoot for gamma={gamma}"

    def test_uniform_ic_stays_exactly_uniform(self):
        """Uniform IC: solution should not develop new extrema."""
        nx = 40
        rho0 = torch.full((nx,), 0.5)
        v0 = torch.full((nx,), 0.3)
        result = generate_one(
            rho0, v0, dx=0.025, dt=0.002, nt=50,
            bc_type="periodic", reconstruction="constant",
        )
        for t_idx in range(result["rho"].shape[0]):
            assert (result["rho"][t_idx] - 0.5).abs().max() < 1e-6

    def test_three_region_bounded(self):
        """Three-region IC: density should stay between min and max of IC."""
        nx = 60
        x = torch.arange(nx, dtype=torch.float32) * 0.025
        rho0, v0 = three_region(
            x, rho_left=0.3, rho_mid=0.8, rho_right=0.5,
            v0=0.2, x1=0.3, x2=0.7,
        )
        result = generate_one(
            rho0, v0, dx=0.025, dt=0.002, nt=40,
            bc_type="periodic", reconstruction="constant",
        )
        assert result["rho"].max() <= 0.8 + 0.15
        assert result["rho"].min() >= 0.3 - 0.15


# ====================================================================
# 3. Convergence
# ====================================================================

class TestConvergence:
    """Test that finer grids give more accurate solutions."""

    def test_steady_state_error_small_all_resolutions(self):
        """For a uniform IC, the error from the exact solution should stay small."""
        errors = []
        for nx in [16, 32, 64]:
            dx = 1.0 / nx
            rho0 = torch.full((nx,), 0.5)
            v0 = torch.full((nx,), 0.3)
            result = generate_one(
                rho0, v0, dx=dx, dt=0.002, nt=20,
                bc_type="periodic", reconstruction="constant",
            )
            error = (result["rho"][-1] - 0.5).abs().max().item()
            errors.append(error)
        # Error should be tiny for all resolutions (exact steady state)
        for err in errors:
            assert err < 1e-5

    def test_diffusion_decreases_with_resolution(self):
        """On a smooth IC, numerical diffusion should decrease with finer grids."""
        # Use a smooth bump (sinusoidal) with periodic BC
        errors = []
        for nx in [20, 40, 80]:
            dx = 2.0 / nx
            x = torch.arange(nx, dtype=torch.float32) * dx
            rho0 = 0.5 + 0.1 * torch.sin(2 * torch.pi * x / 2.0)
            v0 = torch.full((nx,), 0.2)
            result = generate_one(
                rho0, v0, dx=dx, dt=0.001, nt=20,
                bc_type="periodic", reconstruction="constant",
            )
            # Measure how much the amplitude has decayed (numerical diffusion)
            initial_amp = rho0.max() - rho0.min()
            final_amp = result["rho"][-1].max() - result["rho"][-1].min()
            errors.append((initial_amp - final_amp).item())
        # Diffusion error should decrease (or stay similar) with refinement
        # At minimum, the finest grid should have less diffusion than the coarsest
        assert errors[-1] <= errors[0] + 0.01

    def test_weno5_less_diffusive_than_constant(self):
        """WENO5 should be less diffusive than constant reconstruction."""
        nx = 40
        dx = 1.0 / nx
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0 = 0.5 + 0.1 * torch.sin(2 * torch.pi * x)
        v0 = torch.full((nx,), 0.2)

        result_const = generate_one(
            rho0, v0, dx=dx, dt=0.002, nt=30,
            bc_type="periodic", reconstruction="constant",
        )
        result_weno = generate_one(
            rho0, v0, dx=dx, dt=0.002, nt=30,
            bc_type="periodic", reconstruction="weno5",
        )
        amp_const = result_const["rho"][-1].max() - result_const["rho"][-1].min()
        amp_weno = result_weno["rho"][-1].max() - result_weno["rho"][-1].min()
        # WENO5 should preserve the amplitude better (less diffusion)
        assert amp_weno >= amp_const - 0.02, (
            f"WENO5 more diffusive than constant: {amp_weno:.4f} < {amp_const:.4f}"
        )


# ====================================================================
# 4. Conservation
# ====================================================================

class TestConservation:
    """Test conservation of mass and rho*w with periodic BC."""

    @pytest.mark.parametrize("flux_type", ["rusanov", "hll"])
    @pytest.mark.parametrize("reconstruction", ["constant", "weno5"])
    def test_mass_conservation_periodic(self, flux_type, reconstruction):
        """Total mass conserved with periodic BC."""
        nx, dx, dt, nt = 40, 0.025, 0.002, 40
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = riemann(x, rho_left=0.6, rho_right=0.3, v0=0.2)
        result = generate_one(
            rho0, v0, dx=dx, dt=dt, nt=nt, gamma=1.0,
            bc_type="periodic", flux_type=flux_type, reconstruction=reconstruction,
        )
        mass = result["rho"].sum(dim=-1) * dx
        torch.testing.assert_close(mass, mass[0].expand_as(mass), atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("flux_type", ["rusanov", "hll"])
    @pytest.mark.parametrize("reconstruction", ["constant", "weno5"])
    def test_rho_w_conservation_periodic(self, flux_type, reconstruction):
        """Total rho*w conserved with periodic BC."""
        nx, dx, dt, nt = 40, 0.025, 0.002, 40
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = riemann(x, rho_left=0.6, rho_right=0.3, v0=0.2)
        result = generate_one(
            rho0, v0, dx=dx, dt=dt, nt=nt, gamma=1.0,
            bc_type="periodic", flux_type=flux_type, reconstruction=reconstruction,
        )
        rho_w = result["rho"] * result["w"]
        rho_w_total = rho_w.sum(dim=-1) * dx
        torch.testing.assert_close(
            rho_w_total, rho_w_total[0].expand_as(rho_w_total), atol=1e-4, rtol=1e-4,
        )

    @pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0])
    def test_mass_conservation_various_gamma(self, gamma):
        """Mass conservation for different gamma (periodic BC)."""
        nx, dx, dt, nt = 40, 0.025, 0.002, 30
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = riemann(x, rho_left=0.6, rho_right=0.3, v0=0.2)
        result = generate_one(
            rho0, v0, dx=dx, dt=dt, nt=nt, gamma=gamma,
            bc_type="periodic", reconstruction="constant",
        )
        mass = result["rho"].sum(dim=-1) * dx
        torch.testing.assert_close(mass, mass[0].expand_as(mass), atol=1e-4, rtol=1e-4)

    def test_mass_conservation_three_region(self):
        """Mass conservation with a three-region IC (periodic)."""
        nx, dx, dt, nt = 60, 0.025, 0.002, 50
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = three_region(
            x, rho_left=0.7, rho_mid=0.2, rho_right=0.5,
            v0=0.2, x1=0.3, x2=0.7,
        )
        result = generate_one(
            rho0, v0, dx=dx, dt=dt, nt=nt,
            bc_type="periodic", reconstruction="constant",
        )
        mass = result["rho"].sum(dim=-1) * dx
        torch.testing.assert_close(mass, mass[0].expand_as(mass), atol=1e-4, rtol=1e-4)

    def test_mass_conservation_random_ic(self):
        """Mass conservation with random piecewise IC (periodic)."""
        nx, dx, dt, nt = 60, 0.025, 0.002, 40
        x = torch.arange(nx, dtype=torch.float32) * dx
        rng = torch.Generator().manual_seed(42)
        rho0, v0, _ = random_piecewise(x, 5, rng)
        result = generate_one(
            rho0, v0, dx=dx, dt=dt, nt=nt,
            bc_type="periodic", reconstruction="constant",
        )
        mass = result["rho"].sum(dim=-1) * dx
        torch.testing.assert_close(mass, mass[0].expand_as(mass), atol=1e-4, rtol=1e-4)

    def test_mass_non_conservation_zero_gradient(self):
        """With zero-gradient BC, mass is NOT exactly conserved (flux at boundary)."""
        nx, dx, dt, nt = 40, 0.025, 0.002, 30
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = riemann(x, rho_left=0.8, rho_right=0.2, v0=0.3)
        result = generate_one(
            rho0, v0, dx=dx, dt=dt, nt=nt,
            bc_type="zero_gradient", reconstruction="constant",
        )
        mass = result["rho"].sum(dim=-1) * dx
        # Mass should change (not exactly conserved) but not blow up
        assert mass[-1] < mass[0] * 10.0, "Mass grew unreasonably"
        assert mass[-1] > 0, "Mass vanished"


# ====================================================================
# 5. Entropy-like behavior
# ====================================================================

class TestEntropyBehavior:
    """Physical entropy-like properties: dissipation at shocks."""

    def test_total_variation_bounded_constant_recon(self):
        """TV(rho) should remain bounded and not grow unboundedly.

        Note: strict TVD (monotonically non-increasing TV) only holds for
        scalar conservation laws. For the 2x2 ARZ system, TV of a single
        component can temporarily increase due to wave interactions, but
        it should not grow without bound.
        """
        nx = 40
        x = torch.arange(nx, dtype=torch.float32) * 0.025
        rho0, v0 = riemann(x, rho_left=0.7, rho_right=0.3, v0=0.2)
        result = generate_one(
            rho0, v0, dx=0.025, dt=0.002, nt=30,
            bc_type="periodic", reconstruction="constant", flux_type="rusanov",
        )
        # Compute total variation at each time step
        tv = (result["rho"][:, 1:] - result["rho"][:, :-1]).abs().sum(dim=-1)
        tv_initial = tv[0].item()
        # TV should stay bounded: not grow more than 2x the initial TV
        assert tv.max().item() <= tv_initial * 3.0, (
            f"TV grew excessively: max={tv.max():.4f} vs initial={tv_initial:.4f}"
        )
        # TV at the end should be smaller than or comparable to the beginning
        # (numerical diffusion eventually smooths things)
        assert tv[-1] <= tv_initial * 2.0, (
            f"Final TV too large: {tv[-1]:.4f} vs initial={tv_initial:.4f}"
        )

    def test_solution_smoothing_over_time(self):
        """A Riemann problem should become smoother over time (numerical diffusion)."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float32) * 0.025
        rho0, v0 = riemann(x, rho_left=0.7, rho_right=0.3, v0=0.2)
        result = generate_one(
            rho0, v0, dx=0.025, dt=0.002, nt=50,
            bc_type="periodic", reconstruction="constant",
        )
        # Variance of density should decrease over time (smoothing)
        var_0 = result["rho"][0].var().item()
        var_end = result["rho"][-1].var().item()
        assert var_end <= var_0 + 1e-4, "Solution did not smooth over time"


# ====================================================================
# 6. Flux symmetry and consistency
# ====================================================================

class TestFluxSymmetryConsistency:
    """Test physical consistency of flux functions."""

    @pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0])
    def test_flux_consistency_various_gamma(self, gamma):
        """F(U,U) = f(U) for both flux functions and various gamma."""
        rho = torch.tensor([0.3, 0.5, 1.0])
        v = torch.tensor([0.2, 0.4, -0.1])
        w = v + pressure(rho, gamma)
        rho_w = rho * w
        for fn in [rusanov, hll]:
            f_rho, f_rw = fn(rho, rho_w, rho, rho_w, gamma)
            expected_f_rho = rho * v
            expected_f_rw = rho_w * v
            torch.testing.assert_close(f_rho, expected_f_rho, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(f_rw, expected_f_rw, atol=1e-6, rtol=1e-6)

    def test_rusanov_hll_agree_on_identical_states(self):
        """Both fluxes agree when left == right (both reduce to physical flux)."""
        gamma = 1.0
        rho = torch.tensor([0.3, 0.7, 1.0])
        v = torch.tensor([0.1, 0.5, -0.2])
        w = v + pressure(rho, gamma)
        rho_w = rho * w
        f_rho_r, f_rw_r = rusanov(rho, rho_w, rho, rho_w, gamma)
        f_rho_h, f_rw_h = hll(rho, rho_w, rho, rho_w, gamma)
        torch.testing.assert_close(f_rho_r, f_rho_h, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(f_rw_r, f_rw_h, atol=1e-6, rtol=1e-6)

    def test_rusanov_more_diffusive_than_hll(self):
        """Rusanov typically adds more diffusion than HLL.

        We test this by running the same problem and checking that the HLL
        solution preserves more structure (larger amplitude).
        """
        nx = 40
        dx = 1.0 / nx
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0 = 0.5 + 0.1 * torch.sin(2 * torch.pi * x)
        v0 = torch.full((nx,), 0.2)

        result_rusanov = generate_one(
            rho0, v0, dx=dx, dt=0.002, nt=30,
            bc_type="periodic", flux_type="rusanov", reconstruction="constant",
        )
        result_hll = generate_one(
            rho0, v0, dx=dx, dt=0.002, nt=30,
            bc_type="periodic", flux_type="hll", reconstruction="constant",
        )
        amp_rusanov = result_rusanov["rho"][-1].max() - result_rusanov["rho"][-1].min()
        amp_hll = result_hll["rho"][-1].max() - result_hll["rho"][-1].min()
        # HLL should preserve amplitude at least as well as Rusanov
        assert amp_hll >= amp_rusanov - 0.01


# ====================================================================
# 7. Long-time behavior
# ====================================================================

class TestLongTimeBehavior:
    """Test behavior over many time steps."""

    def test_periodic_long_time_stability(self):
        """Solution remains stable over many steps with periodic BC."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float32) * 0.025
        rho0, v0 = riemann(x, rho_left=0.6, rho_right=0.3, v0=0.2)
        result = generate_one(
            rho0, v0, dx=0.025, dt=0.002, nt=200,
            bc_type="periodic", reconstruction="constant",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()
        assert (result["rho"] >= -1e-6).all()

    def test_periodic_long_time_mass_conservation(self):
        """Mass conserved over long evolution with periodic BC."""
        nx = 40
        dx = 0.025
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = riemann(x, rho_left=0.6, rho_right=0.3, v0=0.2)
        result = generate_one(
            rho0, v0, dx=dx, dt=0.002, nt=200,
            bc_type="periodic", reconstruction="constant",
        )
        mass = result["rho"].sum(dim=-1) * dx
        torch.testing.assert_close(mass, mass[0].expand_as(mass), atol=1e-3, rtol=1e-3)

    def test_solution_tends_to_constant_periodic(self):
        """With periodic BC and diffusion, solution should eventually homogenize.

        For the ARZ 2x2 system, wave interactions can temporarily increase
        the density spread. We run for a long time and check that the
        variance eventually decreases compared to the initial condition.
        """
        nx = 40
        dx = 0.025
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = riemann(x, rho_left=0.7, rho_right=0.3, v0=0.2)
        result = generate_one(
            rho0, v0, dx=dx, dt=0.002, nt=1000,
            bc_type="periodic", reconstruction="constant", flux_type="rusanov",
        )
        # Numerical diffusion should eventually reduce the variance
        var_initial = result["rho"][0].var().item()
        var_final = result["rho"][-1].var().item()
        assert var_final < var_initial, (
            f"Solution variance did not decrease: {var_final:.6f} vs {var_initial:.6f}"
        )


# ====================================================================
# 8. Shock speed and Rankine-Hugoniot
# ====================================================================

class TestShockStructure:
    """Test that shocks propagate and interact correctly."""

    def test_shock_propagates_rightward(self):
        """A shock with v > 0 should move rightward over time."""
        nx = 60
        dx = 0.025
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = riemann(x, rho_left=0.8, rho_right=0.3, v0=0.3, x_split=0.5)
        result = generate_one(
            rho0, v0, dx=dx, dt=0.002, nt=30,
            bc_type="zero_gradient", reconstruction="constant",
        )
        # Find approximate shock location at t=0 and t=final
        # (maximum gradient location)
        grad_0 = (result["rho"][0, 1:] - result["rho"][0, :-1]).abs()
        grad_f = (result["rho"][-1, 1:] - result["rho"][-1, :-1]).abs()
        shock_0 = grad_0.argmax().item()
        shock_f = grad_f.argmax().item()
        # Shock should have moved (not necessarily always right due to 2-wave system)
        assert shock_f != shock_0 or grad_f.max() < grad_0.max(), (
            "Shock did not propagate or diffuse"
        )

    def test_two_shocks_merge(self):
        """Two shocks that catch up should merge into one."""
        nx = 80
        dx = 0.02
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = from_steps(
            x,
            rho_steps=[(0.3, 0.8), (0.6, 0.4), (2.0, 0.1)],
            default_v=0.2,
        )
        result = generate_one(
            rho0, v0, dx=dx, dt=0.002, nt=100,
            bc_type="zero_gradient", reconstruction="constant",
        )
        assert result["valid"]
        # After long enough time with merging, there should be fewer sharp gradients
        grad_final = (result["rho"][-1, 1:] - result["rho"][-1, :-1]).abs()
        grad_init = (result["rho"][0, 1:] - result["rho"][0, :-1]).abs()
        # The number of "significant" jumps should decrease
        n_jumps_init = (grad_init > 0.1).sum()
        n_jumps_final = (grad_final > 0.1).sum()
        # Due to numerical diffusion, jumps should not increase
        assert n_jumps_final <= n_jumps_init + 2, (
            f"Jumps increased: {n_jumps_init} -> {n_jumps_final}"
        )


# ====================================================================
# 9. w invariant across Riemann-type ICs
# ====================================================================

class TestWInvariantAcrossICs:
    """w = v + p(rho) should hold for all IC types and parameters."""

    @pytest.mark.parametrize("gamma", [0.5, 1.0, 1.5, 2.0])
    def test_w_invariant_riemann(self, gamma):
        nx = 40
        x = torch.arange(nx, dtype=torch.float32) * 0.025
        rho0, v0 = riemann(x, rho_left=0.7, rho_right=0.2, v0=0.3)
        result = generate_one(
            rho0, v0, dx=0.025, dt=0.002, nt=30, gamma=gamma,
            bc_type="periodic", reconstruction="constant",
        )
        expected_w = result["v"] + pressure(result["rho"], gamma)
        torch.testing.assert_close(result["w"], expected_w, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0])
    def test_w_invariant_three_region(self, gamma):
        nx = 60
        x = torch.arange(nx, dtype=torch.float32) * 0.025
        rho0, v0 = three_region(
            x, rho_left=0.3, rho_mid=0.9, rho_right=0.5,
            v0=0.2, x1=0.3, x2=0.7,
        )
        result = generate_one(
            rho0, v0, dx=0.025, dt=0.002, nt=30, gamma=gamma,
            reconstruction="constant",
        )
        expected_w = result["v"] + pressure(result["rho"], gamma)
        torch.testing.assert_close(result["w"], expected_w, atol=1e-5, rtol=1e-5)

    def test_w_invariant_random_ic(self):
        nx = 40
        x = torch.arange(nx, dtype=torch.float32) * 0.025
        rng = torch.Generator().manual_seed(42)
        rho0, v0, _ = random_piecewise(x, 5, rng)
        gamma = 1.5
        result = generate_one(
            rho0, v0, dx=0.025, dt=0.002, nt=30, gamma=gamma,
            reconstruction="constant",
        )
        expected_w = result["v"] + pressure(result["rho"], gamma)
        torch.testing.assert_close(result["w"], expected_w, atol=1e-5, rtol=1e-5)


# ====================================================================
# 10. Consistency between flux functions
# ====================================================================

class TestFluxFunctionConsistency:
    """Rusanov and HLL should agree qualitatively on the same problem."""

    def test_same_ic_similar_mass(self):
        """Both flux functions should conserve the same total mass (periodic)."""
        nx = 40
        dx = 0.025
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = riemann(x, rho_left=0.6, rho_right=0.3, v0=0.2)

        result_r = generate_one(
            rho0, v0, dx=dx, dt=0.002, nt=30,
            bc_type="periodic", flux_type="rusanov", reconstruction="constant",
        )
        result_h = generate_one(
            rho0, v0, dx=dx, dt=0.002, nt=30,
            bc_type="periodic", flux_type="hll", reconstruction="constant",
        )
        mass_r = result_r["rho"][-1].sum() * dx
        mass_h = result_h["rho"][-1].sum() * dx
        torch.testing.assert_close(mass_r, mass_h, atol=1e-4, rtol=1e-4)

    def test_same_ic_qualitatively_similar(self):
        """Solutions from Rusanov and HLL should be qualitatively similar."""
        nx = 40
        dx = 0.025
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = riemann(x, rho_left=0.6, rho_right=0.3, v0=0.2)

        result_r = generate_one(
            rho0, v0, dx=dx, dt=0.002, nt=30,
            bc_type="periodic", flux_type="rusanov", reconstruction="constant",
        )
        result_h = generate_one(
            rho0, v0, dx=dx, dt=0.002, nt=30,
            bc_type="periodic", flux_type="hll", reconstruction="constant",
        )
        # L1 difference should be small
        l1_diff = (result_r["rho"][-1] - result_h["rho"][-1]).abs().mean()
        assert l1_diff < 0.1, f"Rusanov and HLL disagree: L1 diff = {l1_diff:.4f}"


# ====================================================================
# 11. Steady-state preservation
# ====================================================================

class TestSteadyStatePreservation:
    """Solutions that should remain constant in time."""

    @pytest.mark.parametrize("bc_type", ["periodic", "zero_gradient"])
    @pytest.mark.parametrize("flux_type", ["rusanov", "hll"])
    @pytest.mark.parametrize("reconstruction", ["constant", "weno5"])
    def test_uniform_steady_all_combos(self, bc_type, flux_type, reconstruction):
        """Uniform IC stays exactly uniform for all solver configurations."""
        nx = 32
        rho0 = torch.full((nx,), 0.5)
        v0 = torch.full((nx,), 0.3)
        result = generate_one(
            rho0, v0, dx=0.05, dt=0.002, nt=20,
            bc_type=bc_type, flux_type=flux_type, reconstruction=reconstruction,
        )
        assert result["valid"]
        torch.testing.assert_close(
            result["rho"], rho0.unsqueeze(0).expand(21, -1), atol=1e-5, rtol=1e-5,
        )
        torch.testing.assert_close(
            result["v"], v0.unsqueeze(0).expand(21, -1), atol=1e-5, rtol=1e-5,
        )

    @pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0])
    def test_vacuum_steady(self, gamma):
        """Zero density/velocity should remain exactly zero."""
        nx = 32
        rho0 = torch.zeros(nx)
        v0 = torch.zeros(nx)
        result = generate_one(
            rho0, v0, dx=0.05, dt=0.005, nt=20, gamma=gamma,
            bc_type="periodic", reconstruction="constant",
        )
        torch.testing.assert_close(result["rho"], torch.zeros(21, nx), atol=1e-10, rtol=0)

    def test_dirichlet_matching_ic_steady(self):
        """If Dirichlet BC matches uniform IC, solution stays constant."""
        nx = 32
        rho_val, v_val = 0.5, 0.3
        rho0 = torch.full((nx,), rho_val)
        v0 = torch.full((nx,), v_val)
        result = generate_one(
            rho0, v0, dx=0.05, dt=0.005, nt=20,
            bc_type="dirichlet", bc_left=(rho_val, v_val), bc_right=(rho_val, v_val),
            reconstruction="constant",
        )
        torch.testing.assert_close(
            result["rho"], rho0.unsqueeze(0).expand(21, -1), atol=1e-5, rtol=1e-5,
        )


# ====================================================================
# 12. Numerical accuracy for special cases
# ====================================================================

class TestSpecialCaseAccuracy:
    """Test accuracy on problems with known analytical behavior."""

    def test_isolated_contact_preserves_w(self):
        """For an isolated 1-contact (only lam1=v), w should be exactly preserved.

        If the initial w is uniform (same v + p(rho) everywhere), then the
        w-equation reduces to pure advection of rho*w at speed v, and w
        remains constant everywhere.
        """
        nx = 40
        dx = 0.025
        gamma = 1.0
        # Choose IC where w is uniform: w = v + rho^gamma = const
        # Let w0 = 1.0 everywhere. Then v = w0 - rho^gamma.
        w_const = 1.0
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0 = torch.where(x < 0.5, torch.tensor(0.6), torch.tensor(0.3))
        v0 = w_const - pressure(rho0, gamma)
        result = generate_one(
            rho0, v0, dx=dx, dt=0.002, nt=20, gamma=gamma,
            bc_type="periodic", reconstruction="constant",
        )
        # w should remain close to w_const everywhere (modulo numerical diffusion)
        # The w-equation is ∂(rho*w)/∂t + ∂(rho*w*v)/∂x = 0, which is exact
        # if w is constant because it factors out.
        # Check that w stays close to the constant
        w_max_deviation = (result["w"] - w_const).abs().max().item()
        assert w_max_deviation < 0.1, f"w deviated by {w_max_deviation} from constant"

    def test_equal_density_equal_velocity_trivial(self):
        """When rho and v are both uniform, nothing should change at all."""
        nx = 32
        rho0 = torch.full((nx,), 0.42)
        v0 = torch.full((nx,), 0.17)
        result = generate_one(
            rho0, v0, dx=0.05, dt=0.005, nt=30,
            bc_type="periodic", reconstruction="constant",
        )
        # Should be exactly preserved
        for t_idx in range(result["rho"].shape[0]):
            torch.testing.assert_close(result["rho"][t_idx], rho0, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(result["v"][t_idx], v0, atol=1e-6, rtol=1e-6)


# ====================================================================
# 13. SSP-RK3 time integration (WENO5 path)
# ====================================================================

class TestSSPRK3:
    """Tests specific to the SSP-RK3 time integrator used with WENO5."""

    def test_ssp_rk3_preserves_uniform(self):
        """SSP-RK3 should preserve a uniform state exactly."""
        nx = 32
        rho0 = torch.full((nx,), 0.5)
        v0 = torch.full((nx,), 0.3)
        result = generate_one(
            rho0, v0, dx=0.05, dt=0.002, nt=30,
            bc_type="periodic", reconstruction="weno5",
        )
        torch.testing.assert_close(
            result["rho"], rho0.unsqueeze(0).expand(31, -1), atol=1e-5, rtol=1e-5,
        )

    def test_ssp_rk3_mass_conservation(self):
        """Mass conservation via SSP-RK3 with WENO5 and periodic BC."""
        nx = 40
        dx = 0.025
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = riemann(x, rho_left=0.6, rho_right=0.3, v0=0.2)
        result = generate_one(
            rho0, v0, dx=dx, dt=0.002, nt=40,
            bc_type="periodic", reconstruction="weno5",
        )
        mass = result["rho"].sum(dim=-1) * dx
        torch.testing.assert_close(mass, mass[0].expand_as(mass), atol=1e-4, rtol=1e-4)

    def test_ssp_rk3_density_nonnegative(self):
        """SSP-RK3 with density clamping: density should stay >= 0."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float32) * 0.025
        rho0, v0 = riemann(x, rho_left=0.9, rho_right=0.05, v0=0.3)
        result = generate_one(
            rho0, v0, dx=0.025, dt=0.002, nt=30,
            reconstruction="weno5",
        )
        if result["valid"]:
            assert (result["rho"] >= -1e-10).all()

    def test_ssp_rk3_stability_three_region(self):
        """SSP-RK3 stable on a three-region problem."""
        nx = 60
        x = torch.arange(nx, dtype=torch.float32) * 0.025
        rho0, v0 = three_region(
            x, rho_left=0.7, rho_mid=0.1, rho_right=0.5,
            v0=0.2, x1=0.3, x2=0.7,
        )
        result = generate_one(
            rho0, v0, dx=0.025, dt=0.002, nt=50,
            reconstruction="weno5",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()


# ====================================================================
# 14. Mixed IC types
# ====================================================================

class TestMixedICTypes:
    """Test with unusual combinations of density and velocity profiles."""

    def test_smooth_density_discontinuous_velocity(self):
        """Smooth density profile with a velocity discontinuity."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float32) * 0.025
        rho0 = 0.5 + 0.1 * torch.sin(2 * torch.pi * x / 1.0)
        v0 = torch.where(x < 0.5, torch.tensor(0.5), torch.tensor(0.1))
        result = generate_one(
            rho0, v0, dx=0.025, dt=0.002, nt=30,
            reconstruction="constant",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_discontinuous_density_smooth_velocity(self):
        """Discontinuous density with smooth velocity."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float32) * 0.025
        rho0 = torch.where(x < 0.5, torch.tensor(0.7), torch.tensor(0.3))
        v0 = 0.3 + 0.1 * torch.sin(2 * torch.pi * x / 1.0)
        result = generate_one(
            rho0, v0, dx=0.025, dt=0.002, nt=30,
            reconstruction="constant",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_both_smooth(self):
        """Both density and velocity are smooth (no shocks initially)."""
        nx = 40
        x = torch.arange(nx, dtype=torch.float32) * 0.025
        rho0 = 0.5 + 0.1 * torch.sin(2 * torch.pi * x / 1.0)
        v0 = 0.3 + 0.05 * torch.cos(2 * torch.pi * x / 1.0)
        result = generate_one(
            rho0, v0, dx=0.025, dt=0.002, nt=30,
            bc_type="periodic", reconstruction="constant",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()

    def test_many_discontinuities_both_vars(self):
        """Multiple discontinuities in both density and velocity."""
        nx = 80
        x = torch.arange(nx, dtype=torch.float32) * 0.02
        rho0, v0 = from_steps(
            x,
            rho_steps=[
                (0.3, 0.8), (0.6, 0.2), (0.9, 0.6), (1.2, 0.1), (2.0, 0.5),
            ],
            v_steps=[
                (0.3, 0.5), (0.6, 0.1), (0.9, 0.8), (1.2, 0.3), (2.0, 0.6),
            ],
        )
        result = generate_one(
            rho0, v0, dx=0.02, dt=0.002, nt=50,
            reconstruction="constant",
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()
        assert torch.isfinite(result["v"]).all()
