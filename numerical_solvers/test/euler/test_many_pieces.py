"""Tests for Euler solver with many-piece initial conditions (k >= 5).

Verifies that the solver handles complex piecewise-constant ICs correctly:
output shapes, variable bounds, no NaN/Inf, conservation, and reproducibility.
"""

import numpy as np
import pytest
import torch

from numerical_solvers.src.euler import generate_n, generate_one
from numerical_solvers.src.euler.initial_conditions import from_steps
from numerical_solvers.src.euler.physics import primitive_to_conservative


# ============================================================================
# Hand-crafted many-piece ICs via generate_one
# ============================================================================


@pytest.fixture(
    params=[
        ("hllc", "constant"),
        ("hll", "constant"),
        ("rusanov", "constant"),
    ],
    ids=["hllc-const", "hll-const", "rusanov-const"],
)
def flux_recon(request):
    return request.param


class TestManyPieceGenerateOne:
    """generate_one with hand-crafted ICs having 5+ pieces."""

    def test_staircase_density_5_pieces(self, flux_recon):
        """Monotone ascending density staircase, uniform u and p."""
        flux_type, reconstruction = flux_recon
        nx, dx, dt, nt = 100, 0.01, 0.0005, 50
        x = torch.arange(nx, dtype=torch.float64) * dx
        rho0, u0, p0 = from_steps(
            x,
            rho_steps=[(0.2, 0.3), (0.4, 0.6), (0.6, 1.0), (0.8, 1.5), (2.0, 2.0)],
            u_steps=[(2.0, 0.0)],
            p_steps=[(2.0, 1.0)],
        )
        result = generate_one(
            rho0, u0, p0, dx=dx, dt=dt, nt=nt, gamma=1.4,
            bc_type="extrap", flux_type=flux_type, reconstruction=reconstruction,
        )
        assert result["valid"], f"Failed for {flux_type}/{reconstruction}"
        assert torch.isfinite(result["rho"]).all()
        assert torch.isfinite(result["u"]).all()
        assert torch.isfinite(result["p"]).all()

    def test_alternating_pressure_6_pieces(self, flux_recon):
        """Alternating high/low pressure with uniform density: many shocks."""
        flux_type, reconstruction = flux_recon
        nx, dx, dt, nt = 120, 1.0 / 120, 0.0005, 60
        x = torch.arange(nx, dtype=torch.float64) * dx
        rho0, u0, p0 = from_steps(
            x,
            rho_steps=[(2.0, 1.0)],
            u_steps=[(2.0, 0.0)],
            p_steps=[
                (0.15, 5.0), (0.35, 0.5), (0.55, 4.0),
                (0.7, 0.3), (0.85, 3.0), (2.0, 0.8),
            ],
        )
        result = generate_one(
            rho0, u0, p0, dx=dx, dt=dt, nt=nt, gamma=1.4,
            bc_type="extrap", flux_type=flux_type, reconstruction=reconstruction,
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()
        assert (result["rho"] >= -1e-6).all()

    def test_counter_propagating_flows_5_pieces(self, flux_recon):
        """Alternating velocity directions: head-on collisions."""
        flux_type, reconstruction = flux_recon
        nx, dx, dt, nt = 100, 0.01, 0.0005, 60
        x = torch.arange(nx, dtype=torch.float64) * dx
        rho0, u0, p0 = from_steps(
            x,
            rho_steps=[(0.2, 1.0), (0.4, 0.5), (0.6, 1.0), (0.8, 0.5), (2.0, 1.0)],
            u_steps=[(0.2, 1.0), (0.4, -0.5), (0.6, 0.8), (0.8, -0.3), (2.0, 0.5)],
            p_steps=[(2.0, 1.0)],
        )
        result = generate_one(
            rho0, u0, p0, dx=dx, dt=dt, nt=nt, gamma=1.4,
            bc_type="extrap", flux_type=flux_type, reconstruction=reconstruction,
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()
        assert torch.isfinite(result["u"]).all()

    def test_all_variables_varying_7_pieces(self, flux_recon):
        """All three primitives (rho, u, p) vary across 7 pieces."""
        flux_type, reconstruction = flux_recon
        nx, dx, dt, nt = 140, 1.0 / 140, 0.0003, 50
        x = torch.arange(nx, dtype=torch.float64) * dx
        rho0, u0, p0 = from_steps(
            x,
            rho_steps=[
                (0.15, 1.5), (0.3, 0.3), (0.45, 1.2), (0.55, 0.5),
                (0.7, 1.8), (0.85, 0.4), (2.0, 1.0),
            ],
            u_steps=[
                (0.15, 0.5), (0.3, -0.3), (0.45, 0.8), (0.55, 0.0),
                (0.7, -0.5), (0.85, 0.3), (2.0, -0.2),
            ],
            p_steps=[
                (0.15, 3.0), (0.3, 0.5), (0.45, 2.5), (0.55, 0.8),
                (0.7, 4.0), (0.85, 0.3), (2.0, 1.5),
            ],
        )
        result = generate_one(
            rho0, u0, p0, dx=dx, dt=dt, nt=nt, gamma=1.4,
            bc_type="extrap", flux_type=flux_type, reconstruction=reconstruction,
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()
        assert torch.isfinite(result["p"]).all()
        assert (result["rho"] >= -1e-6).all()

    def test_narrow_high_pressure_pulse_9_pieces(self, flux_recon):
        """Background with a narrow high-pressure pulse: blast-like."""
        flux_type, reconstruction = flux_recon
        nx, dx, dt, nt = 180, 1.0 / 180, 0.0003, 60
        x = torch.arange(nx, dtype=torch.float64) * dx
        rho0, u0, p0 = from_steps(
            x,
            rho_steps=[
                (0.1, 1.0), (0.2, 1.0), (0.3, 1.0), (0.4, 1.0),
                (0.55, 1.0), (0.65, 1.0), (0.75, 1.0), (0.85, 1.0), (2.0, 1.0),
            ],
            u_steps=[(2.0, 0.0)],
            p_steps=[
                (0.1, 1.0), (0.2, 1.0), (0.3, 1.0), (0.4, 1.0),
                (0.55, 10.0), (0.65, 1.0), (0.75, 1.0), (0.85, 1.0), (2.0, 1.0),
            ],
        )
        result = generate_one(
            rho0, u0, p0, dx=dx, dt=dt, nt=nt, gamma=1.4,
            bc_type="extrap", flux_type=flux_type, reconstruction=reconstruction,
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()
        assert (result["rho"] >= -1e-6).all()

    def test_long_time_many_interactions_5_pieces(self, flux_recon):
        """Run long enough for all 5-piece waves to interact multiple times."""
        flux_type, reconstruction = flux_recon
        nx, dx, dt, nt = 100, 0.01, 0.0005, 200
        x = torch.arange(nx, dtype=torch.float64) * dx
        rho0, u0, p0 = from_steps(
            x,
            rho_steps=[(0.2, 1.5), (0.4, 0.3), (0.6, 1.2), (0.8, 0.5), (2.0, 1.0)],
            u_steps=[(2.0, 0.0)],
            p_steps=[(0.2, 3.0), (0.4, 0.5), (0.6, 2.0), (0.8, 0.8), (2.0, 1.5)],
        )
        result = generate_one(
            rho0, u0, p0, dx=dx, dt=dt, nt=nt, gamma=1.4,
            bc_type="extrap", flux_type=flux_type, reconstruction=reconstruction,
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()
        assert (result["rho"] >= -1e-6).all()


# ============================================================================
# Conservation with many-piece periodic ICs
# ============================================================================


class TestManyPieceConservation:
    """Mass/momentum/energy conservation with periodic BCs and many pieces."""

    @pytest.mark.parametrize("k_pieces", [5, 7])
    def test_conservation_periodic(self, k_pieces):
        """All conserved quantities preserved with k-piece periodic ICs."""
        gamma = 1.4
        nx, dx, dt, nt = 80, 1.0 / 80, 0.0005, 50
        x = torch.arange(nx, dtype=torch.float64) * dx

        # Build k-piece IC with moderate contrasts
        piece_width = 1.0 / k_pieces
        rho_steps = [(piece_width * (i + 1), 0.5 + 0.3 * ((-1) ** i)) for i in range(k_pieces)]
        rho_steps[-1] = (2.0, rho_steps[-1][1])  # extend last piece
        p_steps = [(piece_width * (i + 1), 1.0 + 0.4 * ((-1) ** i)) for i in range(k_pieces)]
        p_steps[-1] = (2.0, p_steps[-1][1])

        rho0, u0, p0 = from_steps(
            x,
            rho_steps=rho_steps,
            u_steps=[(2.0, 0.0)],
            p_steps=p_steps,
        )
        result = generate_one(
            rho0, u0, p0, dx=dx, dt=dt, nt=nt, gamma=gamma,
            bc_type="periodic", flux_type="hllc", reconstruction="constant",
        )
        assert result["valid"]

        # Mass conservation
        mass = result["rho"].sum(dim=-1) * dx
        torch.testing.assert_close(mass, mass[0].expand_as(mass), atol=1e-8, rtol=1e-8)

        # Momentum conservation
        momentum = (result["rho"] * result["u"]).sum(dim=-1) * dx
        torch.testing.assert_close(momentum, momentum[0].expand_as(momentum), atol=1e-8, rtol=1e-8)

        # Energy conservation
        rho, u, p = result["rho"], result["u"], result["p"]
        E = p / (gamma - 1.0) + 0.5 * rho * u ** 2
        energy = E.sum(dim=-1) * dx
        torch.testing.assert_close(energy, energy[0].expand_as(energy), atol=1e-8, rtol=1e-8)


# ============================================================================
# generate_n with many pieces
# ============================================================================


class TestManyPieceGenerateN:
    """generate_n with higher piece counts."""

    @pytest.mark.parametrize("k", [5, 8, 10])
    def test_shapes(self, k):
        """Output shapes are correct for various k values."""
        n, nx, nt = 4, 80, 20
        result = generate_n(
            n, k, nx=nx, dx=1.0 / nx, dt=0.001, nt=nt,
            seed=42, show_progress=False, reconstruction="constant",
        )
        assert result["rho"].shape == (n, nt + 1, nx)
        assert result["u"].shape == (n, nt + 1, nx)
        assert result["p"].shape == (n, nt + 1, nx)
        assert result["ic_xs"].shape == (n, k + 1)
        assert result["ic_rho_ks"].shape == (n, k)
        assert result["ic_u_ks"].shape == (n, k)
        assert result["ic_p_ks"].shape == (n, k)

    @pytest.mark.parametrize("k", [5, 8, 10])
    def test_no_nan_inf(self, k):
        """Solutions contain no NaN/Inf for many-piece random ICs."""
        n, nx, nt = 6, 80, 30
        result = generate_n(
            n, k, nx=nx, dx=1.0 / nx, dt=0.001, nt=nt,
            seed=42, show_progress=False, reconstruction="constant",
        )
        assert torch.isfinite(result["rho"]).all(), f"NaN/Inf in rho for k={k}"
        assert torch.isfinite(result["u"]).all(), f"NaN/Inf in u for k={k}"
        assert torch.isfinite(result["p"]).all(), f"NaN/Inf in p for k={k}"

    @pytest.mark.parametrize("k", [5, 8, 10])
    def test_positive_density_pressure(self, k):
        """Density and pressure remain non-negative."""
        n, nx, nt = 6, 80, 30
        result = generate_n(
            n, k, nx=nx, dx=1.0 / nx, dt=0.001, nt=nt,
            seed=42, show_progress=False, reconstruction="constant",
        )
        assert (result["rho"] >= -1e-6).all(), f"Negative density for k={k}"
        assert (result["p"] >= -1e-6).all(), f"Negative pressure for k={k}"

    @pytest.mark.parametrize("k", [5, 10])
    def test_reproducibility(self, k):
        """Same seed produces identical results for many-piece ICs."""
        kwargs = dict(
            nx=60, dx=1.0 / 60, dt=0.001, nt=15,
            seed=99, show_progress=False, reconstruction="constant",
        )
        r1 = generate_n(3, k, **kwargs)
        r2 = generate_n(3, k, **kwargs)
        torch.testing.assert_close(r1["rho"], r2["rho"])
        torch.testing.assert_close(r1["u"], r2["u"])
        torch.testing.assert_close(r1["p"], r2["p"])
        np.testing.assert_array_equal(r1["ic_rho_ks"], r2["ic_rho_ks"])

    def test_weno5_fallback_high_k(self):
        """With k > nx//8, reconstruction falls back to constant automatically."""
        nx = 40
        k = nx // 8 + 1  # just above the threshold
        result = generate_n(
            2, k, nx=nx, dx=1.0 / nx, dt=0.001, nt=10,
            seed=42, show_progress=False, reconstruction="weno5",
        )
        # Should not blow up — fallback to constant handles it
        assert torch.isfinite(result["rho"]).all()

    def test_breakpoints_ordered(self):
        """Breakpoints are strictly increasing for all samples."""
        n, k, nx, nt = 8, 10, 100, 10
        result = generate_n(
            n, k, nx=nx, dx=1.0 / nx, dt=0.001, nt=nt,
            seed=42, show_progress=False, reconstruction="constant",
        )
        for i in range(n):
            xs = result["ic_xs"][i]
            assert len(xs) == k + 1
            for j in range(len(xs) - 1):
                assert xs[j] < xs[j + 1], f"Breakpoints not sorted: sample {i}"

    def test_ic_values_in_range(self):
        """Piece values respect the specified ranges."""
        n, k = 6, 8
        rho_range = (0.2, 1.5)
        u_range = (-1.0, 1.0)
        p_range = (0.2, 3.0)
        result = generate_n(
            n, k, nx=80, dx=1.0 / 80, dt=0.001, nt=10,
            rho_range=rho_range, u_range=u_range, p_range=p_range,
            seed=42, show_progress=False, reconstruction="constant",
        )
        for i in range(n):
            assert all(rho_range[0] <= v <= rho_range[1] for v in result["ic_rho_ks"][i])
            assert all(u_range[0] <= v <= u_range[1] for v in result["ic_u_ks"][i])
            assert all(p_range[0] <= v <= p_range[1] for v in result["ic_p_ks"][i])

    def test_initial_row_matches_ic(self):
        """First time row of output matches the piecewise IC."""
        n, k, nx, nt = 3, 5, 60, 10
        result = generate_n(
            n, k, nx=nx, dx=1.0 / nx, dt=0.001, nt=nt,
            seed=42, show_progress=False, reconstruction="constant",
        )
        # rho at t=0 should be piecewise constant matching ic_rho_ks
        for i in range(n):
            rho_t0 = result["rho"][i, 0]  # (nx,)
            assert torch.isfinite(rho_t0).all()
            # Each cell value should be one of the piece values
            unique_vals = set(rho_t0.unique().tolist())
            ic_vals = set(result["ic_rho_ks"][i].tolist())
            assert unique_vals.issubset(ic_vals), (
                f"t=0 density contains values not in IC: "
                f"{unique_vals - ic_vals}"
            )
