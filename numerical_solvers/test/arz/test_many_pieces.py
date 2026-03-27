"""Tests for ARZ solver with many-piece initial conditions (k >= 5).

Verifies that the solver handles complex piecewise-constant ICs correctly:
output shapes, variable bounds, no NaN/Inf, conservation, and reproducibility.
"""

import numpy as np
import pytest
import torch

from numerical_solvers.src.arz import generate_n, generate_one
from numerical_solvers.src.arz.initial_conditions import from_steps
from numerical_solvers.src.arz.physics import pressure


# ============================================================================
# Hand-crafted many-piece ICs via generate_one
# ============================================================================


@pytest.fixture(
    params=[
        ("rusanov", "constant"),
        ("hll", "constant"),
        ("hll", "weno5"),
    ],
    ids=["rusanov-const", "hll-const", "hll-weno5"],
)
def flux_recon(request):
    return request.param


class TestManyPieceGenerateOne:
    """generate_one with hand-crafted ICs having 5+ pieces."""

    def test_staircase_density_5_pieces(self, flux_recon):
        """Monotone ascending density staircase, uniform velocity."""
        flux_type, reconstruction = flux_recon
        nx, dx, dt, nt = 100, 0.01, 0.002, 50
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = from_steps(
            x,
            rho_steps=[(0.2, 0.2), (0.4, 0.4), (0.6, 0.6), (0.8, 0.8), (2.0, 0.95)],
            default_v=0.3,
        )
        result = generate_one(
            rho0, v0, dx=dx, dt=dt, nt=nt, gamma=1.0,
            bc_type="zero_gradient", flux_type=flux_type, reconstruction=reconstruction,
        )
        assert result["valid"], f"Failed for {flux_type}/{reconstruction}"
        assert torch.isfinite(result["rho"]).all()
        assert torch.isfinite(result["v"]).all()

    def test_alternating_density_6_pieces(self, flux_recon):
        """Alternating high/low density with uniform velocity."""
        flux_type, reconstruction = flux_recon
        nx, dx, dt, nt = 120, 1.0 / 120, 0.002, 60
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = from_steps(
            x,
            rho_steps=[
                (0.15, 0.9), (0.35, 0.15), (0.55, 0.85),
                (0.7, 0.2), (0.85, 0.8), (2.0, 0.25),
            ],
            default_v=0.3,
        )
        result = generate_one(
            rho0, v0, dx=dx, dt=dt, nt=nt, gamma=1.0,
            bc_type="zero_gradient", flux_type=flux_type, reconstruction=reconstruction,
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()
        assert (result["rho"] >= -1e-6).all()

    def test_varying_velocity_5_pieces(self, flux_recon):
        """Both rho and v vary across 5 pieces."""
        flux_type, reconstruction = flux_recon
        nx, dx, dt, nt = 100, 0.01, 0.002, 60
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = from_steps(
            x,
            rho_steps=[(0.2, 0.7), (0.4, 0.3), (0.6, 0.8), (0.8, 0.2), (2.0, 0.6)],
            v_steps=[(0.2, 0.8), (0.4, 0.2), (0.6, 0.7), (0.8, 0.1), (2.0, 0.5)],
        )
        result = generate_one(
            rho0, v0, dx=dx, dt=dt, nt=nt, gamma=1.0,
            bc_type="zero_gradient", flux_type=flux_type, reconstruction=reconstruction,
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()
        assert torch.isfinite(result["v"]).all()

    def test_all_variables_varying_7_pieces(self, flux_recon):
        """7 pieces with both rho and v varying — complex wave pattern."""
        flux_type, reconstruction = flux_recon
        nx, dx, dt, nt = 140, 1.0 / 140, 0.002, 50
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = from_steps(
            x,
            rho_steps=[
                (0.15, 0.8), (0.3, 0.2), (0.45, 0.7), (0.55, 0.3),
                (0.7, 0.9), (0.85, 0.15), (2.0, 0.5),
            ],
            v_steps=[
                (0.15, 0.6), (0.3, 0.2), (0.45, 0.7), (0.55, 0.1),
                (0.7, 0.5), (0.85, 0.3), (2.0, 0.4),
            ],
        )
        result = generate_one(
            rho0, v0, dx=dx, dt=dt, nt=nt, gamma=1.0,
            bc_type="zero_gradient", flux_type=flux_type, reconstruction=reconstruction,
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()
        assert torch.isfinite(result["v"]).all()
        assert (result["rho"] >= -1e-6).all()

    def test_narrow_density_pulse_9_pieces(self, flux_recon):
        """Low background with a narrow high-density pulse."""
        flux_type, reconstruction = flux_recon
        nx, dx, dt, nt = 180, 1.0 / 180, 0.001, 60
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = from_steps(
            x,
            rho_steps=[
                (0.1, 0.2), (0.2, 0.2), (0.3, 0.2), (0.4, 0.2),
                (0.55, 0.95), (0.65, 0.2), (0.75, 0.2), (0.85, 0.2), (2.0, 0.2),
            ],
            default_v=0.3,
        )
        result = generate_one(
            rho0, v0, dx=dx, dt=dt, nt=nt, gamma=1.0,
            bc_type="zero_gradient", flux_type=flux_type, reconstruction=reconstruction,
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()
        assert (result["rho"] >= -1e-6).all()

    def test_long_time_5_pieces(self, flux_recon):
        """Run long enough for all 5-piece waves to interact."""
        flux_type, reconstruction = flux_recon
        nx, dx, dt, nt = 100, 0.01, 0.002, 200
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = from_steps(
            x,
            rho_steps=[(0.2, 0.8), (0.4, 0.2), (0.6, 0.7), (0.8, 0.3), (2.0, 0.6)],
            default_v=0.25,
        )
        result = generate_one(
            rho0, v0, dx=dx, dt=dt, nt=nt, gamma=1.0,
            bc_type="zero_gradient", flux_type=flux_type, reconstruction=reconstruction,
        )
        assert result["valid"]
        assert torch.isfinite(result["rho"]).all()
        assert (result["rho"] >= -1e-6).all()

    def test_w_invariant_many_pieces(self, flux_recon):
        """w = v + p(rho) holds at all timesteps for 6-piece IC."""
        flux_type, reconstruction = flux_recon
        nx, dx, dt, nt = 120, 1.0 / 120, 0.002, 40
        gamma = 1.0
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = from_steps(
            x,
            rho_steps=[
                (0.15, 0.7), (0.35, 0.3), (0.55, 0.8),
                (0.7, 0.2), (0.85, 0.6), (2.0, 0.4),
            ],
            default_v=0.25,
        )
        result = generate_one(
            rho0, v0, dx=dx, dt=dt, nt=nt, gamma=gamma,
            bc_type="zero_gradient", flux_type=flux_type, reconstruction=reconstruction,
        )
        assert result["valid"]
        expected_w = result["v"] + pressure(result["rho"], gamma)
        torch.testing.assert_close(result["w"], expected_w, atol=1e-5, rtol=1e-5)


# ============================================================================
# Conservation with many-piece periodic ICs
# ============================================================================


class TestManyPieceConservation:
    """Mass and rho*w conservation with periodic BCs and many pieces."""

    @pytest.mark.parametrize("k_pieces", [5, 7])
    def test_mass_conservation_periodic(self, k_pieces):
        """Mass conserved with k-piece periodic ICs."""
        nx, dx, dt, nt = 80, 1.0 / 80, 0.002, 40
        x = torch.arange(nx, dtype=torch.float32) * dx

        piece_width = 1.0 / k_pieces
        rho_steps = [
            (piece_width * (i + 1), 0.3 + 0.3 * ((-1) ** i))
            for i in range(k_pieces)
        ]
        rho_steps[-1] = (2.0, rho_steps[-1][1])

        rho0, v0 = from_steps(x, rho_steps=rho_steps, default_v=0.25)
        result = generate_one(
            rho0, v0, dx=dx, dt=dt, nt=nt, gamma=1.0,
            bc_type="periodic", flux_type="hll", reconstruction="constant",
        )
        assert result["valid"]

        mass = result["rho"].sum(dim=-1) * dx
        torch.testing.assert_close(mass, mass[0].expand_as(mass), atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("k_pieces", [5, 7])
    def test_rho_w_conservation_periodic(self, k_pieces):
        """rho*w conserved with k-piece periodic ICs."""
        nx, dx, dt, nt = 80, 1.0 / 80, 0.002, 40
        x = torch.arange(nx, dtype=torch.float32) * dx

        piece_width = 1.0 / k_pieces
        rho_steps = [
            (piece_width * (i + 1), 0.3 + 0.3 * ((-1) ** i))
            for i in range(k_pieces)
        ]
        rho_steps[-1] = (2.0, rho_steps[-1][1])

        rho0, v0 = from_steps(x, rho_steps=rho_steps, default_v=0.25)
        result = generate_one(
            rho0, v0, dx=dx, dt=dt, nt=nt, gamma=1.0,
            bc_type="periodic", flux_type="hll", reconstruction="constant",
        )
        assert result["valid"]

        rho_w = result["rho"] * result["w"]
        rho_w_total = rho_w.sum(dim=-1) * dx
        torch.testing.assert_close(
            rho_w_total, rho_w_total[0].expand_as(rho_w_total), atol=1e-4, rtol=1e-4,
        )


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
            n, k, nx=nx, dx=1.0 / nx, dt=0.005, nt=nt,
            seed=42, show_progress=False, reconstruction="constant",
        )
        assert result["rho"].shape == (n, nt + 1, nx)
        assert result["v"].shape == (n, nt + 1, nx)
        assert result["w"].shape == (n, nt + 1, nx)
        assert result["ic_xs"].shape == (n, k + 1)
        assert result["ic_rho_ks"].shape == (n, k)
        assert result["ic_v_ks"].shape == (n, k)

    @pytest.mark.parametrize("k", [5, 8, 10])
    def test_no_nan_inf(self, k):
        """Solutions contain no NaN/Inf for many-piece random ICs."""
        n, nx, nt = 6, 80, 30
        result = generate_n(
            n, k, nx=nx, dx=1.0 / nx, dt=0.005, nt=nt,
            seed=42, show_progress=False, reconstruction="constant",
        )
        assert torch.isfinite(result["rho"]).all(), f"NaN/Inf in rho for k={k}"
        assert torch.isfinite(result["v"]).all(), f"NaN/Inf in v for k={k}"
        assert torch.isfinite(result["w"]).all(), f"NaN/Inf in w for k={k}"

    @pytest.mark.parametrize("k", [5, 8, 10])
    def test_positive_density(self, k):
        """Density remains non-negative."""
        n, nx, nt = 6, 80, 30
        result = generate_n(
            n, k, nx=nx, dx=1.0 / nx, dt=0.005, nt=nt,
            seed=42, show_progress=False, reconstruction="constant",
        )
        assert (result["rho"] >= -1e-6).all(), f"Negative density for k={k}"

    @pytest.mark.parametrize("k", [5, 10])
    def test_reproducibility(self, k):
        """Same seed produces identical results for many-piece ICs."""
        kwargs = dict(
            nx=60, dx=1.0 / 60, dt=0.005, nt=15,
            seed=99, show_progress=False, reconstruction="constant",
        )
        r1 = generate_n(3, k, **kwargs)
        r2 = generate_n(3, k, **kwargs)
        torch.testing.assert_close(r1["rho"], r2["rho"])
        torch.testing.assert_close(r1["v"], r2["v"])
        torch.testing.assert_close(r1["w"], r2["w"])
        np.testing.assert_array_equal(r1["ic_rho_ks"], r2["ic_rho_ks"])

    @pytest.mark.parametrize("k", [5, 10])
    def test_w_invariant(self, k):
        """w = v + p(rho) holds for all generated samples."""
        n, nx, nt = 4, 60, 20
        gamma = 1.0
        result = generate_n(
            n, k, nx=nx, dx=1.0 / nx, dt=0.005, nt=nt,
            seed=42, show_progress=False, reconstruction="constant",
        )
        expected_w = result["v"] + pressure(result["rho"], gamma)
        torch.testing.assert_close(result["w"], expected_w, atol=1e-5, rtol=1e-5)

    def test_breakpoints_ordered(self):
        """Breakpoints are strictly increasing for all samples."""
        n, k, nx, nt = 8, 10, 100, 10
        result = generate_n(
            n, k, nx=nx, dx=1.0 / nx, dt=0.005, nt=nt,
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
        rho_range = (0.15, 0.85)
        v_range = (0.1, 0.8)
        result = generate_n(
            n, k, nx=80, dx=1.0 / 80, dt=0.005, nt=10,
            rho_range=rho_range, v_range=v_range,
            seed=42, show_progress=False, reconstruction="constant",
        )
        for i in range(n):
            assert all(rho_range[0] <= v <= rho_range[1] for v in result["ic_rho_ks"][i])
            assert all(v_range[0] <= v <= v_range[1] for v in result["ic_v_ks"][i])
