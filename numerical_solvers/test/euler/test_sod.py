"""Tests for the Sod shock tube benchmark."""

import pytest
import torch

from numerical_solvers.src.euler import generate_one
from numerical_solvers.src.euler.initial_conditions import sod


class TestSodQualitative:
    """Qualitative checks on the Sod shock tube solution."""

    @pytest.fixture
    def sod_result(self):
        nx, dx, dt, nt = 100, 0.01, 0.001, 50
        x = torch.arange(nx, dtype=torch.float64) * dx
        rho0, u0, p0 = sod(x, x_split=0.5)
        return generate_one(
            rho0, u0, p0, dx=dx, dt=dt, nt=nt, gamma=1.4,
            bc_type="extrap", flux_type="hllc", reconstruction="weno5",
        )

    def test_valid(self, sod_result):
        assert sod_result["valid"]

    def test_undisturbed_left(self, sod_result):
        """Far-left region should still be close to left state."""
        rho = sod_result["rho"][-1]  # final time
        x = sod_result["x"]
        left_mask = x < 0.15
        if left_mask.sum() > 0:
            torch.testing.assert_close(
                rho[left_mask], torch.full((left_mask.sum(),), 1.0, dtype=torch.float64),
                atol=0.05, rtol=0.05,
            )

    def test_undisturbed_right(self, sod_result):
        """Far-right region should still be close to right state."""
        rho = sod_result["rho"][-1]
        x = sod_result["x"]
        right_mask = x > 0.9
        if right_mask.sum() > 0:
            torch.testing.assert_close(
                rho[right_mask], torch.full((right_mask.sum(),), 0.125, dtype=torch.float64),
                atol=0.05, rtol=0.05,
            )

    def test_velocity_zero_far_away(self, sod_result):
        """Undisturbed regions should have u ≈ 0."""
        u = sod_result["u"][-1]
        x = sod_result["x"]
        far_mask = (x < 0.1) | (x > 0.95)
        if far_mask.sum() > 0:
            assert u[far_mask].abs().max() < 0.1

    @pytest.mark.parametrize("flux_type", ["hllc", "hll", "rusanov"])
    def test_all_fluxes_agree_undisturbed(self, flux_type):
        """All flux types should agree on undisturbed regions."""
        nx, dx, dt, nt = 80, 0.0125, 0.001, 40
        x = torch.arange(nx, dtype=torch.float64) * dx
        rho0, u0, p0 = sod(x)
        result = generate_one(
            rho0, u0, p0, dx=dx, dt=dt, nt=nt, gamma=1.4,
            bc_type="extrap", flux_type=flux_type, reconstruction="constant",
        )
        rho = result["rho"][-1]
        right_mask = x > 0.9
        if right_mask.sum() > 0:
            torch.testing.assert_close(
                rho[right_mask], torch.full((right_mask.sum(),), 0.125, dtype=torch.float64),
                atol=0.05, rtol=0.05,
            )
