"""Tests for LWR density bounds and solution quality."""

import torch

from numerical_solvers.lwr import generate_one


class TestDensityBounds:
    def test_density_bounded_0_1_shock(self):
        """Shock wave: rho_left > rho_right (for Greenshields)."""
        result = generate_one([0.8, 0.2], [0, 0.5, 1], nx=50, nt=50, dx=0.02, dt=0.005)
        rho = result["rho"]
        assert rho.min() >= -1e-10, f"Density below 0: {rho.min()}"
        assert rho.max() <= 1.0 + 1e-10, f"Density above rho_max: {rho.max()}"

    def test_density_bounded_0_1_rarefaction(self):
        """Rarefaction wave: rho_left < rho_right."""
        result = generate_one([0.2, 0.8], [0, 0.5, 1], nx=50, nt=50, dx=0.02, dt=0.005)
        rho = result["rho"]
        assert rho.min() >= -1e-10
        assert rho.max() <= 1.0 + 1e-10

    def test_no_nan_inf(self):
        result = generate_one([0.3, 0.7, 0.1], [0, 0.3, 0.7, 1], nx=50, nt=50, dx=0.02, dt=0.005)
        rho = result["rho"]
        assert torch.isfinite(rho).all(), "Solution contains NaN or Inf"
