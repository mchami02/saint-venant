"""Tests for LWR wave interaction scenarios (multiple colliding waves)."""

import torch

from numerical_solvers.src.lwr import generate_one


class TestMultipleShockCollision:
    """Multiple discontinuities producing waves that collide."""

    def test_three_piece_converging_shocks(self):
        """High-low-high IC: two shocks propagate inward and collide."""
        ks = [0.8, 0.1, 0.8]
        xs = [0, 0.3, 0.7, 1]
        result = generate_one(ks, xs, nx=100, nt=200, dx=0.01, dt=0.002)
        rho = result["rho"]
        assert torch.isfinite(rho).all(), "NaN/Inf after shock collision"
        assert rho.min() >= -1e-10, f"Negative density: {rho.min()}"
        assert rho.max() <= 1.0 + 1e-10, f"Density exceeds rho_max: {rho.max()}"

    def test_five_piece_multiple_interactions(self):
        """Five-piece IC with alternating high/low: many wave interactions."""
        ks = [0.9, 0.1, 0.7, 0.2, 0.8]
        xs = [0, 0.2, 0.4, 0.6, 0.8, 1]
        result = generate_one(ks, xs, nx=100, nt=300, dx=0.01, dt=0.002)
        rho = result["rho"]
        assert torch.isfinite(rho).all(), "NaN/Inf after multiple wave interactions"
        assert rho.min() >= -1e-10
        assert rho.max() <= 1.0 + 1e-10

    def test_rarefaction_shock_collision(self):
        """Low-high-low IC: rarefaction meets shock."""
        ks = [0.2, 0.9, 0.2]
        xs = [0, 0.3, 0.7, 1]
        result = generate_one(ks, xs, nx=100, nt=200, dx=0.01, dt=0.002)
        rho = result["rho"]
        assert torch.isfinite(rho).all()
        assert rho.min() >= -1e-10
        assert rho.max() <= 1.0 + 1e-10

    def test_long_time_many_collisions(self):
        """Run long enough for waves to cross entire domain multiple times."""
        ks = [0.7, 0.3, 0.6, 0.2]
        xs = [0, 0.25, 0.5, 0.75, 1]
        result = generate_one(ks, xs, nx=80, nt=500, dx=0.0125, dt=0.002)
        rho = result["rho"]
        assert torch.isfinite(rho).all(), "NaN/Inf after long-time evolution"
        assert rho.min() >= -1e-10
        assert rho.max() <= 1.0 + 1e-10
