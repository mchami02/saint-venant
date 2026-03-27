"""Tests for LWR steady state preservation."""

import torch

from numerical_solvers.src.lwr import generate_one


class TestSteadyState:
    def test_uniform_zero(self):
        """rho=0 everywhere should stay zero (zero flux)."""
        result = generate_one([0.0], nx=20, nt=20, dx=0.05, dt=0.01)
        rho = result["rho"]
        torch.testing.assert_close(
            rho, torch.zeros_like(rho), atol=1e-10, rtol=1e-10
        )

    def test_uniform_max(self):
        """rho=1 (rho_max) everywhere should stay constant."""
        result = generate_one([1.0], nx=20, nt=20, dx=0.05, dt=0.01)
        rho = result["rho"]
        torch.testing.assert_close(
            rho, torch.ones_like(rho), atol=1e-10, rtol=1e-10
        )

    def test_uniform_mid(self):
        """rho=0.5 everywhere should stay constant (uniform, no gradient)."""
        result = generate_one([0.5], nx=20, nt=20, dx=0.05, dt=0.01)
        rho = result["rho"]
        torch.testing.assert_close(
            rho, torch.full_like(rho, 0.5), atol=1e-10, rtol=1e-10
        )
