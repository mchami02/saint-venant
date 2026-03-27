"""Tests for WENO-5 spatial reconstruction."""

import torch

from numerical_solvers.src.arz.weno import weno5_reconstruct


class TestWENO5:
    def test_constant_data_exact(self):
        """Constant data should be reconstructed exactly."""
        val = 3.14
        v = torch.full((20,), val)
        v_minus, v_plus = weno5_reconstruct(v)
        torch.testing.assert_close(v_minus, torch.full_like(v_minus, val), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(v_plus, torch.full_like(v_plus, val), atol=1e-5, rtol=1e-5)

    def test_linear_data_exact(self):
        """Linear data should be reconstructed exactly by 5th-order scheme."""
        N = 20
        v = torch.arange(N, dtype=torch.float64) * 0.5 + 1.0
        v_minus, v_plus = weno5_reconstruct(v)
        ni = N - 7
        # For linear data, interface values should match the linear interpolation
        # v_minus[j] reconstructs from left at interface j+4 (in the original indexing)
        # For linear f(x) = 0.5*x + 1.0, the interface value at j+3.5 is 0.5*(j+3.5) + 1.0
        expected = torch.arange(ni, dtype=torch.float64) * 0.5 + 0.5 * 3.5 + 1.0
        torch.testing.assert_close(v_minus, expected, atol=1e-10, rtol=1e-10)

    def test_output_shapes(self):
        N = 16
        v = torch.rand(N)
        v_minus, v_plus = weno5_reconstruct(v)
        assert v_minus.shape == (N - 7,)
        assert v_plus.shape == (N - 7,)

    def test_discontinuous_data_no_nan(self):
        v = torch.cat([torch.ones(8), torch.full((8,), 0.1)])
        v_minus, v_plus = weno5_reconstruct(v)
        assert torch.isfinite(v_minus).all()
        assert torch.isfinite(v_plus).all()
