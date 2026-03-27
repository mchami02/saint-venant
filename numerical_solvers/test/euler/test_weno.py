"""Tests for WENO-5 spatial reconstruction (Euler module)."""

import torch

from numerical_solvers.src.euler.weno import weno5_reconstruct


class TestWENO5:
    def test_constant_data_exact(self):
        val = 2.71
        v = torch.full((20,), val, dtype=torch.float64)
        v_minus, v_plus = weno5_reconstruct(v)
        torch.testing.assert_close(v_minus, torch.full_like(v_minus, val), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(v_plus, torch.full_like(v_plus, val), atol=1e-5, rtol=1e-5)

    def test_linear_data_exact(self):
        N = 20
        v = torch.arange(N, dtype=torch.float64) * 0.5 + 1.0
        v_minus, v_plus = weno5_reconstruct(v)
        ni = N - 7
        expected = torch.arange(ni, dtype=torch.float64) * 0.5 + 0.5 * 3.5 + 1.0
        torch.testing.assert_close(v_minus, expected, atol=1e-10, rtol=1e-10)

    def test_output_shapes(self):
        N = 16
        v = torch.rand(N, dtype=torch.float64)
        v_minus, v_plus = weno5_reconstruct(v)
        assert v_minus.shape == (N - 7,)
        assert v_plus.shape == (N - 7,)

    def test_discontinuity_no_nan(self):
        v = torch.cat([torch.ones(8, dtype=torch.float64), torch.full((8,), 0.1, dtype=torch.float64)])
        v_minus, v_plus = weno5_reconstruct(v)
        assert torch.isfinite(v_minus).all()
        assert torch.isfinite(v_plus).all()
