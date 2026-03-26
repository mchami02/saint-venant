"""Tests for LWR generate_one / generate_n API contracts."""

import torch

from numerical_solvers.lwr import generate_n, generate_one


class TestGenerateOne:
    def test_return_keys(self):
        result = generate_one([0.8, 0.2], [0, 0.5, 1], nx=20, nt=10, dx=0.05, dt=0.01)
        assert set(result.keys()) == {"rho", "x", "t", "dx", "dt", "nt"}

    def test_rho_shape(self):
        result = generate_one([0.8, 0.2], [0, 0.5, 1], nx=20, nt=10, dx=0.05, dt=0.01)
        assert result["rho"].shape == (10, 20)

    def test_x_t_shapes(self):
        result = generate_one([0.8, 0.2], [0, 0.5, 1], nx=20, nt=10, dx=0.05, dt=0.01)
        assert result["x"].shape == (20,)
        assert result["t"].shape == (10,)

    def test_dtype_float64(self):
        result = generate_one([0.8, 0.2], [0, 0.5, 1], nx=20, nt=10, dx=0.05, dt=0.01)
        assert result["rho"].dtype == torch.float64
        assert result["x"].dtype == torch.float64
        assert result["t"].dtype == torch.float64

    def test_metadata_passthrough(self):
        result = generate_one([0.5], nx=15, nt=8, dx=0.1, dt=0.02)
        assert result["dx"] == 0.1
        assert result["dt"] == 0.02
        assert result["nt"] == 8


class TestGenerateN:
    def test_return_keys(self):
        result = generate_n(2, 2, nx=20, nt=10, dx=0.05, dt=0.01, seed=42, show_progress=False)
        assert set(result.keys()) == {"rho", "x", "t", "dx", "dt", "nt", "ic_xs", "ic_ks"}

    def test_rho_shape(self):
        result = generate_n(3, 2, nx=20, nt=10, dx=0.05, dt=0.01, seed=42, show_progress=False)
        assert result["rho"].shape == (3, 10, 20)

    def test_ic_params_shapes(self):
        k = 3
        n = 4
        result = generate_n(n, k, nx=20, nt=10, dx=0.05, dt=0.01, seed=42, show_progress=False)
        assert result["ic_xs"].shape == (n, k + 1)
        assert result["ic_ks"].shape == (n, k)

    def test_reproducibility(self):
        r1 = generate_n(2, 2, nx=20, nt=10, dx=0.05, dt=0.01, seed=123, show_progress=False)
        r2 = generate_n(2, 2, nx=20, nt=10, dx=0.05, dt=0.01, seed=123, show_progress=False)
        torch.testing.assert_close(r1["rho"], r2["rho"])

    def test_different_seeds_differ(self):
        r1 = generate_n(2, 2, nx=20, nt=10, dx=0.05, dt=0.01, seed=1, show_progress=False)
        r2 = generate_n(2, 2, nx=20, nt=10, dx=0.05, dt=0.01, seed=2, show_progress=False)
        assert not torch.equal(r1["rho"], r2["rho"])
