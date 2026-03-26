"""Tests for Euler generate_one / generate_n API contracts."""

import torch

from numerical_solvers.euler import generate_n, generate_one
from numerical_solvers.euler.initial_conditions import sod


class TestGenerateOne:
    def test_return_keys(self):
        nx, dx, dt, nt = 20, 0.05, 0.002, 10
        x = torch.arange(nx, dtype=torch.float64) * dx
        rho0, u0, p0 = sod(x)
        result = generate_one(rho0, u0, p0, dx=dx, dt=dt, nt=nt)
        assert set(result.keys()) == {"rho", "u", "p", "x", "t", "dx", "dt", "nt", "valid"}

    def test_shapes(self):
        nx, dx, dt, nt = 20, 0.05, 0.002, 10
        x = torch.arange(nx, dtype=torch.float64) * dx
        rho0, u0, p0 = sod(x)
        result = generate_one(rho0, u0, p0, dx=dx, dt=dt, nt=nt)
        assert result["rho"].shape == (nt + 1, nx)
        assert result["u"].shape == (nt + 1, nx)
        assert result["p"].shape == (nt + 1, nx)
        assert result["x"].shape == (nx,)
        assert result["t"].shape == (nt + 1,)

    def test_dtype_float64(self):
        nx, dx, dt, nt = 20, 0.05, 0.002, 10
        x = torch.arange(nx, dtype=torch.float64) * dx
        rho0, u0, p0 = sod(x)
        result = generate_one(rho0, u0, p0, dx=dx, dt=dt, nt=nt)
        assert result["rho"].dtype == torch.float64
        assert result["u"].dtype == torch.float64
        assert result["p"].dtype == torch.float64

    def test_initial_row_matches_input(self):
        nx, dx, dt, nt = 20, 0.05, 0.002, 10
        x = torch.arange(nx, dtype=torch.float64) * dx
        rho0, u0, p0 = sod(x)
        result = generate_one(rho0, u0, p0, dx=dx, dt=dt, nt=nt)
        torch.testing.assert_close(result["rho"][0], rho0)
        torch.testing.assert_close(result["u"][0], u0)
        torch.testing.assert_close(result["p"][0], p0)

    def test_metadata(self):
        nx, dx, dt, nt = 20, 0.05, 0.002, 10
        x = torch.arange(nx, dtype=torch.float64) * dx
        rho0, u0, p0 = sod(x)
        result = generate_one(rho0, u0, p0, dx=dx, dt=dt, nt=nt)
        assert result["dx"] == dx
        assert result["dt"] == dt
        assert result["nt"] == nt


class TestGenerateN:
    def test_return_keys(self):
        result = generate_n(2, 2, nx=20, dx=0.05, dt=0.002, nt=10, seed=42, show_progress=False, reconstruction="constant")
        expected_keys = {"rho", "u", "p", "x", "t", "dx", "dt", "nt", "ic_xs", "ic_rho_ks", "ic_u_ks", "ic_p_ks"}
        assert set(result.keys()) == expected_keys

    def test_shapes(self):
        n, k, nx, nt = 3, 2, 20, 10
        result = generate_n(n, k, nx=nx, dx=0.05, dt=0.002, nt=nt, seed=42, show_progress=False, reconstruction="constant")
        assert result["rho"].shape == (n, nt + 1, nx)
        assert result["u"].shape == (n, nt + 1, nx)
        assert result["p"].shape == (n, nt + 1, nx)

    def test_ic_params_shapes(self):
        n, k = 4, 3
        result = generate_n(n, k, nx=20, dx=0.05, dt=0.002, nt=10, seed=42, show_progress=False, reconstruction="constant")
        assert result["ic_xs"].shape == (n, k + 1)
        assert result["ic_rho_ks"].shape == (n, k)
        assert result["ic_u_ks"].shape == (n, k)
        assert result["ic_p_ks"].shape == (n, k)

    def test_dtype_float64(self):
        result = generate_n(2, 2, nx=20, dx=0.05, dt=0.002, nt=10, seed=42, show_progress=False, reconstruction="constant")
        assert result["rho"].dtype == torch.float64

    def test_reproducibility(self):
        kwargs = dict(nx=20, dx=0.05, dt=0.002, nt=10, seed=123, show_progress=False, reconstruction="constant")
        r1 = generate_n(2, 2, **kwargs)
        r2 = generate_n(2, 2, **kwargs)
        torch.testing.assert_close(r1["rho"], r2["rho"])
