"""Tests for ARZ generate_one / generate_n API contracts."""

import torch

from numerical_solvers.arz import generate_n, generate_one
from numerical_solvers.arz.initial_conditions import riemann
from numerical_solvers.arz.physics import pressure


class TestGenerateOne:
    def test_return_keys(self):
        nx, dx, dt, nt = 20, 0.05, 0.005, 10
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = riemann(x)
        result = generate_one(rho0, v0, dx=dx, dt=dt, nt=nt)
        assert set(result.keys()) == {"rho", "v", "w", "x", "t", "dx", "dt", "nt", "valid"}

    def test_shapes(self):
        nx, dx, dt, nt = 20, 0.05, 0.005, 10
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = riemann(x)
        result = generate_one(rho0, v0, dx=dx, dt=dt, nt=nt)
        assert result["rho"].shape == (nt + 1, nx)
        assert result["v"].shape == (nt + 1, nx)
        assert result["w"].shape == (nt + 1, nx)
        assert result["x"].shape == (nx,)
        assert result["t"].shape == (nt + 1,)

    def test_w_equals_v_plus_pressure(self):
        nx, dx, dt, nt = 20, 0.05, 0.005, 10
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = riemann(x)
        gamma = 1.0
        result = generate_one(rho0, v0, dx=dx, dt=dt, nt=nt, gamma=gamma)
        expected_w = result["v"] + pressure(result["rho"], gamma)
        torch.testing.assert_close(result["w"], expected_w, atol=1e-6, rtol=1e-6)

    def test_initial_row_matches_input(self):
        nx, dx, dt, nt = 20, 0.05, 0.005, 10
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = riemann(x)
        result = generate_one(rho0, v0, dx=dx, dt=dt, nt=nt)
        torch.testing.assert_close(result["rho"][0], rho0)
        torch.testing.assert_close(result["v"][0], v0)

    def test_metadata(self):
        nx, dx, dt, nt = 20, 0.05, 0.005, 10
        x = torch.arange(nx, dtype=torch.float32) * dx
        rho0, v0 = riemann(x)
        result = generate_one(rho0, v0, dx=dx, dt=dt, nt=nt)
        assert result["dx"] == dx
        assert result["dt"] == dt
        assert result["nt"] == nt


class TestGenerateN:
    def test_return_keys(self):
        result = generate_n(2, 2, nx=20, dx=0.05, dt=0.005, nt=10, seed=42, show_progress=False, reconstruction="constant")
        expected_keys = {"rho", "v", "w", "x", "t", "dx", "dt", "nt", "ic_xs", "ic_rho_ks", "ic_v_ks"}
        assert set(result.keys()) == expected_keys

    def test_shapes(self):
        n, k, nx, nt = 3, 2, 20, 10
        result = generate_n(n, k, nx=nx, dx=0.05, dt=0.005, nt=nt, seed=42, show_progress=False, reconstruction="constant")
        assert result["rho"].shape == (n, nt + 1, nx)
        assert result["v"].shape == (n, nt + 1, nx)
        assert result["w"].shape == (n, nt + 1, nx)

    def test_ic_params_shapes(self):
        n, k = 4, 3
        result = generate_n(n, k, nx=20, dx=0.05, dt=0.005, nt=10, seed=42, show_progress=False, reconstruction="constant")
        assert result["ic_xs"].shape == (n, k + 1)
        assert result["ic_rho_ks"].shape == (n, k)
        assert result["ic_v_ks"].shape == (n, k)

    def test_reproducibility(self):
        kwargs = dict(nx=20, dx=0.05, dt=0.005, nt=10, seed=123, show_progress=False, reconstruction="constant")
        r1 = generate_n(2, 2, **kwargs)
        r2 = generate_n(2, 2, **kwargs)
        torch.testing.assert_close(r1["rho"], r2["rho"])
