"""Tests for Burgers generate_one / generate_n API contracts."""

import torch

from numerical_solvers.src.burgers import generate_n, generate_one, riemann


class TestGenerateOne:
    def test_return_keys(self):
        nx, dx, dt, nt = 20, 0.05, 0.002, 10
        x = torch.arange(nx, dtype=torch.float64) * dx
        u0 = riemann(x, u_left=1.0, u_right=0.0, x_split=0.5)
        result = generate_one(u0, dx=dx, dt=dt, nt=nt)
        assert set(result.keys()) == {"u", "x", "t", "dx", "dt", "nt", "valid"}

    def test_shapes(self):
        nx, dx, dt, nt = 20, 0.05, 0.002, 10
        x = torch.arange(nx, dtype=torch.float64) * dx
        u0 = riemann(x, u_left=1.0, u_right=0.0, x_split=0.5)
        result = generate_one(u0, dx=dx, dt=dt, nt=nt)
        assert result["u"].shape == (nt + 1, nx)
        assert result["x"].shape == (nx,)
        assert result["t"].shape == (nt + 1,)

    def test_dtype_float64(self):
        nx, dx, dt, nt = 20, 0.05, 0.002, 10
        x = torch.arange(nx, dtype=torch.float64) * dx
        u0 = riemann(x, u_left=1.0, u_right=0.0, x_split=0.5)
        result = generate_one(u0, dx=dx, dt=dt, nt=nt)
        assert result["u"].dtype == torch.float64

    def test_initial_row_matches_input(self):
        nx, dx, dt, nt = 20, 0.05, 0.002, 10
        x = torch.arange(nx, dtype=torch.float64) * dx
        u0 = riemann(x, u_left=1.0, u_right=0.0, x_split=0.5)
        result = generate_one(u0, dx=dx, dt=dt, nt=nt)
        torch.testing.assert_close(result["u"][0], u0)

    def test_metadata(self):
        nx, dx, dt, nt = 20, 0.05, 0.002, 10
        x = torch.arange(nx, dtype=torch.float64) * dx
        u0 = riemann(x, u_left=1.0, u_right=0.0, x_split=0.5)
        result = generate_one(u0, dx=dx, dt=dt, nt=nt)
        assert result["dx"] == dx
        assert result["dt"] == dt
        assert result["nt"] == nt
        assert result["valid"] is True


class TestGenerateN:
    def test_return_keys(self):
        result = generate_n(
            2, 2, nx=20, dx=0.05, dt=0.002, nt=10,
            seed=42, show_progress=False, reconstruction="constant",
        )
        assert set(result.keys()) == {
            "u", "x", "t", "dx", "dt", "nt", "ic_xs", "ic_u_ks",
        }

    def test_shapes(self):
        n, k, nx, nt = 3, 2, 20, 10
        result = generate_n(
            n, k, nx=nx, dx=0.05, dt=0.002, nt=nt,
            seed=42, show_progress=False, reconstruction="constant",
        )
        assert result["u"].shape == (n, nt + 1, nx)

    def test_ic_params_shapes(self):
        n, k = 4, 3
        result = generate_n(
            n, k, nx=20, dx=0.05, dt=0.002, nt=10,
            seed=42, show_progress=False, reconstruction="constant",
        )
        assert result["ic_xs"].shape == (n, k + 1)
        assert result["ic_u_ks"].shape == (n, k)

    def test_reproducibility(self):
        kwargs = dict(
            nx=20, dx=0.05, dt=0.002, nt=10,
            seed=123, show_progress=False, reconstruction="constant",
        )
        r1 = generate_n(2, 2, **kwargs)
        r2 = generate_n(2, 2, **kwargs)
        torch.testing.assert_close(r1["u"], r2["u"])
