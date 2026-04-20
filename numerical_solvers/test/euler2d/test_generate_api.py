"""Tests for Euler2D generate_one / generate_n API contracts."""

import torch

from numerical_solvers.src.euler2d import generate_n, generate_one, sod_x


class TestGenerateOne:
    def test_return_keys(self):
        nx, ny, dx, dy, dt, nt = 16, 16, 0.06, 0.06, 0.001, 5
        x = torch.arange(nx, dtype=torch.float64) * dx
        y = torch.arange(ny, dtype=torch.float64) * dy
        rho0, u0, v0, p0 = sod_x(x, y, x_split=0.5)
        result = generate_one(rho0, u0, v0, p0, dx=dx, dy=dy, dt=dt, nt=nt)
        assert set(result.keys()) == {
            "rho", "u", "v", "p", "x", "y", "t",
            "dx", "dy", "dt", "nt", "valid",
        }

    def test_shapes(self):
        nx, ny, dx, dy, dt, nt = 16, 16, 0.06, 0.06, 0.001, 5
        x = torch.arange(nx, dtype=torch.float64) * dx
        y = torch.arange(ny, dtype=torch.float64) * dy
        rho0, u0, v0, p0 = sod_x(x, y, x_split=0.5)
        result = generate_one(rho0, u0, v0, p0, dx=dx, dy=dy, dt=dt, nt=nt)
        assert result["rho"].shape == (nt + 1, ny, nx)
        assert result["u"].shape == (nt + 1, ny, nx)
        assert result["v"].shape == (nt + 1, ny, nx)
        assert result["p"].shape == (nt + 1, ny, nx)
        assert result["x"].shape == (nx,)
        assert result["y"].shape == (ny,)
        assert result["t"].shape == (nt + 1,)

    def test_initial_rows(self):
        nx, ny, dx, dy, dt, nt = 16, 16, 0.06, 0.06, 0.001, 5
        x = torch.arange(nx, dtype=torch.float64) * dx
        y = torch.arange(ny, dtype=torch.float64) * dy
        rho0, u0, v0, p0 = sod_x(x, y, x_split=0.5)
        result = generate_one(rho0, u0, v0, p0, dx=dx, dy=dy, dt=dt, nt=nt)
        torch.testing.assert_close(result["rho"][0], rho0)
        torch.testing.assert_close(result["u"][0], u0)
        torch.testing.assert_close(result["v"][0], v0)
        torch.testing.assert_close(result["p"][0], p0)


class TestGenerateN:
    def test_return_keys(self):
        result = generate_n(
            2, 2, 2, nx=20, ny=20, dx=0.05, dy=0.05, dt=0.001, nt=5,
            seed=42, show_progress=False,
        )
        assert set(result.keys()) == {
            "rho", "u", "v", "p", "x", "y", "t",
            "dx", "dy", "dt", "nt",
            "ic_xs", "ic_ys",
            "ic_rho_ks", "ic_u_ks", "ic_v_ks", "ic_p_ks",
        }

    def test_shapes(self):
        n, kx, ky, nx, ny, nt = 3, 2, 3, 20, 20, 5
        result = generate_n(
            n, kx, ky, nx=nx, ny=ny, dx=0.05, dy=0.05, dt=0.001, nt=nt,
            seed=42, show_progress=False,
        )
        assert result["rho"].shape == (n, nt + 1, ny, nx)
        assert result["u"].shape == (n, nt + 1, ny, nx)
        assert result["v"].shape == (n, nt + 1, ny, nx)
        assert result["p"].shape == (n, nt + 1, ny, nx)
        assert result["ic_xs"].shape == (n, kx + 1)
        assert result["ic_ys"].shape == (n, ky + 1)
        assert result["ic_rho_ks"].shape == (n, ky, kx)

    def test_reproducibility(self):
        kwargs = dict(
            nx=20, ny=20, dx=0.05, dy=0.05, dt=0.001, nt=5,
            seed=123, show_progress=False,
        )
        r1 = generate_n(2, 2, 2, **kwargs)
        r2 = generate_n(2, 2, 2, **kwargs)
        torch.testing.assert_close(r1["rho"], r2["rho"])
