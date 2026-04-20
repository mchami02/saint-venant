"""Tests for 2D Euler initial-condition generators."""

import torch

from numerical_solvers.src.euler2d.initial_conditions import (
    four_quadrant,
    liska_wendroff,
    random_piecewise,
    random_piecewise_batch,
    sod_x,
    sod_y,
)


def _grid(nx=20, ny=30):
    dx = 1.0 / nx; dy = 1.0 / ny
    x = torch.arange(nx, dtype=torch.float64) * dx
    y = torch.arange(ny, dtype=torch.float64) * dy
    return x, y


def test_sod_x_y_invariant():
    x, y = _grid()
    rho, u, v, p = sod_x(x, y, x_split=0.5)
    # All y-rows must be identical
    for j in range(rho.shape[0]):
        torch.testing.assert_close(rho[j], rho[0])
        torch.testing.assert_close(p[j], p[0])
    # Velocities zero
    assert (u == 0).all() and (v == 0).all()


def test_sod_y_x_invariant():
    x, y = _grid()
    rho, u, v, p = sod_y(x, y, y_split=0.5)
    # All x-columns must be identical
    for i in range(rho.shape[1]):
        torch.testing.assert_close(rho[:, i], rho[:, 0])


def test_four_quadrant_values():
    x, y = _grid()
    states = (
        {"rho": 1.0, "u": 0.1, "v": 0.2, "p": 1.1},
        {"rho": 2.0, "u": 0.3, "v": 0.4, "p": 1.2},
        {"rho": 3.0, "u": 0.5, "v": 0.6, "p": 1.3},
        {"rho": 4.0, "u": 0.7, "v": 0.8, "p": 1.4},
    )
    rho, _, _, _ = four_quadrant(x, y, states, x_split=0.5, y_split=0.5)

    ny, nx = rho.shape
    x_mid_idx = int(0.5 / (1.0 / nx))
    y_mid_idx = int(0.5 / (1.0 / ny))
    assert rho[0, 0].item() == 1.0  # BL
    assert rho[0, -1].item() == 2.0  # BR
    assert rho[-1, 0].item() == 3.0  # TL
    assert rho[-1, -1].item() == 4.0  # TR
    # Sanity on split cells (just not NaN)
    assert torch.isfinite(rho[y_mid_idx, x_mid_idx])


def test_liska_wendroff_positive_rho_p():
    x, y = _grid()
    for cfg in (3, 4, 6):
        rho, u, v, p = liska_wendroff(x, y, config=cfg)
        assert (rho > 0).all()
        assert (p > 0).all()
        assert torch.isfinite(rho).all()
        assert torch.isfinite(u).all()
        assert torch.isfinite(v).all()
        assert torch.isfinite(p).all()


def test_random_piecewise_shapes():
    rng = torch.Generator(); rng.manual_seed(0)
    x, y = _grid()
    rho, u, v, p, ic = random_piecewise(
        x, y, kx=3, ky=2, rng=rng,
        rho_range=(0.2, 1.5), u_range=(-1.0, 1.0),
        v_range=(-1.0, 1.0), p_range=(0.3, 2.0),
    )
    assert rho.shape == (y.shape[0], x.shape[0])
    assert len(ic["xs"]) == 4
    assert len(ic["ys"]) == 3
    assert len(ic["rho_ks"]) == 2 and len(ic["rho_ks"][0]) == 3


def test_random_piecewise_batch_shapes():
    rng = torch.Generator(); rng.manual_seed(0)
    x, y = _grid()
    rhos, us, vs, ps, params = random_piecewise_batch(x, y, 2, 2, 4, rng)
    assert rhos.shape == (4, y.shape[0], x.shape[0])
    assert len(params) == 4
