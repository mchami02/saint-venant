"""Initial condition generators for the LWR system.

Provides a PiecewiseRandom IC class (extending nfv's PiecewiseConstant)
and helper functions that return (ks, xs) tuples suitable for generate_one.
"""

import numpy as np
from nfv.initial_conditions import PiecewiseConstant


class PiecewiseRandom(PiecewiseConstant):
    """Piecewise constant IC with random breakpoint locations.

    At most one breakpoint is placed per computational cell, so every
    discontinuity is resolved on the grid.

    Parameters
    ----------
    ks : list[float]
        Piece values.
    x_noise : bool
        Passed through to PiecewiseConstant.
    rng : np.random.Generator or None
        If provided, used for reproducible random breakpoints.
        If None, falls back to np.random global state (legacy behavior).
    nx : int
        Number of spatial cells.  Required so that breakpoints can be
        distributed with at most one per cell.
    """

    def __init__(self, ks, x_noise=False, rng=None, nx=None):
        super().__init__(ks, x_noise)
        n_breaks = len(ks) - 1
        if n_breaks == 0:
            self.xs = np.array([0.0, 1.0])
            return
        if nx is None:
            raise ValueError("nx is required for PiecewiseRandom")
        if n_breaks > nx:
            raise ValueError(
                f"Cannot place {n_breaks} breakpoints in {nx} cells "
                "(need n_breaks <= nx)"
            )
        # Pick n_breaks distinct cells, then place one breakpoint uniformly
        # within each chosen cell so no two breakpoints share a cell.
        if rng is not None:
            cells = rng.choice(nx, size=n_breaks, replace=False)
            offsets = rng.random(n_breaks)
        else:
            cells = np.random.choice(nx, size=n_breaks, replace=False)
            offsets = np.random.rand(n_breaks)
        cells = np.sort(cells)
        xs = (cells + offsets) / nx
        self.xs = np.concatenate([[0], xs, [1]])


def riemann(
    rho_left: float = 0.8,
    rho_right: float = 0.2,
    x_split: float = 0.5,
) -> tuple[list[float], list[float]]:
    """Two-region Riemann problem IC.

    Returns (ks, xs) tuple.
    """
    return [rho_left, rho_right], [0, x_split, 1]


def random_piecewise(
    k: int,
    rng: np.random.Generator | None = None,
    rho_range: tuple[float, float] = (0.0, 1.0),
    nx: int | None = None,
) -> tuple[list[float], list[float]]:
    """Generate a random k-piecewise-constant IC.

    Parameters
    ----------
    k : int
        Number of constant pieces.
    rng : np.random.Generator or None
        If provided, used for reproducibility. Falls back to np.random.rand.
    rho_range : (min, max)
        Range for sampled density values.
    nx : int or None
        Number of spatial cells.  When provided, breakpoints are placed with
        at most one per cell.  Required when k > 1.

    Returns
    -------
    (ks, xs) tuple.
    """
    rho_lo, rho_hi = rho_range
    if rng is not None:
        ks = (rng.random(k) * (rho_hi - rho_lo) + rho_lo).tolist()
    else:
        ks = (np.random.rand(k) * (rho_hi - rho_lo) + rho_lo).tolist()

    n_breaks = k - 1
    if n_breaks == 0:
        return ks, [0.0, 1.0]
    if nx is None:
        raise ValueError("nx is required for random_piecewise when k > 1")
    if n_breaks > nx:
        raise ValueError(
            f"Cannot place {n_breaks} breakpoints in {nx} cells "
            "(need n_breaks <= nx)"
        )
    if rng is not None:
        cells = rng.choice(nx, size=n_breaks, replace=False)
        offsets = rng.random(n_breaks)
    else:
        cells = np.random.choice(nx, size=n_breaks, replace=False)
        offsets = np.random.rand(n_breaks)
    cells = np.sort(cells)
    xs = ((cells + offsets) / nx).tolist()
    xs = [0.0] + xs + [1.0]
    return ks, xs


def from_steps(
    ks: list[float],
    xs: list[float] | None = None,
) -> tuple[list[float], list[float] | None]:
    """Pass-through helper: return (ks, xs) with uniform breakpoints if xs is None.

    Parameters
    ----------
    ks : list[float]
        Piece values.
    xs : list[float] or None
        Breakpoint positions. If None, generate_one will use nfv's default
        (uniform breakpoints).

    Returns
    -------
    (ks, xs) tuple.
    """
    return ks, xs
