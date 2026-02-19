"""Initial condition generators for the LWR system.

Provides a PiecewiseRandom IC class (extending nfv's PiecewiseConstant)
and helper functions that return (ks, xs) tuples suitable for generate_one.
"""

import numpy as np
from nfv.initial_conditions import PiecewiseConstant


class PiecewiseRandom(PiecewiseConstant):
    """Piecewise constant IC with random breakpoint locations.

    Parameters
    ----------
    ks : list[float]
        Piece values.
    x_noise : bool
        Passed through to PiecewiseConstant.
    rng : np.random.Generator or None
        If provided, used for reproducible random breakpoints.
        If None, falls back to np.random.rand (legacy behavior).
    """

    def __init__(self, ks, x_noise=False, rng=None):
        super().__init__(ks, x_noise)
        if rng is not None:
            xs = rng.random(len(ks) - 1)
        else:
            xs = np.random.rand(len(ks) - 1)
        xs = np.sort(xs)
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

    Returns
    -------
    (ks, xs) tuple.
    """
    rho_lo, rho_hi = rho_range
    if rng is not None:
        ks = (rng.random(k) * (rho_hi - rho_lo) + rho_lo).tolist()
        xs = rng.random(k - 1)
    else:
        ks = (np.random.rand(k) * (rho_hi - rho_lo) + rho_lo).tolist()
        xs = np.random.rand(k - 1)
    xs = np.sort(xs)
    xs = np.concatenate([[0], xs, [1]]).tolist()
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
