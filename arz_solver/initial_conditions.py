"""Initial conditions for the ARZ solver."""

from typing import List, Tuple, Optional
import numpy as np
from .solver import pressure


def _piecewise_constant(x: np.ndarray, steps: List[Tuple[float, float]]) -> np.ndarray:
    """steps = [(x_end_1, value_1), (x_end_2, value_2), ...]; value applies on [prev_x, x_end)."""
    steps = sorted(steps, key=lambda p: p[0])
    out = np.full_like(x, steps[-1][1], dtype=float)
    for i in range(len(steps) - 2, -1, -1):
        x_end, val = steps[i]
        out[x < x_end] = val
    return out


def initial_condition_from_steps(
    x: np.ndarray,
    rho_steps: List[Tuple[float, float]],
    v_steps: Optional[List[Tuple[float, float]]] = None,
    default_v: float = 0.1,
    gamma: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (rho0, w0) from piecewise-constant steps.

    Parameters
    ----------
    x : 1D array
        Grid points.
    rho_steps : list of (x_end, rho_value)
        Density: value applies from previous x_end up to this x_end.
        Example: [(0.2, 0.3), (0.5, 0.8), (1.0, 0.2)] gives
        rho=0.3 on [0, 0.2), 0.8 on [0.2, 0.5), 0.2 on [0.5, 1.0].
    v_steps : list of (x_end, v_value), optional
        Same format for velocity. If None, velocity is constant default_v.
    default_v : float
        Used when v_steps is None.
    gamma : float
        Pressure exponent for w = v + p(rho).

    Returns
    -------
    rho0, w0 : 1D arrays
    """
    rho0 = _piecewise_constant(x, rho_steps)
    if v_steps is None:
        v0 = np.full_like(x, default_v)
    else:
        v0 = _piecewise_constant(x, v_steps)
    w0 = v0 + pressure(rho0, gamma)
    return rho0, w0


def initial_condition_riemann(
    x: np.ndarray,
    rho_left: float = 0.8,
    rho_right: float = 0.2,
    v0: float = 0.1,
    x_split: float = 0.5,
    gamma: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Piecewise-constant (Riemann-type): (rho_left, v0) for x < x_split, (rho_right, v0) for x >= x_split.
    """
    return initial_condition_from_steps(
        x,
        rho_steps=[(x_split, rho_left), (x.max() + 1.0, rho_right)],
        default_v=v0,
        gamma=gamma,
    )


def initial_condition_three_region(
    x: np.ndarray,
    rho_left: float = 0.3,
    rho_mid: float = 0.8,
    rho_right: float = 0.2,
    v0: float = 0.1,
    x1: float = 0.2,
    x2: float = 0.5,
    gamma: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Three-region piecewise-constant IC: rho_left on [0, x1), rho_mid on [x1, x2), rho_right on [x2, L].
    """
    return initial_condition_from_steps(
        x,
        rho_steps=[(x1, rho_left), (x2, rho_mid), (x.max() + 1.0, rho_right)],
        default_v=v0,
        gamma=gamma,
    )

