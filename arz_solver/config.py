"""Configuration dataclass for ARZ solver."""

from dataclasses import dataclass
from typing import Optional, Tuple, Callable
import numpy as np


@dataclass
class ARZConfig:
    """Grid, time, and model parameters for the ARZ solver.
    L and T are optional; defaults give domain [0, 1] and run to t=1.
    """

    nx: int = 200
    L: float = 1.0  # domain length (x in [0, L])
    T: float = 1.0  # final time
    gamma: float = 1.0
    cfl_factor: float = 0.1
    # Boundary state for Dirichlet / inflow (rho, v) at left and right
    bc_left: Optional[Tuple[float, float]] = (0.5, 1.0)
    bc_right: Optional[Tuple[float, float]] = (0.3, 0.5)
    # Time-varying left BC: (t) -> (rho, v); used when bc_type is time_varying_inflow
    bc_left_time: Optional[Callable[[float], Tuple[float, float]]] = None

    def __post_init__(self):
        self.x = np.linspace(0, self.L, self.nx)
        self.dx = self.x[1] - self.x[0]
        self.dt = self.cfl_factor * self.dx
        self.nt = int(self.T / self.dt)
