from .boundary_cond import get_bc
from .initial_cond import get_ic
import numpy as np

class Grid:
    def __init__(self, nx: int, nt: int, dx: float, dt: float, initial_condition: str, boundary_condition: str, **kwargs):
        self.nx = nx
        self.nt = nt
        self.dx = dx
        self.dt = dt
        
        # Separate kwargs for IC and BC
        bc_params = {'left_value', 'right_value'}
        ic_kwargs = {k: v for k, v in kwargs.items() if k not in bc_params}
        bc_kwargs = {k: v for k, v in kwargs.items() if k in bc_params}
        
        self.grid = get_ic(initial_condition, nx, nt, **ic_kwargs).get_grid()
        self.boundary_condition = get_bc(boundary_condition, **bc_kwargs)
        self.kwargs = kwargs

    def get(self, i: int, n: int, val: str = "h") -> float:
        if n < 0:
            return self.boundary_condition.get_left(i=i, grid=self.grid[val])
        elif n >= self.nx:
            return self.boundary_condition.get_right(i=i, grid=self.grid[val])
        else:
            return self.grid[val][i, n]

    def get_array(self, val: str) -> np.ndarray:
        return self.grid[val]

    def set(self, i: int, n: int, val: str, value: float):
        self.grid[val][i, n] = value

    def values(self) -> list[str]:
        return list(self.grid.keys())

    def get_state(self, i: int, n: int) -> np.ndarray:
        return np.array([self.get(i, n, val) for val in self.values()])

    def set_state(self, i: int, n: int, state: np.ndarray):
        for val, value in zip(self.values(), state):
            self.set(i, n, val, value)
