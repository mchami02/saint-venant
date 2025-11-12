from .boundary_cond import get_bc
from .initial_cond import get_ic


class Grid:
    def __init__(self, nx: int, nt: int, dx: float, dt: float, initial_condition: str, boundary_condition: str, **kwargs):
        self.nx = nx
        self.nt = nt
        self.dx = dx
        self.dt = dt
        self.grid = get_ic(initial_condition, nx, nt, **kwargs).get_grid()
        self.boundary_condition = get_bc(boundary_condition, **kwargs)

    def get(self, i: int, n: int, val: str = "h") -> float:
        if n < 0:
            return self.boundary_condition.get_left(i=i, grid=self.grid[val])
        elif n >= self.nx:
            return self.boundary_condition.get_right(i=i, grid=self.grid[val])
        else:
            return self.grid[val][i, n]

    def set(self, i: int, n: int, val: str, value: float):
        self.grid[val][i, n] = value

