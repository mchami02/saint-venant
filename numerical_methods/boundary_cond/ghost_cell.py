from .boundary_cond import BoundaryCondition
import numpy as np

class GhostCellBoundary(BoundaryCondition):
    def __init__(self):
        super().__init__()

    def get_left(self, i: int, grid: np.ndarray) -> float:
        return grid[i, 0]

    def get_right(self, i: int, grid: np.ndarray) -> float:
        return grid[i, -1]