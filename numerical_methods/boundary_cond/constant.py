from .boundary_cond import BoundaryCondition
import numpy as np

class ConstantBoundary(BoundaryCondition):
    def __init__(self, left_value: float, right_value: float):
        self.left_value = left_value
        self.right_value = right_value

    def get_left(self, i: int, grid: np.ndarray) -> float:
        return self.left_value

    def get_right(self, i: int, grid: np.ndarray) -> float:
        return self.right_value