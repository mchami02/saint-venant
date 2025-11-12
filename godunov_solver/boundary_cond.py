from abc import ABC, abstractmethod
import numpy as np


class BoundaryCondition(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_left(self, i: int, grid: np.ndarray) -> float:
        pass

    @abstractmethod
    def get_right(self, i: int, grid: np.ndarray) -> float:
        pass

class ConstantBoundary(BoundaryCondition):
    def __init__(self, left_value: float, right_value: float):
        self.left_value = left_value
        self.right_value = right_value

    def get_left(self, i: int, grid: np.ndarray) -> float:
        return self.left_value

    def get_right(self, i: int, grid: np.ndarray) -> float:
        return self.right_value

class GhostCellBoundary(BoundaryCondition):
    def __init__(self):
        super().__init__()

    def get_left(self, i: int, grid: np.ndarray) -> float:
        return grid[i, 0]

    def get_right(self, i: int, grid: np.ndarray) -> float:
        return grid[i, -1]


def get_bc(bc_type: str, **kwargs) -> BoundaryCondition:
    if bc_type == "Constant":
        return ConstantBoundary(kwargs['left_value'], kwargs['right_value'])
    elif bc_type == "GhostCell":
        return GhostCellBoundary()
    else:
        raise ValueError(f"Invalid boundary condition type: {bc_type}")