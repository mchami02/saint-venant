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
