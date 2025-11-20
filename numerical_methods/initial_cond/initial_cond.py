from abc import ABC, abstractmethod
import numpy as np

class InitialCondition(ABC):
    def __init__(self, nx: int, nt: int, vals: list[str]):
        self.grid = {
            v : np.zeros((nt, nx)) for v in vals
        }
        self.nx = nx
        self.nt = nt

    def get_grid(self) -> np.ndarray:
        return self.grid