from abc import ABC, abstractmethod
import numpy as np

class Flux(ABC):
    @abstractmethod
    def __call__(self, rho: float) -> float:
        pass

    @abstractmethod
    def rho_crit(self) -> float:
        pass