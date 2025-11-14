from abc import ABC, abstractmethod
import numpy as np

class Flux(ABC):
    @abstractmethod
    def __call__(self, rho: float) -> float:
        pass

    @abstractmethod
    def rho_crit(self) -> float:
        pass

class Greenshields(Flux):
    def __init__(self, vmax: float, rho_max: float):
        self.vmax = vmax
        self.rho_max = rho_max

    def __call__(self, rho: float) -> float:
        rho = np.clip(rho, 0, self.rho_max)
        return self.vmax * rho * (1 - rho / self.rho_max) if rho < self.rho_max else 0

    def rho_crit(self) -> float:
        return self.rho_max / 2.0

class Triangular(Flux):
    def __init__(self, vf: float, w: float):
        self.vf = vf
        self.w = w

    def __call__(self, rho: float) -> float:
        rho = np.clip(rho, 0, 1)
        return np.minimum(self.vf * rho, self.w * (1 - rho))

    def rho_crit(self) -> float:
        return self.w / (self.vf + self.w)