from .flux import Flux
import numpy as np

class Triangular(Flux):
    def __init__(self, vf: float, w: float):
        self.vf = vf
        self.w = w

    def __call__(self, rho: float) -> float:
        rho = np.clip(rho, 0, 1)
        return np.minimum(self.vf * rho, self.w * (1 - rho))

    def rho_crit(self) -> float:
        return self.w / (self.vf + self.w)