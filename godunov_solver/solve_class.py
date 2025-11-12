from abc import ABC, abstractmethod
from .grid import Grid
import numpy as np

class Solver(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def flux(self, grid, i: int, n: int, val: str):
        pass

    def solve(self, grid):
        for i in range(1, grid.nt):
            for n in range(grid.nx):
                new_h = grid.get(i-1, n, "h") - (grid.dt / grid.dx) * (self.flux(grid, i-1, n+1, "h") - self.flux(grid, i-1, n, "h"))
                grid.set(i, n, "h", new_h)

class Flux(ABC):
    @abstractmethod
    def __call__(self, rho: float) -> float:
        pass

    @abstractmethod
    def rho_crit(self) -> float:
        pass

class GreenshieldsFlux(Flux):
    def __init__(self, vmax: float, rho_max: float):
        self.vmax = vmax
        self.rho_max = rho_max

    def __call__(self, rho: float) -> float:
        rho = np.clip(rho, 0, self.rho_max)
        return self.vmax * rho * (1 - rho / self.rho_max) if rho < self.rho_max else 0

    def rho_crit(self) -> float:
        return 0.5

class TriangularFlux(Flux):
    def __init__(self, vf: float, w: float):
        self.vf = vf
        self.w = w

    def __call__(self, rho: float) -> float:
        rho = np.clip(rho, 0, 1)
        return np.minimum(self.vf * rho, self.w * (1 - rho))

    def rho_crit(self) -> float:
        return self.w / (self.vf + self.w)

class GodunovLWR(Solver):
    def __init__(self, flux: Flux):
        super().__init__()
        self.flux_fn = flux

    def flux(self, grid, i: int, n: int, val: str):
        rhoL = grid.get(i, n-1, val)
        rhoR = grid.get(i, n, val)
        fL = self.flux_fn(rhoL)
        fR = self.flux_fn(rhoR)

        rho_crit = self.flux_fn.rho_crit()

        # Shock case
        if rhoL <= rhoR:
            return min(fL, fR)
        # Rarefaction case
        else:
            if rhoR > rho_crit:
                return fR
            elif rhoL < rho_crit:
                return fL
            else:  # Critical density is between rhoL and rhoR
                return self.flux_fn(rho_crit)