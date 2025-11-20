from .solver import Solver
from ..flux import Flux
import numpy as np

class RiemannSolver:
    def flux(self, UL, UR):
        raise NotImplementedError("RiemannSolver is an abstract class")

class LWRRiemannSolver(RiemannSolver):
    def __init__(self, flux_fn):
        self.flux_fn = flux_fn

    def flux(self, rhoL, rhoR):
        fL = self.flux_fn(rhoL)
        fR = self.flux_fn(rhoR)
        rho_crit = self.flux_fn.rho_crit()

        # shock
        if rhoL <= rhoR:
            return min(fL, fR)
        # rarefaction
        if rhoR > rho_crit:
            return fR
        elif rhoL < rho_crit:
            return fL
        else:
            return self.flux_fn(rho_crit)

class SVERiemannSolver(RiemannSolver):
    def __init__(self, g=9.81):
        self.g = g

    def flux(self, UL, UR):
        """
        HLL Riemann solver for shallow water equations.
        U = [h, u] where h is height and u is velocity.
        F(U) = [h*u, h*u^2 + g*h^2/2]
        """
        # Extract left and right states
        hL, uL = UL[0], UL[1]
        hR, uR = UR[0], UR[1]
        
        # Handle dry states (zero height)
        eps = 1e-10
        if hL < eps:
            hL = 0.0
            uL = 0.0
        if hR < eps:
            hR = 0.0
            uR = 0.0
        
        # Compute discharges
        qL = hL * uL
        qR = hR * uR
        
        # Compute physical fluxes: F = [hu, hu^2 + gh^2/2]
        FL = np.array([qL, hL*uL*uL + 0.5*self.g*hL*hL])
        FR = np.array([qR, hR*uR*uR + 0.5*self.g*hR*hR])
        
        # Compute wave speeds using Einfeldt estimates
        cL = np.sqrt(self.g * hL) if hL > eps else 0.0
        cR = np.sqrt(self.g * hR) if hR > eps else 0.0
        
        SL = min(uL - cL, uR - cR)
        SR = max(uL + cL, uR + cR)
        
        # HLL flux
        if SL >= 0:
            return FL
        elif SR <= 0:
            return FR
        else:
            # Middle state
            return (SR * FL - SL * FR + SL * SR * (UR - UL)) / (SR - SL)
        


class GodunovSolver(Solver):
    def __init__(self, riemann_solver):
        self.riemann = riemann_solver

    def solve(self, grid):
        for i in range(1, grid.nt):
            F = [None] * (grid.nx + 1)

            # compute flux at cell interfaces
            for n in range(grid.nx + 1):
                UL = grid.get_state(i-1, n-1)   # vector or scalar
                UR = grid.get_state(i-1, n)
                F[n] = self.riemann.flux(UL, UR)

            # update the cell averages
            for n in range(grid.nx):
                U_old = grid.get_state(i-1, n)
                U_new = U_old - (grid.dt / grid.dx) * (F[n+1] - F[n])
                grid.set_state(i, n, U_new)
