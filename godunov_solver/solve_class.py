import numpy as np
from .flux import Flux

class Solver():
    def __init__(self):
        pass

    def update(self, grid, i: int, n: int):
        raise NotImplementedError("Subclasses must implement this method")

    def solve(self, grid):
        for i in range(1, grid.nt):
            for n in range(grid.nx):
                self.update(grid, i, n)

class Godunov(Solver):
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
    
    def update(self, grid, i: int, n: int):
        for val in grid.values():
            new_val = grid.get(i-1, n, val) - (grid.dt / grid.dx) * (self.flux(grid, i-1, n+1, val) - self.flux(grid, i-1, n, val))
            grid.set(i, n, val, new_val)

class SVESolver(Solver):
    def __init__(self, g: float = 9.81):
        super().__init__()
        self.g = g

    def rusanov_flux(self, grid, i: int, n: int, val: str):
        uL = grid.get(i, n-1, "u")
        uR = grid.get(i, n, "u")
        hL = grid.get(i, n-1, "h")
        hR = grid.get(i, n, "h")
        
        UL = np.array([hL, hL*uL])
        UR = np.array([hR, hR*uR])
        FL = np.array([hL*uL, hL*uL*uL + 0.5*self.g*hL*hL])
        FR = np.array([hR*uR, hR*uR*uR + 0.5*self.g*hR*hR])
        
        smax = max(abs(uL) + np.sqrt(self.g*hL), abs(uR) + np.sqrt(self.g*hR))
        return 0.5*(FL + FR) - 0.5*smax*(UR - UL)
    
    def solve(self, grid):
        for i in range(1, grid.nt):
            # compute fluxes at interfaces
            Fh = np.zeros(grid.nx+1)
            Fq = np.zeros(grid.nx+1)
            for n in range(grid.nx+1):
                F = self.rusanov_flux(grid, i-1, n, "h")
                Fh[n] = F[0]
                Fq[n] = F[1]
    
            for n in range(grid.nx):
                # --- continuity --- (FIXED: correct flux difference)
                h_new = grid.get(i-1, n, "h") - (grid.dt / grid.dx) * (Fh[n+1] - Fh[n])
                # Positivity preservation
                # h_new = max(h_new, 1e-10)
                grid.set(i, n, "h", h_new)
                
                # --- momentum ---
                q_old = grid.get(i-1, n, "h") * grid.get(i-1, n, "u")
                q_new = q_old - (grid.dt / grid.dx) * (Fq[n+1] - Fq[n])
                u_new = q_new / (h_new + 1e-10)
                
                grid.set(i, n, "u", u_new)
