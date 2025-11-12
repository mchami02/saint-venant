import numpy as np

g = 9.81

def flux(h, u):
    """Physical flux F(U) for Saint-Venant in (h,u) form."""
    return np.array([h*u, h*u*u + 0.5*g*h*h])

def rusanov_flux(hL, uL, hR, uR):
    """Rusanov (LLF) flux computed from left/right (h,u)."""
    UL = np.array([hL, hL*uL])
    UR = np.array([hR, hR*uR])
    FL = flux(hL, uL)
    FR = flux(hR, uR)
    smax = max(abs(uL) + np.sqrt(g*hL), abs(uR) + np.sqrt(g*hR))
    return 0.5*(FL + FR) - 0.5*smax*(UR - UL)

class Solver():
    def __init__(self, N_x, T, d_x, d_t) -> None:
        self.N_x = N_x
        self.T = T
        self.d_x = d_x
        self.d_t = d_t
        self.time_steps = int(T // d_t)

    def init_arrays(self, left_boundary, right_boundary):
        assert isinstance(left_boundary, np.ndarray) or isinstance(left_boundary, float), "Left boundary must be a numpy array or a float"
        assert isinstance(right_boundary, np.ndarray) or isinstance(right_boundary, float), "Right boundary must be a numpy array or a float"
        if isinstance(left_boundary, np.ndarray):
            assert len(left_boundary) == self.time_steps, "Left boundary array must be the same length as the number of cells + 2"
        if isinstance(right_boundary, np.ndarray):
            assert len(right_boundary) == self.time_steps, "Right boundary array must be the same length as the number of time steps"
        h = np.zeros((self.time_steps, self.N_x + 2))
        u = np.zeros((self.time_steps, self.N_x + 2))
        h[:, 0] = left_boundary
        h[:, -1] = right_boundary
        return h, u

    def solve(self, left_boundary, right_boundary):
        h, u = self.init_arrays(left_boundary, right_boundary)
        for i in range(1, self.time_steps):
                # compute fluxes at interfaces
            Fh = np.zeros(self.N_x+1)
            Fq = np.zeros(self.N_x+1)
            for n in range(self.N_x+1):
                F = rusanov_flux(h[i-1, n], u[i-1, n], h[i-1, n+1], u[i-1, n+1])
                Fh[n], Fq[n] = F[0], F[1]
    
            for n in range(1, self.N_x+1):
                # --- continuity ---
                h[i, n] = h[i-1, n] - (self.d_t / self.d_x) * (Fh[n] - Fh[n-1])
    
                # --- momentum ---
                q_old = h[i-1, n] * u[i-1, n]
                q_new = q_old - (self.d_t / self.d_x) * (Fq[n] - Fq[n-1])
                u[i, n] = q_new / (h[i, n] + 1e-6)
                
        return h, u

