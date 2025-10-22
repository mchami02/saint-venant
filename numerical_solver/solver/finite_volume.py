"""
Finite Volume numerical flux schemes for shallow water equations.

Implements LaxFriedrichs, Rusanov, and HLL numerical fluxes with
first and second order MUSCL reconstruction.
"""

import numpy as np
from abc import ABC, abstractmethod
from solver.data_file import DataFile
from solver.mesh import Mesh
from solver.physics import Physics
from solver.numba_kernels import lax_friedrichs_flux_kernel, rusanov_flux_kernel, hll_flux_kernel


class FiniteVolume(ABC):
    """Base class for finite volume numerical flux schemes.
    
    Attributes:
        flux_name: Name of the numerical flux scheme
        flux_vector: Vector storing fluxes at each cell
    """
    
    def __init__(self, data_file: DataFile = None, mesh: Mesh = None, physics: Physics = None):
        """Initialize FiniteVolume scheme.
        
        Args:
            data_file: DataFile with parameters
            mesh: Mesh object
            physics: Physics object
        """
        self._DF = data_file
        self._mesh = mesh
        self._physics = physics
        self.flux_name = "Base"
        
        if mesh is not None:
            self.flux_vector = np.zeros((mesh.get_number_of_cells(), 2))
        else:
            self.flux_vector = None
    
    def initialize(self, data_file: DataFile, mesh: Mesh, physics: Physics):
        """Initialize with new parameters.
        
        Args:
            data_file: DataFile with parameters
            mesh: Mesh object
            physics: Physics object
        """
        self._DF = data_file
        self._mesh = mesh
        self._physics = physics
        self.flux_vector = np.zeros((mesh.get_number_of_cells(), 2))
    
    @abstractmethod
    def num_flux(self, sol_g: np.ndarray, sol_d: np.ndarray) -> np.ndarray:
        """Compute numerical flux at an interface.
        
        Calculates the numerical flux F̂(U_L, U_R) at a cell interface given the 
        left and right states. This is the key component of a finite volume scheme.
        
        Args:
            sol_g: Left (gauche) state vector of shape (2,) containing [h, q] where:
                  - h: water height (m)
                  - q: discharge = h*u (m²/s)
            sol_d: Right (droite) state vector of shape (2,) containing [h, q]
            
        Returns:
            np.ndarray: Numerical flux vector of shape (2,) representing [F_h, F_q]
                       where F_h is mass flux and F_q is momentum flux.
        """
        pass
    
    def build_flux_vector(self, t: float, sol: np.ndarray):
        """Build flux vector using reconstruction and numerical flux.
        
        Constructs the discrete flux contribution for the finite volume method:
            dU_i/dt = -(F̂_{i+1/2} - F̂_{i-1/2})/dx + S_i
        
        This function computes the term (F̂_{i+1/2} - F̂_{i-1/2}) for each cell i.
        
        For first-order schemes (scheme_order=1):
            - Uses piecewise constant reconstruction (cell-centered values)
            - U_{i-1/2}^L = U_{i-1}, U_{i-1/2}^R = U_i
        
        For second-order schemes (scheme_order=2):
            - Uses MUSCL reconstruction with minmod slope limiter
            - U_{i+1/2}^L = U_i + 0.5*dx*σ_i
            - U_{i+1/2}^R = U_{i+1} - 0.5*dx*σ_{i+1}
            - where σ_i = minmod((U_i - U_{i-1})/dx, (U_{i+1} - U_i)/dx)
        
        Args:
            t: Current simulation time (s), used for time-dependent boundary conditions
            sol: Current solution array of shape (n_cells, 2) containing [h, q] at each cell
            
        Returns:
            None. Updates self.flux_vector in place with shape (n_cells, 2).
        """
        # Reset flux
        self.flux_vector.fill(0.0)
        
        # Get mesh parameters
        n_cells = self._mesh.get_number_of_cells()
        dx = self._mesh.get_space_step()
        
        # Reconstructed values at interfaces
        sol_d = np.zeros((n_cells + 1, 2))
        sol_g = np.zeros((n_cells + 1, 2))
        
        # Select order of scheme
        if self._DF.scheme_order == 1:
            # First order: reconstructed values are cell-centered
            # Left boundary
            sol_g[0, :] = self._physics.left_boundary_function(t + self._DF.time_step, sol)
            sol_d[0, :] = sol[0, :]
            # Right boundary
            sol_g[n_cells, :] = sol[n_cells - 1, :]
            sol_d[n_cells, :] = self._physics.right_boundary_function(t + self._DF.time_step, sol)
            # Interior edges
            for i in range(1, n_cells):
                sol_g[i, :] = sol[i - 1, :]
                sol_d[i, :] = sol[i, :]
        
        elif self._DF.scheme_order == 2:
            # Second order MUSCL with minmod limiter
            slopes = np.zeros((n_cells + 1, 2))
            lim_slopes = np.zeros((n_cells, 2))
            
            # Compute slopes
            # Left boundary
            left_boundary_sol = self._physics.left_boundary_function(t + self._DF.time_step, sol)
            slopes[0, 0] = (sol[0, 0] - left_boundary_sol[0]) / dx
            slopes[0, 1] = (sol[0, 1] - left_boundary_sol[1]) / dx
            # Right boundary
            right_boundary_sol = self._physics.right_boundary_function(t + self._DF.time_step, sol)
            slopes[n_cells, 0] = (right_boundary_sol[0] - sol[n_cells - 1, 0]) / dx
            slopes[n_cells, 1] = (right_boundary_sol[1] - sol[n_cells - 1, 1]) / dx
            # Interior edges
            for i in range(1, n_cells):
                slopes[i, :] = (sol[i, :] - sol[i - 1, :]) / dx
            
            # Limit slopes using minmod limiter
            for i in range(n_cells - 1):
                lim_slopes[i, 0] = self.minmod(slopes[i, 0], slopes[i + 1, 0])
                lim_slopes[i, 1] = self.minmod(slopes[i, 1], slopes[i + 1, 1])
            
            # Reconstruct values at edges
            # Left boundary
            sol_g[0, :] = left_boundary_sol
            sol_d[0, :] = sol[0, :] - 0.5 * dx * lim_slopes[0, :]
            # Right boundary
            sol_g[n_cells, :] = sol[n_cells - 1, :] + 0.5 * dx * lim_slopes[n_cells - 1, :]
            sol_d[n_cells, :] = right_boundary_sol
            # Interior edges
            for i in range(1, n_cells):
                sol_g[i, :] = sol[i - 1, :] + 0.5 * dx * lim_slopes[i - 1, :]
                sol_d[i, :] = sol[i, :] - 0.5 * dx * lim_slopes[i, :]
        
        # Build flux vector using reconstructed values
        # Left boundary contribution
        self.flux_vector[0, :] += self.num_flux(sol_g[0, :], sol_d[0, :])
        # Interior fluxes
        for i in range(1, n_cells):
            flux = self.num_flux(sol_g[i, :], sol_d[i, :])
            self.flux_vector[i - 1, :] -= flux
            self.flux_vector[i, :] += flux
        # Right boundary contribution
        self.flux_vector[n_cells - 1, :] -= self.num_flux(sol_g[n_cells, :], sol_d[n_cells, :])
    
    def minmod(self, a: float, b: float) -> float:
        """Minmod slope limiter for TVD (Total Variation Diminishing) schemes.
        
        The minmod limiter is defined as:
            minmod(a, b) = { 0           if a*b <= 0  (different signs)
                           { a           if |a| < |b|
                           { b           otherwise
        
        This limiter prevents spurious oscillations near discontinuities while
        maintaining second-order accuracy in smooth regions.
        
        Args:
            a: First slope estimate (typically upwind slope)
            b: Second slope estimate (typically downwind slope)
            
        Returns:
            float: Limited slope value. Returns 0 at extrema, preserving monotonicity.
        """
        if a * b < 0:
            return 0.0
        elif abs(a) < abs(b):
            return a
        else:
            return b
    
    def get_flux_name(self) -> str:
        """Get flux scheme name."""
        return self.flux_name
    
    def get_flux_vector(self) -> np.ndarray:
        """Get flux vector."""
        return self.flux_vector


class LaxFriedrichs(FiniteVolume):
    """Lax-Friedrichs numerical flux scheme.
    
    A simple and robust monotone numerical flux based on central differencing
    with artificial viscosity. The Lax-Friedrichs flux is defined as:
    
        F̂(U_L, U_R) = 1/2 * [F(U_L) + F(U_R) - α(U_R - U_L)]
    
    where α = dx/dt is the artificial viscosity coefficient and F(U) is the
    physical flux. This scheme is highly dissipative but very stable.
    """
    
    def __init__(self, data_file: DataFile = None, mesh: Mesh = None, physics: Physics = None):
        """Initialize Lax-Friedrichs flux.
        
        Args:
            data_file: DataFile with simulation parameters (dt, dx, g)
            mesh: Mesh object for spatial discretization
            physics: Physics object for physical flux computation
        """
        super().__init__(data_file, mesh, physics)
        self.flux_name = "LF"
    
    def num_flux(self, sol_g: np.ndarray, sol_d: np.ndarray) -> np.ndarray:
        """Compute Lax-Friedrichs numerical flux (Numba-accelerated).
        
        Implements: F̂ = 1/2 * [F(U_L) + F(U_R) - (dx/dt)(U_R - U_L)]
        
        Mathematical formulation:
            For shallow water equations with U = [h, q]:
            F̂_h = 1/2 * [q_L + q_R - (dx/dt)(h_R - h_L)]
            F̂_q = 1/2 * [F_q(U_L) + F_q(U_R) - (dx/dt)(q_R - q_L)]
            where F_q = q²/h + 0.5*g*h²
        
        Args:
            sol_g: Left state [h, q] where h > 0 (m) and q (m²/s)
            sol_d: Right state [h, q]
            
        Returns:
            np.ndarray: Numerical flux [F̂_h, F̂_q] of shape (2,)
        """
        dt = self._DF.time_step
        dx = self._DF.dx
        g = self._DF.g
        
        return lax_friedrichs_flux_kernel(sol_g, sol_d, dt, dx, g)


class Rusanov(FiniteVolume):
    """Rusanov (local Lax-Friedrichs) numerical flux scheme.
    
    An improved version of Lax-Friedrichs that uses local wave speed estimates
    instead of a global artificial viscosity. The Rusanov flux is:
    
        F̂(U_L, U_R) = 1/2 * [F(U_L) + F(U_R) - α(U_R - U_L)]
    
    where α = max(|λ_L|, |λ_R|) with λ = u ± √(gh) being the eigenvalues of
    the flux Jacobian. This provides less dissipation than Lax-Friedrichs while
    maintaining robustness.
    """
    
    def __init__(self, data_file: DataFile = None, mesh: Mesh = None, physics: Physics = None):
        """Initialize Rusanov flux.
        
        Args:
            data_file: DataFile with simulation parameters (gravity constant g)
            mesh: Mesh object for spatial discretization
            physics: Physics object for flux and wave speed computation
        """
        super().__init__(data_file, mesh, physics)
        self.flux_name = "Rusanov"
    
    def num_flux(self, sol_g: np.ndarray, sol_d: np.ndarray) -> np.ndarray:
        """Compute Rusanov numerical flux (Numba-accelerated).
        
        Implements: F̂ = 1/2 * [F(U_L) + F(U_R) - α(U_R - U_L)]
        where α = max(|u_L - √(gh_L)|, |u_L + √(gh_L)|, 
                      |u_R - √(gh_R)|, |u_R + √(gh_R)|)
        
        The wave speeds λ = u ± √(gh) are the eigenvalues of the shallow water
        flux Jacobian, corresponding to the left-going and right-going gravity waves.
        
        Special handling for dry states (h ≈ 0):
            - If h_L ≈ 0: uses only right state contribution
            - If h_R ≈ 0: uses only left state contribution
            - If both ≈ 0: returns zero flux
        
        Args:
            sol_g: Left state [h, q] with h in meters, q in m²/s
            sol_d: Right state [h, q]
            
        Returns:
            np.ndarray: Numerical flux vector [F̂_h, F̂_q] of shape (2,)
        """
        g = self._DF.g
        return rusanov_flux_kernel(sol_g, sol_d, g)


class HLL(FiniteVolume):
    """HLL (Harten-Lax-van Leer) numerical flux scheme.
    
    A three-wave approximate Riemann solver that assumes a simplified wave structure
    with two waves separating three constant states. The HLL flux is:
    
        F̂(U_L, U_R) = { F(U_L)                                    if 0 ≤ λ_1
                      { (λ_2*F(U_L) - λ_1*F(U_R) + λ_1*λ_2*(U_R - U_L))/(λ_2 - λ_1)  if λ_1 < 0 < λ_2
                      { F(U_R)                                    if λ_2 ≤ 0
    
    where λ_1 and λ_2 are estimates of the minimum and maximum wave speeds.
    This scheme resolves isolated shocks and rarefactions better than Rusanov
    while maintaining positivity of water height.
    """
    
    def __init__(self, data_file: DataFile = None, mesh: Mesh = None, physics: Physics = None):
        """Initialize HLL flux.
        
        Args:
            data_file: DataFile with simulation parameters (gravity constant g)
            mesh: Mesh object for spatial discretization
            physics: Physics object for flux and wave speed computation
        """
        super().__init__(data_file, mesh, physics)
        self.flux_name = "HLL"
    
    def num_flux(self, sol_g: np.ndarray, sol_d: np.ndarray) -> np.ndarray:
        """Compute HLL numerical flux (Numba-accelerated).
        
        Mathematical formulation:
            λ_1 = min(u_L - √(gh_L), u_R - √(gh_R))  [slowest wave]
            λ_2 = max(u_L + √(gh_L), u_R + √(gh_R))  [fastest wave]
        
        Then:
            - If λ_1 ≥ 0: supersonic from left, use F(U_L)
            - If λ_2 ≤ 0: supersonic from right, use F(U_R)
            - If λ_1 < 0 < λ_2: subsonic, use weighted average
        
        For shallow water, this correctly captures:
            - Hydraulic jumps (shocks)
            - Rarefaction waves
            - Transcritical flows
            - Wet/dry interfaces
        
        Args:
            sol_g: Left state [h, q] with h ≥ 0 (m), q (m²/s), u = q/h (m/s)
            sol_d: Right state [h, q]
            
        Returns:
            np.ndarray: Numerical flux vector [F̂_h, F̂_q] of shape (2,)
        """
        g = self._DF.g
        return hll_flux_kernel(sol_g, sol_d, g)

