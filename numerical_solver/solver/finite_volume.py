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
        
        Args:
            sol_g: Left state [h, q]
            sol_d: Right state [h, q]
            
        Returns:
            Numerical flux vector
        """
        pass
    
    def build_flux_vector(self, t: float, sol: np.ndarray):
        """Build flux vector using reconstruction and numerical flux.
        
        Args:
            t: Current time
            sol: Current solution array (n_cells x 2)
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
        """Minmod slope limiter.
        
        Args:
            a: First slope
            b: Second slope
            
        Returns:
            Limited slope
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
    """Lax-Friedrichs numerical flux scheme."""
    
    def __init__(self, data_file: DataFile = None, mesh: Mesh = None, physics: Physics = None):
        """Initialize Lax-Friedrichs flux.
        
        Args:
            data_file: DataFile with parameters
            mesh: Mesh object
            physics: Physics object
        """
        super().__init__(data_file, mesh, physics)
        self.flux_name = "LF"
    
    def num_flux(self, sol_g: np.ndarray, sol_d: np.ndarray) -> np.ndarray:
        """Compute Lax-Friedrichs numerical flux (Numba-accelerated).
        
        Args:
            sol_g: Left state [h, q]
            sol_d: Right state [h, q]
            
        Returns:
            Numerical flux vector
        """
        dt = self._DF.time_step
        dx = self._DF.dx
        g = self._DF.g
        
        return lax_friedrichs_flux_kernel(sol_g, sol_d, dt, dx, g)


class Rusanov(FiniteVolume):
    """Rusanov (local Lax-Friedrichs) numerical flux scheme."""
    
    def __init__(self, data_file: DataFile = None, mesh: Mesh = None, physics: Physics = None):
        """Initialize Rusanov flux.
        
        Args:
            data_file: DataFile with parameters
            mesh: Mesh object
            physics: Physics object
        """
        super().__init__(data_file, mesh, physics)
        self.flux_name = "Rusanov"
    
    def num_flux(self, sol_g: np.ndarray, sol_d: np.ndarray) -> np.ndarray:
        """Compute Rusanov numerical flux (Numba-accelerated).
        
        Args:
            sol_g: Left state [h, q]
            sol_d: Right state [h, q]
            
        Returns:
            Numerical flux vector
        """
        g = self._DF.g
        return rusanov_flux_kernel(sol_g, sol_d, g)


class HLL(FiniteVolume):
    """HLL (Harten-Lax-van Leer) numerical flux scheme."""
    
    def __init__(self, data_file: DataFile = None, mesh: Mesh = None, physics: Physics = None):
        """Initialize HLL flux.
        
        Args:
            data_file: DataFile with parameters
            mesh: Mesh object
            physics: Physics object
        """
        super().__init__(data_file, mesh, physics)
        self.flux_name = "HLL"
    
    def num_flux(self, sol_g: np.ndarray, sol_d: np.ndarray) -> np.ndarray:
        """Compute HLL numerical flux (Numba-accelerated).
        
        Args:
            sol_g: Left state [h, q]
            sol_d: Right state [h, q]
            
        Returns:
            Numerical flux vector
        """
        g = self._DF.g
        return hll_flux_kernel(sol_g, sol_d, g)

