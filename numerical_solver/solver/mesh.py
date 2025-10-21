"""
Mesh class for 1D cartesian mesh generation and management.
"""

import numpy as np
from solver.data_file import DataFile


class Mesh:
    """1D Cartesian mesh for finite volume discretization.
    
    Attributes:
        xmin: Minimum coordinate
        xmax: Maximum coordinate
        dx: Spatial step size
        number_of_cells: Number of cells in the mesh
        cell_centers: Array of cell center coordinates
    """
    
    def __init__(self, data_file: DataFile = None):
        """Initialize mesh from DataFile.
        
        Args:
            data_file: DataFile object containing mesh parameters
        """
        self._DF = data_file
        self.xmin = 0.0
        self.xmax = 1.0
        self.dx = 0.1
        self.number_of_cells = 10
        self.cell_centers = None
        
        if data_file is not None:
            self.xmin = data_file.xmin
            self.xmax = data_file.xmax
            self.dx = data_file.dx
            self.number_of_cells = data_file.Nx
            self.cell_centers = np.zeros(self.number_of_cells)
    
    def initialize(self, data_file: DataFile = None, verbosity: int = 1):
        """Initialize the mesh and compute cell centers.
        
        Args:
            data_file: Optional DataFile to reinitialize from
            verbosity: Level of output verbosity
        """
        if data_file is not None:
            self._DF = data_file
            self.xmin = data_file.xmin
            self.xmax = data_file.xmax
            self.dx = data_file.dx
            self.number_of_cells = data_file.Nx
            self.cell_centers = np.zeros(self.number_of_cells)
        
        if verbosity > 0:
            print("Generating a 1D cartesian mesh...")
        
        # Compute cell centers
        for i in range(self.number_of_cells):
            self.cell_centers[i] = self.xmin + (i + 0.5) * self.dx
        
        if verbosity > 0:
            print("\033[92mSUCCESS::MESH : Mesh generated successfully !\033[0m")
            print()
    
    def get_cell_centers(self) -> np.ndarray:
        """Get array of cell centers."""
        return self.cell_centers
    
    def get_number_of_cells(self) -> int:
        """Get number of cells."""
        return self.number_of_cells
    
    def get_space_step(self) -> float:
        """Get spatial step size."""
        return self.dx
    
    def get_xmin(self) -> float:
        """Get minimum x coordinate."""
        return self.xmin
    
    def get_xmax(self) -> float:
        """Get maximum x coordinate."""
        return self.xmax

