"""
Physics class for shallow water equations.

Handles initial conditions, boundary conditions, topography, source terms,
exact solutions, and physical fluxes.
"""

import numpy as np
import re
from typing import Tuple
from solver.data_file import DataFile
from solver.mesh import Mesh
from solver.numba_kernels import build_source_term_bump, build_source_term_thacker, build_source_term_file


class Physics:
    """Physical model for 1D shallow water (Saint-Venant) equations.
    
    Manages:
    - Initial conditions
    - Boundary conditions  
    - Topography and source terms
    - Physical flux computation
    - Exact solutions for test cases
    
    Attributes:
        g: Gravitational acceleration
        n_cells: Number of computational cells
        sol0: Initial condition (n_cells x 2 array: [h, q])
        topography: Topography/bathymetry vector
        source: Source term vector
        exact_sol: Exact solution for comparison
        exp_boundary_data: Experimental boundary data
    """
    
    def __init__(self, data_file: DataFile = None, mesh: Mesh = None):
        """Initialize Physics object.
        
        Args:
            data_file: DataFile with simulation parameters
            mesh: Mesh object
        """
        self._DF = data_file
        self._mesh = mesh
        self.xmin = 0.0
        self.xmax = 1.0
        self.g = 9.81
        self.n_cells = 0
        self._i = 0  # Index for experimental boundary data
        
        self.sol0 = None
        self.topography = None
        self.source = None
        self.exact_sol = None
        self.exp_boundary_data = None
        self.file_topography = None
        
        if data_file is not None and mesh is not None:
            self.xmin = mesh.get_xmin()
            self.xmax = mesh.get_xmax()
            self.g = data_file.g
            self.n_cells = mesh.get_number_of_cells()
    
    def initialize(self, data_file: DataFile = None, mesh: Mesh = None, verbosity: int = 1):
        """Initialize physics: build topography, initial conditions, boundary data.
        
        Args:
            data_file: Optional DataFile to initialize from
            mesh: Optional Mesh to initialize from
            verbosity: Output verbosity level
        """
        if data_file is not None:
            self._DF = data_file
            self.g = data_file.g
        if mesh is not None:
            self._mesh = mesh
            self.xmin = mesh.get_xmin()
            self.xmax = mesh.get_xmax()
            self.n_cells = mesh.get_number_of_cells()
        
        self._i = 0
        
        if verbosity > 0:
            print("Building topography, initial condition, and experimental data...")
        
        # Initialize arrays
        self.source = np.zeros((self.n_cells, 2))
        
        # Build components
        self.build_topography(verbosity)
        self.build_initial_condition(verbosity)
        
        if self._DF.left_BC == "DataFile" or self._DF.right_BC == "DataFile":
            self.build_exp_boundary_data(verbosity)
        
        if verbosity > 0:
            print("\033[92mSUCCESS::FUNCTION : Everything was successfully built.\033[0m")
            print()
    
    def build_topography(self, verbosity: int = 1):
        """Build the topography/bathymetry profile.
        
        Args:
            verbosity: Output verbosity level
        """
        self.topography = np.zeros(self.n_cells)
        cell_centers = self._mesh.get_cell_centers()
        
        topo_type = self._DF.topography_type
        
        if topo_type == "FlatBottom":
            # Already zero
            pass
        
        elif topo_type == "Thacker":
            # Thacker test case topography
            xmin, xmax = self._DF.xmin, self._DF.xmax
            L = xmax - xmin
            a, h0 = 1.0, 0.5
            for i in range(self.n_cells):
                x = cell_centers[i]
                self.topography[i] = h0 * (1.0 / a**2 * (x - 0.5 * L)**2 - 1.0)
        
        elif topo_type == "Bump":
            # Bump topography
            for i in range(self.n_cells):
                x = cell_centers[i]
                if 8 < x < 12:
                    self.topography[i] = 0.2 - 0.05 * (x - 10)**2
        
        elif topo_type == "File":
            # Read from file
            topo_file = self._DF.topography_file
            try:
                if verbosity > 0:
                    print(f"Building the topography from file : {topo_file}")
                
                with open(topo_file, 'r') as f:
                    lines = f.readlines()
                
                size = int(lines[0].strip())
                self.file_topography = np.zeros((size + 1, 2))
                
                for i, line in enumerate(lines[1:size+2]):
                    # Replace commas with spaces
                    line = re.sub(r',', ' ', line)
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        self.file_topography[i, 0] = float(parts[0])
                        self.file_topography[i, 1] = float(parts[1])
                
                # Interpolate topography to mesh
                x1, z1 = self.file_topography[0, 0], self.file_topography[0, 1]
                x2, z2 = self.file_topography[1, 0], self.file_topography[1, 1]
                j = 0
                
                for k in range(self.n_cells):
                    x = cell_centers[k]
                    # Find the interval containing x
                    while not (x1 < x <= x2):
                        j += 1
                        x1 = self.file_topography[j, 0]
                        x2 = self.file_topography[j + 1, 0]
                    z1 = self.file_topography[j, 1]
                    z2 = self.file_topography[j + 1, 1]
                    # Linear interpolation
                    self.topography[k] = z1 + (x - x1) * (z2 - z1) / (x2 - x1)
                
                # Set minimum topography to 0
                topo_min = self.topography.min()
                self.topography -= topo_min
            
            except Exception as e:
                print(f"\033[91mERROR::TOPOGRAPHY : Unable to open the topography file : {topo_file}\033[0m")
                print(f"Error: {e}")
                exit(-1)
        
        else:
            print(f"\033[91mERROR::TOPOGRAPHY : Case {topo_type} not implemented\033[0m")
            exit(-1)
        
        if verbosity > 0:
            print("\033[92mSUCCESS::TOPOGRAPHY : Topography was successfully built.\033[0m")
    
    def build_initial_condition(self, verbosity: int = 1):
        """Build the initial condition.
        
        Args:
            verbosity: Output verbosity level
        """
        self.sol0 = np.zeros((self.n_cells, 2))
        cell_centers = self._mesh.get_cell_centers()
        ic_type = self._DF.initial_condition
        
        if ic_type == "UniformHeightAndDischarge":
            H0 = self._DF.initial_height
            q0 = self._DF.initial_discharge
            for i in range(self.n_cells):
                self.sol0[i, 0] = max(H0 - self.topography[i], 0.0)
                self.sol0[i, 1] = q0
        
        elif ic_type == "DamBreakWet":
            self.sol0[:, 1] = 0.0
            Hl, Hr = 2.0, 1.0
            for i in range(self.n_cells):
                if cell_centers[i] < 0.5 * (self.xmax + self.xmin):
                    self.sol0[i, 0] = max(Hl - self.topography[i], 0.0)
                else:
                    self.sol0[i, 0] = max(Hr - self.topography[i], 0.0)
        
        elif ic_type == "DamBreakDry":
            self.sol0[:, 1] = 0.0
            Hl, Hr = 2.0, 0.0
            for i in range(self.n_cells):
                if cell_centers[i] < 0.5 * (self.xmax + self.xmin):
                    self.sol0[i, 0] = max(Hl - self.topography[i], 0.0)
                else:
                    self.sol0[i, 0] = max(Hr - self.topography[i], 0.0)
        
        elif ic_type == "Thacker":
            self.sol0[:, 1] = 0.0
            xmin, xmax = self._DF.xmin, self._DF.xmax
            L = xmax - xmin
            a, h0 = 1.0, 0.5
            x1 = -0.5 - a + 0.5 * L
            x2 = -0.5 + a + 0.5 * L
            for i in range(self.n_cells):
                x = cell_centers[i]
                if x1 <= x <= x2:
                    self.sol0[i, 0] = -h0 * ((1.0 / a * (x - 0.5 * L) + 1.0 / (2.0 * a))**2 - 1)
                else:
                    self.sol0[i, 0] = 0.0
        
        elif ic_type == "SinePerturbation":
            self.sol0[:, 1] = 0.0
            H = 2.0
            for i in range(self.n_cells):
                x = cell_centers[i]
                if -1 < x < 1:
                    self.sol0[i, 0] = max(H + 0.2 * np.cos(np.pi * x) - self.topography[i], 0.0)
                else:
                    self.sol0[i, 0] = max(1.8 - self.topography[i], 0.0)
        
        elif ic_type == "File":
            init_file = self._DF.init_file
            try:
                with open(init_file, 'r') as f:
                    lines = f.readlines()
                
                # Skip first line (comments)
                for i, line in enumerate(lines[1:]):
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        self.sol0[i, 0] = float(parts[2])
                        self.sol0[i, 1] = float(parts[4])
            except Exception as e:
                print(f"\033[91mERROR::INITIALCONDITION : Unable to read file {init_file}\033[0m")
                print(f"Error: {e}")
                exit(-1)
        
        else:
            print(f"\033[91mERROR::INITIALCONDITION : Case {ic_type} not implemented\033[0m")
            exit(-1)
        
        if verbosity > 0:
            print("\033[92mSUCCESS::INITIALCONDITION : Initial Condition was successfully built.\033[0m")
    
    def build_exp_boundary_data(self, verbosity: int = 1):
        """Build experimental boundary data from file.
        
        Args:
            verbosity: Output verbosity level
        """
        exp_data_file = self._DF.left_BC_data_file
        
        try:
            if verbosity > 0:
                print(f"Building the experimental data from file : {exp_data_file}")
            
            with open(exp_data_file, 'r') as f:
                lines = f.readlines()
            
            # First line has the number of data points
            size = int(lines[0].strip())
            self.exp_boundary_data = np.zeros((size, 2))
            
            # Read data
            for i, line in enumerate(lines[1:size+1]):
                line = re.sub(r',', ' ', line)
                parts = line.strip().split()
                if len(parts) >= 2:
                    self.exp_boundary_data[i, 0] = float(parts[0])
                    self.exp_boundary_data[i, 1] = float(parts[1])
            
            if verbosity > 0:
                print("\033[92mSUCCESS::EXPDATA : Experimental data was successfully built.\033[0m")
        
        except Exception as e:
            print(f"\033[91mERROR::EXPDATA : Unable to open the experimental data file : {exp_data_file}\033[0m")
            print(f"Error: {e}")
            exit(-1)
    
    def build_source_term(self, sol: np.ndarray):
        """Build source term based on topography (Numba-accelerated).
        
        Args:
            sol: Current solution array (n_cells x 2)
        """
        topo_type = self._DF.topography_type
        cell_centers = self._mesh.get_cell_centers()
        
        if topo_type == "FlatBottom":
            # No source term
            self.source.fill(0.0)
        
        elif topo_type == "Bump":
            # Use JIT-compiled parallel version
            build_source_term_bump(self.source, sol, cell_centers, self.g)
        
        elif topo_type == "Thacker":
            # Use JIT-compiled parallel version
            build_source_term_thacker(self.source, sol, cell_centers, 
                                       self._DF.xmin, self._DF.xmax, self.g)
        
        elif topo_type == "File":
            # Use JIT-compiled parallel version
            dx = self._mesh.get_space_step()
            build_source_term_file(self.source, sol, self.topography, dx, self.g)
        
        else:
            print(f"\033[91mERROR::SOURCETERM : Case {topo_type} not implemented.\033[0m")
            exit(-1)
    
    def physical_flux(self, sol: np.ndarray) -> np.ndarray:
        """Compute physical flux for shallow water equations.
        
        Args:
            sol: Solution vector [h, q]
            
        Returns:
            Flux vector [q, q²/h + 0.5*g*h²]
        """
        flux = np.zeros(2)
        h, qx = sol[0], sol[1]
        
        if h <= 0.0:
            qx = 0.0
        
        flux[0] = qx
        flux[1] = qx * qx / h + 0.5 * self.g * h * h
        
        return flux
    
    def compute_wave_speed(self, sol_g: np.ndarray, sol_d: np.ndarray) -> Tuple[float, float]:
        """Compute the eigenvalues of the flux Jacobian (wave speeds).
        
        Args:
            sol_g: Left state [h, q]
            sol_d: Right state [h, q]
            
        Returns:
            Tuple of (lambda1, lambda2) - minimum and maximum wave speeds
        """
        h_g, h_d = sol_g[0], sol_d[0]
        u_g = sol_g[1] / h_g if h_g >= 1e-6 else 0.0
        u_d = sol_d[1] / h_d if h_d >= 1e-6 else 0.0
        
        lambda1 = min(u_g - np.sqrt(self.g * h_g), u_d - np.sqrt(self.g * h_d))
        lambda2 = max(u_g + np.sqrt(self.g * h_g), u_d + np.sqrt(self.g * h_d))
        
        return lambda1, lambda2
    
    def left_boundary_function(self, t: float, sol: np.ndarray) -> np.ndarray:
        """Compute left boundary condition.
        
        Args:
            t: Current time
            sol: Current solution array
            
        Returns:
            Boundary state vector [h, q]
        """
        sol_g = np.zeros(2)
        
        # Compute Froude number at boundary
        h, q = sol[0, 0], sol[0, 1]
        Fr = abs(q) / (h * np.sqrt(self.g * h)) if h > 1e-10 else 0.0
        
        bc_type = self._DF.left_BC
        
        if bc_type == "Neumann":
            sol_g[0] = sol[0, 0]
            sol_g[1] = sol[0, 1]
        
        elif bc_type == "Wall":
            sol_g[0] = sol[0, 0]
            sol_g[1] = 0.0
        
        elif bc_type == "ImposedConstantDischarge":
            if Fr < 1:  # Subcritical inflow/outflow
                sol_g[0] = sol[0, 0]
                sol_g[1] = self._DF.left_BC_imposed_discharge
            elif Fr > 1 and q < 0:  # Supercritical outflow
                sol_g[0] = sol[0, 0]
                sol_g[1] = sol[0, 1]
            elif Fr > 1 and q > 0:  # Supercritical inflow
                sol_g[0] = self._DF.left_BC_imposed_height
                sol_g[1] = self._DF.left_BC_imposed_discharge
        
        elif bc_type in ["PeriodicWaves", "DataFile", "ImposedConstantHeight"]:
            # Characteristic boundary condition
            h1, h2 = sol[0, 0], sol[1, 0]
            u1 = sol[0, 1] / h1 if h1 > 1e-10 else 0.0
            u2 = sol[1, 1] / h2 if h2 > 1e-10 else 0.0
            dx, dt = self._DF.dx, self._DF.time_step
            x1 = self._DF.xmin + 0.5 * dx
            
            a = (1 + dt / dx * (u2 - u1))**2
            b = 2 * dt * (u1 - x1 / dx * (u2 - u1)) * (1 + dt / dx * (u2 - u1)) - dt * dt * self.g * (h2 - h1) / dx
            c = (dt * u1 - dt / dx * x1 * (u2 - u1))**2 - dt * dt * self.g * (h1 - x1 / dx * (h2 - h1))
            
            xe = self.find_root(a, b, c)
            u_xe = u1 + (xe - x1) * (u2 - u1) / dx
            h_xe = h1 + (xe - x1) * (h2 - h1) / dx
            source_xe = self.find_source_x(xe)
            beta_minus_xe_tn = u_xe - 2 * np.sqrt(self.g * h_xe)
            beta_minus_0_tnplus1 = beta_minus_xe_tn - self.g * dt * source_xe
            
            if bc_type == "ImposedConstantHeight":
                if Fr < 1:  # Subcritical
                    sol_g[0] = self._DF.left_BC_imposed_height
                    sol_g[1] = sol_g[0] * (beta_minus_0_tnplus1 + 2 * np.sqrt(self.g * sol_g[0]))
                elif Fr > 1 and q < 0:  # Supercritical outflow
                    sol_g[0] = sol[0, 0]
                    sol_g[1] = sol[0, 1]
                elif Fr > 1 and q > 0:  # Supercritical inflow
                    sol_g[0] = self._DF.left_BC_imposed_height
                    sol_g[1] = self._DF.left_BC_imposed_discharge
            
            elif bc_type == "PeriodicWaves":
                sol_g[0] = 3.0 + 0.1 * np.sin(5 * np.pi * t)
                sol_g[1] = sol_g[0] * (beta_minus_0_tnplus1 + 2 * np.sqrt(self.g * sol_g[0]))
            
            elif bc_type == "DataFile":
                i_max = self.exp_boundary_data.shape[0]
                temps1 = self.exp_boundary_data[self._i, 0]
                temps2 = self.exp_boundary_data[self._i + 1, 0]
                
                while not (temps1 < t <= temps2) and self._i < i_max - 2:
                    self._i += 1
                    temps1 = self.exp_boundary_data[self._i, 0]
                    temps2 = self.exp_boundary_data[self._i + 1, 0]
                
                if self._i >= i_max - 2:
                    print("No matching time step found --> time step too large?")
                    return sol_g
                
                hauteur1 = self.exp_boundary_data[self._i, 1]
                hauteur2 = self.exp_boundary_data[self._i + 1, 1]
                sol_g[0] = hauteur1 + (t - temps1) * (hauteur2 - hauteur1) / (temps2 - temps1)
                sol_g[1] = sol_g[0] * (beta_minus_0_tnplus1 + 2 * np.sqrt(self.g * sol_g[0]))
        
        return sol_g
    
    def right_boundary_function(self, t: float, sol: np.ndarray) -> np.ndarray:
        """Compute right boundary condition.
        
        Args:
            t: Current time
            sol: Current solution array
            
        Returns:
            Boundary state vector [h, q]
        """
        sol_d = np.zeros(2)
        
        # Compute Froude number at boundary
        h, q = sol[self.n_cells - 1, 0], sol[self.n_cells - 1, 1]
        Fr = abs(q) / (h * np.sqrt(self.g * h)) if h > 1e-10 else 0.0
        
        bc_type = self._DF.right_BC
        
        if bc_type == "Neumann":
            sol_d[0] = sol[self.n_cells - 1, 0]
            sol_d[1] = sol[self.n_cells - 1, 1]
        
        elif bc_type == "Wall":
            sol_d[0] = sol[self.n_cells - 1, 0]
            sol_d[1] = 0.0
        
        elif bc_type == "ImposedConstantDischarge":
            if Fr < 1:  # Subcritical
                sol_d[0] = sol[self.n_cells - 1, 0]
                sol_d[1] = self._DF.right_BC_imposed_discharge
            elif Fr > 1 and q > 0:  # Supercritical outflow
                sol_d[0] = sol[self.n_cells - 1, 0]
                sol_d[1] = sol[self.n_cells - 1, 1]
            elif Fr > 1 and q < 0:  # Supercritical inflow
                sol_d[0] = self._DF.right_BC_imposed_height - self.topography[self.n_cells - 1]
                sol_d[1] = self._DF.right_BC_imposed_discharge
        
        elif bc_type in ["PeriodicWaves", "DataFile", "ImposedConstantHeight"]:
            h1 = sol[self.n_cells - 1, 0]
            u1 = sol[self.n_cells - 1, 1] / h1 if h1 > 1e-10 else 0.0
            
            if bc_type == "ImposedConstantHeight":
                if Fr < 1:  # Subcritical
                    sol_d[0] = self._DF.right_BC_imposed_height - self.topography[self.n_cells - 1]
                    sol_d[1] = sol_d[0] * (u1 + 2.0 * np.sqrt(self.g * h1) - 2.0 * np.sqrt(self.g * sol_d[0]))
                elif Fr > 1 and q > 0:  # Supercritical outflow
                    sol_d[0] = sol[self.n_cells - 1, 0]
                    sol_d[1] = sol[self.n_cells - 1, 1]
                elif Fr > 1 and q < 0:  # Supercritical inflow
                    sol_d[0] = self._DF.right_BC_imposed_height - self.topography[self.n_cells - 1]
                    sol_d[1] = self._DF.right_BC_imposed_discharge
            
            elif bc_type == "PeriodicWaves":
                sol_d[0] = 3.0 + 0.1 * np.sin(5 * np.pi * t)
                sol_d[1] = sol_d[0] * (u1 + 2.0 * np.sqrt(self.g * h1) - 2.0 * np.sqrt(self.g * sol_d[0]))
            
            elif bc_type == "DataFile":
                i_max = self.exp_boundary_data.shape[0] - self._i
                temps1 = self.exp_boundary_data[self._i, 0]
                temps2 = self.exp_boundary_data[self._i + 1, 0]
                
                while not (temps1 < t <= temps2) and self._i < i_max:
                    self._i += 1
                    temps1 = self.exp_boundary_data[self._i, 0]
                    temps2 = self.exp_boundary_data[self._i + 1, 0]
                
                if self._i >= i_max:
                    print("No matching time step found --> time step too large?")
                    return sol_d
                
                hauteur1 = self.exp_boundary_data[self._i, 1]
                hauteur2 = self.exp_boundary_data[self._i + 1, 1]
                sol_d[0] = (t - temps1) * (hauteur2 - hauteur1) / (temps2 - temps1) - hauteur1
                sol_d[1] = sol_d[0] * (u1 + 2.0 * np.sqrt(self.g * h1) - 2.0 * np.sqrt(self.g * sol_d[0]))
        
        return sol_d
    
    def find_root(self, a: float, b: float, c: float) -> float:
        """Find root of quadratic equation ax² + bx + c = 0.
        
        Args:
            a, b, c: Quadratic coefficients
            
        Returns:
            The larger root
        """
        delta = b * b - 4 * a * c
        
        if delta < 0:
            print("No real root")
            exit(1)
        elif delta == 0:
            return -b / (2 * a)
        else:
            r1 = (-b - np.sqrt(delta)) / (2 * a)
            r2 = (-b + np.sqrt(delta)) / (2 * a)
            return r2
    
    def find_source_x(self, x: float) -> float:
        """Find source term at position x by interpolation.
        
        Args:
            x: Position
            
        Returns:
            Interpolated source term value
        """
        i = 0
        dx = self._DF.dx
        x1 = self.xmin + (i + 0.5) * dx
        
        while x1 < x:
            i += 1
            x1 += dx
        
        x2 = x1 + dx
        source1 = self.source[i, 1]
        source2 = self.source[i + 1, 1]
        source = source1 + (x - x1) * (source2 - source1) / (x2 - x1)
        
        return source
    
    def get_initial_condition(self) -> np.ndarray:
        """Get initial condition array."""
        return self.sol0
    
    def get_topography(self) -> np.ndarray:
        """Get topography array."""
        return self.topography
    
    def get_source_term(self) -> np.ndarray:
        """Get source term array."""
        return self.source
    
    def get_exact_solution(self) -> np.ndarray:
        """Get exact solution array."""
        return self.exact_sol
    
    def get_experimental_boundary_data(self) -> np.ndarray:
        """Get experimental boundary data."""
        return self.exp_boundary_data
    
    def build_exact_solution(self, t: float):
        """Build exact solution for test cases (placeholder for complex cases).
        
        Args:
            t: Current time
        """
        # This is a simplified version - full implementation would include
        # dam break, Thacker, and stationary flow solutions
        self.exact_sol = np.zeros((self.n_cells, 2))
        test_case = self._DF.test_case
        
        if test_case == "RestingLake":
            for i in range(self.n_cells):
                self.exact_sol[i, 0] = max(self._DF.left_BC_imposed_height - self.topography[i], 0.0)
                self.exact_sol[i, 1] = 0.0
        
        # Additional test cases would be implemented here
        # (DamBreakWet, DamBreakDry, Thacker, SubcriticalFlow, etc.)
    
    def save_exact_solution(self, file_name: str):
        """Save exact solution to file.
        
        Args:
            file_name: Output file path
        """
        cell_centers = self._mesh.get_cell_centers()
        
        with open(file_name, 'w') as f:
            f.write("# x  H=h+z   h       u       q       Fr=|u|/sqrt(gh)\n")
            for i in range(self.exact_sol.shape[0]):
                h = self.exact_sol[i, 0]
                q = self.exact_sol[i, 1]
                u = q / h if h > 1e-10 else 0.0
                Fr = abs(u) / np.sqrt(self.g * h) if h > 1e-10 else 0.0
                
                f.write(f"{cell_centers[i]} {h + self.topography[i]} {h} {u} {q} {Fr}\n")

