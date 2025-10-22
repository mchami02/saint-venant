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
        """Build the bed topography/bathymetry profile z(x).
        
        Supported topography types:
            - "FlatBottom": z(x) = 0 (horizontal bed)
            - "Bump": Parabolic bump for subcritical flow test
                      z(x) = 0.2 - 0.05*(x-10)² for 8 < x < 12
            - "Thacker": Parabolic bowl for oscillating water test
                        z(x) = h₀*[(x-L/2)²/a² - 1]
            - "File": Read from file with linear interpolation to mesh
        
        The topography affects:
            - Initial condition (h = H - z where H is total surface elevation)
            - Source term (S = -g*h*dz/dx in momentum equation)
            - Well-balanced property (stationary solutions with ∂h/∂x = -∂z/∂x)
        
        Args:
            verbosity: Output verbosity level (0=silent, 1=normal)
            
        Returns:
            None. Initializes self.topography array of shape (n_cells,)
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
        """Build the initial condition U(x, t=0) = [h(x,0), q(x,0)]ᵀ.
        
        Supported initial condition types:
            - "UniformHeightAndDischarge": Constant h and q everywhere
            - "DamBreakWet": Piecewise constant with h_L ≠ h_R, q = 0
                            Classic Riemann problem (discontinuous IC)
            - "DamBreakDry": Dam break into dry bed (h_R = 0)
                            Tests wet/dry interface handling
            - "Thacker": Analytical solution for parabolic bowl
                        Tests well-balanced property and C-property
            - "SinePerturbation": Smooth cosine perturbation
                                 Tests dispersion and dissipation
            - "File": Read from file (for restarting simulations)
        
        Important: Water height h is measured from bed, so:
            h(x) = H(x) - z(x)
        where H is the total surface elevation and z is bed elevation.
        
        Args:
            verbosity: Output verbosity level (0=silent, 1=normal)
            
        Returns:
            None. Initializes self.sol0 array of shape (n_cells, 2)
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
        """Build experimental/measured boundary condition data from file.
        
        Reads time series data for boundary conditions from an ASCII file.
        Format: [time, water_height] pairs
        
        The data is used with "DataFile" boundary condition type to impose
        measured water surface elevations. Linear interpolation is used
        between data points during simulation.
        
        Typical use cases:
            - Validation against laboratory experiments
            - Real-world river flow with measured stage
            - Tide-driven flows with water level recordings
        
        Args:
            verbosity: Output verbosity level (0=silent, 1=normal)
            
        Returns:
            None. Initializes self.exp_boundary_data array of shape (n_points, 2)
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
        """Build source term vector based on bed topography (Numba-accelerated).
        
        For shallow water with variable topography z(x), the source term arises
        from the bed slope in the momentum equation:
        
            S = [0, -g*h*dz/dx]ᵀ
        
        The derivative dz/dx is computed using:
            - Analytical formulas for Bump and Thacker test cases
            - Finite differences (2nd-order) for arbitrary topography from file
        
        This is called at each time step to update the source term based on
        the current water height h.
        
        Args:
            sol: Current solution array of shape (n_cells, 2) containing [h, q]
            
        Returns:
            None. Updates self.source in place.
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
        """Compute physical flux for 1D shallow water equations.
        
        The shallow water (Saint-Venant) system in conservative form:
            ∂U/∂t + ∂F(U)/∂x = S
        
        where U = [h, q]ᵀ and the physical flux is:
            F(U) = [q, q²/h + 0.5*g*h²]ᵀ
        
        Components:
            F₁ = q: mass flux (continuity)
            F₂ = q²/h + 0.5*g*h²: momentum flux
                 = h*u² + 0.5*g*h²  (pressure + advection)
        
        Args:
            sol: Solution vector of shape (2,) containing:
                - sol[0] = h: water height (m)
                - sol[1] = q = h*u: discharge (m²/s)
            
        Returns:
            np.ndarray: Physical flux vector of shape (2,):
                       [q, q²/h + 0.5*g*h²]
                       
        Note:
            Sets q = 0 if h ≤ 0 to handle dry states safely.
        """
        flux = np.zeros(2)
        h, qx = sol[0], sol[1]
        
        if h <= 0.0:
            qx = 0.0
        
        flux[0] = qx
        flux[1] = qx * qx / h + 0.5 * self.g * h * h
        
        return flux
    
    def compute_wave_speed(self, sol_g: np.ndarray, sol_d: np.ndarray) -> Tuple[float, float]:
        """Compute eigenvalues of the shallow water flux Jacobian (characteristic wave speeds).
        
        For the shallow water system, the flux Jacobian ∂F/∂U has eigenvalues:
            λ₁ = u - c  (left-going characteristic)
            λ₂ = u + c  (right-going characteristic)
        
        where c = √(gh) is the gravity wave celerity.
        
        Physical interpretation:
            - λ₁, λ₂ are the speeds at which information propagates
            - |λ| > c: supercritical flow (Fr > 1)
            - |λ| < c: subcritical flow (Fr < 1)
            - λ₁ < 0 < λ₂: subsonic regime (both characteristics present)
        
        For numerical methods, we need:
            λ_min = min(u_L - c_L, u_R - c_R)
            λ_max = max(u_L + c_L, u_R + c_R)
        
        Args:
            sol_g: Left state vector [h, q] of shape (2,)
            sol_d: Right state vector [h, q] of shape (2,)
            
        Returns:
            Tuple[float, float]: (lambda1, lambda2) where:
                - lambda1: minimum wave speed (can be negative)
                - lambda2: maximum wave speed (≥ lambda1)
        """
        h_g, h_d = sol_g[0], sol_d[0]
        u_g = sol_g[1] / h_g if h_g >= 1e-6 else 0.0
        u_d = sol_d[1] / h_d if h_d >= 1e-6 else 0.0
        
        lambda1 = min(u_g - np.sqrt(self.g * h_g), u_d - np.sqrt(self.g * h_d))
        lambda2 = max(u_g + np.sqrt(self.g * h_g), u_d + np.sqrt(self.g * h_d))
        
        return lambda1, lambda2
    
    def left_boundary_function(self, t: float, sol: np.ndarray) -> np.ndarray:
        """Compute left boundary condition (ghost cell state).
        
        Implements various boundary condition types based on flow regime (Froude number):
        
        Supported BC types:
            - "Neumann": Zero-gradient (∂U/∂x = 0) - natural outflow
            - "Wall": Reflective wall (u = 0) - solid boundary
            - "ImposedConstantDischarge": Prescribed flow rate
            - "ImposedConstantHeight": Prescribed water surface elevation
            - "PeriodicWaves": Time-periodic height variation
            - "DataFile": Experimental time series from file
        
        For characteristic BCs, the implementation uses Riemann invariants:
            β⁻ = u - 2√(gh)  (left-going characteristic)
            β⁺ = u + 2√(gh)  (right-going characteristic)
        
        The number of boundary conditions needed depends on Froude number:
            - Subcritical (Fr < 1): 1 BC needed (one characteristic leaves domain)
            - Supercritical (Fr > 1): 2 BCs needed (both characteristics leave domain)
        
        Args:
            t: Current simulation time (s)
            sol: Current solution array of shape (n_cells, 2)
            
        Returns:
            np.ndarray: Ghost cell state [h, q] of shape (2,) used for flux calculation
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
        """Compute right boundary condition (ghost cell state).
        
        Implements various boundary condition types at the right (downstream) boundary.
        See left_boundary_function for detailed description of BC types.
        
        Key difference from left boundary:
            - Uses β⁺ = u + 2√(gh) characteristic for subcritical outflow
            - Flow direction matters for determining supercritical vs subcritical
        
        Typical configurations:
            - River outflow: "Neumann" or "ImposedConstantHeight" (subcritical)
            - Channel exit: "ImposedConstantDischarge" (if known)
            - Reservoir: "ImposedConstantHeight" (water level control)
        
        Args:
            t: Current simulation time (s)
            sol: Current solution array of shape (n_cells, 2)
            
        Returns:
            np.ndarray: Ghost cell state [h, q] of shape (2,) for flux calculation
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
        """Solve quadratic equation ax² + bx + c = 0 using discriminant method.
        
        Uses the quadratic formula:
            x = [-b ± √(b² - 4ac)] / (2a)
        
        This is used in characteristic boundary condition calculations where
        we need to find the foot of the characteristic curve.
        
        Args:
            a: Coefficient of x² term
            b: Coefficient of x term
            c: Constant term
            
        Returns:
            float: The larger root r₂ = [-b + √(b² - 4ac)] / (2a)
            
        Raises:
            SystemExit: If discriminant < 0 (no real roots)
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
        """Interpolate source term value at arbitrary position x.
        
        Uses linear interpolation between cell centers:
            S(x) = S_i + (x - x_i) * (S_{i+1} - S_i) / Δx
        
        where x_i ≤ x ≤ x_{i+1}. This is needed for characteristic boundary
        conditions where we evaluate the source along characteristic curves.
        
        Args:
            x: Position at which to evaluate source term (m)
            
        Returns:
            float: Interpolated momentum source term S(x)
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
        """Get initial condition array.
        
        Returns:
            np.ndarray: Initial solution of shape (n_cells, 2) with [h, q]
        """
        return self.sol0
    
    def get_topography(self) -> np.ndarray:
        """Get bed topography/bathymetry array.
        
        Returns:
            np.ndarray: Bed elevation z(x) of shape (n_cells,) in meters
        """
        return self.topography
    
    def get_source_term(self) -> np.ndarray:
        """Get current source term array.
        
        Returns:
            np.ndarray: Source term of shape (n_cells, 2):
                       S[:, 0] = 0 (no mass source)
                       S[:, 1] = -g*h*dz/dx (bed slope + other sources)
        """
        return self.source
    
    def get_exact_solution(self) -> np.ndarray:
        """Get exact analytical solution (for test cases).
        
        Returns:
            np.ndarray: Exact solution of shape (n_cells, 2) with [h, q],
                       or None if not a test case.
        """
        return self.exact_sol
    
    def get_experimental_boundary_data(self) -> np.ndarray:
        """Get experimental boundary condition data from file.
        
        Returns:
            np.ndarray: Boundary data of shape (n_points, 2) with [time, height],
                       or None if not using experimental BC.
        """
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

