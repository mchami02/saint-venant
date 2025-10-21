"""
Time integration schemes for shallow water equations.

Implements Explicit Euler and RK2 (Heun's method) time stepping.
"""

import numpy as np
import h5py
from abc import ABC, abstractmethod
from solver.data_file import DataFile
from solver.mesh import Mesh
from solver.physics import Physics
from solver.finite_volume import FiniteVolume
from solver.numba_kernels import explicit_euler_step, rk2_step
from tqdm import tqdm


class TimeScheme(ABC):
    """Base class for time integration schemes.
    
    Attributes:
        sol: Solution array (n_cells x 2: [h, q])
        time_step: Time step size
        initial_time: Start time
        final_time: End time
        current_time: Current simulation time
        n_probes: Number of measurement probes
        probes_ref: Reference indices for probes
        probes_pos: Physical positions of probes
        probes_indices: Cell indices for probes
    """
    
    def __init__(self, data_file: DataFile = None, mesh: Mesh = None, 
                 physics: Physics = None, fin_vol: FiniteVolume = None):
        """Initialize time scheme.
        
        Args:
            data_file: DataFile with parameters
            mesh: Mesh object
            physics: Physics object
            fin_vol: FiniteVolume flux scheme
        """
        self._DF = data_file
        self._mesh = mesh
        self._physics = physics
        self._finVol = fin_vol
        
        self.sol = None
        self.time_step = 0.01
        self.initial_time = 0.0
        self.final_time = 1.0
        self.current_time = 0.0
        self.n_probes = 0
        self.probes_ref = []
        self.probes_pos = []
        self.probes_indices = []
        
        if data_file is not None and physics is not None:
            self.sol = physics.get_initial_condition().copy()
            self.time_step = data_file.time_step
            self.initial_time = data_file.initial_time
            self.final_time = data_file.final_time
            self.current_time = self.initial_time
            self.n_probes = data_file.n_probes
            self.probes_ref = data_file.probes_references.copy()
            self.probes_pos = data_file.probes_positions.copy()
            self.probes_indices = [0] * self.n_probes
    
    def initialize(self, data_file: DataFile, mesh: Mesh, physics: Physics, fin_vol: FiniteVolume):
        """Initialize with new parameters.
        
        Args:
            data_file: DataFile with parameters
            mesh: Mesh object
            physics: Physics object
            fin_vol: FiniteVolume flux scheme
        """
        self._DF = data_file
        self._mesh = mesh
        self._physics = physics
        self._finVol = fin_vol
        self.sol = physics.get_initial_condition().copy()
        self.time_step = data_file.time_step
        self.initial_time = data_file.initial_time
        self.final_time = data_file.final_time
        self.current_time = self.initial_time
        self.n_probes = data_file.n_probes
        self.probes_ref = data_file.probes_references.copy()
        self.probes_pos = data_file.probes_positions.copy()
        self.probes_indices = [0] * self.n_probes
    
    def build_probes_cell_indices(self):
        """Find cell indices corresponding to probe positions."""
        nb_cells = self._mesh.get_number_of_cells()
        cell_centers = self._mesh.get_cell_centers()
        
        for i in range(self.n_probes):
            pos = self.probes_pos[i]
            index = 0
            dist_min = abs(pos - cell_centers[0])
            
            for k in range(nb_cells):
                x = cell_centers[k]
                dist = abs(pos - x)
                if dist < dist_min:
                    dist_min = dist
                    index = k
            
            self.probes_indices[i] = index
    
    def save_current_solution(self, file_name: str, verbosity: int = 1):
        """Save current solution to file.
        
        Args:
            file_name: Output file path
            verbosity: Output verbosity level
        """
        if verbosity > 0:
            print(f"Saving solution at t = {self.current_time}")
        
        cell_centers = self._mesh.get_cell_centers()
        g = self._DF.g
        topography = self._physics.get_topography()
        
        with open(file_name, 'w') as f:
            f.write("# x  H=h+z   h       u       q       Fr=|u|/sqrt(gh)\n")
            for i in range(self.sol.shape[0]):
                h = self.sol[i, 0]
                q = self.sol[i, 1]
                u = q / h if h > 1e-10 else 0.0
                Fr = abs(u) / np.sqrt(g * h) if h > 1e-10 else 0.0
                
                f.write(f"{cell_centers[i]} {h + topography[i]} {h} {u} {q} {Fr}\n")
    
    def save_probes(self):
        """Save probe data at current time."""
        nb_probes = self._DF.n_probes
        g = self._DF.g
        topography = self._physics.get_topography()
        
        for i in range(nb_probes):
            file_name = f"{self._DF.results_dir}/probe_{self.probes_ref[i]}.txt"
            index = self.probes_indices[i]
            
            h = self.sol[index, 0]
            q = self.sol[index, 1]
            u = q / h if h > 1e-10 else 0.0
            Fr = abs(u) / np.sqrt(g * h) if h > 1e-10 else 0.0
            
            with open(file_name, 'a') as f:
                f.write(f"{self.current_time},{h + topography[index]},{h},{u},{q},{Fr}\n")
    
    @abstractmethod
    def one_step(self):
        """Perform one time step (must be implemented by subclasses)."""
        pass
    
    def solve(self, verbosity: int = 1, output_filename: str = None):
        """Main time loop to solve the problem (saves to HDF5).
        
        Args:
            verbosity: Output verbosity level
            output_filename: Optional custom output filename (default: solution_FLUX.h5)
        """
        results_dir = self._DF.results_dir
        flux_name = self._finVol.get_flux_name()
        
        # Find probe cell indices
        self.build_probes_cell_indices()
        
        # Calculate number of time steps
        n_steps = int((self.final_time - self.initial_time) / self.time_step)
        save_every = max(1, self._DF.save_frequency)
        n_saves = n_steps // save_every + 1
        
        # Prepare HDF5 filename (always in results_dir)
        import os
        if output_filename is None:
            h5_filename = f"{results_dir}/solution_{flux_name}.h5"
        else:
            # Extract just the filename (no path) and save in results_dir
            output_basename = os.path.basename(output_filename)
            h5_filename = f"{results_dir}/{output_basename}"
        
        # Pre-allocate arrays in memory for solution snapshots
        n_cells = self._mesh.get_number_of_cells()
        solution_h = np.zeros((n_saves, n_cells))
        solution_q = np.zeros((n_saves, n_cells))
        solution_time = np.zeros(n_saves)
        
        # Save initial condition in memory
        solution_h[0, :] = self.sol[:, 0]
        solution_q[0, :] = self.sol[:, 1]
        solution_time[0] = self.current_time
        
        # Time loop with progress bar
        n = 0
        save_idx = 1
        
        pbar = tqdm(total=n_steps, desc="Solving", unit="steps", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        while self.current_time < self.final_time:
            self.one_step()
            n += 1
            self.current_time += self.time_step
            pbar.update(1)
            
            # Store solution snapshot in memory
            if not self._DF.is_save_final_time_only and n % save_every == 0:
                if save_idx < n_saves:
                    solution_h[save_idx, :] = self.sol[:, 0]
                    solution_q[save_idx, :] = self.sol[:, 1]
                    solution_time[save_idx] = self.current_time
                    save_idx += 1
        
        pbar.close()
        
        # Save final solution in memory if needed
        if self._DF.is_save_final_time_only:
            solution_h[save_idx, :] = self.sol[:, 0]
            solution_q[save_idx, :] = self.sol[:, 1]
            solution_time[save_idx] = self.current_time
        
        # Now write everything to HDF5 file in one go
        print(f"Writing results to file...")
        with h5py.File(h5_filename, 'w') as h5f:
            # Store mesh information
            cell_centers = self._mesh.get_cell_centers()
            topography = self._physics.get_topography()
            h5f.create_dataset('mesh/x', data=cell_centers)
            h5f.create_dataset('mesh/topography', data=topography)
            h5f.attrs['dx'] = self._mesh.get_space_step()
            h5f.attrs['n_cells'] = n_cells
            
            # Store simulation parameters
            h5f.attrs['initial_time'] = self.initial_time
            h5f.attrs['final_time'] = self.final_time
            h5f.attrs['time_step'] = self.time_step
            h5f.attrs['n_steps'] = n_steps
            h5f.attrs['save_frequency'] = save_every
            h5f.attrs['flux_scheme'] = flux_name
            h5f.attrs['time_scheme'] = self._DF.time_scheme
            h5f.attrs['gravity'] = self._DF.g
            
            # Write solution data (all at once)
            h5f.create_dataset('solution/h', data=solution_h)
            h5f.create_dataset('solution/q', data=solution_q)
            h5f.create_dataset('solution/time', data=solution_time)
            
            # Save exact solution if test case
            if self._DF.is_test_case:
                self._physics.build_exact_solution(self.current_time)
                exact_sol = self._physics.get_exact_solution()
                h5f.create_dataset('exact/h', data=exact_sol[:, 0])
                h5f.create_dataset('exact/q', data=exact_sol[:, 1])
                
                L2_error = self.compute_L2_error()
                L1_error = self.compute_L1_error()
                
                h5f.attrs['L2_error_h'] = L2_error[0]
                h5f.attrs['L2_error_q'] = L2_error[1]
                h5f.attrs['L1_error_h'] = L1_error[0]
                h5f.attrs['L1_error_q'] = L1_error[1]
        
        print(f"Results saved to: {h5_filename}")
    
    def compute_L2_error(self) -> np.ndarray:
        """Compute L2 error compared to exact solution.
        
        Returns:
            Array of [h_error, q_error]
        """
        error = np.zeros(2)
        exact_sol = self._physics.get_exact_solution()
        
        error[0] = np.linalg.norm(self.sol[:, 0] - exact_sol[:, 0])
        error[1] = np.linalg.norm(self.sol[:, 1] - exact_sol[:, 1])
        error *= self._DF.dx
        
        return error
    
    def compute_L1_error(self) -> np.ndarray:
        """Compute L1 error compared to exact solution.
        
        Returns:
            Array of [h_error, q_error]
        """
        error = np.zeros(2)
        exact_sol = self._physics.get_exact_solution()
        
        for i in range(self.sol.shape[0]):
            error[0] += abs(self.sol[i, 0] - exact_sol[i, 0])
            error[1] += abs(self.sol[i, 1] - exact_sol[i, 1])
        
        error *= self._DF.dx
        
        return error
    
    def get_solution(self) -> np.ndarray:
        """Get current solution array."""
        return self.sol
    
    def get_time_step(self) -> float:
        """Get time step size."""
        return self.time_step
    
    def get_initial_time(self) -> float:
        """Get initial time."""
        return self.initial_time
    
    def get_final_time(self) -> float:
        """Get final time."""
        return self.final_time
    
    def get_current_time(self) -> float:
        """Get current time."""
        return self.current_time


class ExplicitEuler(TimeScheme):
    """Explicit Euler time integration scheme."""
    
    def __init__(self, data_file: DataFile = None, mesh: Mesh = None, 
                 physics: Physics = None, fin_vol: FiniteVolume = None):
        """Initialize Explicit Euler scheme.
        
        Args:
            data_file: DataFile with parameters
            mesh: Mesh object
            physics: Physics object
            fin_vol: FiniteVolume flux scheme
        """
        super().__init__(data_file, mesh, physics, fin_vol)
    
    def one_step(self):
        """Perform one Explicit Euler time step (Numba-accelerated).
        
        Updates solution: Sol^{n+1} = Sol^n + dt * (F/dx + S)
        where F is the numerical flux and S is the source term.
        """
        dt = self.time_step
        dx = self._mesh.get_space_step()
        
        # Build flux vector and source term
        self._finVol.build_flux_vector(self.current_time, self.sol)
        self._physics.build_source_term(self.sol)
        
        # Get flux and source
        source = self._physics.get_source_term()
        flux_vector = self._finVol.get_flux_vector()
        
        # Update solution using JIT-compiled kernel
        self.sol = explicit_euler_step(self.sol, flux_vector, source, dt, dx)


class RK2(TimeScheme):
    """Runge-Kutta 2 (Heun's method) time integration scheme."""
    
    def __init__(self, data_file: DataFile = None, mesh: Mesh = None, 
                 physics: Physics = None, fin_vol: FiniteVolume = None):
        """Initialize RK2 scheme.
        
        Args:
            data_file: DataFile with parameters
            mesh: Mesh object
            physics: Physics object
            fin_vol: FiniteVolume flux scheme
        """
        super().__init__(data_file, mesh, physics, fin_vol)
    
    def one_step(self):
        """Perform one RK2 time step (Numba-accelerated).
        
        Updates solution using Heun's method:
        k1 = F(Sol^n)
        k2 = F(Sol^n + dt*k1)
        Sol^{n+1} = Sol^n + 0.5*dt*(k1 + k2)
        """
        dt = self.time_step
        dx = self._mesh.get_space_step()
        
        # Compute k1
        self._finVol.build_flux_vector(self.current_time, self.sol)
        self._physics.build_source_term(self.sol)
        flux_vector1 = self._finVol.get_flux_vector()
        source1 = self._physics.get_source_term()
        k1 = flux_vector1 / dx + source1
        
        # Compute k2
        sol_temp = self.sol + dt * k1
        self._physics.build_source_term(sol_temp)
        self._finVol.build_flux_vector(self.current_time + dt, sol_temp)
        source2 = self._physics.get_source_term()
        flux_vector2 = self._finVol.get_flux_vector()
        k2 = flux_vector2 / dx + source2
        
        # Update solution using JIT-compiled kernel
        self.sol = rk2_step(self.sol, k1, k2, dt)

