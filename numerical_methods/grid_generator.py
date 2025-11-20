import numpy as np
from .grid import Grid
from .solvers.solver import Solver
from .solvers import Godunov, LWRRiemannSolver, SVERiemannSolver

class GridGenerator:
    """
    A class to generate grids with different initial and boundary conditions.
    
    Parameters
    ----------
    solver : Solver
        The solver instance to use for generating solutions
    nx : int
        Number of spatial cells
    nt : int
        Number of time steps
    dx : float
        Spatial step size
    dt : float
        Time step size
    """
    
    def __init__(self, solver: Solver, nx: int, nt: int, dx: float, dt: float, randomize=True, CFL=0.2):
        self.solver = solver
        self.nx = nx
        self.nt = nt
        self.dx = dx
        self.dt = dt
        self.randomize = randomize
        self.CFL = CFL

    def __call__(self, initial_condition=None, boundary_condition=None, ic_kwargs=None, bc_kwargs=None):
        """
        Generate a grid and solve it using the stored solver.
        
        Parameters
        ----------
        initial_condition : str, optional
            Type of initial condition. If None, will generate a random IC appropriate for the solver.
            For SVESolver: "SVERiemannProblem"
            For Godunov: "Constant", "RiemannProblem", "PiecewiseConstant"
        boundary_condition : str, optional
            Type of boundary condition. If None, defaults to "GhostCell".
            Options: "Constant", "GhostCell"
        ic_kwargs : dict, optional
            Additional keyword arguments for the initial condition
        bc_kwargs : dict, optional
            Additional keyword arguments for the boundary condition
            
        Returns
        -------
        Grid
            The solved grid
        """
        if ic_kwargs is None:
            ic_kwargs = {}
        if bc_kwargs is None:
            bc_kwargs = {}
        
        # Handle initial conditions
        if initial_condition is None:
            initial_condition = self._generate_random_ic()
            ic_kwargs = self._generate_random_ic_kwargs(initial_condition)
        
        # Handle boundary conditions
        if boundary_condition is None:
            boundary_condition = "GhostCell"
        
        nx = self.nx
        nt = self.nt
        if self.randomize:
            dx = np.random.uniform(self.dx / 2, self.dx * 2)
            dt = np.random.uniform(self.dt / 2, min(self.dt * 2, self.CFL * dx))
        else:
            dx = self.dx
            dt = self.dt
        # Create grid
        grid = Grid(
            nx=nx,
            nt=nt,
            dx=dx,
            dt=dt,
            initial_condition=initial_condition,
            boundary_condition=boundary_condition,
            **{**ic_kwargs, **bc_kwargs}
        )
        
        # Solve
        self.solver.solve(grid)
        
        return grid
    
    def _generate_random_ic(self):
        """Randomly select an initial condition type based on the solver."""
        
        if isinstance(self.solver.riemann, SVERiemannSolver):
            # SVESolver needs ICs with both h and u values
            ic_types = ["SVERiemann"]
        elif isinstance(self.solver.riemann, LWRRiemannSolver):
            # Godunov (LWR) needs ICs with only density/height
            ic_types = ["LWRPiecewise"]
        else:
            # Default fallback
            ic_types = ["LWRConstant"]
        
        return np.random.choice(ic_types)
    
    def _generate_random_ic_kwargs(self, ic_type):
        """Generate random parameters for the initial condition."""
        if ic_type == "SVERiemann":
            return {
                'hL': np.random.uniform(0.5, 2.0),
                'hR': np.random.uniform(0.5, 2.0),
                'uL': np.random.uniform(-1.0, 1.0),
                'uR': np.random.uniform(-1.0, 1.0),
                'step': np.random.randint(self.nx // 4, 3 * self.nx // 4)
            }
        elif ic_type == "LWRRiemann":
            return {
                'uL': np.random.rand(),
                'uR': np.random.rand(),
                'step': np.random.randint(0, self.nx)
            }
        elif ic_type == "LWRConstant":
            return {
                'value': np.random.rand()
            }
        elif ic_type == "LWRPiecewise":
            num_pieces = np.random.randint(2, 5)
            steps = sorted(np.random.choice(range(self.nx), num_pieces - 1, replace=False).tolist())
            values = np.random.rand(num_pieces).tolist()
            return {
                'values': values,
                'steps': steps
            }
        else:
            return {}
    
    def _generate_random_bc_kwargs(self):
        """Generate random parameters for constant boundary conditions."""
        return {
            'left_value': np.random.uniform(0.5, 2.0),
            'right_value': np.random.uniform(0.5, 2.0)
        }
    
    def generate_batch(self, n_samples, **kwargs):
        """
        Generate a batch of grids.
        
        Parameters
        ----------
        n_samples : int
            Number of grids to generate
        **kwargs
            Additional arguments passed to __call__
            
        Returns
        -------
        list[Grid]
            List of solved grids
        """
        return [self.__call__(**kwargs) for _ in range(n_samples)]

