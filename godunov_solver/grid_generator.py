import numpy as np
from .grid import Grid
from .solve_class import Solver


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
    
    def __init__(self, solver: Solver, nx: int, nt: int, dx: float, dt: float):
        self.solver = solver
        self.nx = nx
        self.nt = nt
        self.dx = dx
        self.dt = dt
    
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
        
        # Create grid
        grid = Grid(
            nx=self.nx,
            nt=self.nt,
            dx=self.dx,
            dt=self.dt,
            initial_condition=initial_condition,
            boundary_condition=boundary_condition,
            **{**ic_kwargs, **bc_kwargs}
        )
        
        # Solve
        self.solver.solve(grid)
        
        return grid
    
    def _generate_random_ic(self):
        """Randomly select an initial condition type based on the solver."""
        from .solve_class import SVESolver, Godunov
        
        if isinstance(self.solver, SVESolver):
            # SVESolver needs ICs with both h and u values
            ic_types = ["SVERiemannProblem"]
        elif isinstance(self.solver, Godunov):
            # Godunov (LWR) needs ICs with only density/height
            ic_types = ["RiemannProblem", "Constant", "PiecewiseConstant"]
        else:
            # Default fallback
            ic_types = ["Constant"]
        
        return np.random.choice(ic_types)
    
    def _generate_random_ic_kwargs(self, ic_type):
        """Generate random parameters for the initial condition."""
        if ic_type == "SVERiemannProblem":
            return {
                'hL': np.random.uniform(0.5, 2.0),
                'hR': np.random.uniform(0.5, 2.0),
                'uL': np.random.uniform(-1.0, 1.0),
                'uR': np.random.uniform(-1.0, 1.0),
                'step': np.random.randint(self.nx // 4, 3 * self.nx // 4)
            }
        elif ic_type == "RiemannProblem":
            return {
                'uL': np.random.rand(),
                'uR': np.random.rand(),
                'step': np.random.randint(0, self.nx)
            }
        elif ic_type == "Constant":
            return {
                'value': np.random.rand()
            }
        elif ic_type == "PiecewiseConstant":
            num_pieces = np.random.randint(2, 5)
            steps = sorted(np.random.choice(range(1, self.nx), num_pieces-1, replace=False).tolist())
            values = np.random.rand(num_pieces).tolist()
            return {
                'values': values,
                'steps': [0] + steps
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

