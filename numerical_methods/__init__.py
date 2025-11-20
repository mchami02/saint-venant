"""
Numerical Methods Package

This package provides numerical methods for solving PDEs,
including grid management, solvers, initial conditions, boundary conditions,
flux functions, and plotting utilities.
"""

# Core modules
from .grid import Grid
from .grid_generator import GridGenerator
from .plotter import plot_grid_density, animate_density, plot_comparison

# Boundary conditions
from .boundary_cond import (
    BoundaryCondition,
    ConstantBoundary,
    GhostCellBoundary,
    get_bc
)

# Flux functions
from .flux import (
    Flux,
    Greenshields,
    Triangular
)

# Initial conditions
from .initial_cond import (
    InitialCondition,
    LWRConstantIC,
    LWRRiemannProblem,
    SVERiemannProblem,
    LWRPiecewiseConstant,
    get_ic
)

# Solvers
from .solvers import (
    Solver,
    Godunov,
    LWRRiemannSolver,
    SVERiemannSolver
)

__all__ = [
    # Core modules
    "Grid",
    "GridGenerator",
    
    # Plotting functions
    "plot_grid_density",
    "animate_density",
    "plot_comparison",
    
    # Boundary conditions
    "BoundaryCondition",
    "ConstantBoundary",
    "GhostCellBoundary",
    "get_bc",
    
    # Flux functions
    "Flux",
    "Greenshields",
    "Triangular",
    
    # Initial conditions
    "InitialCondition",
    "LWRConstantIC",
    "LWRRiemannProblem",
    "SVERiemannProblem",
    "LWRPiecewiseConstant",
    "get_ic",
    
    # Solvers
    "Solver",
    "Godunov",
    "LWRRiemannSolver",
    "SVERiemannSolver",
]

