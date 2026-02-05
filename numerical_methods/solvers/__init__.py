from .solver import Solver
from .godunov import GodunovSolver, LWRRiemannSolver, SVERiemannSolver

# Alias for backward compatibility
Godunov = GodunovSolver

__all__ = ["Solver", "Godunov", "GodunovSolver", "LWRRiemannSolver", "SVERiemannSolver"]

