from .boundary_cond import BoundaryCondition
from .constant import ConstantBoundary
from .ghost_cell import GhostCellBoundary

__all__ = ["BoundaryCondition", "ConstantBoundary", "GhostCellBoundary"]

def get_bc(bc_type: str, **kwargs) -> BoundaryCondition:
    if bc_type == "Constant":
        return ConstantBoundary(kwargs['left_value'], kwargs['right_value'])
    elif bc_type == "GhostCell":
        return GhostCellBoundary()
    else:
        raise ValueError(f"Invalid boundary condition type: {bc_type}")