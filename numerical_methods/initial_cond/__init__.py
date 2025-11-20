from .constant import LWRConstantIC
from .riemann import LWRRiemannProblem, SVERiemannProblem
from .piecewise import LWRPiecewiseConstant, SVPiecewiseConstant
from .initial_cond import InitialCondition

__all__ = ["InitialCondition", "LWRConstantIC", "LWRRiemannProblem", "SVERiemannProblem", "LWRPiecewiseConstant"]

def get_ic(ic_type: str, nx: int, nt: int, **kwargs) -> InitialCondition:
    if ic_type == "LWRConstant":
        return LWRConstantIC(nx, nt, **kwargs)
    elif ic_type == "LWRRiemann":
        return LWRRiemannProblem(nx, nt, **kwargs)
    elif ic_type == "SVERiemann":
        return SVERiemannProblem(nx, nt, **kwargs)
    elif ic_type == "LWRPiecewise":
        return LWRPiecewiseConstant(nx, nt, **kwargs)
    elif ic_type == "SVPiecewise":
        return SVPiecewiseConstant(nx, nt, **kwargs)
    else:
        raise ValueError(f"Invalid initial condition type: {ic_type}")