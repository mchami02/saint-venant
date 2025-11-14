from abc import ABC, abstractmethod
import numpy as np

class InitialCondition(ABC):
    def __init__(self, nx: int, nt: int, vals: list[str]):
        self.grid = {
            v : np.zeros((nt, nx)) for v in vals
        }
        self.nx = nx
        self.nt = nt

    def get_grid(self) -> np.ndarray:
        return self.grid

class ConstantIC(InitialCondition):
    def __init__(self, nx: int, nt: int, value: float):
        super().__init__(nx, nt, ["h"])
        self.grid["h"][0, :] = value

class RiemannProblem(InitialCondition):
    def __init__(self, nx: int, nt: int, uL: float, uR: float, step: int):
        assert step < nx, "Step must be less than the number of cells"
        super().__init__(nx, nt, ["h"])
        self.grid["h"][0, :step] = uL
        self.grid["h"][0, step:] = uR

class PiecewiseConstant(InitialCondition):
    def __init__(self, nx: int, nt: int, values: list[float], steps: list[int]):
        assert len(values) == len(steps), "Values and steps must have the same length"
        assert all(step < nx for step in steps), "Steps must be less than the number of cells"
        assert all(step >= 0 for step in steps), "Steps must be non-negative"
        assert sorted(steps) == steps, "Steps must be sorted in ascending order"
        super().__init__(nx, nt, ["h"])

        for i, value in enumerate(values):
            if i == len(values) - 1:
                self.grid["h"][0, steps[i]:] = value
            else:
                self.grid["h"][0, steps[i]:steps[i+1]] = value

class SVERiemannProblem(InitialCondition):
    def __init__(self, nx: int, nt: int, hL: float = np.random.rand(), hR: float = np.random.rand(), uL: float = 0, uR: float = 0, step: int = None):
        if step is None:
            step = nx//2
        assert step < nx, "Step must be less than the number of cells"
        super().__init__(nx, nt, ["h", "u"])
        self.grid["h"][0, :step] = hL
        self.grid["h"][0, step:] = hR
        self.grid["u"][0, :step] = uL
        self.grid["u"][0, step:] = uR

def get_ic(ic_type: str, nx, nt, **kwargs) -> InitialCondition:
    if ic_type == "Constant":
        return ConstantIC(nx, nt, kwargs['value'])
    elif ic_type == "RiemannProblem":
        return RiemannProblem(nx, nt, kwargs['uL'], kwargs['uR'], kwargs['step'])
    elif ic_type == "PiecewiseConstant":
        return PiecewiseConstant(nx, nt, kwargs['values'], kwargs['steps'])
    elif ic_type == "SVERiemannProblem":
        return SVERiemannProblem(nx, nt, **kwargs)
    else:
        raise ValueError(f"Invalid initial condition type: {ic_type}")