from .initial_cond import InitialCondition
import numpy as np

class LWRRiemannProblem(InitialCondition):
    def __init__(self, nx: int, nt: int, uL: float, uR: float, step: int):
        assert step < nx, "Step must be less than the number of cells"
        super().__init__(nx, nt, ["h"])
        self.grid["h"][0, :step] = uL
        self.grid["h"][0, step:] = uR

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