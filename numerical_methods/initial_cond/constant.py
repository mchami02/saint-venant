from .initial_cond import InitialCondition

class LWRConstantIC(InitialCondition):
    def __init__(self, nx: int, nt: int, value: float):
        super().__init__(nx, nt, ["h"])
        self.grid["h"][0, :] = value