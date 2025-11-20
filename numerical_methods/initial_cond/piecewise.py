from .initial_cond import InitialCondition

class LWRPiecewiseConstant(InitialCondition):
    def __init__(self, nx: int, nt: int, values: list[float], steps: list[int]):
        assert len(values) == len(steps) + 1, "Values and steps must have the same length + 1"
        assert all(step < nx for step in steps), "Steps must be less than the number of cells"
        assert all(step >= 0 for step in steps), "Steps must be non-negative"
        assert sorted(steps) == steps, "Steps must be sorted in ascending order"
        super().__init__(nx, nt, ["h"])
        steps = [0] + steps + [nx]
        for i in range(len(steps) - 1):
            self.grid["h"][0, steps[i]:steps[i+1]] = values[i]

class SVPiecewiseConstant(InitialCondition):
    def __init__(self, nx: int, nt: int, values: list[float], steps: list[int]):
        assert len(values) == len(steps) + 1, "Values and steps must have the same length + 1"
        assert all(step < nx for step in steps), "Steps must be less than the number of cells"
        assert all(step >= 0 for step in steps), "Steps must be non-negative"
        assert sorted(steps) == steps, "Steps must be sorted in ascending order"
        super().__init__(nx, nt, ["h", "u"])
        steps = [0] + steps + [nx]
        for i in range(len(steps) - 1):
            self.grid["h"][0, steps[i]:steps[i+1]] = values[i]
