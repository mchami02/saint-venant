from solver import Solver
from plotter import Plotter

def main():
    N_x = 100
    T = 1
    d_x = 0.01
    d_t = 0.001
    solver = Solver(N_x, T, d_x, d_t)
    left_boundary = 2.0
    right_boundary = 2.0
    h, u = solver.solve(left_boundary, right_boundary)
    plotter = Plotter(N_x, d_x, d_t, T)
    plotter.plot_density(h)
    plotter.animate_density(h)

if __name__ == "__main__":
    main()