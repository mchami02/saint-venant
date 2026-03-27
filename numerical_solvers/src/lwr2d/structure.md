# numerical_solvers/lwr2d/ — Structure

2D LWR (Lighthill-Whitham-Richards) traffic flow solver using unsplit first-order Godunov finite volume with Forward Euler time stepping (PyTorch).

## Mathematical Model

Scalar conservation law in 2D:
```
∂ρ/∂t + ∂f(ρ)/∂x + ∂g(ρ)/∂y = 0
```
with Greenshields or triangular flux, applied independently in x and y (potentially with different max speeds).

## Files

| File | Purpose |
|---|---|
| `__init__.py` | Public API: `generate_one`, `generate_n` |
| `physics.py` | Greenshields & triangular flux functions, derivatives, CFL utility |
| `flux.py` | Exact Godunov numerical flux for scalar conservation laws |
| `boundary.py` | 2D ghost cell BCs (zero_gradient, periodic, dirichlet) |
| `initial_conditions.py` | IC generators: riemann_x/y, four_quadrant, gaussian_bump, random_piecewise |
| `timestepper.py` | Main 2D solve loop (Forward Euler, unsplit Godunov) |
| `test_api_lwr2d.ipynb` | Interactive notebook testing the 2D LWR solver API |
| `structure.md` | This file |

## Tests

| File | What it tests |
|---|---|
| `tests/test_mass_conservation.py` | Periodic BC conserves mass; zero-gradient doesn't blow up |
| `tests/test_1d_consistency.py` | x/y-Riemann ICs produce row/column-identical solutions |
| `tests/test_symmetry.py` | Diagonal symmetry for symmetric IC; uniform stays uniform |
| `tests/test_steady_state.py` | rho=0, rho=rho_max, constant rho are steady states |
| `tests/test_cfl.py` | CFL utility correctness; solver stability at CFL-limited dt |
