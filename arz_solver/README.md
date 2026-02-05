# ARZ solver

Clean interface for the Aw–Rascle–Zhang (ARZ) traffic flow model: Rusanov flux, finite-volume time stepping, and configurable boundary conditions.

## Quick use

```python
from arz_solver import ARZConfig, run, plot_results, initial_condition_riemann

config = ARZConfig(nx=200, L=1.0, T=1.0, gamma=1.0)
rho0, w0 = initial_condition_riemann(config.x, rho_left=0.8, rho_right=0.2, v0=0.1)
rho_hist, w_hist, v_hist = run(rho0, w0, config, bc_type="zero_gradient")
plot_results(rho_hist, w_hist, v_hist, config)
```

## Configuration

- **ARZConfig**: `nx`, `L`, `T`, `gamma`, `cfl_factor`, optional `bc_left` / `bc_right` `(rho, v)` for Dirichlet/inflow, optional `bc_left_time(t)` for time-varying inflow.
- **Boundary types**: `"zero_gradient"`, `"periodic"`, `"inflow_outflow"`, `"time_varying_inflow"`, `"dirichlet"`.

## Initial conditions

- **initial_condition_from_steps**: pass lists of (x_end, value) for piecewise-constant ρ and optional v:
  ```python
  # ρ = 0.3 on [0, 0.2), 0.8 on [0.2, 0.5), 0.2 on [0.5, 1]; v = 0.1 constant
  rho0, w0 = initial_condition_from_steps(
      config.x,
      rho_steps=[(0.2, 0.3), (0.5, 0.8), (1.0, 0.2)],
      default_v=0.1,
      gamma=config.gamma,
  )
  # Optional v_steps: same format, e.g. v_steps=[(0.5, 0.1), (1.0, 0.2)]
  ```
- **initial_condition_riemann**: one discontinuity at `x_split`.

Run from the project root so `from arz_solver import ...` works.
