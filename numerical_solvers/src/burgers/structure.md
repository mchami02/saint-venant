# Burgers Solver Module Structure

```
burgers/
├── __init__.py              # Public API: generate_one, generate_n
├── physics.py               # flux(u) = u^2/2, max_wave_speed(u) = |u|
├── flux.py                  # Godunov (with transonic entropy fix), Rusanov
├── boundary.py              # Ghost cells: extrap / periodic / wall (sign flip)
├── timestepper.py           # Time integration: Forward Euler / SSP-RK3
├── weno.py                  # Re-export of src/euler/weno.py (equation-agnostic)
├── initial_conditions.py    # IC generators: riemann, from_steps, random_piecewise(_batch)
└── structure.md             # This file
```

## Overview

1D inviscid Burgers equation solver using finite volume method with PyTorch.
Riemann flux ported from clawpack/riemann `burgers_1D_py.py`.

Conserved variable: `u` (scalar).
Flux: `f(u) = 0.5 * u^2`.

## Flux types

- `godunov`: exact convex Godunov flux with transonic entropy fix at `u = 0` — default.
- `rusanov`: Local Lax-Friedrichs.

## Reconstruction

- `constant`: 1st order Godunov + Forward Euler.
- `weno5`: 5th order WENO + SSP-RK3 (reuses `src/euler/weno.py`).
