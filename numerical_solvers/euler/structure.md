# Euler Solver Module Structure

```
euler/
├── __init__.py              # Public API: generate_one, generate_n
├── physics.py               # EOS, primitive <-> conservative conversions
├── flux.py                  # Riemann solvers: HLLC, HLL, Rusanov (ported from clawpack)
├── boundary.py              # Ghost cell BCs: extrap, periodic, wall
├── timestepper.py           # Time integration: Forward Euler / SSP-RK3
├── weno.py                  # WENO-5 spatial reconstruction
├── initial_conditions.py    # IC generators: sod, riemann, from_steps, random_piecewise
└── structure.md             # This file
```

## Overview

1D compressible Euler equations solver using finite volume method with PyTorch.
Riemann solver algorithms ported from clawpack/riemann `euler_1D_py.py`.

Conservative variables: (rho, rho*u, E)
Primitive variables: (rho, u, p)
EOS: p = (gamma - 1) * (E - 0.5 * rho * u^2)

## Flux types

- `hllc`: HLLC (3-wave, contact-restoring) — default, recommended
- `hll`: HLL (2-wave)
- `rusanov`: Local Lax-Friedrichs

## Reconstruction

- `constant`: 1st order Godunov + Forward Euler
- `weno5`: 5th order WENO + SSP-RK3
