# 2D Euler Solver Module Structure

```
euler2d/
├── __init__.py              # Public API: generate_one, generate_n
├── physics.py               # EOS, primitive <-> conservative for (rho, u, v, p)
├── flux.py                  # HLLC/HLL/Rusanov with passive tangential scalar
├── boundary.py              # 2D ghost cells: extrap / periodic / wall
├── timestepper.py           # Unsplit FV + adaptive CFL sub-stepping (FE / SSP-RK3)
├── weno.py                  # Per-direction WENO-5 wrappers around src/euler/weno.py
├── initial_conditions.py    # sod_x/y, four_quadrant, liska_wendroff, random_piecewise(_batch)
└── structure.md             # This file
```

## Overview

2D compressible Euler equations solver using unsplit finite-volume method
with PyTorch. Riemann kernels ported from clawpack/riemann
`euler_1D_py.py`, extended with a passive tangential momentum scalar for
dimensional fluxes (PyClaw pattern). Benchmark ICs from Liska-Wendroff
(SIAM J. Sci. Comput. 2003).

Conservative variables: (rho, rho*u, rho*v, E)
Primitive variables:    (rho, u, v, p)

## Flux types

- `hllc`: HLLC (3-wave, contact-restoring) — default.
- `hll`: HLL (2-wave).
- `rusanov`: Local Lax-Friedrichs.

## Reconstruction

- `constant`: 1st order Godunov + Forward Euler.
- `weno5`: 5th order WENO + SSP-RK3 (per-direction reconstruction).

## Benchmark ICs

- `sod_x`, `sod_y`: 2D Sod shock tube aligned with an axis.
- `four_quadrant`: generic four-quadrant Riemann.
- `liska_wendroff(config)`: configurations 3, 4, 6 from Liska-Wendroff 2003.
