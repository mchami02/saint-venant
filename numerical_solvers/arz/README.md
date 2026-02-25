# ARZ Traffic Flow Solver

A modular, fully-vectorized solver for the **Aw-Rascle-Zhang (ARZ)** traffic flow equations, implemented in PyTorch.

The ARZ system is a second-order macroscopic traffic model that couples density and velocity through a pressure law:

$$
\partial_t \rho + \partial_x (\rho v) = 0
$$
$$
\partial_t (\rho w) + \partial_x (\rho w v) = 0
$$

where $w = v + p(\rho)$ is the Lagrangian marker (conserved along characteristics) and $p(\rho) = \rho^\gamma$ is the anticipation pressure.

## Quick Start

```python
from numerical_solvers.arz import generate_one, generate_n, riemann
import torch

# Solve a single Riemann problem
nx, dx, dt, nt = 200, 0.005, 0.0005, 2000
x = torch.arange(nx) * dx
rho0, v0 = riemann(x, rho_left=0.8, rho_right=0.2, v0=0.1)
result = generate_one(rho0, v0, dx=dx, dt=dt, nt=nt, gamma=1.0)

result["rho"]  # (nt+1, 200) — density over space-time
result["v"]    # (nt+1, 200) — velocity over space-time

# Generate a batch of random samples
data = generate_n(100, k=3, nx=nx, dx=dx, dt=dt, nt=nt, seed=42)

data["rho"]  # (100, nt+1, 200) — batch of density fields
```

## API Reference

### `generate_one(rho0, v0, *, ...)`

Solve one ARZ problem from given initial conditions.

```python
generate_one(
    rho0,                          # (nx,) tensor — initial density
    v0,                            # (nx,) tensor — initial velocity
    *,
    dx,                            # cell width
    dt,                            # time step
    nt,                            # number of time steps
    gamma=1.0,                     # pressure exponent p(rho) = rho^gamma
    bc_type="zero_gradient",       # boundary condition type
    flux_type="hll",               # numerical flux
    reconstruction="weno5",        # spatial reconstruction
    bc_left=None,                  # (rho, v) for left Dirichlet BC
    bc_right=None,                 # (rho, v) for right Dirichlet BC
    bc_left_time=None,             # callable(t) -> (rho, v) for time-varying inflow
)
```

**Returns** a dict:

| Key | Shape | Description |
|---|---|---|
| `rho` | `(nt+1, nx)` | Density |
| `v` | `(nt+1, nx)` | Velocity |
| `w` | `(nt+1, nx)` | Lagrangian marker $w = v + p(\rho)$ |
| `x` | `(nx,)` | Spatial grid |
| `t` | `(nt+1,)` | Time grid |
| `dx` | `float` | Cell width |
| `dt` | `float` | Time step |
| `nt` | `int` | Number of time steps |

`nx` is inferred from `len(rho0)` — no need to pass it separately.

### `generate_n(n, k, *, ...)`

Generate `n` samples with random `k`-piecewise-constant initial conditions.

```python
generate_n(
    n,                             # number of samples
    k,                             # pieces per piecewise-constant IC
    *,
    nx=200,                        # spatial resolution
    dx,                            # cell width
    dt,                            # time step
    nt,                            # number of time steps
    gamma=1.0,                     # pressure exponent p(rho) = rho^gamma
    bc_type="zero_gradient",
    flux_type="hll",
    reconstruction="weno5",
    rho_range=(0.1, 1.0),          # (min, max) for sampled density values
    v_range=(0.0, 1.0),            # (min, max) for sampled velocity values
    seed=None,                     # random seed for reproducibility
    show_progress=True,            # show tqdm progress bar
    device="cpu",                  # torch device
)
```

**Returns** a dict with the same keys as `generate_one`, but with an extra batch dimension: `rho` has shape `(n, nt+1, nx)`.

## Initial Condition Helpers

All IC functions return `(rho0, v0)` tensors — physical velocity, not the conserved variable `w`.

```python
from numerical_solvers.arz import riemann, three_region, from_steps, random_piecewise

# Two-region Riemann problem
rho0, v0 = riemann(x, rho_left=0.8, rho_right=0.2, v0=0.1, x_split=0.5)

# Three-region piecewise constant
rho0, v0 = three_region(x, rho_left=0.3, rho_mid=0.8, rho_right=0.2, x1=0.2, x2=0.5)

# Arbitrary piecewise constant from step specifications
rho0, v0 = from_steps(
    x,
    rho_steps=[(0.3, 0.5), (0.7, 0.9), (1.1, 0.2)],
    v_steps=[(0.5, 0.3), (1.1, 0.8)],
)

# Random k-piecewise-constant
rng = torch.Generator().manual_seed(42)
rho0, v0 = random_piecewise(x, k=4, rng=rng, rho_range=(0.1, 1.0), v_range=(0.0, 1.0))
```

## Solver Options

### Boundary Conditions (`bc_type`)

| Value | Description |
|---|---|
| `"zero_gradient"` | Extrapolate interior values (default, non-reflective) |
| `"periodic"` | Wrap-around boundaries |
| `"inflow_outflow"` | Fixed left state (`bc_left`), zero-gradient right |
| `"dirichlet"` | Fixed states on both sides (`bc_left`, `bc_right`) |
| `"time_varying_inflow"` | Time-dependent left state via `bc_left_time(t)` callable |

### Numerical Flux (`flux_type`)

| Value | Description |
|---|---|
| `"hll"` | Harten-Lax-van Leer — less diffusive (default) |
| `"rusanov"` | Local Lax-Friedrichs — most diffusive, most stable |

### Spatial Reconstruction (`reconstruction`)

| Value | Order | Time Integration |
|---|---|---|
| `"constant"` | 1st (Godunov) | Forward Euler |
| `"weno5"` | 5th (WENO) | SSP-RK3 |

## File Overview

```
numerical_solvers/arz/
├── __init__.py             # Public API: generate_one, generate_n
├── physics.py              # pressure, dp_drho, eigenvalues
├── flux.py                 # Vectorized rusanov and hll fluxes
├── weno.py                 # WENO-5 reconstruction (equation-agnostic)
├── boundary.py             # Ghost cell boundary conditions
├── timestepper.py          # Solve loop (Forward Euler / SSP-RK3)
├── initial_conditions.py   # IC generators
└── README.md
```

### Module Dependencies

```
__init__.py → timestepper.py → flux.py → physics.py
                             → weno.py
                             → boundary.py → physics.py
           → initial_conditions.py
           → physics.py
```
