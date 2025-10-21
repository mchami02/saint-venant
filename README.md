# 1D Saint-Venant Equations Solver

A Python implementation of a finite volume solver for the 1D shallow water (Saint-Venant) equations.

## Overview

This solver implements various numerical schemes for solving the 1D Saint-Venant equations:

```
∂h/∂t + ∂q/∂x = 0
∂q/∂t + ∂(q²/h + gh²/2)/∂x = -gh ∂z/∂x
```

where:
- `h` is the water height
- `q` is the discharge (q = hu)
- `g` is the gravitational acceleration
- `z` is the topography/bathymetry

## Features

### Numerical Fluxes
- **Lax-Friedrichs**: Simple and robust first-order flux
- **Rusanov**: Local Lax-Friedrichs with wave speed estimation
- **HLL**: Harten-Lax-van Leer approximate Riemann solver

### Time Integration
- **Explicit Euler**: First-order explicit time stepping
- **RK2**: Second-order Runge-Kutta (Heun's method)

### Spatial Discretization
- **First order**: Piecewise constant reconstruction
- **Second order**: MUSCL scheme with minmod slope limiter (TVD)

### Boundary Conditions
- Neumann (zero gradient)
- Wall (reflective)
- Imposed constant height
- Imposed constant discharge
- Data file (time-dependent from file)
- Periodic waves

### Initial Conditions
- Uniform height and discharge
- Dam break (wet/dry)
- Thacker test case
- Sine perturbation
- From file

### Topography
- Flat bottom
- Analytical bump
- Thacker parabolic
- From file (interpolated)

### Test Cases
Several analytical solutions are implemented for validation:
- Resting lake
- Subcritical flow over a bump
- Transcritical flow with/without shock
- Dam break (wet/dry)
- Thacker oscillating lake

## Installation

### Using uv (recommended)

```bash
cd saint-venant
uv run python main.py parameters.txt
```

### Using pip

```bash
cd saint-venant
pip install -r requirements.txt
python main.py parameters.txt
```

## Usage

Run the solver with a parameter file:

```bash
python main.py parameters.txt
```

Or with uv:

```bash
uv run python main.py parameters.txt
```

### Parameter File Format

The parameter file is a text file with key-value pairs. See `parameters.txt` for a complete example with all available options.

Key sections:
- **Numerical parameters**: Time scheme, numerical flux, spatial order
- **Mesh parameters**: Domain bounds, spatial step
- **Time parameters**: Start/end times, time step
- **Initial conditions**: Type and values
- **Boundary conditions**: Left and right BC types
- **Topography**: Type and data file
- **Output**: Results directory, save frequency, probes

### Example

```txt
# Time scheme: ExplicitEuler or RK2
TimeScheme
ExplicitEuler

# Numerical flux: LaxFriedrichs, Rusanov, or HLL
NumericalFlux
Rusanov

# Spatial order: 1 or 2
Order
2

# Domain
xmin
0.0
xmax
25.0
dx
0.25

# Time parameters
InitialTime
0.0
FinalTime
300.0
TimeStep
0.001

# Initial condition
InitialCondition
UniformHeightAndDischarge
InitialHeight
2.0
InitialDischarge
0.0

# Boundary conditions
LeftBoundaryCondition
ImposedConstantDischarge
RightBoundaryCondition
ImposedConstantHeight

# Topography
IsTopography
1
TopographyType
Bump
```

## Project Structure

```
saint-venant/
├── main.py              # Entry point
├── data_file.py         # Parameter file reader
├── mesh.py              # 1D mesh generation
├── physics.py           # Physics: IC, BC, fluxes, source terms
├── finite_volume.py     # Numerical flux schemes
├── time_scheme.py       # Time integration schemes
├── parameters.txt       # Example parameter file
├── pyproject.toml       # Project configuration
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Output

Results are saved in the directory specified by `ResultsDir` in the parameter file:

- `solution_<flux>_<n>.txt`: Solution snapshots at different times
- `topography.txt`: Topography profile
- `probe_<i>.txt`: Time series at probe locations (if configured)
- `solution_exacte.txt`: Exact solution (for test cases)
- `parameters.txt`: Copy of input parameters

### Output Format

Each solution file contains columns:
```
# x  H=h+z   h       u       q       Fr=|u|/sqrt(gh)
```

where:
- `x`: spatial coordinate
- `H`: free surface elevation
- `h`: water height
- `u`: velocity
- `q`: discharge
- `Fr`: Froude number

## Validation

The solver has been validated against analytical solutions for various test cases:

- **Resting lake**: Verifies well-balanced property (lake at rest should remain at rest)
- **Subcritical flow**: Steady flow over a bump
- **Transcritical flow**: Flow transitions through critical depth
- **Dam break**: Riemann problem with known analytical solution
- **Thacker**: Oscillating water in a parabolic basin

Run a test case by setting `IsTestCase = 1` and choosing the appropriate test in the parameter file.

## Physical Model

The code solves the 1D Saint-Venant equations, which are a hyperbolic system of conservation laws representing shallow water flow. The equations are derived from depth-averaging the Navier-Stokes equations under the shallow water approximation (horizontal length scale >> vertical length scale).

The system conserves:
1. Mass: `∂h/∂t + ∂q/∂x = 0`
2. Momentum: `∂q/∂t + ∂(q²/h + gh²/2)/∂x = S`

where `S = -gh ∂z/∂x` is the source term due to topography.

## Numerical Method

The solver uses:
1. **Finite volume discretization** in space
2. **Explicit time integration** (Euler or RK2)
3. **Approximate Riemann solvers** for numerical fluxes
4. **MUSCL reconstruction** with minmod limiter for second-order accuracy
5. **Source term treatment** using centered differences

## Dependencies

- Python >= 3.8
- NumPy >= 1.20.0

## Credits

This Python implementation is a translation of a C++ finite volume solver for shallow water equations, preserving all numerical methods and physical models.

## License

See LICENSE file for details.

## References

1. Toro, E. F. (2009). Riemann Solvers and Numerical Methods for Fluid Dynamics. Springer.
2. LeVeque, R. J. (2002). Finite Volume Methods for Hyperbolic Problems. Cambridge University Press.
3. Godunov, S. K. (1959). A difference method for numerical calculation of discontinuous solutions of the equations of hydrodynamics.
