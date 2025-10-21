# Parameters Guide

Complete reference for all configuration parameters in `parameters.json`.

## Table of Contents

1. [Output Settings](#output-settings)
2. [Probes](#probes)
3. [Test Cases](#test-cases)
4. [Initial Conditions](#initial-conditions)
5. [Mesh Parameters](#mesh-parameters)
6. [Numerical Schemes](#numerical-schemes)
7. [Time Parameters](#time-parameters)
8. [Physics](#physics)
9. [Boundary Conditions](#boundary-conditions)
10. [Topography](#topography)

---

## Output Settings

**Note**: All results are automatically saved in the `results/` directory. This cannot be changed.

### `save_final_time_only`
- **Type**: `boolean`
- **Default**: `false`
- **Description**: 
  - `true`: Save only the final time step
  - `false`: Save solution snapshots at regular intervals
- **Example**: `false`

### `save_frequency`
- **Type**: `integer`
- **Default**: `1`
- **Description**: Save solution every N timesteps (ignored if `save_final_time_only` is `true`)
- **Example**: `500` (saves every 500 timesteps)
- **Note**: Higher values = less frequent saves = smaller output files

---

## Probes

### `probes`
- **Type**: `array of objects`
- **Default**: `[]`
- **Description**: Water height measurement probes at specific locations
- **Format**: Each probe has:
  - `ref` (integer): Reference number for the probe
  - `position` (float): Physical x-position in meters
- **Example**:
  ```json
  "probes": [
    {"ref": 1, "position": 5.0},
    {"ref": 2, "position": 10.0},
    {"ref": 3, "position": 15.0}
  ]
  ```
- **Note**: Probe data is saved as time series in the HDF5 file

---

## Test Cases

### `is_test_case`
- **Type**: `boolean`
- **Default**: `false`
- **Description**: Whether this simulation has a known analytical solution
- **Example**: `true`

### `test_case`
- **Type**: `string`
- **Default**: `"None"`
- **Description**: Name of the test case (only used if `is_test_case` is `true`)
- **Options**:
  - `"RestingLake"` - Lake at rest (well-balanced test)
  - `"SubcriticalFlow"` - Steady subcritical flow over a bump
  - `"TranscriticalFlowWithoutShock"` - Flow transitions through critical depth
  - `"TranscriticalFlowWithShock"` - Transcritical with hydraulic jump
  - `"DamBreakWet"` - Dam break into wet domain (Riemann problem)
  - `"DamBreakDry"` - Dam break into dry domain
  - `"Thacker"` - Oscillating water in parabolic basin
  - `"None"` - No test case
- **Example**: `"SubcriticalFlow"`

---

## Initial Conditions

### `initial_condition`
- **Type**: `string`
- **Default**: `"none"`
- **Description**: Type of initial condition
- **Options**:
  - `"UniformHeightAndDischarge"` - Constant height and discharge
  - `"DamBreakWet"` - Dam break setup (h_left=2m, h_right=1m)
  - `"DamBreakDry"` - Dam break into dry bed (h_left=2m, h_right=0m)
  - `"Thacker"` - Thacker test case initial condition
  - `"SinePerturbation"` - Sinusoidal perturbation
  - `"File"` - Read from file
- **Example**: `"UniformHeightAndDischarge"`

### `initial_height`
- **Type**: `float` (meters)
- **Default**: `0.0`
- **Description**: Initial water surface elevation H = h + z
- **Example**: `2.0`
- **Used with**: `UniformHeightAndDischarge`

### `initial_discharge`
- **Type**: `float` (m²/s)
- **Default**: `0.0`
- **Description**: Initial discharge q = h*u
- **Example**: `0.0`
- **Used with**: `UniformHeightAndDischarge`

### `init_file`
- **Type**: `string`
- **Default**: `""`
- **Description**: Path to file containing initial conditions
- **Example**: `"results/solution_LF.h5"`
- **Used with**: `"File"` initial condition
- **Format**: Must be a text file with columns: x, H, h, u, q

---

## Mesh Parameters

### `xmin`
- **Type**: `float` (meters)
- **Default**: `0.0`
- **Description**: Minimum x-coordinate of computational domain
- **Example**: `0.0`

### `xmax`
- **Type**: `float` (meters)
- **Default**: `1.0`
- **Description**: Maximum x-coordinate of computational domain
- **Example**: `25.0`

### `dx`
- **Type**: `float` (meters)
- **Default**: `0.1`
- **Description**: Target spatial step size
- **Example**: `0.25`
- **Note**: Will be adjusted slightly to fit exactly in `[xmin, xmax]`
- **Grid resolution**: Number of cells = `(xmax - xmin) / dx`

---

## Numerical Schemes

### `numerical_flux`
- **Type**: `string`
- **Default**: `"LaxFriedrichs"`
- **Description**: Numerical flux scheme for solving the Riemann problem
- **Options**:
  - `"LaxFriedrichs"` - Lax-Friedrichs (simple, robust, dissipative)
  - `"Rusanov"` - Local Lax-Friedrichs (less dissipative)
  - `"HLL"` - Harten-Lax-van Leer (accurate, handles shocks well)
- **Example**: `"Rusanov"`
- **Recommendation**: Use `"HLL"` for best accuracy

### `scheme_order`
- **Type**: `integer`
- **Default**: `1`
- **Description**: Spatial order of accuracy
- **Options**:
  - `1` - First order (piecewise constant reconstruction)
  - `2` - Second order (MUSCL with minmod limiter, TVD)
- **Example**: `2`
- **Recommendation**: Use `2` for better accuracy (less numerical diffusion)

### `time_scheme`
- **Type**: `string`
- **Default**: `"ExplicitEuler"`
- **Description**: Time integration method
- **Options**:
  - `"ExplicitEuler"` - First order explicit Euler
  - `"RK2"` - Second order Runge-Kutta (Heun's method)
- **Example**: `"RK2"`
- **Recommendation**: Use `"RK2"` for better temporal accuracy

---

## Time Parameters

### `initial_time`
- **Type**: `float` (seconds)
- **Default**: `0.0`
- **Description**: Start time of simulation
- **Example**: `0.0`

### `final_time`
- **Type**: `float` (seconds)
- **Default**: `1.0`
- **Description**: End time of simulation
- **Example**: `100.0`

### `time_step`
- **Type**: `float` (seconds)
- **Default**: `0.01`
- **Description**: Time step size Δt
- **Example**: `0.02`
- **Important**: Must satisfy CFL condition for stability:
  ```
  CFL = (|u| + √(gh)) * dt / dx < 1
  ```
- **Typical values**:
  - `dt = 0.001` - Very safe but slow (300,000 steps for 300s)
  - `dt = 0.01` - Good balance (30,000 steps for 300s)
  - `dt = 0.02` - Faster (15,000 steps for 300s)
  - `dt = 0.05` - Aggressive but often stable (6,000 steps for 300s)

### `CFL`
- **Type**: `float`
- **Default**: `0.9`
- **Description**: CFL number (currently not used for adaptive timestepping)
- **Example**: `0.9`
- **Note**: For future implementation of adaptive time stepping

---

## Physics

### `gravity`
- **Type**: `float` (m/s²)
- **Default**: `9.81`
- **Description**: Gravitational acceleration
- **Example**: `9.81`
- **Note**: 
  - Earth: 9.81 m/s²
  - Moon: 1.62 m/s²
  - Mars: 3.71 m/s²

---

## Boundary Conditions

### Left Boundary

#### `left_BC`
- **Type**: `string`
- **Default**: `"Neumann"`
- **Description**: Type of left boundary condition
- **Options**:
  - `"Neumann"` - Zero-gradient (∂u/∂x = 0, ∂h/∂x = 0)
  - `"Wall"` - Reflective wall (u = 0)
  - `"ImposedConstantHeight"` - Impose constant water height
  - `"ImposedConstantDischarge"` - Impose constant discharge
  - `"DataFile"` - Time-dependent from file
  - `"PeriodicWaves"` - Sinusoidal waves
- **Example**: `"ImposedConstantDischarge"`

#### `left_BC_imposed_height`
- **Type**: `float` (meters)
- **Default**: `0.0`
- **Description**: Imposed water surface elevation at left boundary
- **Example**: `0.1`
- **Used with**: `"ImposedConstantHeight"` or supercritical `"ImposedConstantDischarge"`

#### `left_BC_imposed_discharge`
- **Type**: `float` (m²/s)
- **Default**: `0.0`
- **Description**: Imposed discharge at left boundary
- **Example**: `4.42`
- **Used with**: `"ImposedConstantDischarge"` or supercritical `"ImposedConstantHeight"`

#### `left_BC_data_file`
- **Type**: `string`
- **Default**: `""`
- **Description**: CSV file with time-dependent boundary data
- **Example**: `"exp_data/inflow.csv"`
- **Used with**: `"DataFile"`
- **Format**: CSV with columns: time, height

### Right Boundary

#### `right_BC`
- **Type**: `string`
- **Default**: `"Neumann"`
- **Description**: Type of right boundary condition
- **Options**: Same as `left_BC`
- **Example**: `"ImposedConstantHeight"`

#### `right_BC_imposed_height`
- **Type**: `float` (meters)
- **Default**: `0.0`
- **Description**: Imposed water surface elevation at right boundary
- **Example**: `2.0`

#### `right_BC_imposed_discharge`
- **Type**: `float` (m²/s)
- **Default**: `0.0`
- **Description**: Imposed discharge at right boundary
- **Example**: `1.0`

#### `right_BC_data_file`
- **Type**: `string`
- **Default**: `""`
- **Description**: CSV file with time-dependent boundary data
- **Example**: `"exp_data/outflow.csv"`

### Boundary Condition Notes

The solver automatically detects subcritical vs supercritical flow:
- **Subcritical (Fr < 1)**:
  - One characteristic enters domain → impose 1 condition
  - `ImposedConstantHeight`: imposes height, computes discharge
  - `ImposedConstantDischarge`: imposes discharge, computes height
- **Supercritical (Fr > 1)**:
  - All characteristics enter domain → impose all conditions
  - Must specify both height and discharge

---

## Topography

### `is_topography`
- **Type**: `boolean`
- **Default**: `false`
- **Description**: Whether topography/bathymetry is present
- **Example**: `true`

### `topography_type`
- **Type**: `string`
- **Default**: `"FlatBottom"`
- **Description**: Type of bottom topography
- **Options**:
  - `"FlatBottom"` - Flat bed (z = 0)
  - `"Bump"` - Analytical bump (parabola between x=8m and x=12m)
  - `"Thacker"` - Parabolic basin for Thacker test
  - `"File"` - Read from file
- **Example**: `"Bump"`

### `topography_file`
- **Type**: `string`
- **Default**: `""`
- **Description**: Path to file containing topography data
- **Example**: `"exp_data/bathymetry.csv"`
- **Used with**: `"File"` topography type
- **Format**: CSV with first line = number of points, then: x, z

---

## Example Configurations

### Quick Test (Fast)
```json
{
  "final_time": 10.0,
  "time_step": 0.05,
  "dx": 0.5,
  "numerical_flux": "HLL",
  "scheme_order": 2,
  "time_scheme": "RK2"
}
```
→ Only 200 timesteps, coarse grid

### Accurate Simulation (Moderate)
```json
{
  "final_time": 100.0,
  "time_step": 0.02,
  "dx": 0.1,
  "numerical_flux": "HLL",
  "scheme_order": 2,
  "time_scheme": "RK2"
}
```
→ 5,000 timesteps, fine grid

### High Accuracy (Slow)
```json
{
  "final_time": 300.0,
  "time_step": 0.01,
  "dx": 0.05,
  "numerical_flux": "HLL",
  "scheme_order": 2,
  "time_scheme": "RK2"
}
```
→ 30,000 timesteps, very fine grid

---

## Performance Tips

1. **Reduce timesteps**: Increase `time_step` (respect CFL!)
2. **Coarser grid**: Increase `dx` 
3. **Less saves**: Increase `save_frequency`
4. **Shorter simulation**: Decrease `final_time`
5. **Use Numba**: Already enabled for 10-50x speedup
6. **First order**: Set `scheme_order = 1` (faster but less accurate)

## Stability Guidelines

### CFL Condition
Always check that your timestep satisfies:
```
dt < CFL * dx / (|u| + √(gh))
```

For typical shallow water flows:
- Wave speed: c = √(gh) ≈ 3-6 m/s
- Safe CFL ≈ 0.9
- **Rule of thumb**: `dt ≈ 0.03 * dx`

### Grid Resolution
Minimum recommended cells to resolve features:
- Shock waves: 5-10 cells
- Smooth gradients: 20-50 cells
- Total domain: 50-500 cells typical

---

## Units

All parameters use SI units:
- Length: meters (m)
- Time: seconds (s)
- Velocity: meters per second (m/s)
- Discharge: square meters per second (m²/s)
- Acceleration: meters per second squared (m/s²)

