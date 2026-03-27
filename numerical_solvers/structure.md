# numerical_solvers/ — Structure

Top-level package for numerical PDE solvers (PyTorch).

## Layout

| Directory | Purpose |
|---|---|
| `src/` | Source code for all equation solvers |
| `test/` | Comprehensive pytest test suite |

## src/ — Solver Subpackages

| Directory | Purpose |
|---|---|
| `src/arz/` | ARZ (Aw-Rascle-Zhang) traffic flow solver |
| `src/euler/` | 1D compressible Euler equations solver (HLLC/HLL/Rusanov, PyTorch) |
| `src/lwr/` | LWR (Lighthill-Whitham-Richards) traffic flow solver (nfv Lax-Hopf) |
| `src/lwr2d/` | 2D LWR traffic flow solver (Godunov FV, PyTorch) |

## test/ — Test Suite

| Directory | Purpose |
|---|---|
| `test/arz/` | ARZ solver tests |
| `test/euler/` | Euler solver tests |
| `test/lwr/` | LWR solver tests |
| `test/lwr2d/` | 2D LWR solver tests |
| `test/conftest.py` | Shared pytest fixtures |

## Notable Files

| File | Purpose |
|---|---|
| `src/arz/test_api_arz.ipynb` | Interactive notebook testing the ARZ solver API |
| `src/lwr/test_api_lwr.ipynb` | Interactive notebook testing the LWR solver API |
| `src/lwr2d/test_api_lwr2d.ipynb` | Interactive notebook testing the 2D LWR solver API |
| `src/euler/test_api_euler.ipynb` | Interactive notebook testing the Euler solver API |
