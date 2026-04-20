# numerical_solvers/ — Structure

Top-level package for numerical PDE solvers (PyTorch).

## Layout

| Directory | Purpose |
|---|---|
| `src/` | Source code for all equation solvers |
| `test/` | Comprehensive pytest test suite |
| `notebooks/` | Interactive API test notebooks for each solver |

## src/ — Solver Subpackages

| Directory | Purpose |
|---|---|
| `src/arz/` | ARZ (Aw-Rascle-Zhang) traffic flow solver |
| `src/burgers/` | 1D inviscid Burgers equation solver (Godunov + entropy fix / Rusanov, PyTorch) |
| `src/euler/` | 1D compressible Euler equations solver (HLLC/HLL/Rusanov, PyTorch) |
| `src/euler2d/` | 2D compressible Euler equations solver (HLLC/HLL/Rusanov, unsplit FV, PyTorch) |
| `src/lwr/` | LWR (Lighthill-Whitham-Richards) traffic flow solver (nfv Lax-Hopf) |
| `src/lwr2d/` | 2D LWR traffic flow solver (Godunov FV, PyTorch) |

## test/ — Test Suite

| Directory | Purpose |
|---|---|
| `test/arz/` | ARZ solver tests |
| `test/burgers/` | Burgers solver tests |
| `test/euler/` | Euler solver tests |
| `test/euler2d/` | 2D Euler solver tests |
| `test/lwr/` | LWR solver tests |
| `test/lwr2d/` | 2D LWR solver tests |
| `test/conftest.py` | Shared pytest fixtures |

## notebooks/ — API Test Notebooks

| File | Purpose |
|---|---|
| `notebooks/test_api_arz.ipynb` | Interactive notebook testing the ARZ solver API |
| `notebooks/test_api_burgers.ipynb` | Interactive notebook testing the Burgers solver API |
| `notebooks/test_api_euler.ipynb` | Interactive notebook testing the 1D Euler solver API |
| `notebooks/test_api_euler2d.ipynb` | Interactive notebook testing the 2D Euler solver API |
| `notebooks/test_api_lwr.ipynb` | Interactive notebook testing the LWR solver API |
| `notebooks/test_api_lwr2d.ipynb` | Interactive notebook testing the 2D LWR solver API |
