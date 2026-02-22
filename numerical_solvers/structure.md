# numerical_solvers/ — Structure

Top-level package for numerical PDE solvers (PyTorch).

## Subpackages

| Directory | Purpose |
|---|---|
| `arz/` | ARZ (Aw-Rascle-Zhang) traffic flow solver |
| `lwr/` | LWR (Lighthill-Whitham-Richards) traffic flow solver (nfv Lax-Hopf) |
| `lwr2d/` | 2D LWR traffic flow solver (Godunov FV, PyTorch) |

## Notable Files

| File | Purpose |
|---|---|
| `arz/test_api.ipynb` | Interactive notebook testing the ARZ solver API |
| `lwr/test_api_lwr.ipynb` | Interactive notebook testing the LWR solver API |
| `lwr2d/test_api_lwr2d.ipynb` | Interactive notebook testing the 2D LWR solver API |
