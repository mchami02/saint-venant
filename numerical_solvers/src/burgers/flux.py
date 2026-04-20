"""Numerical fluxes for the inviscid Burgers equation (PyTorch).

Godunov flux for a strictly convex scalar flux f(u) = u^2/2 is:

    if uL >= uR  (shock):
        F = max(f(uL), f(uR))
    elif uL < 0 < uR  (transonic rarefaction):
        F = f(0) = 0
    else:
        F = min(f(uL), f(uR))

This is the entropy-consistent upwind flux. Ported from clawpack/riemann
``burgers_1D_py.py`` where the same logic is written as a flux-difference
splitting with an explicit ``efix`` (entropy fix) branch.

Reference:
    clawpack/riemann — burgers_1D_py.py
    https://github.com/clawpack/riemann/blob/master/riemann/burgers_1D_py.py
"""

import torch

from .physics import flux as _flux


def godunov(uL: torch.Tensor, uR: torch.Tensor) -> torch.Tensor:
    """Vectorised Godunov flux for Burgers (convex, entropy-consistent)."""
    fL = _flux(uL)
    fR = _flux(uR)
    shock = uL >= uR
    transonic = (uL < 0.0) & (uR > 0.0)
    return torch.where(
        shock,
        torch.maximum(fL, fR),
        torch.where(transonic, torch.zeros_like(fL), torch.minimum(fL, fR)),
    )


def rusanov(uL: torch.Tensor, uR: torch.Tensor) -> torch.Tensor:
    """Vectorised Rusanov (local Lax-Friedrichs) flux."""
    fL = _flux(uL)
    fR = _flux(uR)
    smax = torch.maximum(uL.abs(), uR.abs())
    return 0.5 * (fL + fR) - 0.5 * smax * (uR - uL)
