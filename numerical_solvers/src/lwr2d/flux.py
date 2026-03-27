"""Exact Godunov numerical flux for scalar conservation laws (PyTorch)."""

import torch

from .physics import (
    greenshields_critical_density,
    greenshields_flux,
    triangular_critical_density,
    triangular_flux,
)


def godunov_flux_greenshields(
    rho_L: torch.Tensor,
    rho_R: torch.Tensor,
    v_max: float,
    rho_max: float,
) -> torch.Tensor:
    """Exact Godunov flux for the Greenshields (concave) flux function.

    For a concave flux f with maximum at rho_crit:
    - rho_L <= rho_R  =>  F = min(f(rho_L), f(rho_R))
    - rho_L >  rho_R  and rho_crit in [rho_R, rho_L]  =>  F = f(rho_crit)
    - otherwise        =>  F = max(f(rho_L), f(rho_R))
    """
    rho_crit = greenshields_critical_density(rho_max)
    f_L = greenshields_flux(rho_L, v_max, rho_max)
    f_R = greenshields_flux(rho_R, v_max, rho_max)
    f_crit = greenshields_flux(
        torch.tensor(rho_crit, device=rho_L.device, dtype=rho_L.dtype),
        v_max,
        rho_max,
    )

    # Case 1: rho_L <= rho_R  (shock or contact)
    case1 = torch.minimum(f_L, f_R)

    # Case 2: rho_L > rho_R with transonic rarefaction
    transonic = (rho_R <= rho_crit) & (rho_crit <= rho_L)
    case2 = torch.where(transonic, f_crit, torch.maximum(f_L, f_R))

    return torch.where(rho_L <= rho_R, case1, case2)


def godunov_flux_triangular(
    rho_L: torch.Tensor,
    rho_R: torch.Tensor,
    v_f: float,
    w: float,
    rho_max: float,
) -> torch.Tensor:
    """Exact Godunov flux for the triangular (piecewise-linear concave) flux.

    Same logic as Greenshields but with the triangular flux and its critical
    density rho_crit = w * rho_max / (v_f + w).
    """
    rho_crit = triangular_critical_density(v_f, w, rho_max)
    f_L = triangular_flux(rho_L, v_f, w, rho_max)
    f_R = triangular_flux(rho_R, v_f, w, rho_max)
    f_crit = triangular_flux(
        torch.tensor(rho_crit, device=rho_L.device, dtype=rho_L.dtype),
        v_f,
        w,
        rho_max,
    )

    case1 = torch.minimum(f_L, f_R)

    transonic = (rho_R <= rho_crit) & (rho_crit <= rho_L)
    case2 = torch.where(transonic, f_crit, torch.maximum(f_L, f_R))

    return torch.where(rho_L <= rho_R, case1, case2)
