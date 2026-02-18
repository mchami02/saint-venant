"""Vectorized numerical fluxes for the ARZ system (PyTorch)."""

import torch

from .physics import dp_drho, pressure


def rusanov(
    rhoL: torch.Tensor,
    rho_wL: torch.Tensor,
    rhoR: torch.Tensor,
    rho_wR: torch.Tensor,
    gamma: float,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized Rusanov (local Lax-Friedrichs) flux for ARZ.

    All inputs are 1-D tensors of matching length (one entry per interface).
    Returns (flux_rho, flux_rw) of the same length.
    """
    wL = torch.where(rhoL > eps, rho_wL / rhoL, torch.zeros_like(rhoL))
    wR = torch.where(rhoR > eps, rho_wR / rhoR, torch.zeros_like(rhoR))
    vL = wL - pressure(rhoL, gamma)
    vR = wR - pressure(rhoR, gamma)

    # Physical fluxes
    fL_rho = rhoL * vL
    fL_rw = rho_wL * vL
    fR_rho = rhoR * vR
    fR_rw = rho_wR * vR

    # Max wave speed at each interface
    dpL = dp_drho(rhoL.clamp(min=eps), gamma)
    dpR = dp_drho(rhoR.clamp(min=eps), gamma)

    sL = torch.maximum(vL.abs(), (vL - rhoL * dpL).abs())
    sR = torch.maximum(vR.abs(), (vR - rhoR * dpR).abs())
    smax = torch.maximum(sL, sR)

    flux_rho = 0.5 * (fL_rho + fR_rho) - 0.5 * smax * (rhoR - rhoL)
    flux_rw = 0.5 * (fL_rw + fR_rw) - 0.5 * smax * (rho_wR - rho_wL)
    return flux_rho, flux_rw


def hll(
    rhoL: torch.Tensor,
    rho_wL: torch.Tensor,
    rhoR: torch.Tensor,
    rho_wR: torch.Tensor,
    gamma: float,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized HLL flux for ARZ.

    All inputs are 1-D tensors of matching length (one entry per interface).
    Returns (flux_rho, flux_rw) of the same length.
    """
    wL = torch.where(rhoL > eps, rho_wL / rhoL, torch.zeros_like(rhoL))
    wR = torch.where(rhoR > eps, rho_wR / rhoR, torch.zeros_like(rhoR))
    vL = wL - pressure(rhoL, gamma)
    vR = wR - pressure(rhoR, gamma)

    # Physical fluxes
    fL_rho = rhoL * vL
    fL_rw = rho_wL * vL
    fR_rho = rhoR * vR
    fR_rw = rho_wR * vR

    # Eigenvalues
    dpL = dp_drho(rhoL.clamp(min=eps), gamma)
    dpR = dp_drho(rhoR.clamp(min=eps), gamma)

    lam1L, lam2L = vL, vL - rhoL * dpL
    lam1R, lam2R = vR, vR - rhoR * dpR

    sL = torch.minimum(torch.minimum(lam1L, lam2L), torch.minimum(lam1R, lam2R))
    sR = torch.maximum(torch.maximum(lam1L, lam2L), torch.maximum(lam1R, lam2R))

    denom = sR - sL
    denom = torch.where(denom.abs() < 1e-14, torch.full_like(denom, 1e-14), denom)

    hll_rho = (sR * fL_rho - sL * fR_rho + sL * sR * (rhoR - rhoL)) / denom
    hll_rw = (sR * fL_rw - sL * fR_rw + sL * sR * (rho_wR - rho_wL)) / denom

    flux_rho = torch.where(sL >= 0, fL_rho, torch.where(sR <= 0, fR_rho, hll_rho))
    flux_rw = torch.where(sL >= 0, fL_rw, torch.where(sR <= 0, fR_rw, hll_rw))

    return flux_rho, flux_rw
