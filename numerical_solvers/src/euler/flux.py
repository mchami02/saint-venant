"""Vectorized numerical fluxes for the 1D Euler system (PyTorch).

Port of clawpack/riemann euler_1D_py.py Riemann solvers to PyTorch.
All operations are fully vectorized (no Python loops over interfaces).

Reference:
    clawpack/riemann — euler_1D_py.py
    https://github.com/clawpack/riemann/blob/master/riemann/euler_1D_py.py
"""

import torch

from .physics import pressure_from_conservative


# ------------------------------------------------------------------ helpers
def _roe_averages(
    rhoL: torch.Tensor,
    rho_uL: torch.Tensor,
    EL: torch.Tensor,
    rhoR: torch.Tensor,
    rho_uR: torch.Tensor,
    ER: torch.Tensor,
    gamma1: float,
    eps: float = 1e-12,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Compute Roe-averaged quantities at each interface.

    Matches clawpack ``roe_averages`` exactly.

    Returns
    -------
    u_roe, a_roe, H_roe, pL, pR
    """
    sqrtL = torch.sqrt(rhoL.clamp(min=eps))
    sqrtR = torch.sqrt(rhoR.clamp(min=eps))

    pL = gamma1 * (EL - 0.5 * rho_uL**2 / rhoL.clamp(min=eps))
    pR = gamma1 * (ER - 0.5 * rho_uR**2 / rhoR.clamp(min=eps))

    denom = sqrtL + sqrtR
    u_roe = (rho_uL / sqrtL + rho_uR / sqrtR) / denom
    H_roe = ((EL + pL) / sqrtL + (ER + pR) / sqrtR) / denom
    a_roe = torch.sqrt((gamma1 * (H_roe - 0.5 * u_roe**2)).clamp(min=eps))

    return u_roe, a_roe, H_roe, pL, pR


# ------------------------------------------------------------------ HLLC
def hllc(
    rhoL: torch.Tensor,
    rho_uL: torch.Tensor,
    EL: torch.Tensor,
    rhoR: torch.Tensor,
    rho_uR: torch.Tensor,
    ER: torch.Tensor,
    gamma: float,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized HLLC flux for the 1D Euler equations.

    Mirrors clawpack ``euler_hllc_1D``.
    All inputs are 1-D tensors of matching length (one entry per interface).
    Returns (flux_rho, flux_rho_u, flux_E) of the same length.
    """
    gamma1 = gamma - 1.0

    # Roe averages
    u_roe, a_roe, _, pL, pR = _roe_averages(
        rhoL, rho_uL, EL, rhoR, rho_uR, ER, gamma1, eps
    )

    # Left / right primitive speeds
    uL = torch.where(rhoL > eps, rho_uL / rhoL, torch.zeros_like(rhoL))
    uR = torch.where(rhoR > eps, rho_uR / rhoR, torch.zeros_like(rhoR))
    HL = (EL + pL) / rhoL.clamp(min=eps)
    HR = (ER + pR) / rhoR.clamp(min=eps)
    aL = torch.sqrt((gamma1 * (HL - 0.5 * uL**2)).clamp(min=eps))
    aR = torch.sqrt((gamma1 * (HR - 0.5 * uR**2)).clamp(min=eps))

    # Einfeldt wave speed estimates (matches clawpack)
    sL = torch.minimum(
        torch.minimum(u_roe - a_roe, u_roe + a_roe),
        torch.minimum(uL - aL, uL + aL),
    )
    sR = torch.maximum(
        torch.maximum(u_roe - a_roe, u_roe + a_roe),
        torch.maximum(uR - aR, uR + aR),
    )

    # Contact wave speed
    denom_sm = rhoL * (sL - uL) - rhoR * (sR - uR)
    denom_sm = torch.where(
        denom_sm.abs() < 1e-14, torch.full_like(denom_sm, 1e-14), denom_sm
    )
    sM = (pR - pL + rhoL * uL * (sL - uL) - rhoR * uR * (sR - uR)) / denom_sm

    # Left star state
    fac_L = rhoL * (sL - uL) / (sL - sM + 1e-30)
    q_hat_L_rho = fac_L
    q_hat_L_rho_u = fac_L * sM
    q_hat_L_E = fac_L * (
        EL / rhoL.clamp(min=eps)
        + (sM - uL) * (sM + pL / (rhoL * (sL - uL) + 1e-30))
    )

    # Right star state
    fac_R = rhoR * (sR - uR) / (sR - sM + 1e-30)
    q_hat_R_rho = fac_R
    q_hat_R_rho_u = fac_R * sM
    q_hat_R_E = fac_R * (
        ER / rhoR.clamp(min=eps)
        + (sM - uR) * (sM + pR / (rhoR * (sR - uR) + 1e-30))
    )

    # Physical fluxes
    fL_rho = rho_uL
    fL_rho_u = rho_uL * uL + pL
    fL_E = uL * (EL + pL)

    fR_rho = rho_uR
    fR_rho_u = rho_uR * uR + pR
    fR_E = uR * (ER + pR)

    # Star-region fluxes: F* = F_K + s_K * (Q*_K - Q_K)
    fstar_L_rho = fL_rho + sL * (q_hat_L_rho - rhoL)
    fstar_L_rho_u = fL_rho_u + sL * (q_hat_L_rho_u - rho_uL)
    fstar_L_E = fL_E + sL * (q_hat_L_E - EL)

    fstar_R_rho = fR_rho + sR * (q_hat_R_rho - rhoR)
    fstar_R_rho_u = fR_rho_u + sR * (q_hat_R_rho_u - rho_uR)
    fstar_R_E = fR_E + sR * (q_hat_R_E - ER)

    # Select flux based on wave speeds
    flux_rho = torch.where(
        sL >= 0,
        fL_rho,
        torch.where(sM >= 0, fstar_L_rho, torch.where(sR <= 0, fR_rho, fstar_R_rho)),
    )
    flux_rho_u = torch.where(
        sL >= 0,
        fL_rho_u,
        torch.where(
            sM >= 0, fstar_L_rho_u, torch.where(sR <= 0, fR_rho_u, fstar_R_rho_u)
        ),
    )
    flux_E = torch.where(
        sL >= 0,
        fL_E,
        torch.where(sM >= 0, fstar_L_E, torch.where(sR <= 0, fR_E, fstar_R_E)),
    )

    return flux_rho, flux_rho_u, flux_E


# ------------------------------------------------------------------ HLL
def hll(
    rhoL: torch.Tensor,
    rho_uL: torch.Tensor,
    EL: torch.Tensor,
    rhoR: torch.Tensor,
    rho_uR: torch.Tensor,
    ER: torch.Tensor,
    gamma: float,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized HLL flux for the 1D Euler equations.

    Mirrors clawpack ``euler_hll_1D``.
    """
    gamma1 = gamma - 1.0

    u_roe, a_roe, _, pL, pR = _roe_averages(
        rhoL, rho_uL, EL, rhoR, rho_uR, ER, gamma1, eps
    )

    uL = torch.where(rhoL > eps, rho_uL / rhoL, torch.zeros_like(rhoL))
    uR = torch.where(rhoR > eps, rho_uR / rhoR, torch.zeros_like(rhoR))
    HL = (EL + pL) / rhoL.clamp(min=eps)
    HR = (ER + pR) / rhoR.clamp(min=eps)
    aL = torch.sqrt((gamma1 * (HL - 0.5 * uL**2)).clamp(min=eps))
    aR = torch.sqrt((gamma1 * (HR - 0.5 * uR**2)).clamp(min=eps))

    # Einfeldt wave speed estimates
    sL = torch.minimum(
        torch.minimum(u_roe - a_roe, u_roe + a_roe),
        torch.minimum(uL - aL, uL + aL),
    )
    sR = torch.maximum(
        torch.maximum(u_roe - a_roe, u_roe + a_roe),
        torch.maximum(uR - aR, uR + aR),
    )

    # Physical fluxes
    fL_rho = rho_uL
    fL_rho_u = rho_uL * uL + pL
    fL_E = uL * (EL + pL)

    fR_rho = rho_uR
    fR_rho_u = rho_uR * uR + pR
    fR_E = uR * (ER + pR)

    denom = sR - sL
    denom = torch.where(denom.abs() < 1e-14, torch.full_like(denom, 1e-14), denom)

    hll_rho = (sR * fL_rho - sL * fR_rho + sL * sR * (rhoR - rhoL)) / denom
    hll_rho_u = (
        sR * fL_rho_u - sL * fR_rho_u + sL * sR * (rho_uR - rho_uL)
    ) / denom
    hll_E = (sR * fL_E - sL * fR_E + sL * sR * (ER - EL)) / denom

    flux_rho = torch.where(sL >= 0, fL_rho, torch.where(sR <= 0, fR_rho, hll_rho))
    flux_rho_u = torch.where(
        sL >= 0, fL_rho_u, torch.where(sR <= 0, fR_rho_u, hll_rho_u)
    )
    flux_E = torch.where(sL >= 0, fL_E, torch.where(sR <= 0, fR_E, hll_E))

    return flux_rho, flux_rho_u, flux_E


# ------------------------------------------------------------------ Rusanov
def rusanov(
    rhoL: torch.Tensor,
    rho_uL: torch.Tensor,
    EL: torch.Tensor,
    rhoR: torch.Tensor,
    rho_uR: torch.Tensor,
    ER: torch.Tensor,
    gamma: float,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized Rusanov (local Lax-Friedrichs) flux for 1D Euler."""
    gamma1 = gamma - 1.0

    uL = torch.where(rhoL > eps, rho_uL / rhoL, torch.zeros_like(rhoL))
    uR = torch.where(rhoR > eps, rho_uR / rhoR, torch.zeros_like(rhoR))

    pL = gamma1 * (EL - 0.5 * rhoL * uL**2)
    pR = gamma1 * (ER - 0.5 * rhoR * uR**2)

    aL = torch.sqrt((gamma * pL.clamp(min=0.0) / rhoL.clamp(min=eps)))
    aR = torch.sqrt((gamma * pR.clamp(min=0.0) / rhoR.clamp(min=eps)))

    smax = torch.maximum(uL.abs() + aL, uR.abs() + aR)

    # Physical fluxes
    fL_rho = rho_uL
    fL_rho_u = rho_uL * uL + pL
    fL_E = uL * (EL + pL)

    fR_rho = rho_uR
    fR_rho_u = rho_uR * uR + pR
    fR_E = uR * (ER + pR)

    flux_rho = 0.5 * (fL_rho + fR_rho) - 0.5 * smax * (rhoR - rhoL)
    flux_rho_u = 0.5 * (fL_rho_u + fR_rho_u) - 0.5 * smax * (rho_uR - rho_uL)
    flux_E = 0.5 * (fL_E + fR_E) - 0.5 * smax * (ER - EL)

    return flux_rho, flux_rho_u, flux_E
