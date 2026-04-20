"""Vectorized numerical fluxes for the 2D Euler system via dimensional sweeps.

Each kernel takes the state at one set of interfaces:
    (rho, rho_n, rho_t, E)
where ``rho_n`` is the normal momentum (rho*u for x-sweep, rho*v for y-sweep)
and ``rho_t`` is the tangential momentum (advected as a passive scalar at
the interface velocity).

Physical flux along the normal direction:
    F = (rho*u_n, rho*u_n^2 + p, rho*u_n*u_t, u_n*(E + p))

Riemann logic ported from clawpack/riemann ``euler_1D_py.py`` (HLLC/HLL/
Rusanov) and extended with the passive scalar following the PyClaw
dimensionally-split approach for 2D Euler.

Reference:
    clawpack/riemann — euler_1D_py.py
    https://github.com/clawpack/riemann/blob/master/riemann/euler_1D_py.py
"""

import torch


def _roe_averages(
    rhoL, rho_nL, EL, rhoR, rho_nR, ER, pL, pR, gamma1, eps=1e-12,
):
    sqrtL = torch.sqrt(rhoL.clamp(min=eps))
    sqrtR = torch.sqrt(rhoR.clamp(min=eps))
    denom = sqrtL + sqrtR
    u_roe = (rho_nL / sqrtL + rho_nR / sqrtR) / denom
    H_roe = ((EL + pL) / sqrtL + (ER + pR) / sqrtR) / denom
    a_roe = torch.sqrt((gamma1 * (H_roe - 0.5 * u_roe * u_roe)).clamp(min=eps))
    return u_roe, a_roe


def _pressures(rhoL, rho_nL, rho_tL, EL, rhoR, rho_nR, rho_tR, ER, gamma1, eps=1e-12):
    inv_rhoL = 1.0 / rhoL.clamp(min=eps)
    inv_rhoR = 1.0 / rhoR.clamp(min=eps)
    kinL = 0.5 * (rho_nL * rho_nL + rho_tL * rho_tL) * inv_rhoL
    kinR = 0.5 * (rho_nR * rho_nR + rho_tR * rho_tR) * inv_rhoR
    pL = (gamma1 * (EL - kinL)).clamp(min=0.0)
    pR = (gamma1 * (ER - kinR)).clamp(min=0.0)
    return pL, pR, inv_rhoL, inv_rhoR


def hllc(
    rhoL, rho_nL, rho_tL, EL,
    rhoR, rho_nR, rho_tR, ER,
    gamma: float, eps: float = 1e-12,
):
    """Vectorised HLLC flux with a passive tangential momentum scalar."""
    gamma1 = gamma - 1.0

    pL, pR, inv_rhoL, inv_rhoR = _pressures(
        rhoL, rho_nL, rho_tL, EL, rhoR, rho_nR, rho_tR, ER, gamma1, eps,
    )

    uL = rho_nL * inv_rhoL  # normal velocity
    uR = rho_nR * inv_rhoR
    vL = rho_tL * inv_rhoL  # tangential velocity
    vR = rho_tR * inv_rhoR
    HL = (EL + pL) * inv_rhoL
    HR = (ER + pR) * inv_rhoR

    # Roe averages on the (rho, rho_n, E) subsystem (tangential term enters
    # through H, so include full kinetic energy via pL/pR which already
    # accounts for both components).
    u_roe, a_roe = _roe_averages(
        rhoL, rho_nL, EL, rhoR, rho_nR, ER, pL, pR, gamma1, eps,
    )

    # Einfeldt speed estimates (1D character along the normal direction)
    aL = torch.sqrt((gamma1 * (HL - 0.5 * (uL * uL + vL * vL))).clamp(min=eps))
    aR = torch.sqrt((gamma1 * (HR - 0.5 * (uR * uR + vR * vR))).clamp(min=eps))
    sL = torch.minimum(
        torch.minimum(u_roe - a_roe, u_roe + a_roe),
        torch.minimum(uL - aL, uL + aL),
    )
    sR = torch.maximum(
        torch.maximum(u_roe - a_roe, u_roe + a_roe),
        torch.maximum(uR - aR, uR + aR),
    )

    denom_sm = rhoL * (sL - uL) - rhoR * (sR - uR)
    denom_sm = torch.where(
        denom_sm.abs() < 1e-14, torch.full_like(denom_sm, 1e-14), denom_sm,
    )
    sM = (pR - pL + rhoL * uL * (sL - uL) - rhoR * uR * (sR - uR)) / denom_sm

    # Star states (tangential velocity is continuous across the contact wave)
    fac_L = rhoL * (sL - uL) / (sL - sM + 1e-30)
    q_star_rhoL = fac_L
    q_star_rho_nL = fac_L * sM
    q_star_rho_tL = fac_L * vL  # tangential advected
    q_star_EL = fac_L * (
        EL * inv_rhoL + (sM - uL) * (sM + pL / (rhoL * (sL - uL) + 1e-30))
    )

    fac_R = rhoR * (sR - uR) / (sR - sM + 1e-30)
    q_star_rhoR = fac_R
    q_star_rho_nR = fac_R * sM
    q_star_rho_tR = fac_R * vR
    q_star_ER = fac_R * (
        ER * inv_rhoR + (sM - uR) * (sM + pR / (rhoR * (sR - uR) + 1e-30))
    )

    # Physical fluxes along the normal direction
    fL_rho = rho_nL
    fL_rho_n = rho_nL * uL + pL
    fL_rho_t = rho_nL * vL
    fL_E = uL * (EL + pL)

    fR_rho = rho_nR
    fR_rho_n = rho_nR * uR + pR
    fR_rho_t = rho_nR * vR
    fR_E = uR * (ER + pR)

    # Star-region fluxes: F* = F_K + s_K * (Q*_K - Q_K)
    fstar_L_rho = fL_rho + sL * (q_star_rhoL - rhoL)
    fstar_L_rho_n = fL_rho_n + sL * (q_star_rho_nL - rho_nL)
    fstar_L_rho_t = fL_rho_t + sL * (q_star_rho_tL - rho_tL)
    fstar_L_E = fL_E + sL * (q_star_EL - EL)

    fstar_R_rho = fR_rho + sR * (q_star_rhoR - rhoR)
    fstar_R_rho_n = fR_rho_n + sR * (q_star_rho_nR - rho_nR)
    fstar_R_rho_t = fR_rho_t + sR * (q_star_rho_tR - rho_tR)
    fstar_R_E = fR_E + sR * (q_star_ER - ER)

    def _select(fL, fstarL, fstarR, fR):
        return torch.where(
            sL >= 0, fL,
            torch.where(sM >= 0, fstarL, torch.where(sR <= 0, fR, fstarR)),
        )

    return (
        _select(fL_rho, fstar_L_rho, fstar_R_rho, fR_rho),
        _select(fL_rho_n, fstar_L_rho_n, fstar_R_rho_n, fR_rho_n),
        _select(fL_rho_t, fstar_L_rho_t, fstar_R_rho_t, fR_rho_t),
        _select(fL_E, fstar_L_E, fstar_R_E, fR_E),
    )


def hll(
    rhoL, rho_nL, rho_tL, EL,
    rhoR, rho_nR, rho_tR, ER,
    gamma: float, eps: float = 1e-12,
):
    """Vectorised HLL flux with a passive tangential momentum scalar."""
    gamma1 = gamma - 1.0

    pL, pR, inv_rhoL, inv_rhoR = _pressures(
        rhoL, rho_nL, rho_tL, EL, rhoR, rho_nR, rho_tR, ER, gamma1, eps,
    )

    uL = rho_nL * inv_rhoL
    uR = rho_nR * inv_rhoR
    vL = rho_tL * inv_rhoL
    vR = rho_tR * inv_rhoR
    HL = (EL + pL) * inv_rhoL
    HR = (ER + pR) * inv_rhoR

    u_roe, a_roe = _roe_averages(
        rhoL, rho_nL, EL, rhoR, rho_nR, ER, pL, pR, gamma1, eps,
    )
    aL = torch.sqrt((gamma1 * (HL - 0.5 * (uL * uL + vL * vL))).clamp(min=eps))
    aR = torch.sqrt((gamma1 * (HR - 0.5 * (uR * uR + vR * vR))).clamp(min=eps))
    sL = torch.minimum(
        torch.minimum(u_roe - a_roe, u_roe + a_roe),
        torch.minimum(uL - aL, uL + aL),
    )
    sR = torch.maximum(
        torch.maximum(u_roe - a_roe, u_roe + a_roe),
        torch.maximum(uR - aR, uR + aR),
    )

    fL_rho = rho_nL
    fL_rho_n = rho_nL * uL + pL
    fL_rho_t = rho_nL * vL
    fL_E = uL * (EL + pL)

    fR_rho = rho_nR
    fR_rho_n = rho_nR * uR + pR
    fR_rho_t = rho_nR * vR
    fR_E = uR * (ER + pR)

    denom = sR - sL
    denom = torch.where(denom.abs() < 1e-14, torch.full_like(denom, 1e-14), denom)

    def _hll_flux(fL, fR, qL, qR):
        return (sR * fL - sL * fR + sL * sR * (qR - qL)) / denom

    h_rho = _hll_flux(fL_rho, fR_rho, rhoL, rhoR)
    h_rho_n = _hll_flux(fL_rho_n, fR_rho_n, rho_nL, rho_nR)
    h_rho_t = _hll_flux(fL_rho_t, fR_rho_t, rho_tL, rho_tR)
    h_E = _hll_flux(fL_E, fR_E, EL, ER)

    def _select(fL, fR, mid):
        return torch.where(sL >= 0, fL, torch.where(sR <= 0, fR, mid))

    return (
        _select(fL_rho, fR_rho, h_rho),
        _select(fL_rho_n, fR_rho_n, h_rho_n),
        _select(fL_rho_t, fR_rho_t, h_rho_t),
        _select(fL_E, fR_E, h_E),
    )


def rusanov(
    rhoL, rho_nL, rho_tL, EL,
    rhoR, rho_nR, rho_tR, ER,
    gamma: float, eps: float = 1e-12,
):
    """Vectorised Rusanov (local Lax-Friedrichs) flux with passive tangential scalar."""
    gamma1 = gamma - 1.0

    pL, pR, inv_rhoL, inv_rhoR = _pressures(
        rhoL, rho_nL, rho_tL, EL, rhoR, rho_nR, rho_tR, ER, gamma1, eps,
    )
    uL = rho_nL * inv_rhoL
    uR = rho_nR * inv_rhoR
    vL = rho_tL * inv_rhoL
    vR = rho_tR * inv_rhoR

    aL = torch.sqrt(gamma * pL / rhoL.clamp(min=eps))
    aR = torch.sqrt(gamma * pR.clamp(min=0.0) / rhoR.clamp(min=eps))

    smax = torch.maximum(uL.abs() + aL, uR.abs() + aR)

    fL_rho = rho_nL
    fL_rho_n = rho_nL * uL + pL
    fL_rho_t = rho_nL * vL
    fL_E = uL * (EL + pL)

    fR_rho = rho_nR
    fR_rho_n = rho_nR * uR + pR
    fR_rho_t = rho_nR * vR
    fR_E = uR * (ER + pR)

    f_rho = 0.5 * (fL_rho + fR_rho) - 0.5 * smax * (rhoR - rhoL)
    f_rho_n = 0.5 * (fL_rho_n + fR_rho_n) - 0.5 * smax * (rho_nR - rho_nL)
    f_rho_t = 0.5 * (fL_rho_t + fR_rho_t) - 0.5 * smax * (rho_tR - rho_tL)
    f_E = 0.5 * (fL_E + fR_E) - 0.5 * smax * (ER - EL)

    return f_rho, f_rho_n, f_rho_t, f_E
