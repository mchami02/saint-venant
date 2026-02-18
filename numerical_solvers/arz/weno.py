"""WENO-5 spatial reconstruction (equation-agnostic, PyTorch)."""

import torch


def _weno5_weights(
    b0: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    d0: float,
    d1: float,
    d2: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute WENO nonlinear weights from smoothness indicators."""
    eps = 1e-6
    w0 = d0 / (eps + b0) ** 2
    w1 = d1 / (eps + b1) ** 2
    w2 = d2 / (eps + b2) ** 2
    ws = w0 + w1 + w2
    return w0 / ws, w1 / ws, w2 / ws


def weno5_reconstruct(
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """WENO-5 reconstruction from cell averages (vectorized).

    Given cell averages v[0..N-1] with 4 ghost cells on each side,
    returns (v_minus, v_plus) at cell interfaces.

    v_minus[j] = value from left of interface j
    v_plus[j]  = value from right of interface j
    """
    N = v.shape[0]
    ni = N - 7  # number of interfaces

    # --- v^- (left-biased) ---
    a = v[1 : 1 + ni]
    b = v[2 : 2 + ni]
    c = v[3 : 3 + ni]
    d = v[4 : 4 + ni]
    e = v[5 : 5 + ni]

    q0 = (1 / 3) * a - (7 / 6) * b + (11 / 6) * c
    q1 = -(1 / 6) * b + (5 / 6) * c + (1 / 3) * d
    q2 = (1 / 3) * c + (5 / 6) * d - (1 / 6) * e

    b0 = (13 / 12) * (a - 2 * b + c) ** 2 + (1 / 4) * (a - 4 * b + 3 * c) ** 2
    b1 = (13 / 12) * (b - 2 * c + d) ** 2 + (1 / 4) * (b - d) ** 2
    b2 = (13 / 12) * (c - 2 * d + e) ** 2 + (1 / 4) * (3 * c - 4 * d + e) ** 2

    w0, w1, w2 = _weno5_weights(b0, b1, b2, 1 / 10, 6 / 10, 3 / 10)
    v_minus = w0 * q0 + w1 * q1 + w2 * q2

    # --- v^+ (right-biased) ---
    p = v[2 : 2 + ni]
    q = v[3 : 3 + ni]
    r = v[4 : 4 + ni]
    s = v[5 : 5 + ni]
    t = v[6 : 6 + ni]

    q0r = (1 / 3) * t - (7 / 6) * s + (11 / 6) * r
    q1r = -(1 / 6) * s + (5 / 6) * r + (1 / 3) * q
    q2r = (1 / 3) * r + (5 / 6) * q - (1 / 6) * p

    b0r = (13 / 12) * (t - 2 * s + r) ** 2 + (1 / 4) * (t - 4 * s + 3 * r) ** 2
    b1r = (13 / 12) * (s - 2 * r + q) ** 2 + (1 / 4) * (s - q) ** 2
    b2r = (13 / 12) * (r - 2 * q + p) ** 2 + (1 / 4) * (3 * r - 4 * q + p) ** 2

    w0r, w1r, w2r = _weno5_weights(b0r, b1r, b2r, 1 / 10, 6 / 10, 3 / 10)
    v_plus = w0r * q0r + w1r * q1r + w2r * q2r

    return v_minus, v_plus
