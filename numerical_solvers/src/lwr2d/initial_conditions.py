"""Initial condition generators for 2D LWR (PyTorch).

All functions return rho0 of shape (ny, nx).
"""

import torch


def riemann_x(
    x: torch.Tensor,
    y: torch.Tensor,
    rho_left: float = 0.8,
    rho_right: float = 0.2,
    x_split: float | None = None,
) -> torch.Tensor:
    """Riemann problem with discontinuity in x, uniform in y.

    Parameters
    ----------
    x : 1-D tensor (nx,).
    y : 1-D tensor (ny,).
    rho_left, rho_right : densities on each side.
    x_split : location of the discontinuity (defaults to domain midpoint).
    """
    if x_split is None:
        x_split = (x.min() + x.max()).item() / 2.0
    # (ny, nx) via broadcasting
    xx = x.unsqueeze(0).expand(y.shape[0], -1)
    return torch.where(xx < x_split, rho_left, rho_right)


def riemann_y(
    x: torch.Tensor,
    y: torch.Tensor,
    rho_bottom: float = 0.8,
    rho_top: float = 0.2,
    y_split: float | None = None,
) -> torch.Tensor:
    """Riemann problem with discontinuity in y, uniform in x.

    Parameters
    ----------
    x : 1-D tensor (nx,).
    y : 1-D tensor (ny,).
    rho_bottom, rho_top : densities below / above the split.
    y_split : location of the discontinuity (defaults to domain midpoint).
    """
    if y_split is None:
        y_split = (y.min() + y.max()).item() / 2.0
    yy = y.unsqueeze(1).expand(-1, x.shape[0])
    return torch.where(yy < y_split, rho_bottom, rho_top)


def four_quadrant(
    x: torch.Tensor,
    y: torch.Tensor,
    rho_bl: float = 0.8,
    rho_br: float = 0.3,
    rho_tl: float = 0.5,
    rho_tr: float = 0.1,
    x_split: float | None = None,
    y_split: float | None = None,
) -> torch.Tensor:
    """Four-quadrant Riemann problem.

    Parameters
    ----------
    x : 1-D tensor (nx,).
    y : 1-D tensor (ny,).
    rho_bl, rho_br, rho_tl, rho_tr : densities in bottom-left, bottom-right,
        top-left, top-right quadrants.
    x_split, y_split : discontinuity locations (default to midpoints).
    """
    if x_split is None:
        x_split = (x.min() + x.max()).item() / 2.0
    if y_split is None:
        y_split = (y.min() + y.max()).item() / 2.0
    xx = x.unsqueeze(0).expand(y.shape[0], -1)
    yy = y.unsqueeze(1).expand(-1, x.shape[0])
    left = xx < x_split
    bottom = yy < y_split
    rho = torch.where(
        bottom,
        torch.where(left, rho_bl, rho_br),
        torch.where(left, rho_tl, rho_tr),
    )
    return rho


def gaussian_bump(
    x: torch.Tensor,
    y: torch.Tensor,
    rho_bg: float = 0.1,
    rho_peak: float = 0.8,
    cx: float | None = None,
    cy: float | None = None,
    sigma: float = 0.1,
) -> torch.Tensor:
    """Smooth Gaussian bump on a uniform background.

    Parameters
    ----------
    x : 1-D tensor (nx,).
    y : 1-D tensor (ny,).
    rho_bg : background density.
    rho_peak : peak density at the bump centre.
    cx, cy : centre of the bump (defaults to domain midpoints).
    sigma : standard deviation of the Gaussian.
    """
    if cx is None:
        cx = (x.min() + x.max()).item() / 2.0
    if cy is None:
        cy = (y.min() + y.max()).item() / 2.0
    xx = x.unsqueeze(0).expand(y.shape[0], -1)
    yy = y.unsqueeze(1).expand(-1, x.shape[0])
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return rho_bg + (rho_peak - rho_bg) * torch.exp(-r2 / (2.0 * sigma**2))


def random_piecewise(
    x: torch.Tensor,
    y: torch.Tensor,
    kx: int,
    ky: int,
    rng: torch.Generator,
    rho_range: tuple[float, float] = (0.1, 0.9),
) -> torch.Tensor:
    """Random kx * ky piecewise-constant initial condition.

    Parameters
    ----------
    x : 1-D tensor (nx,).
    y : 1-D tensor (ny,).
    kx, ky : number of pieces in x and y directions.
    rng : PyTorch random generator for reproducibility.
    rho_range : (min, max) for sampled density values.
    """
    nx = x.shape[0]
    ny = y.shape[0]
    rho_lo, rho_hi = rho_range

    # Sample a (ky, kx) block of random values
    vals = torch.rand(ky, kx, generator=rng) * (rho_hi - rho_lo) + rho_lo

    # Assign each grid cell to a block via integer division
    ix = torch.arange(nx, device=x.device) * kx // nx  # (nx,)
    iy = torch.arange(ny, device=y.device) * ky // ny  # (ny,)
    # Index into the block values
    rho = vals.to(device=x.device)[iy.unsqueeze(1), ix.unsqueeze(0)]
    return rho
