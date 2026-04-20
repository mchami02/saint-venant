"""Initial condition generators for the 2D compressible Euler system (PyTorch).

All public functions return primitives ``(rho0, u0, v0, p0)`` as four
tensors of shape ``(ny, nx)``.

The Liska-Wendroff 2D Riemann configurations are taken from:
    R. Liska and B. Wendroff, "Comparison of Several Difference Schemes on
    1D and 2D Test Problems for the Euler Equations", SIAM J. Sci. Comput.,
    25(3), 995-1017, 2003.
"""

import torch


def _broadcast(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xx = x.unsqueeze(0).expand(y.shape[0], -1)
    yy = y.unsqueeze(1).expand(-1, x.shape[0])
    return xx, yy


def sod_x(
    x: torch.Tensor,
    y: torch.Tensor,
    x_split: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """2D Sod shock tube with discontinuity in x, uniform in y.

    Left:  (rho, u, v, p) = (1.0, 0.0, 0.0, 1.0)
    Right: (rho, u, v, p) = (0.125, 0.0, 0.0, 0.1)
    """
    if x_split is None:
        x_split = (x.min() + x.max()).item() / 2.0
    xx, _ = _broadcast(x, y)
    left = xx < x_split
    rho = torch.where(left, torch.ones_like(xx), 0.125 * torch.ones_like(xx))
    u = torch.zeros_like(xx)
    v = torch.zeros_like(xx)
    p = torch.where(left, torch.ones_like(xx), 0.1 * torch.ones_like(xx))
    return rho, u, v, p


def sod_y(
    x: torch.Tensor,
    y: torch.Tensor,
    y_split: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """2D Sod shock tube with discontinuity in y, uniform in x."""
    if y_split is None:
        y_split = (y.min() + y.max()).item() / 2.0
    _, yy = _broadcast(x, y)
    bottom = yy < y_split
    rho = torch.where(bottom, torch.ones_like(yy), 0.125 * torch.ones_like(yy))
    u = torch.zeros_like(yy)
    v = torch.zeros_like(yy)
    p = torch.where(bottom, torch.ones_like(yy), 0.1 * torch.ones_like(yy))
    return rho, u, v, p


def four_quadrant(
    x: torch.Tensor,
    y: torch.Tensor,
    states: tuple[dict, dict, dict, dict],
    x_split: float | None = None,
    y_split: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Four-quadrant 2D Riemann problem.

    Parameters
    ----------
    states : 4-tuple of dicts with keys ``rho, u, v, p`` in order
        (bottom-left, bottom-right, top-left, top-right).
    """
    if x_split is None:
        x_split = (x.min() + x.max()).item() / 2.0
    if y_split is None:
        y_split = (y.min() + y.max()).item() / 2.0
    xx, yy = _broadcast(x, y)
    left = xx < x_split
    bottom = yy < y_split

    bl, br, tl, tr = states

    def _field(key: str) -> torch.Tensor:
        return torch.where(
            bottom,
            torch.where(
                left,
                torch.full_like(xx, bl[key]),
                torch.full_like(xx, br[key]),
            ),
            torch.where(
                left,
                torch.full_like(xx, tl[key]),
                torch.full_like(xx, tr[key]),
            ),
        )

    return _field("rho"), _field("u"), _field("v"), _field("p")


# Liska-Wendroff configurations. Values taken from the paper (Table 4.2 of
# Liska-Wendroff 2003) and the PyClaw quadrants example. Quadrant order is
# (top-right, top-left, bottom-left, bottom-right) in the paper, but here
# we pass them to ``four_quadrant`` in (BL, BR, TL, TR) order.
_LISKA_WENDROFF = {
    # Config 3: four shocks meeting at the centre.
    3: dict(
        tr={"rho": 1.5, "u": 0.0, "v": 0.0, "p": 1.5},
        tl={"rho": 0.5323, "u": 1.206, "v": 0.0, "p": 0.3},
        bl={"rho": 0.138, "u": 1.206, "v": 1.206, "p": 0.029},
        br={"rho": 0.5323, "u": 0.0, "v": 1.206, "p": 0.3},
    ),
    # Config 4: four shocks, symmetrical with swirl.
    4: dict(
        tr={"rho": 1.1, "u": 0.0, "v": 0.0, "p": 1.1},
        tl={"rho": 0.5065, "u": 0.8939, "v": 0.0, "p": 0.35},
        bl={"rho": 1.1, "u": 0.8939, "v": 0.8939, "p": 1.1},
        br={"rho": 0.5065, "u": 0.0, "v": 0.8939, "p": 0.35},
    ),
    # Config 6: four contact discontinuities.
    6: dict(
        tr={"rho": 1.0, "u": 0.75, "v": -0.5, "p": 1.0},
        tl={"rho": 2.0, "u": 0.75, "v": 0.5, "p": 1.0},
        bl={"rho": 1.0, "u": -0.75, "v": 0.5, "p": 1.0},
        br={"rho": 3.0, "u": -0.75, "v": -0.5, "p": 1.0},
    ),
}


def liska_wendroff(
    x: torch.Tensor,
    y: torch.Tensor,
    config: int = 3,
    x_split: float | None = None,
    y_split: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Liska-Wendroff (2003) 2D Riemann configuration.

    Supports config 3 (four-shock quadrant test used in the PyClaw gallery),
    config 4 (four shocks with swirl), and config 6 (four contacts).
    """
    if config not in _LISKA_WENDROFF:
        raise ValueError(
            f"Unsupported Liska-Wendroff config {config}; "
            f"available: {sorted(_LISKA_WENDROFF.keys())}"
        )
    s = _LISKA_WENDROFF[config]
    return four_quadrant(
        x, y, (s["bl"], s["br"], s["tl"], s["tr"]),
        x_split=x_split, y_split=y_split,
    )


def _random_breaks_1d(
    n_cells: int,
    n_breaks: int,
    d: float,
    origin: float,
    rng: torch.Generator,
) -> list[float]:
    """Place ``n_breaks`` breakpoints at random offsets inside n_breaks
    distinct cells of a uniform grid. Mirrors the 1D convention in
    ``src/burgers/initial_conditions.py`` and ``src/euler/initial_conditions.py``.
    """
    if n_breaks > n_cells:
        raise ValueError(
            f"Cannot place {n_breaks} breakpoints in {n_cells} cells"
        )
    if n_breaks == 0:
        return []
    cells = torch.randperm(n_cells, generator=rng)[:n_breaks].sort().values
    offsets = torch.rand(n_breaks, generator=rng)
    return ((cells + offsets) * d + origin).tolist()


def _tile_index(x: torch.Tensor, breaks: list[float], dx: float) -> torch.Tensor:
    """Return the tile index (0..len(breaks)) for each grid cell.

    Uses cell centres (``x + dx/2``) so that a breakpoint placed inside
    cell ``i`` is honoured at the interface rather than snapped to a face.
    """
    if len(breaks) == 0:
        return torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    b = torch.tensor(breaks, dtype=x.dtype, device=x.device)
    return torch.bucketize(x + dx / 2, b)


def random_piecewise(
    x: torch.Tensor,
    y: torch.Tensor,
    kx: int,
    ky: int,
    rng: torch.Generator,
    rho_range: tuple[float, float] = (0.1, 2.0),
    u_range: tuple[float, float] = (-2.0, 2.0),
    v_range: tuple[float, float] = (-2.0, 2.0),
    p_range: tuple[float, float] = (0.1, 5.0),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Random block-constant (kx * ky tiles) primitive state.

    Tiles are axis-aligned rectangles whose widths and heights are random:
    we place ``kx - 1`` x-breakpoints and ``ky - 1`` y-breakpoints inside
    randomly-chosen distinct cells (at a random sub-cell offset), then
    assign each tile an independent random (rho, u, v, p).
    """
    nx = x.shape[0]
    ny = y.shape[0]
    x_min = x.min().item()
    y_min = y.min().item()
    dx = (x[1] - x[0]).item() if nx > 1 else 0.0
    dy = (y[1] - y[0]).item() if ny > 1 else 0.0

    breaks_x = _random_breaks_1d(nx, kx - 1, dx, x_min, rng)
    breaks_y = _random_breaks_1d(ny, ky - 1, dy, y_min, rng)

    ix = _tile_index(x, breaks_x, dx)
    iy = _tile_index(y, breaks_y, dy)

    rho_lo, rho_hi = rho_range
    u_lo, u_hi = u_range
    v_lo, v_hi = v_range
    p_lo, p_hi = p_range

    rho_vals = torch.rand(ky, kx, generator=rng) * (rho_hi - rho_lo) + rho_lo
    u_vals = torch.rand(ky, kx, generator=rng) * (u_hi - u_lo) + u_lo
    v_vals = torch.rand(ky, kx, generator=rng) * (v_hi - v_lo) + v_lo
    p_vals = torch.rand(ky, kx, generator=rng) * (p_hi - p_lo) + p_lo

    gather_rows = iy.unsqueeze(1).expand(ny, nx)
    gather_cols = ix.unsqueeze(0).expand(ny, nx)
    rho = rho_vals.to(device=x.device)[gather_rows, gather_cols]
    u = u_vals.to(device=x.device)[gather_rows, gather_cols]
    v = v_vals.to(device=x.device)[gather_rows, gather_cols]
    p = p_vals.to(device=x.device)[gather_rows, gather_cols]

    domain_right = x_min + nx * dx
    domain_top = y_min + ny * dy
    ic_params = {
        "xs": [x_min] + breaks_x + [domain_right],
        "ys": [y_min] + breaks_y + [domain_top],
        "rho_ks": rho_vals.tolist(),
        "u_ks": u_vals.tolist(),
        "v_ks": v_vals.tolist(),
        "p_ks": p_vals.tolist(),
    }
    return rho, u, v, p, ic_params


def random_piecewise_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    kx: int,
    ky: int,
    n: int,
    rng: torch.Generator,
    rho_range: tuple[float, float] = (0.1, 2.0),
    u_range: tuple[float, float] = (-2.0, 2.0),
    v_range: tuple[float, float] = (-2.0, 2.0),
    p_range: tuple[float, float] = (0.1, 5.0),
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict]
]:
    """Generate *n* random block-constant ICs as a batch."""
    rhos, us, vs, ps = [], [], [], []
    params: list[dict] = []
    for _ in range(n):
        rho, u, v, p, ic = random_piecewise(
            x, y, kx, ky, rng,
            rho_range=rho_range, u_range=u_range,
            v_range=v_range, p_range=p_range,
        )
        rhos.append(rho); us.append(u); vs.append(v); ps.append(p)
        params.append(ic)
    return (
        torch.stack(rhos), torch.stack(us),
        torch.stack(vs), torch.stack(ps),
        params,
    )
