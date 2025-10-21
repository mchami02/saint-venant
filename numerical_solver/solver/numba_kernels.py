"""
Numba-accelerated computational kernels for shallow water solver.

These JIT-compiled functions provide significant speedup for performance-critical operations.
"""

import numpy as np
from numba import njit, prange


@njit(fastmath=True)
def physical_flux_kernel(h: float, q: float, g: float) -> np.ndarray:
    """Compute physical flux (JIT-compiled).
    
    Args:
        h: Water height
        q: Discharge
        g: Gravity
        
    Returns:
        Flux vector [q, q²/h + 0.5*g*h²]
    """
    flux = np.zeros(2)
    qx = q if h > 0.0 else 0.0
    flux[0] = qx
    flux[1] = qx * qx / h + 0.5 * g * h * h
    return flux


@njit(fastmath=True)
def compute_wave_speed_kernel(h_g: float, q_g: float, h_d: float, q_d: float, g: float):
    """Compute wave speeds (JIT-compiled).
    
    Returns:
        Tuple of (lambda1, lambda2)
    """
    u_g = q_g / h_g if h_g >= 1e-6 else 0.0
    u_d = q_d / h_d if h_d >= 1e-6 else 0.0
    
    lambda1 = min(u_g - np.sqrt(g * h_g), u_d - np.sqrt(g * h_d))
    lambda2 = max(u_g + np.sqrt(g * h_g), u_d + np.sqrt(g * h_d))
    
    return lambda1, lambda2


@njit(fastmath=True)
def lax_friedrichs_flux_kernel(sol_g: np.ndarray, sol_d: np.ndarray, 
                                 dt: float, dx: float, g: float) -> np.ndarray:
    """Lax-Friedrichs numerical flux (JIT-compiled)."""
    b = dx / dt
    flux_g = physical_flux_kernel(sol_g[0], sol_g[1], g)
    flux_d = physical_flux_kernel(sol_d[0], sol_d[1], g)
    return 0.5 * ((flux_d + flux_g) - b * (sol_d - sol_g))


@njit(fastmath=True)
def rusanov_flux_kernel(sol_g: np.ndarray, sol_d: np.ndarray, g: float) -> np.ndarray:
    """Rusanov numerical flux (JIT-compiled)."""
    flux = np.zeros(2)
    
    lambda1, lambda2 = compute_wave_speed_kernel(sol_g[0], sol_g[1], sol_d[0], sol_d[1], g)
    b = max(abs(lambda1), abs(lambda2))
    
    h_g, h_d = sol_g[0], sol_d[0]
    
    if h_g > 1e-6 and h_d > 1e-6:
        flux_g = physical_flux_kernel(sol_g[0], sol_g[1], g)
        flux_d = physical_flux_kernel(sol_d[0], sol_d[1], g)
        flux = 0.5 * ((flux_d + flux_g) - b * (sol_d - sol_g))
    elif h_g < 1e-6 and h_d > 1e-6:
        flux_d = physical_flux_kernel(sol_d[0], sol_d[1], g)
        flux = 0.5 * (flux_d - b * sol_d)
    elif h_d < 1e-6 and h_g > 1e-6:
        flux_g = physical_flux_kernel(sol_g[0], sol_g[1], g)
        flux = 0.5 * (flux_g + b * sol_g)
    
    return flux


@njit(fastmath=True)
def hll_flux_kernel(sol_g: np.ndarray, sol_d: np.ndarray, g: float) -> np.ndarray:
    """HLL numerical flux (JIT-compiled)."""
    flux = np.zeros(2)
    
    lambda1, lambda2 = compute_wave_speed_kernel(sol_g[0], sol_g[1], sol_d[0], sol_d[1], g)
    
    h_g, h_d = sol_g[0], sol_d[0]
    
    if 0 <= lambda1:
        if h_g >= 1e-6:
            flux = physical_flux_kernel(sol_g[0], sol_g[1], g)
    elif lambda1 < 0 < lambda2:
        if h_g > 1e-6 and h_d > 1e-6:
            flux_g = physical_flux_kernel(sol_g[0], sol_g[1], g)
            flux_d = physical_flux_kernel(sol_d[0], sol_d[1], g)
            flux = (lambda2 * flux_g - lambda1 * flux_d + lambda2 * lambda1 * (sol_d - sol_g)) / (lambda2 - lambda1)
        elif h_g < 1e-6 and h_d > 1e-6:
            flux_d = physical_flux_kernel(sol_d[0], sol_d[1], g)
            flux = (-lambda1 * flux_d + lambda2 * lambda1 * sol_d) / (lambda2 - lambda1)
        elif h_d < 1e-6 and h_g > 1e-6:
            flux_g = physical_flux_kernel(sol_g[0], sol_g[1], g)
            flux = (lambda2 * flux_g - lambda2 * lambda1 * sol_g) / (lambda2 - lambda1)
    elif lambda2 <= 0:
        if h_d >= 1e-6:
            flux = physical_flux_kernel(sol_d[0], sol_d[1], g)
    
    return flux


@njit(fastmath=True)
def minmod(a: float, b: float) -> float:
    """Minmod slope limiter (JIT-compiled)."""
    if a * b < 0:
        return 0.0
    elif abs(a) < abs(b):
        return a
    else:
        return b


@njit(fastmath=True, parallel=True)
def build_source_term_bump(source: np.ndarray, sol: np.ndarray, 
                            cell_centers: np.ndarray, g: float):
    """Build source term for bump topography (JIT-compiled, parallelized).
    
    Args:
        source: Output source term array (modified in-place)
        sol: Current solution
        cell_centers: Cell center coordinates
        g: Gravitational acceleration
    """
    n_cells = len(cell_centers)
    source.fill(0.0)
    
    for i in prange(n_cells):
        x = cell_centers[i]
        if 8 < x < 12:
            source[i, 1] = g * sol[i, 0] * 0.05 * 2.0 * (x - 10.0)


@njit(fastmath=True, parallel=True)
def build_source_term_thacker(source: np.ndarray, sol: np.ndarray,
                               cell_centers: np.ndarray, xmin: float, 
                               xmax: float, g: float):
    """Build source term for Thacker topography (JIT-compiled, parallelized).
    
    Args:
        source: Output source term array (modified in-place)
        sol: Current solution
        cell_centers: Cell center coordinates
        xmin: Minimum x coordinate
        xmax: Maximum x coordinate
        g: Gravitational acceleration
    """
    n_cells = len(cell_centers)
    source.fill(0.0)
    L = xmax - xmin
    a, h0 = 1.0, 0.5
    
    for i in prange(n_cells):
        x = cell_centers[i]
        source[i, 1] = -g * sol[i, 0] * h0 * (2.0 / (a * a) * (x - 0.5 * L))


@njit(fastmath=True, parallel=True)
def build_source_term_file(source: np.ndarray, sol: np.ndarray, 
                             topography: np.ndarray, dx: float, g: float):
    """Build source term for file topography (JIT-compiled, parallelized).
    
    Uses central differences for interior points, one-sided differences at boundaries.
    
    Args:
        source: Output source term array (modified in-place)
        sol: Current solution
        topography: Topography array
        dx: Spatial step
        g: Gravitational acceleration
    """
    n_cells = len(topography)
    source.fill(0.0)
    
    # First cell (forward difference)
    source[0, 1] = -g * sol[0, 0] * (-topography[2] + 4.0 * topography[1] - 3.0 * topography[0]) / (2.0 * dx)
    
    # Interior cells (central difference) - parallelized
    for i in prange(1, n_cells - 1):
        source[i, 1] = -g * sol[i, 0] * (topography[i + 1] - topography[i - 1]) / (2.0 * dx)
    
    # Last cell (backward difference)
    source[n_cells - 1, 1] = -g * sol[n_cells - 1, 0] * (3.0 * topography[n_cells - 1] - 4.0 * topography[n_cells - 2] + topography[n_cells - 3]) / (2.0 * dx)


@njit(fastmath=True)
def explicit_euler_step(sol: np.ndarray, flux_vector: np.ndarray, 
                         source: np.ndarray, dt: float, dx: float) -> np.ndarray:
    """Explicit Euler time step (JIT-compiled).
    
    Computes: Sol^{n+1} = Sol^n + dt * (F/dx + S)
    
    Args:
        sol: Current solution
        flux_vector: Numerical flux vector
        source: Source term
        dt: Time step
        dx: Spatial step
        
    Returns:
        Updated solution
    """
    return sol + dt * (flux_vector / dx + source)


@njit(fastmath=True)
def rk2_step(sol: np.ndarray, k1: np.ndarray, k2: np.ndarray, dt: float) -> np.ndarray:
    """RK2 (Heun's method) time step (JIT-compiled).
    
    Computes: Sol^{n+1} = Sol^n + 0.5*dt*(k1 + k2)
    
    Args:
        sol: Current solution
        k1: First stage derivative
        k2: Second stage derivative
        dt: Time step
        
    Returns:
        Updated solution
    """
    return sol + 0.5 * dt * (k1 + k2)

