"""
Numba-accelerated computational kernels for shallow water solver.

These JIT-compiled functions provide significant speedup for performance-critical operations.
"""

import numpy as np
from numba import njit, prange


@njit(fastmath=True)
def physical_flux_kernel(h: float, q: float, g: float) -> np.ndarray:
    """Compute physical flux for 1D shallow water equations (JIT-compiled).
    
    The physical flux for the shallow water (Saint-Venant) system is:
        F(U) = [q, q²/h + 0.5*g*h²]ᵀ
    
    where U = [h, q]ᵀ is the conserved variables vector:
        - h: water height (m)
        - q = h*u: discharge (m²/s), where u is velocity
    
    Mathematical derivation:
        Mass conservation: ∂h/∂t + ∂(hu)/∂x = 0
        Momentum conservation: ∂(hu)/∂t + ∂(hu² + 0.5*g*h²)/∂x = S
    
    Args:
        h: Water height (m), must be non-negative
        q: Discharge q = h*u (m²/s)
        g: Gravitational acceleration (m/s²), typically 9.81
        
    Returns:
        np.ndarray: Flux vector of shape (2,) containing:
                   [0] = q (mass flux)
                   [1] = q²/h + 0.5*g*h² (momentum flux)
                   
    Note:
        Sets q = 0 if h ≤ 0 to handle dry states safely.
    """
    flux = np.zeros(2)
    qx = q if h > 0.0 else 0.0
    flux[0] = qx
    flux[1] = qx * qx / h + 0.5 * g * h * h
    return flux


@njit(fastmath=True)
def compute_wave_speed_kernel(h_g: float, q_g: float, h_d: float, q_d: float, g: float):
    """Compute eigenvalues (wave speeds) of the shallow water flux Jacobian (JIT-compiled).
    
    The eigenvalues of the flux Jacobian ∂F/∂U for shallow water are:
        λ₁ = u - c  (left-going wave)
        λ₂ = u + c  (right-going wave)
    
    where c = √(gh) is the gravity wave celerity. These represent the characteristic
    speeds at which information propagates in the shallow water system.
    
    For numerical fluxes, we need the minimum and maximum wave speeds across
    an interface:
        λ_min = min(u_L - c_L, u_R - c_R)
        λ_max = max(u_L + c_L, u_R + c_R)
    
    Args:
        h_g: Left water height (m)
        q_g: Left discharge (m²/s)
        h_d: Right water height (m)
        q_d: Right discharge (m²/s)
        g: Gravitational acceleration (m/s²)
    
    Returns:
        tuple: (lambda1, lambda2) where:
              - lambda1: minimum wave speed (m/s), can be negative
              - lambda2: maximum wave speed (m/s), always ≥ lambda1
              
    Note:
        Uses threshold h ≥ 1e-6 m to avoid division by zero in dry states.
    """
    u_g = q_g / h_g if h_g >= 1e-6 else 0.0
    u_d = q_d / h_d if h_d >= 1e-6 else 0.0
    
    lambda1 = min(u_g - np.sqrt(g * h_g), u_d - np.sqrt(g * h_d))
    lambda2 = max(u_g + np.sqrt(g * h_g), u_d + np.sqrt(g * h_d))
    
    return lambda1, lambda2


@njit(fastmath=True)
def lax_friedrichs_flux_kernel(sol_g: np.ndarray, sol_d: np.ndarray, 
                                 dt: float, dx: float, g: float) -> np.ndarray:
    """Lax-Friedrichs numerical flux kernel (JIT-compiled).
    
    Computes the Lax-Friedrichs (LF) flux with global artificial viscosity:
        F̂_LF(U_L, U_R) = 1/2 * [F(U_L) + F(U_R) - α(U_R - U_L)]
    
    where α = dx/dt is the artificial viscosity coefficient. This is a monotone,
    first-order accurate flux that is very stable but dissipative.
    
    The scheme is consistent with the CFL condition: dt ≤ dx/max|λ|, ensuring
    that α ≥ max|λ| which guarantees stability.
    
    Args:
        sol_g: Left state [h, q] of shape (2,)
        sol_d: Right state [h, q] of shape (2,)
        dt: Time step size (s)
        dx: Spatial step size (m)
        g: Gravitational acceleration (m/s²)
        
    Returns:
        np.ndarray: Numerical flux [F̂_h, F̂_q] of shape (2,)
    """
    b = dx / dt
    flux_g = physical_flux_kernel(sol_g[0], sol_g[1], g)
    flux_d = physical_flux_kernel(sol_d[0], sol_d[1], g)
    return 0.5 * ((flux_d + flux_g) - b * (sol_d - sol_g))


@njit(fastmath=True)
def rusanov_flux_kernel(sol_g: np.ndarray, sol_d: np.ndarray, g: float) -> np.ndarray:
    """Rusanov (local Lax-Friedrichs) numerical flux kernel (JIT-compiled).
    
    Computes the Rusanov flux using local wave speed estimates:
        F̂_Rus(U_L, U_R) = 1/2 * [F(U_L) + F(U_R) - α(U_R - U_L)]
    
    where α = max(|λ₁|, |λ₂|) with λ₁, λ₂ being the minimum and maximum
    eigenvalues at the interface. This provides less numerical dissipation
    than Lax-Friedrichs while maintaining robustness.
    
    Special cases for wet/dry interfaces:
        - Both wet (h_L, h_R > ε): standard Rusanov formula
        - Left dry (h_L ≈ 0): F̂ = 1/2 * (F(U_R) - α*U_R)
        - Right dry (h_R ≈ 0): F̂ = 1/2 * (F(U_L) + α*U_L)
        - Both dry: F̂ = 0
    
    Args:
        sol_g: Left state [h, q] of shape (2,)
        sol_d: Right state [h, q] of shape (2,)
        g: Gravitational acceleration (m/s²)
        
    Returns:
        np.ndarray: Numerical flux [F̂_h, F̂_q] of shape (2,)
        
    Note:
        Dry state threshold: h < 1e-6 m
    """
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
    """HLL (Harten-Lax-van Leer) numerical flux kernel (JIT-compiled).
    
    Computes the HLL approximate Riemann solver flux. The HLL method assumes
    a simplified Riemann problem solution with three constant states separated
    by two waves with speeds λ₁ and λ₂:
    
        F̂_HLL = { F(U_L)                                           if 0 ≤ λ₁
                { [λ₂F(U_L) - λ₁F(U_R) + λ₁λ₂(U_R - U_L)]/(λ₂-λ₁)  if λ₁ < 0 < λ₂
                { F(U_R)                                           if λ₂ ≤ 0
    
    Physical interpretation:
        - λ₁ ≥ 0: Both waves propagate right → use left state flux
        - λ₂ ≤ 0: Both waves propagate left → use right state flux  
        - λ₁ < 0 < λ₂: Transonic flow → use averaged star region flux
    
    The HLL flux:
        - Is entropy-satisfying
        - Preserves positivity of water height
        - Resolves shocks and rarefactions accurately
        - Handles wet/dry fronts robustly
    
    Args:
        sol_g: Left state [h, q] of shape (2,)
        sol_d: Right state [h, q] of shape (2,)
        g: Gravitational acceleration (m/s²)
        
    Returns:
        np.ndarray: Numerical flux [F̂_h, F̂_q] of shape (2,)
        
    Note:
        Dry state threshold: h < 1e-6 m. Special formulas used for wet/dry
        interfaces to ensure physical consistency.
    """
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
    """Minmod slope limiter for TVD schemes (JIT-compiled).
    
    The minmod function returns the argument with the smallest absolute value
    if both have the same sign, zero otherwise:
    
        minmod(a, b) = { 0   if a*b ≤ 0  (opposite signs or zero)
                       { a   if |a| < |b| and a*b > 0
                       { b   if |b| ≤ |a| and a*b > 0
    
    This limiter is used in MUSCL reconstruction to maintain the TVD property
    (Total Variation Diminishing), preventing spurious oscillations.
    
    Args:
        a: First slope value
        b: Second slope value
        
    Returns:
        float: Limited slope, ensuring monotonicity preservation
    """
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
    
    For shallow water with topography z(x), the source term in the momentum equation is:
        S = -g*h * dz/dx
    
    For the bump topography:
        z(x) = { 0.2 - 0.05*(x-10)²  if 8 < x < 12
               { 0                    otherwise
    
    The derivative is:
        dz/dx = { -0.1*(x-10)  if 8 < x < 12
                { 0            otherwise
    
    This gives:
        S = g*h * 0.1*(x-10)  for 8 < x < 12
    
    Args:
        source: Output source term array of shape (n_cells, 2), modified in-place.
                source[:, 0] = 0 (no mass source)
                source[:, 1] = -g*h*dz/dx (momentum source from bed slope)
        sol: Current solution array of shape (n_cells, 2) with [h, q]
        cell_centers: Cell center x-coordinates of shape (n_cells,)
        g: Gravitational acceleration (m/s²)
        
    Returns:
        None. Modifies source array in place.
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
    """Build source term for Thacker parabolic bowl topography (JIT-compiled, parallelized).
    
    The Thacker test case uses a parabolic bathymetry:
        z(x) = h₀ * [(x - L/2)²/a² - 1]
    
    where:
        - h₀ = 0.5 m (characteristic depth)
        - a = 1.0 m (characteristic length)
        - L = xmax - xmin (domain length)
    
    The bed slope derivative is:
        dz/dx = h₀ * 2*(x - L/2)/a²
    
    The momentum source term becomes:
        S = -g*h * dz/dx = -g*h*h₀ * 2*(x - L/2)/a²
    
    This test case admits an analytical oscillating solution, useful for
    verifying the well-balanced property of the scheme.
    
    Args:
        source: Output source term array of shape (n_cells, 2), modified in-place
        sol: Current solution array of shape (n_cells, 2) with [h, q]
        cell_centers: Cell center x-coordinates of shape (n_cells,)
        xmin: Minimum x coordinate of domain (m)
        xmax: Maximum x coordinate of domain (m)
        g: Gravitational acceleration (m/s²)
        
    Returns:
        None. Modifies source array in place.
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
    """Build source term for arbitrary topography from file (JIT-compiled, parallelized).
    
    For general topography z(x), the source term is:
        S = -g*h * dz/dx
    
    The bed slope dz/dx is approximated using finite differences:
    
    Interior cells (i = 1, ..., n-2):
        dz/dx ≈ (z_{i+1} - z_{i-1})/(2*dx)  [2nd-order central difference]
    
    Boundary cells:
        Left (i=0): dz/dx ≈ (-z_2 + 4*z_1 - 3*z_0)/(2*dx)  [2nd-order forward]
        Right (i=n-1): dz/dx ≈ (3*z_{n-1} - 4*z_{n-2} + z_{n-3})/(2*dx)  [2nd-order backward]
    
    These one-sided formulas maintain 2nd-order accuracy at boundaries.
    
    Args:
        source: Output source term array of shape (n_cells, 2), modified in-place
        sol: Current solution array of shape (n_cells, 2) with [h, q]
        topography: Bed elevation z(x) at cell centers, shape (n_cells,)
        dx: Uniform spatial step size (m)
        g: Gravitational acceleration (m/s²)
        
    Returns:
        None. Modifies source array in place with:
              source[:, 0] = 0 (no mass source)
              source[:, 1] = -g*h*dz/dx (bed slope source term)
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
    """Explicit Euler (forward Euler) time integration step (JIT-compiled).
    
    Performs one time step of the explicit Euler method for the semi-discrete
    finite volume scheme:
    
        dU_i/dt = -(F̂_{i+1/2} - F̂_{i-1/2})/dx + S_i
    
    The explicit Euler update is:
        U_i^{n+1} = U_i^n + Δt * [-(F̂_{i+1/2} - F̂_{i-1/2})/dx + S_i]
    
    This is a first-order accurate in time method. Stability requires:
        Δt ≤ dx / (|u| + √(gh))  (CFL condition)
    
    Args:
        sol: Current solution array U^n of shape (n_cells, 2) with [h, q]
        flux_vector: Flux differences (F̂_{i+1/2} - F̂_{i-1/2}) of shape (n_cells, 2)
        source: Source term S of shape (n_cells, 2)
        dt: Time step Δt (s)
        dx: Spatial step Δx (m)
        
    Returns:
        np.ndarray: Updated solution U^{n+1} of shape (n_cells, 2)
    """
    return sol + dt * (flux_vector / dx + source)


@njit(fastmath=True)
def rk2_step(sol: np.ndarray, k1: np.ndarray, k2: np.ndarray, dt: float) -> np.ndarray:
    """RK2 (Heun's method / modified Euler) time integration step (JIT-compiled).
    
    Performs one time step of the second-order Runge-Kutta method (Heun's method):
    
        Stage 1: k₁ = F(U^n)
        Stage 2: U* = U^n + Δt*k₁
                k₂ = F(U*)
        Update:  U^{n+1} = U^n + (Δt/2)*(k₁ + k₂)
    
    where F(U) = -(F̂_{i+1/2} - F̂_{i-1/2})/dx + S_i is the spatial operator.
    
    This method is:
        - Second-order accurate in time
        - More accurate than explicit Euler for same CFL
        - Requires two flux evaluations per time step
        - Stability: Δt ≤ dx / (|u| + √(gh)) (same CFL as Euler)
    
    Args:
        sol: Current solution U^n of shape (n_cells, 2) with [h, q]
        k1: First stage derivative k₁ = F(U^n) of shape (n_cells, 2)
        k2: Second stage derivative k₂ = F(U^n + Δt*k₁) of shape (n_cells, 2)
        dt: Time step Δt (s)
        
    Returns:
        np.ndarray: Updated solution U^{n+1} of shape (n_cells, 2)
    """
    return sol + 0.5 * dt * (k1 + k2)

