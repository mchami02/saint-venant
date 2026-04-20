"""Burgers per-segment attention bias.

Inviscid Burgers is structurally identical to LWR: a scalar conservation
law with piecewise-constant ICs.  The only difference is the flux
(f(u) = u^2/2, convex) versus LWR's Greenshields (f(rho) = rho(1-rho),
concave).  ``LWRBias`` is flux-agnostic, so we simply subclass it and
swap the default flux to ``BurgersFlux``.
"""

from .flux import BurgersFlux
from .lwr_bias import LWRBias


class BurgersBias(LWRBias):
    """Burgers-aware per-segment attention bias (inherits from LWRBias)."""

    def __init__(
        self,
        initial_damping_sharpness: float = 5.0,
        flux=None,
        use_damping: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__(
            initial_damping_sharpness=initial_damping_sharpness,
            flux=flux if flux is not None else BurgersFlux(),
            use_damping=use_damping,
            eps=eps,
        )
