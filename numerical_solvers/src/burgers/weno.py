"""WENO-5 reconstruction for Burgers.

Burgers is scalar and the WENO-5 kernel is equation-agnostic, so we simply
re-export the implementation that already lives in ``src/euler/weno.py``.
"""

from ..euler.weno import weno5_reconstruct

__all__ = ["weno5_reconstruct"]
