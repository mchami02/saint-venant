"""Neural network models for wavefront prediction.

Main models:
- ShockTrajectoryNet: DeepONet-like model for shock trajectory prediction
- HybridDeepONet: Combined trajectory and region prediction

Building blocks are available in the `base` submodule.
"""

from .base import BaseWavefrontModel
from .deeponet import DeepONet, build_deeponet
from .hybrid_deeponet import HybridDeepONet, build_hybrid_deeponet
from .shock_trajectory_net import ShockTrajectoryNet, build_shock_net

__all__ = [
    # Base class
    "BaseWavefrontModel",
    # Main models
    "ShockTrajectoryNet",
    "HybridDeepONet",
    "DeepONet",
    # Builder functions
    "build_shock_net",
    "build_hybrid_deeponet",
    "build_deeponet",
]
