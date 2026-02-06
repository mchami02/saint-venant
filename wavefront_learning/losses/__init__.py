"""Loss functions for wavefront prediction.

This package provides modular loss functions for training wavefront prediction models:

Base:
- BaseLoss: Abstract base class with unified interface

Grid Losses:
- MSELoss: Mean squared error for grid predictions
- ICLoss: Initial condition matching at t=0

Trajectory Losses:
- TrajectoryConsistencyLoss: Enforces Rankine-Hugoniot trajectory consistency
- BoundaryLoss: Penalizes shocks outside domain
- CollisionLoss: Penalizes colliding shocks
- ICAnchoringLoss: Anchors trajectories to IC positions
- SupervisedTrajectoryLoss: Supervised trajectory loss when GT available
- AccelerationLoss: Penalizes low existence where GT has high acceleration

Physics Losses:
- PDEResidualLoss: PDE conservation in smooth regions
- RHResidualLoss: Rankine-Hugoniot residual from sampled densities

Utilities:
- flux: Centralized flux functions (greenshields_flux, compute_shock_speed)
"""

from .acceleration import AccelerationLoss, compute_acceleration
from .base import BaseLoss
from .boundary import BoundaryLoss
from .collision import CollisionLoss
from .existence_regularization import ICAnchoringLoss
from .flux import compute_shock_speed, greenshields_flux, greenshields_flux_derivative
from .ic import ICLoss
from .mse import MSELoss
from .pde_residual import PDEResidualLoss, compute_pde_residual, create_shock_mask
from .rh_residual import RHResidualLoss
from .supervised_trajectory import SupervisedTrajectoryLoss
from .trajectory_consistency import TrajectoryConsistencyLoss

__all__ = [
    # Base
    "BaseLoss",
    # Grid losses
    "MSELoss",
    "ICLoss",
    # Trajectory losses
    "TrajectoryConsistencyLoss",
    "BoundaryLoss",
    "CollisionLoss",
    "ICAnchoringLoss",
    "SupervisedTrajectoryLoss",
    "AccelerationLoss",
    # Physics losses
    "PDEResidualLoss",
    "RHResidualLoss",
    # Utilities
    "greenshields_flux",
    "greenshields_flux_derivative",
    "compute_shock_speed",
    "compute_pde_residual",
    "create_shock_mask",
    "compute_acceleration",
]
