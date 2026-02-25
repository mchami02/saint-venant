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
- RegularizeTrajLoss: Penalizes erratic jumps between consecutive timesteps

Physics Losses:
- PDEResidualLoss: PDE conservation in smooth regions
- PDEShockResidualLoss: PDE residual on GT, penalizing unpredicted shocks
- RHResidualLoss: Rankine-Hugoniot residual from sampled densities

Utilities:
- flux: Centralized flux functions (greenshields_flux, compute_shock_speed)
"""

from .acceleration import AccelerationLoss, compute_acceleration
from .base import BaseLoss
from .cell_avg_mse import CellAverageMSELoss
from .boundary import BoundaryLoss
from .collision import CollisionLoss
from .conservation import ConservationLoss
from .existence_regularization import ICAnchoringLoss
from .flow_matching import FlowMatchingLoss
from .flux import compute_shock_speed, greenshields_flux, greenshields_flux_derivative
from .ic import ICLoss
from .kl_divergence import KLDivergenceLoss
from .mse import MSELoss
from .pde_residual import (
    PDEResidualLoss,
    PDEShockResidualLoss,
    compute_pde_residual,
    create_shock_mask,
)
from .regularize_traj import RegularizeTrajLoss
from .rh_residual import RHResidualLoss
from .selection_supervision import SelectionSupervisionLoss
from .supervised_trajectory import SupervisedTrajectoryLoss
from .trajectory_consistency import TrajectoryConsistencyLoss
from .vae_reconstruction import VAEReconstructionLoss
from .wasserstein import WassersteinLoss

__all__ = [
    # Base
    "BaseLoss",
    # Grid losses
    "MSELoss",
    "CellAverageMSELoss",
    "ICLoss",
    # Trajectory losses
    "TrajectoryConsistencyLoss",
    "BoundaryLoss",
    "CollisionLoss",
    "ICAnchoringLoss",
    "SupervisedTrajectoryLoss",
    "AccelerationLoss",
    "RegularizeTrajLoss",
    # Physics losses
    "PDEResidualLoss",
    "PDEShockResidualLoss",
    "RHResidualLoss",
    # Utilities
    "greenshields_flux",
    "greenshields_flux_derivative",
    "compute_shock_speed",
    "compute_pde_residual",
    "create_shock_mask",
    "compute_acceleration",
    # Latent diffusion losses
    "VAEReconstructionLoss",
    "FlowMatchingLoss",
    # CVAE losses
    "KLDivergenceLoss",
    # CharNO losses
    "WassersteinLoss",
    "ConservationLoss",
    "SelectionSupervisionLoss",
]
