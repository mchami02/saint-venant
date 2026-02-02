"""Rankine-Hugoniot physics loss for unsupervised shock trajectory learning.

This module provides loss functions based on the Rankine-Hugoniot jump conditions
for the LWR traffic flow equation with Greenshields flux.

For Greenshields flux f(rho) = rho * (1 - rho) with v_max = rho_max = 1:
    Shock speed: s = (f(rho_R) - f(rho_L)) / (rho_R - rho_L) = 1 - (rho_L + rho_R)
    Trajectory: x(t) = x_0 + s * t
"""

import torch
import torch.nn as nn


def compute_shock_speed(rho_L: torch.Tensor, rho_R: torch.Tensor) -> torch.Tensor:
    """Compute shock speed from Rankine-Hugoniot condition for Greenshields flux.

    For Greenshields flux f(rho) = rho * (1 - rho):
        s = [f(rho_R) - f(rho_L)] / [rho_R - rho_L]
          = [rho_R(1-rho_R) - rho_L(1-rho_L)] / [rho_R - rho_L]
          = 1 - rho_L - rho_R

    Args:
        rho_L: Left density values.
        rho_R: Right density values.

    Returns:
        Shock speeds with same shape as inputs.
    """
    return 1.0 - rho_L - rho_R


def compute_analytical_trajectory(
    x_0: torch.Tensor,
    rho_L: torch.Tensor,
    rho_R: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    """Compute analytical shock trajectory using Rankine-Hugoniot condition.

    Args:
        x_0: Initial shock positions of shape (B, D).
        rho_L: Left densities of shape (B, D).
        rho_R: Right densities of shape (B, D).
        times: Query times of shape (B, T) or (T,).

    Returns:
        Analytical positions of shape (B, D, T).
    """
    # Compute shock speeds: (B, D)
    speeds = compute_shock_speed(rho_L, rho_R)

    # Handle 1D times
    if times.dim() == 1:
        times = times.unsqueeze(0)  # (1, T)

    # Expand for broadcasting: x_0 (B, D, 1), speeds (B, D, 1), times (B, 1, T)
    x_0 = x_0.unsqueeze(-1)  # (B, D, 1)
    speeds = speeds.unsqueeze(-1)  # (B, D, 1)
    times = times.unsqueeze(1)  # (B, 1, T)

    # Compute trajectory: x(t) = x_0 + s * t
    positions = x_0 + speeds * times  # (B, D, T)

    return positions


def trajectory_consistency_loss(
    predicted_positions: torch.Tensor,
    discontinuities: torch.Tensor,
    query_times: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute loss enforcing Rankine-Hugoniot trajectory consistency.

    The predicted positions should match the analytical trajectory:
        x_pred(t) ≈ x_0 + s * t
    where s = 1 - (rho_L + rho_R).

    Args:
        predicted_positions: Predicted positions of shape (B, D, T).
        discontinuities: Initial discontinuities of shape (B, D, 3)
            where columns are [x_0, rho_L, rho_R].
        query_times: Query times of shape (B, T).
        mask: Validity mask of shape (B, D).

    Returns:
        Scalar loss tensor.
    """
    # Extract initial conditions
    x_0 = discontinuities[..., 0]  # (B, D)
    rho_L = discontinuities[..., 1]  # (B, D)
    rho_R = discontinuities[..., 2]  # (B, D)

    # Compute analytical trajectory
    analytical_positions = compute_analytical_trajectory(x_0, rho_L, rho_R, query_times)

    # Compute squared error
    error = (predicted_positions - analytical_positions) ** 2  # (B, D, T)

    # Apply mask: only count valid discontinuities
    mask_exp = mask.unsqueeze(-1)  # (B, D, 1)
    masked_error = error * mask_exp

    # Average over valid entries
    n_valid = mask.sum() * query_times.shape[-1]
    if n_valid > 0:
        loss = masked_error.sum() / n_valid
    else:
        loss = torch.tensor(0.0, device=predicted_positions.device)

    return loss


def boundary_loss(
    predicted_positions: torch.Tensor,
    predicted_existence: torch.Tensor,
    mask: torch.Tensor,
    domain_min: float = 0.0,
    domain_max: float = 1.0,
) -> torch.Tensor:
    """Compute loss penalizing existence of shocks outside the domain.

    When a shock exits the domain [0, 1], its existence probability should be 0.

    Args:
        predicted_positions: Predicted positions of shape (B, D, T).
        predicted_existence: Predicted existence of shape (B, D, T).
        mask: Validity mask of shape (B, D).
        domain_min: Minimum domain boundary.
        domain_max: Maximum domain boundary.

    Returns:
        Scalar loss tensor.
    """
    # Check which positions are outside the domain
    outside = (predicted_positions < domain_min) | (predicted_positions > domain_max)
    outside = outside.float()  # (B, D, T)

    # Penalize existence when outside
    penalty = outside * (predicted_existence ** 2)

    # Apply mask
    mask_exp = mask.unsqueeze(-1)  # (B, D, 1)
    masked_penalty = penalty * mask_exp

    # Average over valid entries
    n_valid = mask.sum() * predicted_positions.shape[-1]
    if n_valid > 0:
        loss = masked_penalty.sum() / n_valid
    else:
        loss = torch.tensor(0.0, device=predicted_positions.device)

    return loss


def collision_loss(
    predicted_positions: torch.Tensor,
    predicted_existence: torch.Tensor,
    mask: torch.Tensor,
    collision_threshold: float = 0.02,
) -> torch.Tensor:
    """Compute loss penalizing simultaneous existence of colliding shocks.

    When two shocks are very close (within threshold), at most one should exist.
    This encourages the model to predict shock merging.

    Args:
        predicted_positions: Predicted positions of shape (B, D, T).
        predicted_existence: Predicted existence of shape (B, D, T).
        mask: Validity mask of shape (B, D).
        collision_threshold: Distance threshold for collision detection.

    Returns:
        Scalar loss tensor.
    """
    _, D, T = predicted_positions.shape

    if D < 2:
        return torch.tensor(0.0, device=predicted_positions.device)

    total_loss = torch.tensor(0.0, device=predicted_positions.device)
    n_pairs = 0

    # Compare all pairs of discontinuities
    for i in range(D):
        for j in range(i + 1, D):
            # Get positions and existence for this pair
            pos_i = predicted_positions[:, i, :]  # (B, T)
            pos_j = predicted_positions[:, j, :]  # (B, T)
            exist_i = predicted_existence[:, i, :]  # (B, T)
            exist_j = predicted_existence[:, j, :]  # (B, T)
            mask_i = mask[:, i]  # (B,)
            mask_j = mask[:, j]  # (B,)

            # Distance between shocks
            distance = torch.abs(pos_i - pos_j)  # (B, T)

            # Collision indicator: 1 if close, 0 otherwise
            colliding = (distance < collision_threshold).float()

            # Penalty: product of existences when colliding
            penalty = colliding * exist_i * exist_j  # (B, T)

            # Apply mask: both must be valid
            pair_mask = (mask_i * mask_j).unsqueeze(-1)  # (B, 1)
            masked_penalty = penalty * pair_mask

            total_loss = total_loss + masked_penalty.sum()
            n_pairs = n_pairs + pair_mask.sum() * T

    if n_pairs > 0:
        return total_loss / n_pairs
    return total_loss


def existence_regularization(
    predicted_existence: torch.Tensor,
    mask: torch.Tensor,
    target_mean: float = 0.5,
) -> torch.Tensor:
    """Regularization to prevent existence from collapsing to 0 or 1.

    Encourages existence predictions to be varied rather than all zeros/ones.

    Args:
        predicted_existence: Predicted existence of shape (B, D, T).
        mask: Validity mask of shape (B, D).
        target_mean: Target mean existence value.

    Returns:
        Scalar regularization loss.
    """
    mask_exp = mask.unsqueeze(-1)  # (B, D, 1)
    masked_exist = predicted_existence * mask_exp

    n_valid = mask.sum() * predicted_existence.shape[-1]
    if n_valid > 0:
        mean_exist = masked_exist.sum() / n_valid
        return (mean_exist - target_mean) ** 2
    return torch.tensor(0.0, device=predicted_existence.device)


class RankineHugoniotLoss(nn.Module):
    """Combined unsupervised loss for shock trajectory prediction.

    Combines multiple physics-based loss components:
    - Trajectory consistency: enforce Rankine-Hugoniot condition
    - Boundary loss: shocks should vanish outside domain
    - Collision loss: colliding shocks should merge (one vanishes)
    - Existence regularization: prevent trivial solutions

    Args:
        trajectory_weight: Weight for trajectory consistency loss.
        boundary_weight: Weight for boundary loss.
        collision_weight: Weight for collision loss.
        regularization_weight: Weight for existence regularization.
        domain_min: Minimum domain boundary.
        domain_max: Maximum domain boundary.
        collision_threshold: Distance threshold for collision detection.
    """

    def __init__(
        self,
        trajectory_weight: float = 1.0,
        boundary_weight: float = 1.0,
        collision_weight: float = 0.5,
        regularization_weight: float = 0.1,
        domain_min: float = 0.0,
        domain_max: float = 1.0,
        collision_threshold: float = 0.02,
    ):
        super().__init__()
        self.trajectory_weight = trajectory_weight
        self.boundary_weight = boundary_weight
        self.collision_weight = collision_weight
        self.regularization_weight = regularization_weight
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.collision_threshold = collision_threshold

    def forward(
        self,
        model_output: dict[str, torch.Tensor],
        batch_input: dict[str, torch.Tensor],
        target: torch.Tensor = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute total loss and individual components.

        Args:
            model_output: Dict with 'positions' and 'existence' tensors.
            batch_input: Dict with 'discontinuities', 't_coords', and 'disc_mask' tensors.

        Returns:
            Tuple of (total_loss, loss_components_dict).
        """
        positions = model_output["positions"]
        existence = model_output["existence"]

        discontinuities = batch_input["discontinuities"]
        query_times = batch_input["t_coords"][:, 0, :, 0]
        mask = batch_input["disc_mask"]

        # Compute individual losses
        loss_traj = trajectory_consistency_loss(positions, discontinuities, query_times, mask)
        loss_bound = boundary_loss(positions, existence, mask, self.domain_min, self.domain_max)
        loss_coll = collision_loss(positions, existence, mask, self.collision_threshold)
        loss_reg = existence_regularization(existence, mask)

        # Combine with weights
        total_loss = (
            self.trajectory_weight * loss_traj
            + self.boundary_weight * loss_bound
            + self.collision_weight * loss_coll
            + self.regularization_weight * loss_reg
        )

        components = {
            "trajectory": loss_traj.item(),
            "boundary": loss_bound.item(),
            "collision": loss_coll.item(),
            "regularization": loss_reg.item(),
            "total": total_loss.item(),
        }

        return total_loss, components


class SupervisedTrajectoryLoss(nn.Module):
    """Supervised loss for trajectory prediction when ground truth is available.

    Can be combined with physics loss for semi-supervised training.

    Args:
        position_weight: Weight for position MSE loss.
        existence_weight: Weight for existence BCE loss.
    """

    def __init__(
        self,
        position_weight: float = 1.0,
        existence_weight: float = 1.0,
    ):
        super().__init__()
        self.position_weight = position_weight
        self.existence_weight = existence_weight
        self.mse = nn.MSELoss(reduction="none")
        self.bce = nn.BCELoss(reduction="none")

    def forward(
        self,
        model_output: dict[str, torch.Tensor],
        target_positions: torch.Tensor,
        target_existence: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute supervised loss.

        Args:
            model_output: Dict with 'positions' and 'existence' tensors.
            target_positions: Ground truth positions of shape (B, D, T).
            target_existence: Ground truth existence of shape (B, D, T).
            mask: Validity mask of shape (B, D).

        Returns:
            Tuple of (total_loss, loss_components_dict).
        """
        positions = model_output["positions"]
        existence = model_output["existence"]

        # Position MSE loss
        pos_error = self.mse(positions, target_positions)
        mask_exp = mask.unsqueeze(-1)
        pos_loss = (pos_error * mask_exp).sum() / (mask.sum() * positions.shape[-1] + 1e-8)

        # Existence BCE loss
        exist_error = self.bce(existence, target_existence)
        exist_loss = (exist_error * mask_exp).sum() / (mask.sum() * existence.shape[-1] + 1e-8)

        total_loss = self.position_weight * pos_loss + self.existence_weight * exist_loss

        components = {
            "position": pos_loss.item(),
            "existence": exist_loss.item(),
            "total": total_loss.item(),
        }

        return total_loss, components
