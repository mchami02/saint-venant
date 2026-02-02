"""Base class for wavefront prediction losses."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseLoss(nn.Module, ABC):
    """Abstract base class for wavefront prediction losses."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute loss.

        Args:
            prediction: Model prediction tensor.
            target: Ground truth tensor.
            **kwargs: Additional arguments (e.g., dx, dt for physics losses).

        Returns:
            Scalar loss tensor.
        """
        pass


class MSELoss(BaseLoss):
    """Mean squared error loss."""

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute MSE loss.

        Args:
            prediction: Model prediction tensor.
            target: Ground truth tensor.
            **kwargs: Unused.

        Returns:
            Scalar MSE loss tensor.
        """
        # TODO: Implement MSE loss
        pass


class RankineHugoniotLoss(BaseLoss):
    """Physics-informed loss based on Rankine-Hugoniot conditions.

    Args:
        dx: Spatial step size.
        dt: Time step size.
        weight: Weight for this loss component.
    """

    def __init__(self, dx: float, dt: float, weight: float = 1.0):
        super().__init__()
        self.dx = dx
        self.dt = dt
        self.weight = weight

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute Rankine-Hugoniot loss.

        Args:
            prediction: Model prediction tensor.
            target: Ground truth tensor.
            **kwargs: Additional arguments.

        Returns:
            Scalar loss tensor.
        """
        # TODO: Implement Rankine-Hugoniot loss
        pass
