"""Base class for wavefront prediction losses."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseLoss(nn.Module, ABC):
    """Abstract base class for wavefront prediction losses.

    All losses follow a unified interface:
        forward(input_dict, output_dict, target) -> (loss, components)

    Args for forward:
        input_dict: Dictionary of input tensors (discontinuities, coords, masks, etc.)
        output_dict: Dictionary of model outputs (positions, existence, grids, etc.)
        target: Ground truth tensor (typically the target grid).

    Returns:
        Tuple of (loss_tensor, components_dict) where components_dict contains
        named loss component values as floats.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        output_dict: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute loss.

        Args:
            input_dict: Dictionary of input tensors.
            output_dict: Dictionary of model output tensors.
            target: Ground truth tensor.

        Returns:
            Tuple of (scalar loss tensor, dict of component values).
        """
        pass
