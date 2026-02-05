"""Base class for wavefront prediction models."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseWavefrontModel(nn.Module, ABC):
    """Abstract base class for wavefront prediction models.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        hidden_channels: Number of hidden channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, in_channels, nt, nx).

        Returns:
            Output tensor of shape (batch, out_channels, nt, nx).
        """
        pass

    def count_parameters(self) -> int:
        """Count trainable parameters.

        Returns:
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
