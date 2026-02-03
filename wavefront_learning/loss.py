"""Loss function factory for wavefront learning."""

import torch
import torch.nn as nn
from losses.hybrid_loss import HybridDeepONetLoss
from losses.rankine_hugoniot import RankineHugoniotLoss

# Registry of available loss functions
LOSSES = {
    "rankine_hugoniot": RankineHugoniotLoss,
    "hybrid": HybridDeepONetLoss,
}


def get_loss(loss_name: str, **kwargs) -> nn.Module:
    """Create a loss function instance.

    Args:
        loss_name: Name of the loss function.
        **kwargs: Additional arguments for the loss function.

    Returns:
        Instantiated loss function.

    Raises:
        ValueError: If loss_name is not supported.
    """
    if loss_name not in LOSSES:
        raise ValueError(f"Loss {loss_name} not supported")
    return LOSSES[loss_name](**kwargs)


class CombinedLoss(nn.Module):
    """Combines multiple loss functions with weights.

    Args:
        losses: List of (loss_fn, weight) tuples.
    """

    def __init__(self, losses: list[tuple[nn.Module, float]]):
        super().__init__()
        self.losses = nn.ModuleList([loss for loss, _ in losses])
        self.weights = [w for _, w in losses]

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, dict]:
        """Compute combined loss.

        Args:
            prediction: Model prediction.
            target: Ground truth.
            **kwargs: Additional arguments for individual losses.

        Returns:
            Tuple of (total_loss, loss_components_dict).
        """
        # TODO: Implement combined loss computation
        pass


def create_loss_from_args(args) -> nn.Module:
    """Create loss function from command line arguments.

    Args:
        args: Parsed command line arguments.

    Returns:
        Configured loss function.
    """
    # TODO: Implement loss creation from args
    pass
