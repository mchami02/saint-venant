"""Loss function factory for wavefront learning.

This module provides:
- CombinedLoss: Combines multiple losses with configurable weights
- LOSSES: Registry of available loss classes
- LOSS_PRESETS: Pre-configured loss combinations (defined in configs/presets.py)
- get_loss(): Factory function to create loss instances
"""

import torch
import torch.nn as nn
from configs.presets import LOSS_PRESETS
from losses.acceleration import AccelerationLoss
from losses.base import BaseLoss
from losses.boundary import BoundaryLoss
from losses.cell_avg_mse import CellAverageMSELoss
from losses.collision import CollisionLoss
from losses.conservation import ConservationLoss
from losses.entropy import EntropyConditionLoss
from losses.existence_regularization import ICAnchoringLoss
from losses.flow_matching import FlowMatchingLoss
from losses.ic import ICLoss
from losses.kl_divergence import KLDivergenceLoss
from losses.mse import MSELoss
from losses.mse_shock import MSEShockLoss
from losses.pde_residual import PDEResidualLoss, PDEShockResidualLoss
from losses.regularize_traj import RegularizeTrajLoss
from losses.rh_residual import RHResidualLoss
from losses.selection_supervision import SelectionSupervisionLoss
from losses.shock_proximity import ShockProximityLoss
from losses.supervised_trajectory import SupervisedTrajectoryLoss
from losses.trajectory_consistency import TrajectoryConsistencyLoss
from losses.vae_reconstruction import VAEReconstructionLoss
from losses.wasserstein import WassersteinLoss

# Registry of available loss functions
LOSSES: dict[str, type[BaseLoss]] = {
    "mse": MSELoss,
    "cell_avg_mse": CellAverageMSELoss,
    "trajectory": TrajectoryConsistencyLoss,
    "rh_residual": RHResidualLoss,
    "pde_residual": PDEResidualLoss,
    "pde_shock_residual": PDEShockResidualLoss,
    "entropy": EntropyConditionLoss,
    "mse_shock": MSEShockLoss,
    "boundary": BoundaryLoss,
    "collision": CollisionLoss,
    "ic_anchoring": ICAnchoringLoss,
    "supervised_trajectory": SupervisedTrajectoryLoss,
    "ic": ICLoss,
    "acceleration": AccelerationLoss,
    "regularize_traj": RegularizeTrajLoss,
    "wasserstein": WassersteinLoss,
    "conservation": ConservationLoss,
    "selection_supervision": SelectionSupervisionLoss,
    "vae_reconstruction": VAEReconstructionLoss,
    "flow_matching": FlowMatchingLoss,
    "kl_divergence": KLDivergenceLoss,
    "shock_proximity": ShockProximityLoss,
}


class CombinedLoss(BaseLoss):
    """Combines multiple loss functions with weights.

    Args:
        losses: Dict of {name: (loss_fn, weight)} or list of (loss_fn, weight).
    """

    def __init__(
        self,
        losses: dict[str, tuple[nn.Module, float]] | list[tuple[nn.Module, float]],
    ):
        super().__init__()

        if isinstance(losses, list):
            # Convert list to dict with auto-generated names
            losses = {f"loss_{i}": item for i, item in enumerate(losses)}

        self.loss_names = list(losses.keys())
        self.losses = nn.ModuleList([loss_fn for loss_fn, _ in losses.values()])
        self.weights = [weight for _, weight in losses.values()]

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        loss_kwargs: dict[str, dict] | None = None,
    ) -> "CombinedLoss":
        """Create CombinedLoss from a preset configuration.

        Args:
            preset_name: Name of the preset (e.g., 'mse', 'pde_shocks').
            loss_kwargs: Optional dict of {loss_name: kwargs} for customizing
                individual loss instances. These override preset defaults.

        Returns:
            Configured CombinedLoss instance.

        Examples:
            # Preset with per-loss kwargs:
            LOSS_PRESETS = {
                "hybrid": [
                    ("mse", 1.0),                              # No kwargs
                    ("rh_residual", 1.0, {"mode": "gt"}),      # With kwargs
                ],
            }

            # Override preset kwargs at runtime:
            loss = CombinedLoss.from_preset("hybrid", loss_kwargs={
                "rh_residual": {"mode": "per_region"},  # Overrides preset default
            })
        """
        if preset_name not in LOSS_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset_name}'. Available: {list(LOSS_PRESETS.keys())}"
            )

        loss_kwargs = loss_kwargs or {}
        preset = LOSS_PRESETS[preset_name]

        losses = {}
        for entry in preset:
            # Parse 2-element or 3-element tuples
            if len(entry) == 2:
                loss_name, weight = entry
                preset_kwargs = {}
            else:
                loss_name, weight, preset_kwargs = entry

            if loss_name not in LOSSES:
                raise ValueError(
                    f"Unknown loss '{loss_name}' in preset '{preset_name}'"
                )

            # Merge: loss_kwargs overrides preset defaults
            kwargs = {**preset_kwargs, **loss_kwargs.get(loss_name, {})}
            losses[loss_name] = (LOSSES[loss_name](**kwargs), weight)

        return cls(losses)

    @classmethod
    def from_config(
        cls,
        config: list[tuple[str, float] | tuple[str, float, dict]],
        loss_kwargs: dict[str, dict] | None = None,
    ) -> "CombinedLoss":
        """Create CombinedLoss from a custom configuration.

        Args:
            config: List of (loss_name, weight) or (loss_name, weight, kwargs) tuples.
            loss_kwargs: Optional dict of {loss_name: kwargs} for customizing
                individual loss instances. These override config defaults.

        Returns:
            Configured CombinedLoss instance.

        Examples:
            # Config with per-loss kwargs:
            config = [
                ("mse", 1.0),
                ("rh_residual", 1.0, {"mode": "gt", "dt": 0.004}),
            ]
            loss = CombinedLoss.from_config(config)

            # Override config kwargs:
            loss = CombinedLoss.from_config(config, loss_kwargs={
                "rh_residual": {"mode": "per_region"},
            })
        """
        loss_kwargs = loss_kwargs or {}

        losses = {}
        for entry in config:
            # Parse 2-element or 3-element tuples
            if len(entry) == 2:
                loss_name, weight = entry
                config_kwargs = {}
            else:
                loss_name, weight, config_kwargs = entry

            if loss_name not in LOSSES:
                raise ValueError(
                    f"Unknown loss '{loss_name}'. Available: {list(LOSSES.keys())}"
                )

            # Merge: loss_kwargs overrides config defaults
            kwargs = {**config_kwargs, **loss_kwargs.get(loss_name, {})}
            losses[loss_name] = (LOSSES[loss_name](**kwargs), weight)

        return cls(losses)

    def set_kl_beta(self, beta: float) -> None:
        """Set KL beta on all sub-losses that support it (e.g. KLDivergenceLoss)."""
        for loss_fn in self.losses:
            if hasattr(loss_fn, "beta"):
                loss_fn.beta = beta

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        output_dict: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined loss.

        Args:
            input_dict: Dictionary of input tensors.
            output_dict: Dictionary of model output tensors.
            target: Ground truth tensor.

        Returns:
            Tuple of (total_loss, loss_components_dict).
        """
        total = torch.tensor(0.0, device=target.device)
        components: dict[str, float] = {}

        for name, loss_fn, weight in zip(
            self.loss_names, self.losses, self.weights, strict=True
        ):
            loss_val, sub_components = loss_fn(input_dict, output_dict, target)
            total = total + weight * loss_val

            # Store the main loss value
            components[name] = loss_val.item()

            # Store sub-components with prefix (skip duplicates of main loss name)
            for sub_key, sub_val in sub_components.items():
                if sub_key != "total" and sub_key != name:
                    components[f"{name}/{sub_key}"] = sub_val

        components["total"] = total.item()
        return total, components


def get_loss(loss_name: str, **kwargs) -> nn.Module:
    """Create a loss function instance.

    Args:
        loss_name: Name of the loss function or preset.
        **kwargs: Additional arguments for the loss function.
            For presets, pass 'loss_kwargs' dict to customize individual losses.
            For individual losses, kwargs are passed directly to the constructor.

    Returns:
        Instantiated loss function.

    Raises:
        ValueError: If loss_name is not supported.

    Examples:
        # Use a preset
        loss = get_loss("mse")

        # Use a preset with custom parameters
        loss = get_loss("pde_shocks", loss_kwargs={
            "pde_shock_residual": {"dt": 0.004, "dx": 0.02},
            "rh_residual": {"dt": 0.004},
        })
    """

    # Check if it's a preset
    if loss_name in LOSS_PRESETS:
        loss_kwargs = kwargs.pop("loss_kwargs", None)
        return CombinedLoss.from_preset(loss_name, loss_kwargs=loss_kwargs)

    # Check if it's an individual loss
    if loss_name in LOSSES:
        # Extract per-loss kwargs if loss_kwargs dict was passed
        loss_kwargs = kwargs.pop("loss_kwargs", None)
        if loss_kwargs and loss_name in loss_kwargs:
            kwargs.update(loss_kwargs[loss_name])
        return LOSSES[loss_name](**kwargs)

    raise ValueError(
        f"Loss '{loss_name}' not supported. "
        f"Available losses: {list(LOSSES.keys())}. "
        f"Available presets: {list(LOSS_PRESETS.keys())}."
    )


def create_loss_from_args(args) -> nn.Module:
    """Create loss function from command line arguments.

    Args:
        args: Parsed command line arguments with 'loss', 'dt', 'dx' attributes.

    Returns:
        Configured loss function.
    """
    loss_kwargs = {}

    # Build kwargs for losses that need dt/dx
    if hasattr(args, "dt") and hasattr(args, "dx"):
        loss_kwargs["pde_residual"] = {
            "dt": args.dt,
            "dx": args.dx,
        }
        loss_kwargs["rh_residual"] = {
            "dt": args.dt,
        }
        loss_kwargs["pde_shock_residual"] = {
            "dt": args.dt,
            "dx": args.dx,
        }
        loss_kwargs["wasserstein"] = {
            "dx": args.dx,
        }
        loss_kwargs["conservation"] = {
            "dx": args.dx,
        }
        loss_kwargs["entropy"] = {
            "dx": args.dx,
            "min_component_size": getattr(args, "min_component_size", 5),
        }
        loss_kwargs["mse_shock"] = {
            "dx": args.dx,
            "min_component_size": getattr(args, "min_component_size", 5),
        }

    return get_loss(args.loss, loss_kwargs=loss_kwargs)
