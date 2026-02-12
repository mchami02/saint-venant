"""Model factory for wavefront learning."""

from argparse import Namespace

import torch
import torch.nn as nn
from models.deeponet import build_deeponet
from models.encoder_decoder import build_encoder_decoder, build_encoder_decoder_cross
from models.fno_wrapper import build_fno
from models.hybrid_deeponet import build_hybrid_deeponet
from models.shock_trajectory_net import build_shock_net
from models.traj_deeponet import (
    build_classifier_traj_deeponet,
    build_no_traj_deeponet,
    build_traj_deeponet,
)
from models.traj_transformer import (
    build_classifier_all_traj_transformer,
    build_classifier_traj_transformer,
    build_no_traj_transformer,
    build_traj_transformer,
)

# Registry of available models
MODELS = {
    "ShockNet": build_shock_net,
    "HybridDeepONet": build_hybrid_deeponet,
    "TrajDeepONet": build_traj_deeponet,
    "ClassifierTrajDeepONet": build_classifier_traj_deeponet,
    "NoTrajDeepONet": build_no_traj_deeponet,
    "FNO": build_fno,
    "DeepONet": build_deeponet,
    "EncoderDecoder": build_encoder_decoder,
    "EncoderDecoderCross": build_encoder_decoder_cross,
    "TrajTransformer": build_traj_transformer,
    "ClassifierTrajTransformer": build_classifier_traj_transformer,
    "ClassifierAllTrajTransformer": build_classifier_all_traj_transformer,
    "NoTrajTransformer": build_no_traj_transformer,
}

# Registry of per-model transforms (None or a string key into TRANSFORMS in data.py)
MODEL_TRANSFORM = {
    "ShockNet": None,
    "HybridDeepONet": None,
    "TrajDeepONet": None,
    "ClassifierTrajDeepONet": None,
    "NoTrajDeepONet": None,
    "FNO": "ToGridInput",
    "DeepONet": "ToGridInput",
    "EncoderDecoder": "ToGridInput",
    "EncoderDecoderCross": "ToGridInput",
    "TrajTransformer": None,
    "ClassifierTrajTransformer": None,
    "ClassifierAllTrajTransformer": None,
    "NoTrajTransformer": None,
}


def get_model(model_name: str, args: dict) -> nn.Module:
    """Create a model instance based on name and arguments.

    Args:
        model_name: Name of the model (e.g., 'fno', 'wavefront').
        args: Dictionary containing model configuration.

    Returns:
        Instantiated model on the appropriate device.

    Raises:
        ValueError: If model_name is not supported.
    """
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not supported")
    return MODELS[model_name](args)


def load_model(
    model_path: str,
    device: torch.device,
    args: Namespace | dict | None = None,
    logger=None,
) -> nn.Module:
    """Load a trained model from a checkpoint.

    When a logger is provided and active, downloads the model artifact from
    W&B instead of loading from the local path.

    Args:
        model_path: Path to the model checkpoint (used as fallback).
        device: Device to load the model on.
        args: Optional args to override checkpoint config (Namespace or dict).
        logger: Optional WandbLogger to download the model from W&B.

    Returns:
        Loaded model in evaluation mode.
    """
    # Try downloading from W&B if logger is available
    if logger is not None and args is not None:
        args_dict = args if isinstance(args, dict) else vars(args)
        model_name = args_dict.get("model")
        if model_name:
            wandb_path = logger.download_model(model_name)
            if wandb_path is not None:
                model_path = wandb_path

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        config = dict(checkpoint.get("config", {}))
        if args is not None:
            args_dict = args if isinstance(args, dict) else vars(args)
            config.update(args_dict)
        model_name = config.get("model", None)
    else:
        state_dict = checkpoint
        if args is None:
            raise ValueError("args is required when checkpoint has no config")
        config = args if isinstance(args, dict) else vars(args)
        model_name = config.get("model", None)

    if model_name is None or model_name not in MODELS:
        raise ValueError(
            f"Cannot load model: config has no 'model' or '{model_name}' is not in MODELS. "
            f"Available: {list(MODELS.keys())}"
        )

    model = get_model(model_name, config)
    state_dict.pop("_metadata", None)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model


def save_model(
    model: nn.Module,
    save_path: str,
    args: Namespace | dict,
    epoch: int,
    optimizer_state: dict | None = None,
    logger=None,
) -> None:
    """Save model checkpoint.

    Args:
        model: Model to save.
        save_path: Path to save the checkpoint.
        args: Training arguments to store (Namespace or dict).
        epoch: Current epoch number.
        optimizer_state: Optional optimizer state dict.
        logger: Optional WandbLogger to log model artifact.
    """
    config = args if isinstance(args, dict) else vars(args)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "epoch": epoch,
    }
    if optimizer_state is not None:
        checkpoint["optimizer_state_dict"] = optimizer_state
    torch.save(checkpoint, save_path)

    if logger is not None:
        model_name = config.get("model", "model")
        logger.log_model(model, model_name, metadata=config)
