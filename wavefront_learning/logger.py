"""Logging utilities for wavefront learning using Weights & Biases."""

from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import wandb
from dotenv import load_dotenv

# Load WANDB_API_KEY from local .env
load_dotenv(Path(__file__).parent / ".env")


class WandbLogger:
    """Wrapper for Weights & Biases experiment logging.

    Args:
        project: W&B project name.
        name: Run name.
        config: Configuration dict or Namespace.
        tags: Optional list of tags.
        enabled: Whether logging is enabled.
    """

    def __init__(
        self,
        project: str,
        name: str | None = None,
        config: dict | Namespace | None = None,
        tags: list[str] | None = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.run = None

        if enabled:
            config_dict = vars(config) if isinstance(config, Namespace) else config
            self.run = wandb.init(
                project=project,
                name=name,
                config=config_dict,
                tags=tags,
            )

            # Log source code (exclude patterns from .gitignore)
            if self.run is not None:
                self.run.log_code(
                    root=".",
                    include_fn=lambda path: (
                        path.endswith(".py")
                        and "__pycache__" not in path
                        and ".venv" not in path
                        and "wandb/" not in path
                    ),
                )

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """Log scalar metrics.

        Args:
            metrics: Dictionary of metric names to values.
            step: Optional global step number.
        """
        if not self.enabled or self.run is None:
            return
        wandb.log(metrics, step=step)

    def log_image(
        self,
        key: str,
        image,
        step: int | None = None,
        caption: str | None = None,
    ) -> None:
        """Log an image.

        Args:
            key: Image key/name.
            image: Image data (PIL, numpy, matplotlib figure, or path).
            step: Optional global step number.
            caption: Optional image caption.
        """
        if not self.enabled or self.run is None:
            return

        if hasattr(image, "savefig"):
            # Matplotlib figure
            wandb.log({key: wandb.Image(image, caption=caption)}, step=step)
        elif isinstance(image, np.ndarray):
            wandb.log({key: wandb.Image(image, caption=caption)}, step=step)
        else:
            wandb.log({key: wandb.Image(image, caption=caption)}, step=step)

    def log_figure(
        self,
        key: str,
        figure,
        step: int | None = None,
    ) -> None:
        """Log a matplotlib figure.

        Args:
            key: Figure key/name.
            figure: Matplotlib figure.
            step: Optional global step number.
        """
        if not self.enabled or self.run is None:
            return
        wandb.log({key: wandb.Image(figure)}, step=step)

    def log_summary(self, metrics: dict[str, float]) -> None:
        """Log metrics to run summary (final/aggregate values).

        Args:
            metrics: Dictionary of metric names to values.
        """
        if not self.enabled or self.run is None:
            return
        for key, value in metrics.items():
            wandb.run.summary[key] = value

    def log_summary_image(self, key: str, image, caption: str | None = None) -> None:
        """Log an image to run summary.

        Args:
            key: Image key/name.
            image: Image data (PIL, numpy, matplotlib figure, or path).
            caption: Optional image caption.
        """
        if not self.enabled or self.run is None:
            return
        wandb.run.summary[key] = wandb.Image(image, caption=caption)

    def log_summary_video(self, key: str, path: str, fps: int = 20) -> None:
        """Log a video file to run summary.

        Args:
            key: Video key/name.
            path: Path to video file.
            fps: Frames per second.
        """
        if not self.enabled or self.run is None:
            return
        wandb.run.summary[key] = wandb.Video(path, fps=fps, format="gif")

    def log_video(
        self,
        path: str,
        key: str,
        step: int | None = None,
        fps: int = 20,
    ) -> None:
        """Log a video file.

        Args:
            path: Path to video file.
            key: Video key/name.
            step: Optional global step number.
            fps: Frames per second.
        """
        if not self.enabled or self.run is None:
            return
        wandb.log({key: wandb.Video(path, fps=fps, format="gif")}, step=step)

    def log_model(
        self,
        model: torch.nn.Module,
        name: str,
        metadata: dict | None = None,
    ) -> None:
        """Log model artifact.

        Args:
            model: PyTorch model to log.
            name: Artifact name.
            metadata: Optional metadata dict.
        """
        if not self.enabled or self.run is None:
            return

        artifact = wandb.Artifact(name, type="model", metadata=metadata)
        # Save model state dict to temp file
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            torch.save(model.state_dict(), f.name)
            artifact.add_file(f.name, name=f"{name}.pth")
        self.run.log_artifact(artifact)

    def log_table(
        self,
        key: str,
        columns: list[str],
        data: list[list],
    ) -> None:
        """Log a data table.

        Args:
            key: Table key/name.
            columns: Column names.
            data: Table rows.
        """
        if not self.enabled or self.run is None:
            return
        table = wandb.Table(columns=columns, data=data)
        wandb.log({key: table})

    def log_code(self, folder: str | None = None, file_name: str | None = None) -> None:
        """Log code files.

        Args:
            folder: Folder to log.
            file_name: Specific file to log.
        """
        if not self.enabled or self.run is None:
            return
        if folder:
            wandb.run.log_code(root=folder)
        if file_name:
            wandb.save(file_name)

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for context."""
        self._current_epoch = epoch

    def finish(self) -> None:
        """Finish the logging run."""
        if self.enabled and self.run is not None:
            wandb.finish()


def init_logger(args: Namespace, project: str = "wavefront-learning") -> WandbLogger:
    """Initialize logger from command line arguments.

    Args:
        args: Parsed command line arguments.
        project: W&B project name.

    Returns:
        Configured WandbLogger instance.
    """
    tags = [args.model] if hasattr(args, "model") else []
    name = getattr(args, "run_name", None)

    return WandbLogger(
        project=project,
        name=name,
        config=args,
        tags=tags,
        enabled=not getattr(args, "no_wandb", False),
    )


def log_epoch_metrics(
    logger: WandbLogger,
    epoch: int,
    train_loss: float,
    val_loss: float,
    learning_rate: float,
    additional_metrics: dict | None = None,
) -> None:
    """Log standard epoch metrics.

    Args:
        logger: WandbLogger instance.
        epoch: Current epoch number.
        train_loss: Training loss for the epoch.
        val_loss: Validation loss for the epoch.
        learning_rate: Current learning rate.
        additional_metrics: Optional additional metrics to log.
    """
    metrics = {
        "train/loss": train_loss,
        "val/loss": val_loss,
        "train/lr": learning_rate,
        "epoch": epoch,
    }

    if additional_metrics:
        metrics.update(additional_metrics)

    logger.log_metrics(metrics, step=epoch)
