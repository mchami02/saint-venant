"""Plotting utilities for wavefront learning results.

Follows the same style as operator_learning/plot_data.py.
"""

import tempfile

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend (thread-safe, no GUI)
import matplotlib.pyplot as plt
import numpy as np
import torch
from logger import WandbLogger
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure


def plot_prediction_comparison(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    nx: int,
    nt: int,
    dx: float,
    dt: float,
    title: str | None = None,
) -> Figure:
    """Plot ground truth vs prediction comparison as heatmaps.

    Args:
        ground_truth: Ground truth array of shape (nt, nx).
        prediction: Prediction array of shape (nt, nx).
        nx: Number of spatial points.
        nt: Number of time steps.
        dx: Spatial step size.
        dt: Time step size.
        title: Optional plot title.

    Returns:
        Matplotlib figure with comparison plot.
    """
    difference = np.abs(prediction - ground_truth)
    extent = [0, nx * dx, 0, nt * dt]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Ground Truth
    im1 = axes[0].imshow(
        ground_truth, extent=extent, aspect="auto", origin="lower", cmap="jet", vmin=0, vmax=1
    )
    axes[0].set_xlabel("Space x")
    axes[0].set_ylabel("Time t")
    axes[0].set_title("Ground Truth")
    plt.colorbar(im1, ax=axes[0], label="Value")

    # Prediction
    im2 = axes[1].imshow(
        prediction, extent=extent, aspect="auto", origin="lower", cmap="jet", vmin=0, vmax=1
    )
    axes[1].set_xlabel("Space x")
    axes[1].set_ylabel("Time t")
    axes[1].set_title("Prediction")
    plt.colorbar(im2, ax=axes[1], label="Value")

    # Difference
    im3 = axes[2].imshow(
        difference, extent=extent, aspect="auto", origin="lower", cmap="RdBu_r", vmin=0, vmax=1
    )
    axes[2].set_xlabel("Space x")
    axes[2].set_ylabel("Time t")
    axes[2].set_title("Absolute Error")
    plt.colorbar(im3, ax=axes[2], label="Error")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_error_map(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    nx: int,
    nt: int,
    dx: float,
    dt: float,
) -> Figure:
    """Plot spatial-temporal error map.

    Args:
        ground_truth: Ground truth array of shape (nt, nx).
        prediction: Prediction array of shape (nt, nx).
        nx: Number of spatial points.
        nt: Number of time steps.
        dx: Spatial step size.
        dt: Time step size.

    Returns:
        Matplotlib figure with error heatmap.
    """
    error = np.abs(prediction - ground_truth)
    extent = [0, nx * dx, 0, nt * dt]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(error, extent=extent, aspect="auto", origin="lower", cmap="hot")
    ax.set_xlabel("Space x")
    ax.set_ylabel("Time t")
    ax.set_title("Prediction Error Heatmap")
    plt.colorbar(im, ax=ax, label="Absolute Error")

    plt.tight_layout()
    return fig


def _create_comparison_animation(
    gt: np.ndarray,
    pred: np.ndarray,
    nx: int,
    nt: int,
    dx: float,
    dt: float,
    sample_idx: int = 0,
    skip_frames: int = 5,
    fps: int = 20,
):
    """Create an animation showing ground truth vs prediction side by side through time.

    Args:
        gt: Ground truth array (nt, nx).
        pred: Prediction array (nt, nx).
        nx, nt: Grid dimensions.
        dx, dt: Grid spacing.
        sample_idx: Sample index for title.
        skip_frames: Number of frames to skip.
        fps: Frames per second.

    Returns:
        FuncAnimation object and the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.linspace(0, nx * dx, gt.shape[1])

    # Set consistent y-axis limits
    y_min = min(gt.min(), pred.min())
    y_max = max(gt.max(), pred.max())
    y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 0.1

    # Initialize lines
    (line_gt,) = axes[0].plot(x, gt[0], "b-", linewidth=2)
    (line_pred,) = axes[1].plot(x, pred[0], "r-", linewidth=2)

    for ax, title in zip(
        axes,
        [f"Ground Truth (Sample {sample_idx + 1})", f"Prediction (Sample {sample_idx + 1})"],
        strict=True,
    ):
        ax.set_xlim(0, nx * dx)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_xlabel("Position x")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    time_text = axes[0].text(0.02, 0.95, "", transform=axes[0].transAxes, fontsize=10)
    plt.tight_layout()

    def update(frame):
        t = frame * skip_frames
        if t >= nt:
            t = nt - 1
        line_gt.set_ydata(gt[t])
        line_pred.set_ydata(pred[t])
        time_text.set_text(f"Time: {t * dt:.3f} s (step {t}/{nt})")
        return line_gt, line_pred, time_text

    n_frames = max(1, nt // skip_frames)
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps)

    return anim, fig


def plot_wavefront_trajectory(
    prediction: np.ndarray,
    wavefront_positions: np.ndarray | None = None,
    nx: int | None = None,
    nt: int | None = None,
    dx: float | None = None,
    dt: float | None = None,
) -> Figure:
    """Plot wavefront trajectory over time.

    Detects discontinuities and plots their position over time.

    Args:
        prediction: Prediction array of shape (nt, nx).
        wavefront_positions: Optional ground truth wavefront positions.
        nx: Number of spatial points.
        nt: Number of time steps.
        dx: Spatial step size.
        dt: Time step size.

    Returns:
        Matplotlib figure with wavefront trajectory.
    """
    if nt is None:
        nt = prediction.shape[0]
    if nx is None:
        nx = prediction.shape[1]
    if dx is None:
        dx = 1.0 / nx
    if dt is None:
        dt = 1.0 / nt

    fig, ax = plt.subplots(figsize=(8, 6))

    # Detect wavefront positions from prediction (using gradient)
    detected_positions = []
    for t in range(nt):
        grad = np.abs(np.gradient(prediction[t]))
        if grad.max() > 0.1:  # Threshold for detecting discontinuity
            pos = np.argmax(grad) * dx
            detected_positions.append((t * dt, pos))

    if detected_positions:
        times, positions = zip(*detected_positions, strict=True)
        ax.plot(times, positions, "b-", linewidth=2, label="Detected Wavefront")

    if wavefront_positions is not None:
        ax.plot(
            np.arange(nt) * dt,
            wavefront_positions,
            "r--",
            linewidth=2,
            label="True Wavefront",
        )

    ax.set_xlabel("Time t")
    ax.set_ylabel("Position x")
    ax.set_title("Wavefront Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    title: str = "Training Progress",
) -> Figure:
    """Plot training and validation loss curves.

    Args:
        train_losses: List of training losses per epoch.
        val_losses: List of validation losses per epoch.
        title: Plot title.

    Returns:
        Matplotlib figure with loss curves.
    """
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=2)
    ax.plot(epochs, val_losses, "r-", label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    return fig


def plot_sample_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 5,
    nx: int | None = None,
    nt: int | None = None,
    dx: float | None = None,
    dt: float | None = None,
) -> list[Figure]:
    """Generate prediction plots for multiple samples.

    Args:
        model: Trained model.
        dataloader: DataLoader to sample from.
        device: Computation device.
        num_samples: Number of samples to plot.
        nx: Number of spatial points.
        nt: Number of time steps.
        dx: Spatial step size.
        dt: Time step size.

    Returns:
        List of matplotlib figures.
    """
    model.eval()
    figures = []
    count = 0

    with torch.no_grad():
        for batch_input, batch_target in dataloader:
            if isinstance(batch_input, dict):
                batch_input = {k: v.to(device) for k, v in batch_input.items()}
            else:
                batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)

            pred = model(batch_input)

            for i in range(batch_target.shape[0]):
                if count >= num_samples:
                    break

                gt = batch_target[i].squeeze(0).cpu().numpy()
                p = pred[i].squeeze(0).cpu().numpy()

                # Infer dimensions if not provided
                actual_nt, actual_nx = gt.shape
                _nx = nx if nx else actual_nx
                _nt = nt if nt else actual_nt
                _dx = dx if dx else 1.0 / _nx
                _dt = dt if dt else 1.0 / _nt

                fig = plot_prediction_comparison(gt, p, _nx, _nt, _dx, _dt, title=f"Sample {count + 1}")
                figures.append(fig)
                count += 1

            if count >= num_samples:
                break

    return figures


def plot_comparison_wandb(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    nx: int,
    nt: int,
    dx: float,
    dt: float,
    logger: WandbLogger,
    epoch: int,
    mode: str = "val",
) -> None:
    """Create comparison plots and upload to W&B.

    Args:
        ground_truth: (B, nt, nx) or (nt, nx) array.
        prediction: (B, nt, nx) or (nt, nx) array.
        nx, nt: Grid dimensions.
        dx, dt: Grid spacing.
        logger: WandbLogger instance.
        epoch: Current epoch.
        mode: Mode string for logging prefix.
    """
    ground_truth = np.asarray(ground_truth)
    prediction = np.asarray(prediction)

    # Handle 2D input by adding batch dimension
    if ground_truth.ndim == 2:
        ground_truth = ground_truth[np.newaxis, ...]
        prediction = prediction[np.newaxis, ...]

    B = ground_truth.shape[0]
    difference = np.abs(prediction - ground_truth)

    # Create static comparison plots
    fig, axes = plt.subplots(B, 3, figsize=(18, 5 * B))
    if B == 1:
        axes = axes.reshape(1, -1)

    extent = [0, nx * dx, 0, nt * dt]

    for b in range(B):
        # Ground Truth
        im1 = axes[b, 0].imshow(
            ground_truth[b], extent=extent, aspect="auto", origin="lower", cmap="jet", vmin=0, vmax=1
        )
        axes[b, 0].set_xlabel("Space x")
        axes[b, 0].set_ylabel("Time t")
        axes[b, 0].set_title(f"Ground Truth (Sample {b + 1})")
        plt.colorbar(im1, ax=axes[b, 0], label="Value")

        # Prediction
        im2 = axes[b, 1].imshow(
            prediction[b], extent=extent, aspect="auto", origin="lower", cmap="jet", vmin=0, vmax=1
        )
        axes[b, 1].set_xlabel("Space x")
        axes[b, 1].set_ylabel("Time t")
        axes[b, 1].set_title(f"Prediction (Sample {b + 1})")
        plt.colorbar(im2, ax=axes[b, 1], label="Value")

        # Difference
        im3 = axes[b, 2].imshow(
            difference[b], extent=extent, aspect="auto", origin="lower", cmap="RdBu_r", vmin=0, vmax=1
        )
        axes[b, 2].set_xlabel("Space x")
        axes[b, 2].set_ylabel("Time t")
        axes[b, 2].set_title(f"Difference (Sample {b + 1})")
        plt.colorbar(im3, ax=axes[b, 2], label="Error")

    plt.tight_layout()
    logger.log_figure(f"{mode}/comparison_plot", fig, step=epoch)
    plt.close(fig)

    # Create and upload animated GIFs for each sample
    for b in range(B):
        anim, anim_fig = _create_comparison_animation(
            ground_truth[b], prediction[b], nx, nt, dx, dt, sample_idx=b
        )

        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
            anim.save(tmp.name, writer="pillow", fps=20)
            logger.log_video(tmp.name, f"{mode}/animation_sample_{b + 1}", step=epoch)

        plt.close(anim_fig)


def save_figure(fig: Figure, path: str, dpi: int = 150) -> None:
    """Save a matplotlib figure to file.

    Args:
        fig: Matplotlib figure to save.
        path: Output file path.
        dpi: Resolution in dots per inch.
    """
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_shock_trajectories(
    positions: np.ndarray,
    existence: np.ndarray,
    discontinuities: np.ndarray,
    mask: np.ndarray,
    times: np.ndarray,
    sample_idx: int = 0,
    show_analytical: bool = True,
) -> Figure:
    """Plot predicted shock trajectories vs analytical Rankine-Hugoniot trajectories.

    Args:
        positions: Predicted positions of shape (D, T) for one sample.
        existence: Predicted existence of shape (D, T) for one sample.
        discontinuities: Initial discontinuities of shape (D, 3) with [x_0, rho_L, rho_R].
        mask: Validity mask of shape (D,).
        times: Query times of shape (T,).
        sample_idx: Sample index for title.
        show_analytical: Whether to show analytical RH trajectories.

    Returns:
        Matplotlib figure with trajectory plot.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    n_disc = int(mask.sum())
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_disc, 1)))

    for d in range(n_disc):
        x_0 = discontinuities[d, 0]
        rho_L = discontinuities[d, 1]
        rho_R = discontinuities[d, 2]

        # Predicted trajectory with existence as alpha
        pred_pos = positions[d]
        pred_exist = existence[d]

        # Plot predicted trajectory with varying alpha based on existence
        for t_idx in range(len(times) - 1):
            alpha = float(pred_exist[t_idx])
            ax.plot(
                [times[t_idx], times[t_idx + 1]],
                [pred_pos[t_idx], pred_pos[t_idx + 1]],
                color=colors[d],
                alpha=max(0.1, alpha),
                linewidth=2,
            )

        # Mark initial position
        ax.scatter([0], [x_0], color=colors[d], s=100, marker="o", zorder=5)

        # Analytical trajectory (Rankine-Hugoniot)
        if show_analytical:
            shock_speed = 1.0 - rho_L - rho_R
            analytical_pos = x_0 + shock_speed * times
            ax.plot(
                times,
                analytical_pos,
                color=colors[d],
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label=f"RH d={d} (s={shock_speed:.2f})",
            )

    # Domain boundaries
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(y=1, color="gray", linestyle=":", alpha=0.5)
    ax.fill_between(times, -0.1, 0, color="gray", alpha=0.1)
    ax.fill_between(times, 1, 1.1, color="gray", alpha=0.1)

    ax.set_xlabel("Time t", fontsize=12)
    ax.set_ylabel("Position x", fontsize=12)
    ax.set_title(f"Shock Trajectories (Sample {sample_idx + 1})", fontsize=14)
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_existence_heatmap(
    existence: np.ndarray,
    mask: np.ndarray,
    times: np.ndarray,
    sample_idx: int = 0,
) -> Figure:
    """Plot existence probability heatmap for all discontinuities.

    Args:
        existence: Predicted existence of shape (D, T) for one sample.
        mask: Validity mask of shape (D,).
        times: Query times of shape (T,).
        sample_idx: Sample index for title.

    Returns:
        Matplotlib figure with existence heatmap.
    """
    n_disc = int(mask.sum())
    if n_disc == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No valid discontinuities", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=(10, 4))

    # Only show valid discontinuities
    valid_existence = existence[:n_disc]

    im = ax.imshow(
        valid_existence,
        aspect="auto",
        origin="lower",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        extent=[times[0], times[-1], -0.5, n_disc - 0.5],
    )

    ax.set_xlabel("Time t", fontsize=12)
    ax.set_ylabel("Discontinuity Index", fontsize=12)
    ax.set_title(f"Existence Probability (Sample {sample_idx + 1})", fontsize=14)
    ax.set_yticks(range(n_disc))
    plt.colorbar(im, ax=ax, label="P(exists)")

    plt.tight_layout()
    return fig


def plot_trajectory_wandb(
    positions: np.ndarray,
    existence: np.ndarray,
    discontinuities: np.ndarray,
    masks: np.ndarray,
    times: np.ndarray,
    logger: WandbLogger,
    epoch: int,
    mode: str = "val",
) -> None:
    """Create trajectory plots and upload to W&B.

    Args:
        positions: Predicted positions of shape (B, D, T).
        existence: Predicted existence of shape (B, D, T).
        discontinuities: Initial discontinuities of shape (B, D, 3).
        masks: Validity masks of shape (B, D).
        times: Query times of shape (T,) or (B, T).
        logger: WandbLogger instance.
        epoch: Current epoch.
        mode: Mode string for logging prefix.
    """
    B = positions.shape[0]

    # Handle times shape
    if times.ndim == 1:
        times_1d = times
    else:
        times_1d = times[0]  # Assume same times for all samples

    # Create trajectory plots for each sample
    for b in range(min(B, 3)):  # Limit to 3 samples
        # Trajectory plot
        fig_traj = plot_shock_trajectories(
            positions[b],
            existence[b],
            discontinuities[b],
            masks[b],
            times_1d,
            sample_idx=b,
            show_analytical=True,
        )
        logger.log_figure(f"{mode}/trajectory_sample_{b + 1}", fig_traj, step=epoch)
        plt.close(fig_traj)

        # Existence heatmap
        fig_exist = plot_existence_heatmap(
            existence[b],
            masks[b],
            times_1d,
            sample_idx=b,
        )
        logger.log_figure(f"{mode}/existence_sample_{b + 1}", fig_exist, step=epoch)
        plt.close(fig_exist)

    # Create combined summary plot
    fig, axes = plt.subplots(B, 2, figsize=(14, 5 * B))
    if B == 1:
        axes = axes.reshape(1, -1)

    for b in range(B):
        n_disc = int(masks[b].sum())
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_disc, 1)))

        # Left: trajectories
        ax = axes[b, 0]
        for d in range(n_disc):
            x_0 = discontinuities[b, d, 0]
            rho_L = discontinuities[b, d, 1]
            rho_R = discontinuities[b, d, 2]

            # Predicted
            ax.plot(times_1d, positions[b, d], color=colors[d], linewidth=2, label=f"Pred d={d}")

            # Analytical
            shock_speed = 1.0 - rho_L - rho_R
            analytical = x_0 + shock_speed * times_1d
            ax.plot(times_1d, analytical, color=colors[d], linestyle="--", linewidth=1.5, alpha=0.7)

        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.axhline(y=1, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Time t")
        ax.set_ylabel("Position x")
        ax.set_title(f"Sample {b + 1}: Trajectories (solid=pred, dashed=RH)")
        ax.set_ylim(-0.1, 1.1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Right: existence
        ax = axes[b, 1]
        if n_disc > 0:
            im = ax.imshow(
                existence[b, :n_disc],
                aspect="auto",
                origin="lower",
                cmap="RdYlGn",
                vmin=0,
                vmax=1,
                extent=[times_1d[0], times_1d[-1], -0.5, n_disc - 0.5],
            )
            plt.colorbar(im, ax=ax, label="P(exists)")
        ax.set_xlabel("Time t")
        ax.set_ylabel("Discontinuity")
        ax.set_title(f"Sample {b + 1}: Existence Probability")

    plt.tight_layout()
    logger.log_figure(f"{mode}/trajectory_summary", fig, step=epoch)
    plt.close(fig)
