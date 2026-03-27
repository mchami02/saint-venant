"""Test W&B model artifact round-trip: log → download → verify identical output.

Trains a tiny WaveNO for 1 epoch, logs it as an artifact, downloads it back,
and checks that the reloaded model produces bit-identical predictions.

Run:
    cd wavefront_learning && uv run python -m pytest tests/test_wandb_roundtrip.py -v -s
"""

import torch
import pytest

import wandb
from logger import WandbLogger
from model import get_model, load_model


# ── Helpers ──────────────────────────────────────────────────────────────────


def _minimal_waveno_config() -> dict:
    """Return the smallest viable WaveNO config."""
    return {
        "model": "WaveNO",
        "equation": "LWR",
        "hidden_dim": 16,
        "num_self_attn_layers": 1,
        "num_cross_layers": 1,
        "num_heads": 2,
        "dropout": 0.0,
        "num_cross_segment_layers": 0,
    }


def _dummy_batch(batch_size: int = 2, K: int = 3, nt: int = 4, nx: int = 5):
    """Create a synthetic WaveNO input batch."""
    xs = torch.rand(batch_size, K + 1).sort(dim=1).values
    ks = torch.rand(batch_size, K)
    pieces_mask = torch.ones(batch_size, K)
    t_coords = torch.linspace(0, 1, nt).view(1, 1, nt, 1).expand(batch_size, 1, nt, nx)
    x_coords = torch.linspace(0, 1, nx).view(1, 1, 1, nx).expand(batch_size, 1, nt, nx)
    return {
        "xs": xs,
        "ks": ks,
        "pieces_mask": pieces_mask,
        "t_coords": t_coords,
        "x_coords": x_coords,
    }


def _train_one_step(model, batch):
    """Run a single training step so weights diverge from init."""
    target = torch.rand(batch["ks"].shape[0], 1, 4, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    out = model(batch)["output_grid"]
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    optimizer.step()
    model.eval()


# ── Test ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def wandb_logger():
    """Create a real (but short-lived) W&B run for artifact testing."""
    logger = WandbLogger(
        project="wavefront-learning-test",
        name="pytest-roundtrip",
        config={"test": True},
        tags=["test"],
        enabled=True,
    )
    yield logger
    logger.finish()


def test_wandb_model_roundtrip(wandb_logger):
    """Log a model to W&B, download it, and verify identical output."""
    device = torch.device("cpu")
    config = _minimal_waveno_config()

    # 1. Build and train a tiny model
    model = get_model("WaveNO", config)
    model.to(device)
    batch = _dummy_batch()
    _train_one_step(model, batch)

    # 2. Log to W&B
    wandb_logger.log_model(model, "WaveNO", metadata=config)
    # Wait for artifact to be committed
    wandb_logger.run.log_artifact(wandb.Artifact("_noop", type="noop"))
    wandb_logger.run.finish()

    # 3. Start a new run in the same project to download
    dl_logger = WandbLogger(
        project="wavefront-learning-test",
        name="pytest-roundtrip-download",
        config={"test": True},
        tags=["test"],
        enabled=True,
    )
    try:
        # 4. Download and reload
        loaded_model = load_model(
            model_path="unused_fallback.pth",
            device=device,
            args=config,
            logger=dl_logger,
        )

        # 5. Compare state dicts
        orig_sd = model.state_dict()
        loaded_sd = loaded_model.state_dict()
        assert set(orig_sd.keys()) == set(loaded_sd.keys()), "State dict keys differ"
        for key in orig_sd:
            assert torch.equal(orig_sd[key], loaded_sd[key]), (
                f"Parameter '{key}' differs after round-trip"
            )

        # 6. Compare outputs on same input
        model.eval()
        with torch.no_grad():
            out_orig = model(batch)["output_grid"]
            out_loaded = loaded_model(batch)["output_grid"]
        assert torch.equal(out_orig, out_loaded), (
            f"Outputs differ! Max abs diff = {(out_orig - out_loaded).abs().max().item()}"
        )
    finally:
        dl_logger.finish()
