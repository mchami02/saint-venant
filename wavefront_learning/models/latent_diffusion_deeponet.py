"""Latent Diffusion DeepONet.

Generative model for hyperbolic PDE solutions with sharp discontinuities.
Phase 1: Train VAE (encoder + DeepONet decoder) on coarse-grid solutions.
Phase 2: Train flow matching denoiser conditioned on IC, in the frozen latent space.
Inference: Sample noise → ODE solve → DeepONet decode at arbitrary resolution.
"""

import torch
import torch.nn as nn

from .base.deeponet_decoder import DeepONetDecoder
from .base.flow_matching import ConditionEncoder, FlowMatchingDenoiser, HeunODESolver
from .base.vae_encoder import VAEEncoder


class LatentDiffusionDeepONet(nn.Module):
    """Latent Diffusion DeepONet combining VAE and flow matching.

    Training is split into two phases:
    - Phase 1: VAE training (encoder + decoder). Loss = MSE + beta * KL.
    - Phase 2: Flow matching (frozen encoder/decoder). Loss = velocity MSE.
    - Eval: Condition encoder → Heun ODE → decoder at query coordinates.

    Args:
        latent_dim: Dimension of the latent space.
        num_basis: Number of DeepONet basis functions.
        condition_dim: Dimension of the condition embedding.
        max_pieces: Maximum number of IC pieces.
        num_ode_steps: Number of Heun ODE steps at inference.
        trunk_hidden_dim: Hidden dimension for trunk MLP.
        branch_hidden_dim: Hidden dimension for branch MLP.
        denoiser_hidden_dim: Hidden dimension for denoiser MLP.
        num_residual_blocks: Number of residual blocks in denoiser.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        num_basis: int = 64,
        condition_dim: int = 64,
        max_pieces: int = 10,
        num_ode_steps: int = 100,
        trunk_hidden_dim: int = 128,
        branch_hidden_dim: int = 128,
        denoiser_hidden_dim: int = 256,
        num_residual_blocks: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_ode_steps = num_ode_steps
        self.phase = 1

        # VAE components
        self.encoder = VAEEncoder(latent_dim=latent_dim)
        self.decoder = DeepONetDecoder(
            latent_dim=latent_dim,
            num_basis=num_basis,
            trunk_hidden_dim=trunk_hidden_dim,
            branch_hidden_dim=branch_hidden_dim,
        )

        # Flow matching components
        self.condition_encoder = ConditionEncoder(
            max_pieces=max_pieces,
            condition_dim=condition_dim,
        )
        self.denoiser = FlowMatchingDenoiser(
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            hidden_dim=denoiser_hidden_dim,
            num_residual_blocks=num_residual_blocks,
        )

    def set_phase(self, phase: int) -> None:
        """Switch training phase.

        Phase 0: Inference mode (condition → ODE → decode).
        Phase 1: VAE training (all params trainable).
        Phase 2: Flow matching (encoder + decoder frozen).
        """
        self.phase = phase
        if phase == 2:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False

    def forward(self, batch_input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass dispatched by phase.

        Phase 1: VAE encode → reparameterize → decode (train + val).
        Phase 2: Frozen encode → flow matching (train + val).
        Phase 0: Inference via ODE solver (test / generation).

        Args:
            batch_input: Dict with keys including "xs", "ks", "pieces_mask",
                "t_coords", "x_coords", and (during training) "target_grid".

        Returns:
            Dict with phase-dependent keys (see inline comments).
        """
        if self.phase == 1:
            return self._forward_phase1(batch_input)
        elif self.phase == 2:
            return self._forward_phase2(batch_input)
        else:
            return self._forward_inference(batch_input)

    def _forward_phase1(
        self, batch_input: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Phase 1: VAE encode → reparameterize → decode."""
        target_grid = batch_input["target_grid"]  # (B, 1, nt, nx)
        t_coords = batch_input["t_coords"]  # (B, nt, nx)
        x_coords = batch_input["x_coords"]  # (B, nt, nx)

        # Handle coordinate shapes: strip channel dim if present
        if t_coords.dim() == 4:
            t_coords = t_coords[:, 0]
        if x_coords.dim() == 4:
            x_coords = x_coords[:, 0]

        mean, logvar = self.encoder(target_grid)
        z = VAEEncoder.reparameterize(mean, logvar)
        output_grid = self.decoder(z, t_coords, x_coords)

        return {
            "output_grid": output_grid,
            "z_mean": mean,
            "z_logvar": logvar,
        }

    def _forward_phase2(
        self, batch_input: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Phase 2: Frozen encode → sample noise/t → predict velocity."""
        target_grid = batch_input["target_grid"]  # (B, 1, nt, nx)
        t_coords = batch_input["t_coords"]
        x_coords = batch_input["x_coords"]

        if t_coords.dim() == 4:
            t_coords = t_coords[:, 0]
        if x_coords.dim() == 4:
            x_coords = x_coords[:, 0]

        B = target_grid.shape[0]
        dev = target_grid.device

        # Encode target to get z (use mean, no reparameterization)
        with torch.no_grad():
            mean, _ = self.encoder(target_grid)
            z = mean  # deterministic encoding for flow matching targets

        # Sample noise and diffusion time
        noise = torch.randn_like(z)
        t = torch.rand(B, device=dev)

        # OT interpolation: z_t = (1-t)*noise + t*z
        t_expand = t.unsqueeze(-1)  # (B, 1)
        z_t = (1 - t_expand) * noise + t_expand * z

        # Target velocity: z - noise (constant velocity field for OT)
        target_velocity = z - noise

        # Condition on IC
        condition = self.condition_encoder(batch_input)

        # Predict velocity
        predicted_velocity = self.denoiser(z_t, t, condition)

        # Also produce output_grid (detached) for metrics
        with torch.no_grad():
            output_grid = self.decoder(z, t_coords, x_coords)

        return {
            "predicted_velocity": predicted_velocity,
            "target_velocity": target_velocity,
            "output_grid": output_grid.detach(),
        }

    def _forward_inference(
        self, batch_input: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Inference: condition → ODE solve → decode."""
        t_coords = batch_input["t_coords"]
        x_coords = batch_input["x_coords"]

        if t_coords.dim() == 4:
            t_coords = t_coords[:, 0]
        if x_coords.dim() == 4:
            x_coords = x_coords[:, 0]

        B = t_coords.shape[0]
        dev = t_coords.device

        # Encode condition
        condition = self.condition_encoder(batch_input)

        # Sample noise and solve ODE
        noise = torch.randn(B, self.latent_dim, device=dev)
        z = HeunODESolver.solve(self.denoiser, noise, condition, self.num_ode_steps)

        # Decode at query coordinates
        output_grid = self.decoder(z, t_coords, x_coords)

        return {"output_grid": output_grid}

    def sample_multiple(
        self,
        batch_input: dict[str, torch.Tensor],
        num_samples: int = 5,
    ) -> list[dict[str, torch.Tensor]]:
        """Generate multiple samples for uncertainty quantification.

        Args:
            batch_input: Input dict.
            num_samples: Number of independent samples to generate.

        Returns:
            List of output dicts, one per sample.
        """
        self.eval()
        results = []
        for _ in range(num_samples):
            with torch.no_grad():
                results.append(self._forward_inference(batch_input))
        return results


def build_ld_deeponet(args: dict) -> LatentDiffusionDeepONet:
    """Factory function for LatentDiffusionDeepONet.

    Args:
        args: Config dict with optional keys:
            ld_latent_dim, ld_num_basis, ld_condition_dim,
            max_steps, ld_num_ode_steps.

    Returns:
        Configured LatentDiffusionDeepONet instance.
    """

    def get(k, d=None):
        if isinstance(args, dict):
            return args.get(k, d)
        return getattr(args, k, d)

    return LatentDiffusionDeepONet(
        latent_dim=get("ld_latent_dim", 32),
        num_basis=get("ld_num_basis", 64),
        condition_dim=get("ld_condition_dim", 64),
        max_pieces=get("max_discontinuities", 10),
        num_ode_steps=get("ld_num_ode_steps", 100),
    )
