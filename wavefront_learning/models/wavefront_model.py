"""WaveFrontModel: Learned Riemann solver with analytical wave reconstruction.

Predicts wave characteristics (shock vs rarefaction, speeds) for each
discontinuity, handles type-aware wave interactions iteratively, and
reconstructs the full density grid from the resulting wave pattern.

Wave types:
    0 = shock: step function at linear position
    1 = rarefaction: entropy solution fan (single entity)
    2 = bent shock: polynomial trajectory from shock-rarefaction interaction

Architecture:
    (rho_L, rho_R) per disc
      -> MLP encoder (2 -> H)
      -> 2 heads: classifier, shock_head
      -> build_initial_waves: STE for type selection
      -> process_collisions: type-aware (shock-shock, rar-rar, shock-rar)
      -> reconstruct_grid: type-aware reconstruction -> (B, 1, nt, nx)
"""

import torch
import torch.nn as nn

from .base.collision_processor import process_collisions
from .base.flux import Flux, GreenshieldsFlux
from .base.wave_builder import build_initial_waves
from .base.wave_reconstructor import reconstruct_grid


_UNSET = object()


class WaveFrontModel(nn.Module):
    """Learned Riemann solver with analytical wave reconstruction.

    Components:
        disc_encoder: MLP encoder for (rho_L, rho_R) discontinuities
        classifier_head: P(shock) per discontinuity
        shock_head: shock speed (dx/dt) per discontinuity
        bent_shock_head: polynomial coefficients for shock-rar interactions

    Args:
        hidden_dim: Hidden dimension for encoder and heads.
        max_interaction_rounds: Maximum collision processing iterations.
        sigma: Sigmoid sharpness for grid reconstruction.
        dropout: Dropout rate.
        flux: Flux instance (default: GreenshieldsFlux).
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        max_interaction_rounds: int = 5,
        sigma: float = 0.01,
        dropout: float = 0.05,
        flux: Flux | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_interaction_rounds = max_interaction_rounds
        self.sigma = sigma
        self.flux = flux or GreenshieldsFlux()

        # Shared discontinuity encoder: (rho_L, rho_R) -> H
        self.disc_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

        H = hidden_dim

        # Classifier: P(shock) per discontinuity
        self.classifier_head = nn.Sequential(
            nn.Linear(H, H // 2),
            nn.ReLU(),
            nn.Linear(H // 2, 1),
        )

        # Shock speed per discontinuity
        self.shock_head = nn.Sequential(
            nn.Linear(H, H // 2),
            nn.GELU(),
            nn.Linear(H // 2, 1),
        )

        # Bent shock head: 7 features -> (c2, c3, log_duration)
        self.bent_shock_head = nn.Sequential(
            nn.Linear(7, H // 2),
            nn.GELU(),
            nn.Linear(H // 2, 3),
        )

    def _encode_and_predict(
        self,
        discs: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode discontinuities and predict wave parameters.

        Args:
            discs: (B, D, 3) discontinuity features [x, rho_L, rho_R].
            mask: (B, D) validity mask.

        Returns:
            is_shock: (B, D) probability of shock.
            shock_speed: (B, D) predicted shock speed.
        """
        features = discs[:, :, 1:]  # (B, D, 2) — just rho_L, rho_R
        emb = self.disc_encoder(features) * mask.unsqueeze(-1)  # (B, D, H)
        is_shock = torch.sigmoid(self.classifier_head(emb).squeeze(-1)) * mask
        shock_speed = self.shock_head(emb).squeeze(-1)  # (B, D)
        return is_shock, shock_speed

    def forward(self, batch_input: dict) -> dict:
        """Unified forward pass for both training and evaluation.

        Pipeline:
        1. Encode discontinuities -> is_shock, shock_speed
        2. Build initial waves (STE during training for type selection)
        3. Process collisions (type-aware, both train and eval)
        4. Reconstruct grid (type-aware: shock/rarefaction/bent shock)
        """
        discontinuities = batch_input["discontinuities"]  # (B, D, 3)
        disc_mask = batch_input["disc_mask"]  # (B, D)
        ks = batch_input["ks"]  # (B, K)
        t_coords = batch_input["t_coords"]  # (B, 1, nt, nx)
        x_coords = batch_input["x_coords"]  # (B, 1, nt, nx)

        base_density = ks[:, 0]  # (B,)
        T_max = float(t_coords[:, 0, -1, 0].max())

        # 1. Encode and predict
        is_shock, shock_speed = self._encode_and_predict(
            discontinuities, disc_mask
        )

        # 2. Build initial waves
        waves = build_initial_waves(
            discontinuities, disc_mask, is_shock, shock_speed,
            self.flux, self.training,
        )

        # 3. Process collisions (both train and eval)
        waves = process_collisions(
            waves, self.flux, self.bent_shock_head,
            T_max, self.max_interaction_rounds,
        )

        # 4. Reconstruct grid
        sigma = self.sigma if self.training else None
        output_grid = reconstruct_grid(
            waves, base_density, t_coords, x_coords,
            self.flux, sigma,
        )

        result = {"output_grid": output_grid}

        # Add wave data for plotting (eval only)
        if not self.training:
            result.update({
                "wave_origins_x": waves["wave_origin_x"].detach(),
                "wave_origins_t": waves["wave_origin_t"].detach(),
                "wave_left_speed": waves["wave_left_speed"].detach(),
                "wave_right_speed": waves["wave_right_speed"].detach(),
                "wave_active": waves["wave_active"].detach(),
                "wave_types": waves["wave_type"].detach(),
                "wave_poly_c2": waves["wave_poly_c2"].detach(),
                "wave_poly_c3": waves["wave_poly_c3"].detach(),
                "wave_poly_duration": waves["wave_poly_duration"].detach(),
            })

        return result

    def predict(
        self,
        batch_input: dict,
        max_interaction_rounds: int | None = None,
        sigma: float | None = _UNSET,
    ) -> dict:
        """Run the full pipeline with overridable hyperparameters.

        Args:
            batch_input: Dictionary with discontinuities, disc_mask, etc.
            max_interaction_rounds: Override collision rounds.
            sigma: Override sigmoid sharpness.

        Returns:
            Dictionary with output_grid and wave data.
        """
        old_rounds = self.max_interaction_rounds
        old_sigma = self.sigma

        if max_interaction_rounds is not None:
            self.max_interaction_rounds = max_interaction_rounds
        if sigma is not _UNSET:
            self.sigma = sigma

        result = self.forward(batch_input)

        self.max_interaction_rounds = old_rounds
        self.sigma = old_sigma

        return result


def build_wavefront_model(args: dict) -> WaveFrontModel:
    """Factory function for WaveFrontModel.

    Args:
        args: Dictionary or Namespace containing model configuration.

    Returns:
        Configured WaveFrontModel instance.
    """
    if not isinstance(args, dict):
        args = vars(args)
    return WaveFrontModel(
        hidden_dim=args.get("hidden_dim", 64),
        max_interaction_rounds=args.get("max_interaction_rounds", 5),
        sigma=args.get("sigma", 0.01),
        dropout=args.get("dropout", 0.05),
    )
