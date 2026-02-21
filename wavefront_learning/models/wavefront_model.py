"""WaveFrontModel: Learned Riemann solver with analytical wave reconstruction.

Predicts wave characteristics (shock vs rarefaction, speeds) for each
discontinuity, handles wave interactions iteratively, and reconstructs
the full density grid from the resulting wave pattern.

Architecture:
    (rho_L, rho_R) per disc
      -> MLP encoder (2 -> H)
      -> 3 heads: classifier, shock_head, rarefaction_head
      -> build_waves: initial waves + collision processing -> wave dict
      -> get_grid: wave dict + query coords -> density grid (B, 1, nt, nx)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveFrontModel(nn.Module):
    """Learned Riemann solver with analytical wave reconstruction.

    Components:
        disc_encoder: MLP encoder for (rho_L, rho_R) discontinuities
        classifier_head: P(shock) per discontinuity
        shock_head: shock speed (dx/dt) per discontinuity
        rarefaction_head: [speed1, delta] per discontinuity

    Args:
        hidden_dim: Hidden dimension for encoder and heads.
        rarefaction_angles: Number of sub-waves per rarefaction fan.
        max_interaction_rounds: Maximum collision processing iterations.
        sigma: Sigmoid sharpness for grid reconstruction.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        rarefaction_angles: int = 5,
        max_interaction_rounds: int = 5,
        sigma: float = 0.01,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rarefaction_angles = rarefaction_angles
        self.max_interaction_rounds = max_interaction_rounds
        self.sigma = sigma

        # Shared discontinuity encoder: (rho_L, rho_R) -> H
        self.disc_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Three prediction heads
        H = hidden_dim
        self.classifier_head = nn.Sequential(
            nn.Linear(H, H // 2),
            nn.ReLU(),
            nn.Linear(H // 2, 1),
        )

        self.shock_head = nn.Sequential(
            nn.Linear(H, H // 2),
            nn.GELU(),
            nn.Linear(H // 2, 1),
        )

        self.rarefaction_head = nn.Sequential(
            nn.Linear(H, H // 2),
            nn.GELU(),
            nn.Linear(H // 2, 2),
        )

    def _encode_and_predict(
        self,
        discs: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode discontinuities and predict wave parameters.

        Args:
            discs: (B, D, 3) discontinuity features [x, rho_L, rho_R].
            mask: (B, D) validity mask.

        Returns:
            is_shock: (B, D) probability of shock.
            shock_speed: (B, D) predicted shock speed.
            rar_params: (B, D, 2) rarefaction [speed1, delta].
        """
        features = discs[:, :, 1:]  # (B, D, 2) — just rho_L, rho_R
        emb = self.disc_encoder(features) * mask.unsqueeze(-1)  # (B, D, H)
        is_shock = torch.sigmoid(self.classifier_head(emb).squeeze(-1)) * mask
        shock_speed = self.shock_head(emb).squeeze(-1)  # (B, D)
        rar_params = self.rarefaction_head(emb)  # (B, D, 2)
        return is_shock, shock_speed, rar_params

    def _build_initial_waves(
        self,
        discontinuities: torch.Tensor,
        disc_mask: torch.Tensor,
        is_shock: torch.Tensor,
        shock_speed: torch.Tensor,
        rar_params: torch.Tensor,
        rarefaction_angles: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Build initial wave tensors from discontinuity predictions.

        Returns:
            wave_origin_x: (B, W) wave origin x positions
            wave_origin_t: (B, W) wave origin times (all zeros)
            wave_speeds: (B, W) wave speeds
            wave_jumps: (B, W) density jumps
            wave_active: (B, W) soft activity mask
            wave_types: (B, W) 0=shock, 1=rarefaction
        """
        B, D, _ = discontinuities.shape
        N = rarefaction_angles
        device = discontinuities.device

        x_pos = discontinuities[:, :, 0]  # (B, D)
        total_jump = discontinuities[:, :, 2] - discontinuities[:, :, 1]  # (B, D)

        # Build lists of (B,) tensors per wave slot — avoids in-place mutation
        all_origin_x: list[torch.Tensor] = []
        all_speeds: list[torch.Tensor] = []
        all_jumps: list[torch.Tensor] = []
        all_active: list[torch.Tensor] = []
        all_types: list[float] = []

        for d in range(D):
            # Shock wave (1 slot per disc)
            all_origin_x.append(x_pos[:, d])
            all_speeds.append(shock_speed[:, d])
            all_jumps.append(total_jump[:, d] * is_shock[:, d])
            all_active.append(disc_mask[:, d] * is_shock[:, d])
            all_types.append(0.0)

            # Rarefaction fan (N slots per disc)
            speed1 = rar_params[:, d, 0]  # (B,)
            delta = F.softplus(rar_params[:, d, 1])  # (B,)
            rar_weight = (1.0 - is_shock[:, d])  # (B,)

            for n in range(N):
                frac = n / max(N - 1, 1)
                all_origin_x.append(x_pos[:, d])
                all_speeds.append(speed1 + frac * delta)
                all_jumps.append(total_jump[:, d] * rar_weight / N)
                all_active.append(disc_mask[:, d] * rar_weight)
                all_types.append(1.0)

        n_initial = len(all_speeds)  # D * (1 + N)

        # Stack lists → (B, n_initial)
        wave_origin_x = torch.stack(all_origin_x, dim=1)
        wave_speeds = torch.stack(all_speeds, dim=1)
        wave_jumps = torch.stack(all_jumps, dim=1)
        wave_active = torch.stack(all_active, dim=1)

        # No grad dependencies — safe to build directly
        wave_origin_t = torch.zeros(B, n_initial, device=device)
        wave_types = torch.tensor(all_types, device=device).unsqueeze(0).expand(B, -1)

        return (
            wave_origin_x,
            wave_origin_t,
            wave_speeds,
            wave_jumps,
            wave_active,
            wave_types,
        )

    def _process_collisions(
        self,
        wave_origin_x: torch.Tensor,
        wave_origin_t: torch.Tensor,
        wave_speeds: torch.Tensor,
        wave_jumps: torch.Tensor,
        wave_active: torch.Tensor,
        wave_types: torch.Tensor,
        T_max: float,
        rarefaction_angles: int,
        max_interaction_rounds: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Iterative collision processing.

        For each round, find earliest collision between adjacent converging waves,
        spawn new waves from the collision point, and soft-deactivate colliding waves.
        """
        B = wave_origin_x.shape[0]
        N = rarefaction_angles
        device = wave_origin_x.device

        for _ in range(max_interaction_rounds):
            W = wave_origin_x.shape[1]

            # Position of each wave at a reference time (use T_max/2 for sorting)
            t_ref = T_max / 2.0
            pos_at_ref = wave_origin_x + wave_speeds * (t_ref - wave_origin_t)

            # Sort active waves by position at reference time
            # Use large values for inactive waves to push them to the end
            sort_pos = pos_at_ref + (1.0 - wave_active) * 1e6
            sort_idx = torch.argsort(sort_pos, dim=1)  # (B, W)

            # Gather sorted wave properties
            sorted_ox = torch.gather(wave_origin_x, 1, sort_idx)
            sorted_ot = torch.gather(wave_origin_t, 1, sort_idx)
            sorted_sp = torch.gather(wave_speeds, 1, sort_idx)
            sorted_jm = torch.gather(wave_jumps, 1, sort_idx)
            sorted_ac = torch.gather(wave_active, 1, sort_idx)

            # Compute pairwise collision times for adjacent waves
            # Two waves collide when: ox_i + sp_i * (t - ot_i) = ox_j + sp_j * (t - ot_j)
            # t = (ox_j - ox_i + sp_i*ot_i - sp_j*ot_j) / (sp_i - sp_j)
            speed_diff = sorted_sp[:, :-1] - sorted_sp[:, 1:]  # (B, W-1)
            pos_diff = (
                sorted_ox[:, 1:]
                - sorted_ox[:, :-1]
                + sorted_sp[:, :-1] * sorted_ot[:, :-1]
                - sorted_sp[:, 1:] * sorted_ot[:, 1:]
            )

            # Avoid division by zero
            speed_diff_safe = speed_diff + (speed_diff.abs() < 1e-10).float() * 1e-10
            t_coll = pos_diff / speed_diff_safe  # (B, W-1)

            # Collision is valid only if:
            # 1. Both waves active
            # 2. Converging (speed_diff > 0 means left wave faster)
            # 3. Collision time is in [0, T_max]
            both_active = sorted_ac[:, :-1] * sorted_ac[:, 1:]  # (B, W-1)
            converging = (speed_diff > 1e-8).float()
            in_time = ((t_coll > 0) & (t_coll < T_max)).float()
            valid = both_active * converging * in_time  # (B, W-1)

            # Find earliest valid collision per batch
            t_coll_masked = t_coll + (1.0 - valid) * 1e10
            earliest_idx = torch.argmin(t_coll_masked, dim=1)  # (B,)
            earliest_t = torch.gather(t_coll_masked, 1, earliest_idx.unsqueeze(1)).squeeze(1)
            has_collision = (earliest_t < 1e9).float()  # (B,)

            if has_collision.sum() < 0.5:
                break

            # Get collision properties
            idx_left = earliest_idx  # (B,)
            idx_right = earliest_idx + 1

            coll_t = torch.gather(t_coll, 1, idx_left.unsqueeze(1)).squeeze(1)
            left_ox = torch.gather(sorted_ox, 1, idx_left.unsqueeze(1)).squeeze(1)
            left_ot = torch.gather(sorted_ot, 1, idx_left.unsqueeze(1)).squeeze(1)
            left_sp = torch.gather(sorted_sp, 1, idx_left.unsqueeze(1)).squeeze(1)

            # Collision position
            coll_x = left_ox + left_sp * (coll_t - left_ot)

            # Compute outer densities for new discontinuity
            # Cumulative jump up to left wave = sum of all jumps to the left
            cumjump = torch.cumsum(sorted_jm * sorted_ac, dim=1)  # (B, W)

            rho_at_left = torch.gather(cumjump, 1, idx_left.unsqueeze(1)).squeeze(1)
            left_jm = torch.gather(sorted_jm, 1, idx_left.unsqueeze(1)).squeeze(1)
            right_jm = torch.gather(sorted_jm, 1, idx_right.unsqueeze(1)).squeeze(1)

            rho_L_new = rho_at_left - left_jm  # density just left of left wave
            rho_R_new = rho_at_left + right_jm  # density just right of right wave

            # Create new discontinuity for re-encoding
            new_disc = torch.stack(
                [coll_x, rho_L_new, rho_R_new], dim=-1
            ).unsqueeze(1)  # (B, 1, 3)
            new_mask = has_collision.unsqueeze(1)  # (B, 1)

            # Re-encode
            new_is_shock, new_shock_speed, new_rar_params = self._encode_and_predict(
                new_disc, new_mask
            )

            new_jump = rho_R_new - rho_L_new  # (B,)

            # Build spawned wave tensors via list + stack (no in-place ops)
            new_ox_list: list[torch.Tensor] = []
            new_ot_list: list[torch.Tensor] = []
            new_sp_list: list[torch.Tensor] = []
            new_jm_list: list[torch.Tensor] = []
            new_ac_list: list[torch.Tensor] = []

            # Shock wave
            new_ox_list.append(coll_x * has_collision)
            new_ot_list.append(coll_t * has_collision)
            new_sp_list.append(new_shock_speed[:, 0] * has_collision)
            new_jm_list.append(new_jump * new_is_shock[:, 0] * has_collision)
            new_ac_list.append(new_is_shock[:, 0] * has_collision)

            # Rarefaction fan
            s1 = new_rar_params[:, 0, 0]
            d1 = F.softplus(new_rar_params[:, 0, 1])
            rar_w = (1.0 - new_is_shock[:, 0]) * has_collision
            for n in range(N):
                frac = n / max(N - 1, 1)
                new_ox_list.append(coll_x * has_collision)
                new_ot_list.append(coll_t * has_collision)
                new_sp_list.append((s1 + frac * d1) * has_collision)
                new_jm_list.append(new_jump * rar_w / N)
                new_ac_list.append(rar_w)

            # Stack → (B, N+1), then cat onto existing buffers
            spawned_ox = torch.stack(new_ox_list, dim=1)  # (B, N+1)
            spawned_ot = torch.stack(new_ot_list, dim=1)
            spawned_sp = torch.stack(new_sp_list, dim=1)
            spawned_jm = torch.stack(new_jm_list, dim=1)
            spawned_ac = torch.stack(new_ac_list, dim=1)
            spawned_types = torch.full(
                (B, N + 1), 2.0, device=device
            )

            # Soft-deactivate colliding waves (before cat, so one_hot size matches W)
            orig_idx_left = torch.gather(sort_idx, 1, idx_left.unsqueeze(1)).squeeze(1)
            orig_idx_right = torch.gather(sort_idx, 1, idx_right.unsqueeze(1)).squeeze(1)

            deact_mask_left = F.one_hot(orig_idx_left, W).float()
            deact_mask_right = F.one_hot(orig_idx_right, W).float()
            deact = (deact_mask_left + deact_mask_right) * has_collision.unsqueeze(1)
            wave_active = wave_active * (1.0 - deact).clamp(0, 1)

            # Concatenate spawned waves onto buffers
            wave_origin_x = torch.cat([wave_origin_x, spawned_ox], dim=1)
            wave_origin_t = torch.cat([wave_origin_t, spawned_ot], dim=1)
            wave_speeds = torch.cat([wave_speeds, spawned_sp], dim=1)
            wave_jumps = torch.cat([wave_jumps, spawned_jm], dim=1)
            wave_active = torch.cat([wave_active, spawned_ac], dim=1)
            wave_types = torch.cat([wave_types, spawned_types], dim=1)

        return (
            wave_origin_x,
            wave_origin_t,
            wave_speeds,
            wave_jumps,
            wave_active,
            wave_types,
        )

    def _reconstruct_grid(
        self,
        wave_origin_x: torch.Tensor,
        wave_origin_t: torch.Tensor,
        wave_speeds: torch.Tensor,
        wave_jumps: torch.Tensor,
        wave_active: torch.Tensor,
        base_density: torch.Tensor,
        t_coords: torch.Tensor,
        x_coords: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        """Reconstruct density grid from wave pattern.

        density(t, x) = base + sum_w jump_w * sigmoid((x - pos_w(t)) / sigma) * active_w

        Args:
            wave_origin_x: (B, W)
            wave_origin_t: (B, W)
            wave_speeds: (B, W)
            wave_jumps: (B, W)
            wave_active: (B, W)
            base_density: (B,) leftmost density value
            t_coords: (B, 1, nt, nx) time coordinates
            x_coords: (B, 1, nt, nx) space coordinates

        Returns:
            output_grid: (B, 1, nt, nx)
        """
        B, _, nt, nx = t_coords.shape
        W = wave_origin_x.shape[1]

        # Reshape coords: (B, nt, nx)
        t = t_coords[:, 0, :, :]  # (B, nt, nx)
        x = x_coords[:, 0, :, :]  # (B, nt, nx)

        # Expand wave params for broadcasting: (B, W, 1, 1)
        ox = wave_origin_x[:, :, None, None]  # (B, W, 1, 1)
        ot = wave_origin_t[:, :, None, None]  # (B, W, 1, 1)
        sp = wave_speeds[:, :, None, None]  # (B, W, 1, 1)
        jm = wave_jumps[:, :, None, None]  # (B, W, 1, 1)
        ac = wave_active[:, :, None, None]  # (B, W, 1, 1)

        # Expand coords: (B, 1, nt, nx)
        t_exp = t[:, None, :, :]  # (B, 1, nt, nx)
        x_exp = x[:, None, :, :]  # (B, 1, nt, nx)

        # Wave position at each time: (B, W, nt, nx)
        wave_pos = ox + sp * (t_exp - ot)

        # Soft step function: sigmoid((x - wave_pos) / sigma)
        step = torch.sigmoid((x_exp - wave_pos) / sigma)

        # Density contribution per wave: jump * step * activity
        contributions = jm * step * ac  # (B, W, nt, nx)

        # Sum all wave contributions + base density
        density = base_density[:, None, None] + contributions.sum(dim=1)  # (B, nt, nx)

        return density.unsqueeze(1)  # (B, 1, nt, nx)

    def build_waves(
        self,
        batch_input: dict,
        rarefaction_angles: int,
        max_interaction_rounds: int,
    ) -> dict:
        """Encode discontinuities, build initial waves, and process collisions.

        Args:
            batch_input: Dictionary containing:
                - discontinuities: (B, D, 3) [x, rho_L, rho_R]
                - disc_mask: (B, D) validity mask
                - ks: (B, K) piece values (used for base density)
                - t_coords: (B, 1, nt, nx) time coordinates
            rarefaction_angles: Number of sub-waves per rarefaction fan.
            max_interaction_rounds: Maximum collision processing iterations.

        Returns:
            Dictionary containing wave data:
                - wave_origin_x: (B, W) wave origin x positions
                - wave_origin_t: (B, W) wave origin times
                - wave_speeds: (B, W) wave speeds
                - wave_jumps: (B, W) density jumps
                - wave_active: (B, W) soft activity mask
                - wave_types: (B, W) 0=shock, 1=rarefaction, 2=spawned
                - base_density: (B,) leftmost density value
        """
        discontinuities = batch_input["discontinuities"]  # (B, D, 3)
        disc_mask = batch_input["disc_mask"]  # (B, D)
        ks = batch_input["ks"]  # (B, K)
        t_coords = batch_input["t_coords"]  # (B, 1, nt, nx)

        base_density = ks[:, 0]  # (B,)
        T_max = float(t_coords[:, 0, -1, 0].max())

        # Encode and predict wave parameters
        is_shock, shock_speed, rar_params = self._encode_and_predict(
            discontinuities, disc_mask
        )

        # Build initial waves
        (
            wave_origin_x,
            wave_origin_t,
            wave_speeds,
            wave_jumps,
            wave_active,
            wave_types,
        ) = self._build_initial_waves(
            discontinuities, disc_mask, is_shock, shock_speed, rar_params,
            rarefaction_angles,
        )

        # Iterative collision processing
        (
            wave_origin_x,
            wave_origin_t,
            wave_speeds,
            wave_jumps,
            wave_active,
            wave_types,
        ) = self._process_collisions(
            wave_origin_x,
            wave_origin_t,
            wave_speeds,
            wave_jumps,
            wave_active,
            wave_types,
            T_max,
            rarefaction_angles,
            max_interaction_rounds,
        )

        return {
            "wave_origin_x": wave_origin_x,
            "wave_origin_t": wave_origin_t,
            "wave_speeds": wave_speeds,
            "wave_jumps": wave_jumps,
            "wave_active": wave_active,
            "wave_types": wave_types,
            "base_density": base_density,
        }

    def get_grid(
        self,
        waves: dict,
        t_coords: torch.Tensor,
        x_coords: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        """Reconstruct density grid from wave data.

        Args:
            waves: Dictionary from build_waves().
            t_coords: (B, 1, nt, nx) time coordinates.
            x_coords: (B, 1, nt, nx) space coordinates.
            sigma: Sigmoid sharpness for grid reconstruction.

        Returns:
            output_grid: (B, 1, nt, nx) predicted density.
        """
        return self._reconstruct_grid(
            waves["wave_origin_x"],
            waves["wave_origin_t"],
            waves["wave_speeds"],
            waves["wave_jumps"],
            waves["wave_active"],
            waves["base_density"],
            t_coords,
            x_coords,
            sigma,
        )

    def forward(self, batch_input: dict) -> dict:
        """Forward pass — delegates to predict with constructor defaults."""
        return self.predict(
            batch_input,
            rarefaction_angles=self.rarefaction_angles,
            max_interaction_rounds=self.max_interaction_rounds,
            sigma=self.sigma,
        )

    def predict(
        self,
        batch_input: dict,
        rarefaction_angles: int,
        max_interaction_rounds: int,
        sigma: float,
    ) -> dict:
        """Run the full pipeline with overridable hyperparameters.

        Args:
            batch_input: Dictionary containing:
                - discontinuities: (B, D, 3) [x, rho_L, rho_R]
                - disc_mask: (B, D) validity mask
                - t_coords: (B, 1, nt, nx) time coordinates
                - x_coords: (B, 1, nt, nx) space coordinates
                - ks: (B, K) piece values (used for base density)
            rarefaction_angles: Number of sub-waves per rarefaction fan.
            max_interaction_rounds: Maximum collision processing iterations.
            sigma: Sigmoid sharpness for grid reconstruction.

        Returns:
            Dictionary containing:
                - output_grid: (B, 1, nt, nx) predicted density
                - wave_origins_x: (B, W) wave origin x positions
                - wave_origins_t: (B, W) wave origin times
                - wave_speeds: (B, W) wave speeds
                - wave_active: (B, W) wave activity
                - wave_types: (B, W) wave types (0=shock, 1=rarefaction, 2=spawned)
        """
        waves = self.build_waves(batch_input, rarefaction_angles, max_interaction_rounds)
        output_grid = self.get_grid(
            waves, batch_input["t_coords"], batch_input["x_coords"], sigma
        )

        return {
            "output_grid": output_grid,
            "wave_origins_x": waves["wave_origin_x"].detach(),
            "wave_origins_t": waves["wave_origin_t"].detach(),
            "wave_speeds": waves["wave_speeds"].detach(),
            "wave_active": waves["wave_active"].detach(),
            "wave_types": waves["wave_types"].detach(),
        }


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
        rarefaction_angles=args.get("rarefaction_angles", 5),
        max_interaction_rounds=args.get("max_interaction_rounds", 5),
        sigma=args.get("sigma", 0.01),
        dropout=args.get("dropout", 0.05),
    )
