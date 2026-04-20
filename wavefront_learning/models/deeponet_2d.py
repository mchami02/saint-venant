"""DeepONet2D for 2D-spatial wavefront learning.

Branch: encodes the flattened tile IC (``4 * Kx * Ky`` values plus
``Kx + 1`` x-breakpoints and ``Ky + 1`` y-breakpoints) → latent per
output channel.
Trunk: encodes ``(t, x, y)`` per query point → latent.
Output = branch · trunk + bias per channel.
"""

import torch
import torch.nn as nn


class DeepONet2D(nn.Module):
    def __init__(
        self,
        max_steps: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_branch_layers: int = 4,
        num_trunk_layers: int = 4,
        output_dim: int = 4,
        n_ic_vars: int = 4,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.output_dim = output_dim
        self.n_ic_vars = n_ic_vars
        self.latent_dim = latent_dim

        # Branch input: n_ic_vars * Kx * Ky tile values + (Kx+1) xs + (Ky+1) ys
        branch_in = n_ic_vars * max_steps * max_steps + 2 * (max_steps + 1)
        branch_out = latent_dim * output_dim
        layers = [nn.Linear(branch_in, hidden_dim), nn.GELU()]
        for _ in range(num_branch_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, branch_out))
        self.branch = nn.Sequential(*layers)

        # Trunk: (t, x, y) → latent
        layers = [nn.Linear(3, hidden_dim), nn.GELU()]
        for _ in range(num_trunk_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.trunk = nn.Sequential(*layers)

        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, batch_input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        xs = batch_input["xs"]    # (B, Kx+1)
        ys = batch_input["ys"]    # (B, Ky+1)
        ks = batch_input["ks"].flatten(1)        # (B, Kx*Ky)
        ks_u = batch_input["ks_u"].flatten(1)
        ks_v = batch_input["ks_v"].flatten(1)
        ks_p = batch_input["ks_p"].flatten(1)
        t_coords = batch_input["t_coords"]       # (B, nt, ny, nx)
        x_coords = batch_input["x_coords"]
        y_coords = batch_input["y_coords"]

        B, nt, ny, nx = t_coords.shape

        ic_flat = torch.cat([ks, ks_u, ks_v, ks_p, xs, ys], dim=1)  # (B, ...)
        branch_out = self.branch(ic_flat)  # (B, latent_dim * output_dim)
        branch_reshaped = branch_out.reshape(
            B, self.output_dim, self.latent_dim
        )

        coords = torch.stack([t_coords, x_coords, y_coords], dim=-1)  # (B, nt, ny, nx, 3)
        coords_flat = coords.reshape(B, -1, 3)
        trunk_out = self.trunk(coords_flat)  # (B, N, latent_dim)

        out = torch.einsum("bcp,bnp->bcn", branch_reshaped, trunk_out)
        out = out + self.bias.reshape(1, self.output_dim, 1)
        out = out.reshape(B, self.output_dim, nt, ny, nx)
        return {"output_grid": out}


def build_deeponet_2d(args: dict) -> DeepONet2D:
    return DeepONet2D(
        max_steps=args.get("max_steps", 3),
        hidden_dim=args.get("hidden_dim", 128),
        latent_dim=args.get("latent_dim", 64),
        num_branch_layers=args.get("num_branch_layers", 4),
        num_trunk_layers=args.get("num_trunk_layers", 4),
        output_dim=4 if args.get("equation", "Euler2D") == "Euler2D" else 4,
        n_ic_vars=4,
    )
