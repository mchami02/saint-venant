import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class GatedMPNNLayer(MessagePassing):
    """Gated message-passing layer for shock correction."""
    def __init__(self, hidden_dim: int, edge_dim: int, gate_hidden_dim: int = 64):
        super().__init__(aggr="add")

        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Gate uses |u_i - u_j|, |du_dx_i|, |du_dx_j|, |du_dt_i|, |du_dt_j|
        self.gate_mlp = nn.Sequential(
            nn.Linear(7, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.gate_mlp[2].bias, -3.0)

        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self._gate_values = None

    def forward(self, x, edge_index, edge_attr, u, du_dx, du_dt):
        self._gate_values = None
        out = self.propagate(
            edge_index=edge_index,
            x=x,
            edge_attr=edge_attr,
            u=u,
            du_dx=du_dx,
            du_dt=du_dt,
        )
        x = x + self.update_mlp(torch.cat([x, out], dim=-1))
        return self.norm(x), self._gate_values

    def message(self, x_i, x_j, edge_attr, u_i, u_j, du_dx_i, du_dx_j, du_dt_i, du_dt_j):
        gate_input = torch.cat([
            u_i, u_j,
            torch.abs(u_i - u_j),
            du_dx_i, du_dx_j,
            du_dt_i, du_dt_j,
        ], dim=-1)
        gate = self.gate_mlp(gate_input)
        self._gate_values = gate
        msg = self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
        return gate * msg


class ShockGNN(nn.Module):
    """
    Shock corrector GNN with 2D grid connectivity.
    
    Edges connect each cell to its 8 neighbors (4 cardinal + 4 diagonal).
    Edge features are (dx, dt) displacements.
    """
    def __init__(self, nx: int, nt: int, dx: float, dt: float, hidden_dim: int, num_layers: int, device):
        super().__init__()
        self.dx = dx
        self.dt = dt
        self.nx = nx
        self.nt = nt

        self.in_proj = nn.Linear(1, hidden_dim)

        edge_index, edge_attr = self._build_2d_grid_edges(nt, nx, dt, dx, device=device)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_attr", edge_attr)

        self.layers = nn.ModuleList([
            GatedMPNNLayer(hidden_dim=hidden_dim, edge_dim=2)
            for _ in range(num_layers)
        ])

        self.out = nn.Linear(hidden_dim, 1)

    def _build_2d_grid_edges(self, nt: int, nx: int, dt: float, dx: float, device=None):
        """Build edges for a 2D grid with 8-connectivity."""
        src, dst, edge_attrs = [], [], []
        
        for t in range(nt):
            for n in range(nx):
                node_idx = t * nx + n
                neighbors = [
                    (t - 1, n, 0.0, -dt),      # temporal: previous
                    (t + 1, n, 0.0, dt),       # temporal: next
                    (t, n - 1, -dx, 0.0),      # spatial: left
                    (t, n + 1, dx, 0.0),       # spatial: right
                    (t - 1, n - 1, -dx, -dt),  # diagonal: upper-left
                    (t - 1, n + 1, dx, -dt),   # diagonal: upper-right
                    (t + 1, n - 1, -dx, dt),   # diagonal: lower-left
                    (t + 1, n + 1, dx, dt),    # diagonal: lower-right
                ]
                for nt_idx, nn_idx, edge_dx, edge_dt in neighbors:
                    if 0 <= nt_idx < nt and 0 <= nn_idx < nx:
                        neighbor_idx = nt_idx * nx + nn_idx
                        src.append(node_idx)
                        dst.append(neighbor_idx)
                        edge_attrs.append([edge_dx, edge_dt])
        
        edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32, device=device)
        return edge_index, edge_attr

    def _du_dx(self, u):
        du_center = (u[..., 2:, 0] - u[..., :-2, 0]) / (2 * self.dx)
        du_left = (u[..., 1:2, 0] - u[..., 0:1, 0]) / self.dx
        du_right = (u[..., -1:, 0] - u[..., -2:-1, 0]) / self.dx
        return torch.cat([du_left, du_center, du_right], dim=-1).unsqueeze(-1)

    def _du_dt(self, u):
        du_center = (u[:, 2:, :, 0] - u[:, :-2, :, 0]) / (2 * self.dt)
        du_first = (u[:, 1:2, :, 0] - u[:, 0:1, :, 0]) / self.dt
        du_last = (u[:, -1:, :, 0] - u[:, -2:-1, :, 0]) / self.dt
        return torch.cat([du_first, du_center, du_last], dim=1).unsqueeze(-1)
        
    def forward(self, x: torch.Tensor, u: torch.Tensor):
        """
        Args:
            x: (B, T, N, 2) - coordinates (unused except for shape consistency)
            u: (B, T, N, 1) - input field values
        
        Returns:
            delta_u: (B, T, N, 1) - correction
            gate_values: list of gate tensors from each layer
        """
        B, T, N, _ = u.shape
        num_nodes_per_sample = T * N

        du_dx = self._du_dx(u).reshape(B * num_nodes_per_sample, 1)
        du_dt = self._du_dt(u).reshape(B * num_nodes_per_sample, 1)
        u_flat = u.reshape(B * num_nodes_per_sample, 1)
        
        h = self.in_proj(u_flat)

        # Batch edges
        E = self.edge_index.size(1)
        offsets = torch.arange(B, device=h.device).repeat_interleave(E) * num_nodes_per_sample
        edge_index = self.edge_index.repeat(1, B) + offsets
        edge_attr = self.edge_attr.repeat(B, 1)

        all_gate_values = []
        for layer in self.layers:
            h, gate_values = layer(
                x=h,
                edge_index=edge_index,
                edge_attr=edge_attr,
                u=u_flat,
                du_dx=du_dx,
                du_dt=du_dt,
            )
            if gate_values is not None:
                all_gate_values.append(gate_values)

        delta_u = self.out(h)
        return delta_u.view(B, T, N, 1), all_gate_values


if __name__ == "__main__":
    B, T, N = 4, 10, 25
    hidden_dim = 64
    
    x = torch.randn(B, T, N, 2)
    u = torch.randn(B, T, N, 1)
    
    model = ShockGNN(nx=N, nt=T, dx=0.1, dt=0.01, hidden_dim=hidden_dim, num_layers=2, device=torch.device("cpu"))
    delta_u, gate_values = model(x, u)
    
    print(f"Input u shape: {u.shape}")
    print(f"Output delta_u shape: {delta_u.shape}")
    print(f"Number of gate value tensors: {len(gate_values)}")
    if gate_values:
        print(f"Gate values mean: {gate_values[0].mean().item():.4f}")
