# Wavefront Learning Architecture

This document describes the neural network architectures and loss functions used in the wavefront learning module for predicting shock trajectories and solutions in LWR traffic flow equations.

## Table of Contents

1. [Base Components](#base-components)
2. [Main Models](#main-models)
   - [ShockTrajectoryNet](#shocktrajectorynet)
   - [HybridDeepONet](#hybriddeepopnet)
   - [TrajDeepONet](#trajdeeponet)
   - [DeepONet](#deeponet-baseline)
   - [FNO](#fno-baseline)
   - [EncoderDecoder](#encoderdecoder)
   - [TrajTransformer](#trajtransformer)
   - [WaveNO](#waveno)
   - [LatentDiffusionDeepONet](#latentdiffusiondeeponet)
   - [NeuralFVSolver](#neuralfvsolver)
3. [Losses](#losses)
   - [Unified Loss Interface](#unified-loss-interface)
   - [Flux Functions](#flux-functions)
   - [Individual Losses](#individual-losses)
     - [MSELoss](#mseloss)
     - [ICLoss](#icloss)
     - [TrajectoryConsistencyLoss](#trajectoryconsistencyloss)
     - [BoundaryLoss](#boundaryloss)
     - [CollisionLoss](#collisionloss)
     - [ICAnchoringLoss](#icanchoringloss)
     - [SupervisedTrajectoryLoss](#supervisedtrajectoryloss)
     - [PDEResidualLoss](#pderesidualloss)
     - [PDEShockResidualLoss](#pdeshockresidualloss)
     - [RHResidualLoss](#rhresidualloss)
     - [AccelerationLoss](#accelerationloss)
     - [RegularizeTrajLoss](#regularizetrajloss)
   - [CombinedLoss](#combinedloss)
   - [Loss Presets](#loss-presets)

---

## Base Components

All base components are located in `models/base/`.

| File | Components | Description |
|------|------------|-------------|
| `base_model.py` | `BaseWavefrontModel` | Abstract base class for wavefront models |
| `feature_encoders.py` | `FourierFeatures`, `TimeEncoder`, `DiscontinuityEncoder`, `SpaceTimeEncoder` | Input encoding modules |
| `decoders.py` | `TrajectoryDecoder` | Decodes trajectories from branch/trunk embeddings |
| `blocks.py` | `ResidualBlock` | Residual MLP block with LayerNorm |
| `regions.py` | `RegionTrunk`, `RegionTrunkSet` | Density prediction for inter-shock regions |
| `assemblers.py` | `GridAssembler` | Assembles grid from region predictions with soft boundaries |
| `transformer_encoder.py` | `Tokenizer`, `EncoderLayer`, `Encoder` | Transformer encoder for token sequences |
| `axial_decoder.py` | `FourierTokenizer`, `AxialAttention`, `AxialDecoderLayer`, `AxialDecoder` | Factored time/space attention decoder |
| `cross_decoder.py` | `CrossDecoderLayer`, `CrossDecoder` | Cross-attention decoder with Nadaraya-Watson interpolation |
| `shock_gnn.py` | `GatedMPNNLayer`, `ShockGNN` | Gated message-passing GNN for shock correction (requires torch_geometric) |

---

## Main Models

### ShockTrajectoryNet

**Location**: `models/shock_trajectory_net.py`

A DeepONet-style architecture for predicting shock (discontinuity) trajectories in LWR traffic flow using unsupervised physics-based training.

#### Pseudocode

```
discontinuities: (D, 3)  →  DiscontinuityEncoder  →  branch_emb: (D, H)
query_times: (T,)         →  TimeEncoder           →  trunk_emb: (T, H)
(branch_emb, trunk_emb)   →  TrajectoryDecoder     →  positions: (D, T), existence: (D, T)
```

#### Sub-Components

##### FourierFeatures

**Location**: `models/base/feature_encoders.py`

Positional encoding for scalar inputs using sinusoidal features.

**Formula**:
$$\gamma(x) = \left[ \sin(2^0 \pi x), \cos(2^0 \pi x), \sin(2^1 \pi x), \cos(2^1 \pi x), \ldots, \sin(2^{L-1} \pi x), \cos(2^{L-1} \pi x) \right]$$

where $L$ = `num_frequencies` (default: 32).

**Output dimension**: $2L$ (or $2L + 1$ if including original input)

##### DiscontinuityEncoder (Branch Network)

**Location**: `models/base/feature_encoders.py`

Encodes discontinuities using Fourier features + MLP. Each discontinuity is processed independently (no cross-attention).

```
# Per discontinuity: [x_position, rho_L, rho_R]
x_position: (D,)  →  FourierFeatures  →  (D, 2*num_freq + 1)
concat with [rho_L, rho_R]            →  (D, 2*num_freq + 3)
→  MLP([Linear + GELU + LayerNorm] × num_layers)
→  mask × output                       →  (D, output_dim)
```

**Default**: hidden_dim=128, output_dim=128, num_frequencies=16, num_layers=3

##### TimeEncoder (Trunk Network)

**Location**: `models/base/feature_encoders.py`

Encodes query times for trajectory prediction.

```
t: (T,)  →  FourierFeatures  →  (T, 2L+1)
→  MLP([Linear + GELU + LayerNorm] × num_layers)  →  (T, output_dim)
```

**Default**: hidden_dim=128, output_dim=128, num_frequencies=32, num_layers=3

##### TrajectoryDecoder

**Location**: `models/base/decoders.py`

Predicts shock positions and existence probabilities from branch and trunk embeddings.

```
branch_emb: (D, branch_dim), trunk_emb: (T, trunk_dim)
→  BilinearFusion(branch ⊗ trunk) + Skip(branch) + Skip(trunk)  →  (D, T, H)
→  LayerNorm → ResidualBlock × num_res_blocks
→  PositionHead(Linear → GELU → Linear)     →  positions: (D, T)
→  ExistenceHead(Linear → GELU → Linear → Sigmoid)  →  existence: (D, T) ∈ [0, 1]
```

**Bilinear fusion formula**:
$$h_{d,t} = W_{bilinear}(b_d \otimes e_t) + W_{branch}(b_d) + W_{trunk}(e_t)$$

where $b_d$ is the branch embedding for discontinuity $d$ and $e_t$ is the trunk embedding for time $t$.

#### Input/Output Format

**Input** (dictionary):
- `discontinuities`: $(D, 3)$ - initial shock features $[x_0, \rho_L, \rho_R]$
- `disc_mask`: $(D,)$ - validity mask for discontinuities
- `t_coords`: $(1, n_t, n_x)$ - query times

**Output** (dictionary):
- `positions`: $(D, T)$ - predicted x-coordinates of each shock at each time
- `existence`: $(D, T)$ - probability $\in [0, 1]$ that each shock exists

---

### HybridDeepONet

**Location**: `models/hybrid_deeponet.py`

Combined model that predicts both shock trajectories AND full solution grids by assembling per-region density predictions.

#### Pseudocode

```
discontinuities: (D, 3)  →  DiscontinuityEncoder  →  branch_emb: (D, H), pooled: (H,)
query_times: (T,)         →  TimeEncoder           →  trunk_emb: (T, H)
(branch_emb, trunk_emb)   →  TrajectoryDecoder     →  positions: (D, T), existence: (D, T)
(t_coords, x_coords)      →  SpaceTimeEncoder      →  coord_emb: (nt, nx, H)
(pooled, coord_emb)        →  RegionTrunkSet(K)     →  region_densities: (K, nt, nx)
(positions, existence, region_densities)  →  GridAssembler  →  output_grid: (1, nt, nx)
```

#### Sub-Components

##### SpaceTimeEncoder

**Location**: `models/base/feature_encoders.py`

Encodes $(t, x)$ coordinate pairs for region density prediction.

```
t_coords: (nt, nx), x_coords: (nt, nx)
→  FourierFeatures_t(t), FourierFeatures_x(x)
→  concat: (nt, nx, 2L_t + 2L_x + 2)
→  MLP([Linear + GELU + LayerNorm] × num_layers)  →  (nt, nx, output_dim)
```

**Default**: hidden_dim=128, output_dim=128, num_frequencies_t=16, num_frequencies_x=16, num_layers=3

##### RegionTrunk

**Location**: `models/base/regions.py`

Predicts density values for a single region between shocks.

```
branch_emb: (H,), coord_emb: (nt, nx, coord_dim)
→  expand branch to (nt, nx, H)
→  BilinearFusion + SkipPaths → ResidualBlocks
→  DensityHead(Linear → GELU → Linear → Sigmoid)  →  (nt, nx) ∈ [0, 1]
```

##### RegionTrunkSet

**Location**: `models/base/regions.py`

Set of $K = \text{max\_discontinuities} + 1$ region trunks, one for each region.

**Output**: $(K, n_t, n_x)$ - stacked per-region density predictions

##### GridAssembler

**Location**: `models/base/assemblers.py`

Assembles the final solution grid from per-region predictions using soft sigmoid boundaries.

**Region Assignment**: For $D$ discontinuities → $K = D + 1$ regions.

**Soft Boundary Computation** for shock $d$ at position $x_d(t)$:
$$\text{left\_of\_shock}_d(t, x) = \sigma\left(\frac{x_d(t) - x}{\sigma_{soft}}\right) \cdot e_d(t) \cdot m_d$$

where $\sigma_{soft} = 0.02$ is the softness parameter.

**Region Weights**:
$$w_0(t, x) = \text{left\_of\_shock}_0(t, x)$$
$$w_k(t, x) = (1 - \text{left\_of\_shock}_{k-1}(t, x)) \cdot \text{left\_of\_shock}_k(t, x) \quad \text{for } 1 \leq k < K-1$$
$$w_{K-1}(t, x) = 1 - \text{left\_of\_shock}_{D-1}(t, x)$$

**Final Grid Assembly**:
$$\rho(t, x) = \sum_{k=0}^{K-1} w_k(t, x) \cdot \rho_k(t, x)$$

#### Input/Output Format

**Input** (dictionary):
- `discontinuities`: $(D, 3)$ - initial shock features
- `disc_mask`: $(D,)$ - validity mask
- `t_coords`: $(1, n_t, n_x)$ - query times
- `x_coords`: $(1, n_t, n_x)$ - query positions

**Output** (dictionary):
- `positions`: $(D, T)$ - shock positions
- `existence`: $(D, T)$ - shock existence probabilities
- `output_grid`: $(1, n_t, n_x)$ - assembled solution
- `region_densities`: $(K, n_t, n_x)$ - per-region predictions
- `region_weights`: $(K, n_t, n_x)$ - soft region assignments

---

### TrajDeepONet

**Location**: `models/traj_deeponet.py`

Trajectory-conditioned DeepONet that predicts shock trajectories and uses them to condition a single density trunk. Key simplifications over HybridDeepONet:
- **No existence head**: all input discontinuities persist through time
- **Single trunk**: one network conditioned on boundary positions instead of K separate region trunks
- **No GridAssembler**: the trunk directly outputs density

#### Pseudocode

```
discontinuities: (D, 3)  →  DiscontinuityEncoder  →  per_disc: (D, H), pooled: (H,)
query_times: (T,)         →  TimeEncoder           →  trunk_emb: (T, H)
(per_disc, trunk_emb)     →  PositionDecoder       →  positions: (D, T)
(positions, x_coords, disc_mask)  →  compute_boundaries  →  x_left: (nt, nx), x_right: (nt, nx)
(pooled, t, x, x_left, x_right)  →  BoundaryConditionedTrunk  →  output_grid: (1, nt, nx)
```

#### Sub-Components

##### PositionDecoder

**Location**: `models/traj_deeponet.py`

Simplified trajectory decoder that only predicts positions (no existence head).

```
branch_emb: (D, branch_dim), trunk_emb: (T, trunk_dim)
→  BilinearFusion + SkipPaths → LayerNorm → ResidualBlocks
→  PositionHead(Linear → GELU → Linear → clamp[0,1])  →  positions: (D, T)
```

##### compute_boundaries

**Location**: `models/traj_deeponet.py`

Computes left and right boundary discontinuity positions for each grid point.

$$x_{left}(t, x) = \max_{d : x_d(t) \leq x,\ m_d = 1} x_d(t) \quad \text{(or 0 if none)}$$

$$x_{right}(t, x) = \min_{d : x_d(t) > x,\ m_d = 1} x_d(t) \quad \text{(or 1 if none)}$$

- Input: `positions` $(D, n_t)$, `x_coords` $(n_t, n_x)$, `disc_mask` $(D,)$
- Output: `left_bound` $(n_t, n_x)$, `right_bound` $(n_t, n_x)$

##### BoundaryConditionedTrunk

**Location**: `models/traj_deeponet.py`

Single trunk that predicts density conditioned on boundary positions.

```
t: (nt, nx), x: (nt, nx), x_left: (nt, nx), x_right: (nt, nx)
→  FourierEncode: γ_t(t), γ_x(x), γ_x(x_left), γ_x(x_right)
→  concat → CoordMLP([Linear + GELU + LayerNorm] × L)  →  coord_emb: (nt, nx, H)
branch_emb: (H,) + coord_emb  →  BilinearFusion + SkipPaths
→  LayerNorm → ResidualBlocks
→  DensityHead(Linear → GELU → Linear → Sigmoid)  →  (nt, nx) ∈ [0, 1]
```

#### Input/Output Format

**Input** (dictionary):
- `discontinuities`: $(D, 3)$ - initial shock features $[x_0, \rho_L, \rho_R]$
- `disc_mask`: $(D,)$ - validity mask
- `t_coords`: $(1, n_t, n_x)$ - query times
- `x_coords`: $(1, n_t, n_x)$ - query positions

**Output** (dictionary):
- `positions`: $(D, T)$ - shock positions (all discontinuities persist)
- `output_grid`: $(1, n_t, n_x)$ - predicted density grid

---

### DeepONet (Baseline)

**Location**: `models/deeponet.py`

Classic DeepONet architecture used as a baseline. Branch network encodes the initial condition, trunk network encodes query coordinates, and their dot product produces the output.

#### Pseudocode

```
grid_input: (3, nt, nx)  →  extract IC at t=0: (nx,), coords: (nt*nx, 2)
IC: (nx,)           →  BranchMLP([Linear + GELU] × L → Linear)  →  (p,)
coords: (nt*nx, 2)  →  TrunkMLP([Linear + GELU] × L → Linear)  →  (nt*nx, p)
→  dot product + bias: Σ_k branch_k · trunk_k + β  →  reshape to (1, nt, nx)
```

#### Output formula

$$\rho(t, x) = \sum_{k=1}^{p} b_k \cdot \tau_k(t, x) + \beta$$

where $b_k$ is the $k$-th branch output, $\tau_k$ is the $k$-th trunk output, and $\beta$ is a learned bias.

#### Input/Output Format

**Input**: tensor $(3, n_t, n_x)$ from `ToGridInputTransform` — channels: [ic_masked, t_coords, x_coords]

**Output** (dictionary):
- `output_grid`: $(1, n_t, n_x)$ - predicted solution

#### Default Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 128 | Hidden layer width |
| `latent_dim` | 64 | Dot-product dimension |
| `num_branch_layers` | 4 | Branch MLP depth |
| `num_trunk_layers` | 4 | Trunk MLP depth |

---

### FNO (Baseline)

**Location**: `models/fno_wrapper.py`

Fourier Neural Operator baseline wrapped for the wavefront learning dict interface. Wraps `neuralop.models.FNO` and adds dict input/output.

#### Pseudocode

```
grid_input: (3, nt, nx)
→  Lifting(3 → hidden_channels)
→  SpectralConv2d(n_modes_t, n_modes_x) × n_layers
→  Projection(hidden_channels → 1)
→  output_grid: (1, nt, nx)
```

#### Input/Output Format

**Input**: tensor $(3, n_t, n_x)$ from `ToGridInputTransform` — channels: [ic_masked, t_coords, x_coords]

**Output** (dictionary):
- `output_grid`: $(1, n_t, n_x)$ - predicted solution

#### Default Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_modes_t` | 32 | Fourier modes in time dimension |
| `n_modes_x` | 16 | Fourier modes in space dimension |
| `hidden_channels` | 32 | FNO hidden channel width |
| `n_layers` | 2 | Number of spectral convolution layers |

---

### AutoregressiveFNO / AutoregressiveRealFNO

**Location**: `models/autoregressive_fno.py`

Two variants of a 1D spatial FNO that steps autoregressively in time. Instead of applying spectral convolutions over the full 2D space-time grid, these models use a 1D FNO in space only and march forward one timestep at a time. A residual connection predicts the state delta at each step. The model is conditioned on `dt` via an input channel, enabling it to handle different temporal resolutions during high-res testing.

- **AutoregressiveFNO**: wraps `neuralop.models.FNO` (complex-valued spectral weights). Does **not** work with `clip_grad_norm_` on MPS.
- **AutoregressiveRealFNO**: self-contained 1D FNO storing real/imag weight parts as separate real `nn.Parameter` tensors. Works on all backends including MPS.

#### Pseudocode

```
grid_input: (1, nt, nx)  →  extract IC  →  state: (1, nx)
dt: scalar               →  broadcast   →  dt_channel: (1, nx)

for t in 1..nt-1:
    fno_input = cat([state, dt_channel], dim=0)    # (2, nx)
    state = state + FNO_1d(fno_input)              # (1, nx)  [residual]
    outputs.append(state)

stack(outputs, dim=1)  →  output_grid: (1, nt, nx)
```

**RealFNO1d internals** (AutoregressiveRealFNO only):

```
x: (C_in, nx)
→  Lifting: Conv1d(C_in → H, k=1)
→  [SpectralConv1d(H, H, n_modes) + Conv1d(H, H, k=1) skip + GELU] × L
→  Projection: Conv1d(H → C_out, k=1)
→  (C_out, nx)

SpectralConv1d:
    rfft(x)  →  truncate to n_modes  →  einsum(x_ft, weight_real + i*weight_imag)  →  irfft
```

#### Input/Output Format

**Input**: dict with:
- `grid_input`: $(1, n_t, n_x)$ from `ToGridNoCoords` — masked IC (only first row used)
- `dt`: scalar — temporal grid spacing

**Output** (dictionary):
- `output_grid`: $(1, n_t, n_x)$ — predicted solution

#### Default Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_modes_x` | 16 | Fourier modes in space dimension |
| `hidden_channels` | 32 | FNO hidden channel width |
| `n_layers` | 2 | Number of spectral convolution layers per step |
| `domain_padding` | 0.2 | Padding fraction for non-periodic boundary conditions |

---

### AutoregressiveWaveNO

**Location**: `models/autoregressive_waveno.py`

Combines WaveNO's segment encoding and trajectory prediction with autoregressive time-stepping. Both the spatial state and shock positions evolve step-by-step. The FNO receives boundary channels indicating where discontinuities are, and a trajectory MLP predicts position deltas from static breakpoint embeddings.

Key design choices:
- **Boundary channels in FNO** (4 input channels): `[state, dt, left_bound, right_bound]` — boundary positions tell the FNO where discontinuities are.
- **Static breakpoint embeddings**: Computed once from IC segments via concat of adjacent segment embeddings. Time-evolution is handled by the trajectory MLP learning position deltas.
- **Gradient flow**: Boundaries use hard comparisons (`pos <= x`), so grid MSE gradients don't flow to trajectories. Trajectory learning is driven by `traj_regularized` losses (ic_anchoring, boundary, regularize_traj).

#### Pseudocode

```
xs, ks, pieces_mask → SegmentPhysicsEncoder → seg_emb: (K, H)
seg_emb → SelfAttention(EncoderLayer × L) → seg_emb: (K, H)

cat[seg_emb[d], seg_emb[d+1]] → bp_encoder(MLP) → bp_emb: (D, H)

state = grid_input[:, 0, :]         # (1, nx) — IC
positions = discontinuities[:, 0]   # (D,) — initial positions
x_grid = x_coords[0, 0, :]         # (nx,)

for t in 0..nt-2:
    pos → compute_boundaries → left, right: (1, nx)
    fno_in = cat[state, dt, left, right]           # (4, nx)
    state = state + RealFNO1d(fno_in)              # (1, nx)

    traj_in = cat[bp_emb, fourier(pos), dt]        # (D, H+F+1)
    positions = clamp(positions + traj_mlp(traj_in), 0, 1) * disc_mask

stack(states) → output_grid: (1, nt, nx)
stack(positions) → positions: (D, nt)
```

#### Input/Output Format

**Input** (via `ToGridNoCoords` transform): dict with:
- `grid_input`: $(1, n_t, n_x)$ — masked IC (only first row used)
- `xs`, `ks`, `pieces\_mask` — IC segments for SegmentPhysicsEncoder
- `discontinuities`: $(D, 3)$ — `[x, \rho_L, \rho_R]` initial positions
- `disc\_mask`: $(D,)$ — validity mask
- `x\_coords`: $(1, n_t, n_x)$ — spatial grid
- `dt`: scalar — time step

**Output** (dictionary):
- `output_grid`: $(1, n_t, n_x)$ — predicted density
- `positions`: $(D, n_t)$ — breakpoint trajectories

#### Default Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 64 | Segment/breakpoint embedding dimension |
| `num_seg_frequencies` | 8 | Fourier bands for segment encoder |
| `num_seg_mlp_layers` | 2 | MLP depth in segment encoder |
| `num_self_attn_layers` | 2 | Self-attention layers over segments |
| `num_heads` | 4 | Attention heads |
| `num_freq_pos` | 8 | Fourier bands for position encoding in trajectory MLP |
| `fno_hidden` | 32 | FNO hidden channels |
| `fno_modes` | 16 | FNO Fourier modes |
| `fno_layers` | 2 | FNO spectral conv layers |
| `fno_padding` | 0.2 | Domain padding for non-periodic BCs |
| `dropout` | 0.05 | Dropout rate |

---

### EncoderDecoder

**Location**: `models/encoder_decoder.py`

Transformer-based encoder-decoder model. The encoder processes masked IC conditions (non-masked grid points) as a variable-length token sequence. The decoder reconstructs the full space-time solution from query coordinates, using cross-attention to the encoder output. Two decoder variants are available, plus an optional GNN shock correction.

#### Pseudocode

```
grid_input: (3, nt, nx)  →  extract non-masked tokens (ic != -1)  →  conds: (S, 3)
conds: (S, 3)  →  Tokenizer(3 → H) → EncoderLayer(self-attn + FFN) × L_e  →  encoder_out: (S, H)

# Axial decoder variant:
coords: (nt, nx, 2)  →  FourierTokenizer → [AxialAttn(time) + AxialAttn(space) + CrossAttn(encoder)] × L_d
→  Linear → u: (nt, nx, 1)

# Cross decoder variant:
coords: (nt, nx, 2)  →  NadarayaWatson(learnable_grid) → [CrossAttn(encoder)] × L_d
→  MLP → u: (nt, nx, 1)

# Optional ShockGNN correction (if layers_gnn > 0):
u  →  ShockGNN(gated MPNN on 8-connected grid)  →  delta_u: (nt, nx, 1)
output_grid = (u + delta_u): (1, nt, nx)
```

#### Sub-Components

##### Encoder (`base/transformer_encoder.py`)

Standard transformer encoder with tokenizer and stacked self-attention layers.

- **Tokenizer**: Linear(input_dim → hidden_dim)
- **EncoderLayer**: MultiheadAttention + residual + LayerNorm, then FFN(H → 4H → H) + residual + LayerNorm

##### AxialDecoder (`base/axial_decoder.py`)

Memory-efficient decoder using factored attention over time and space dimensions separately.

**FourierTokenizer**: Projects coordinate inputs using sinusoidal Fourier features.

$$\gamma(x) = \left[ x, \sin(2^0 \pi x), \cos(2^0 \pi x), \ldots, \sin(2^{L-1} \pi x), \cos(2^{L-1} \pi x) \right]$$

Then projects via Linear to hidden_dim. Default: $L = 4$ frequencies.

**AxialAttention**: Factored self-attention that attends over T and N dimensions separately. Each axis has its own `MultiheadAttention` and `LayerNorm`.

**AxialDecoderLayer**: Axial self-attention (time + space) → flatten to $(T \cdot N, D)$ → cross-attention to encoder → FFN with residual + LayerNorm.

##### CrossDecoder (`base/cross_decoder.py`)

Decoder using Nadaraya-Watson interpolation over a trainable latent grid, followed by cross-attention.

**Nadaraya-Watson Interpolation**:

For query point $y \in \mathbb{R}^2$, interpolates from learnable grid features $\{x_{ij}\}$ at uniform grid points $\{y_{ij}\} \subset [0, 1]^2$:

$$x'(y) = \sum_{i,j} w_{ij}(y) \cdot x_{ij}, \quad w_{ij}(y) = \frac{\exp(-\beta \|y - y_{ij}\|^2)}{\sum_{i',j'} \exp(-\beta \|y - y_{i'j'}\|^2)}$$

where $\beta$ is the locality hyperparameter (default: 10.0). Larger $\beta$ → more localized weights.

**CrossDecoderLayer** (pre-norm architecture):
$$x'_k = x_{k-1} + \text{MHA}(\text{LN}(x_{k-1}), \text{LN}(z_L), \text{LN}(z_L))$$
$$x_k = x'_k + \text{MLP}(\text{LN}(x'_k))$$

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grid_nx` | 16 | Latent grid points in first dimension |
| `grid_ny` | 16 | Latent grid points in second dimension |
| `beta` | 10.0 | Nadaraya-Watson locality parameter |

##### ShockGNN (`base/shock_gnn.py`)

Optional graph neural network correction for shock regions. Uses gated message passing on a 2D grid with 8-connectivity (4 cardinal + 4 diagonal neighbors).

**GatedMPNNLayer**: Message-passing layer with physics-informed gating:
- Gate inputs: $u_i$, $u_j$, $|u_i - u_j|$, $\partial u/\partial x_i$, $\partial u/\partial x_j$, $\partial u/\partial t_i$, $\partial u/\partial t_j$
- Gate output: $g \in [0, 1]$ via sigmoid (initialized near 0 with bias $= -3.0$)
- Message: $m_{ij} = g \cdot \text{MLP}([h_i, h_j, e_{ij}])$
- Update: $h'_i = h_i + \text{MLP}([h_i, \sum_j m_{ij}])$

Requires `torch_geometric` (optional dependency).

#### Input/Output Format

**Input**: tensor $(3, n_t, n_x)$ from `ToGridInputTransform` — channels: [ic_masked, t_coords, x_coords]

**Output** (dictionary):
- `output_grid`: $(1, n_t, n_x)$ - predicted solution

#### Default Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 64 | Hidden dimension for all components |
| `layers_encoder` | 2 | Encoder transformer layers |
| `decoder_type` | `"axial"` or `"cross"` | Decoder variant |
| `layers_decoder` | 2 | Decoder layers |
| `layers_gnn` | 0 | GNN correction layers (0 = disabled) |

#### Factory Functions

- `build_encoder_decoder(args)`: Builds with axial decoder
- `build_encoder_decoder_cross(args)`: Builds with cross-attention decoder

---

### TrajTransformer

**Location**: `models/traj_transformer.py`

Cross-attention variant of TrajDeepONet. Replaces bilinear fusion with cross-attention throughout: discontinuity embeddings serve as keys/values, while time or spacetime embeddings serve as queries. This avoids the need for branch pooling and enables richer discontinuity-aware feature aggregation.

#### Pseudocode

```
discontinuities: (D, 3)  →  DiscontinuityEncoder  →  (D, H)
→  SelfAttention(EncoderLayer × L)                 →  disc_emb: (D, H)
→  [optional] ClassifierHead(MLP → Sigmoid)         →  existence: (D,)

query_times: (T,)  →  TimeEncoder  →  time_emb: (T, H)
(time_emb as Q, disc_emb as K/V)  →  CrossDecoderLayer × L  →  time_enriched: (T, H)
disc_emb + time_enriched  →  PositionHead  →  positions: (D, T)

(positions, x_coords, disc_mask)  →  compute_boundaries  →  x_left: (nt, nx), x_right: (nt, nx)
(t, x, x_left, x_right)  →  FourierEncode → CoordMLP  →  coord_emb: (nt*nx, H)
(coord_emb as Q, disc_emb as K/V)  →  CrossDecoderLayer × L  →  DensityHead  →  (nt, nx)
→  output_grid: (1, nt, nx)
```

#### Sub-Components

##### TrajectoryDecoderTransformer

Decodes trajectory positions using cross-attention instead of bilinear fusion.

- Time embeddings (queries) attend to discontinuity embeddings (keys/values) via `CrossDecoderLayer × L`
- Enriched time features combined with each disc embedding: `disc_emb + time_enriched`
- Position head: `LayerNorm → Linear(H → H/2) → ReLU → Linear(H/2 → 1) → clamp[0,1]`

##### DensityDecoderTransformer

Predicts density using Fourier-encoded coordinates as queries attending to discontinuity embeddings.

- Fourier encode: $\gamma_t(t)$, $\gamma_x(x)$ [, $\gamma_x(x_{left})$, $\gamma_x(x_{right})$]
- Coord MLP: concatenated features → hidden_dim
- Cross-attention: coord queries attend to disc K/V via `CrossDecoderLayer × L`
- Density head: `Linear(H → H/2) → ReLU → Linear(H/2 → 1) → clamp[0,1]`

##### Classifier Head (optional)

Binary classification per discontinuity: shock (1) vs rarefaction (0).

$$e_d = \sigma(\text{MLP}(h_d)) \cdot m_d$$

Output is constant across time: $(D,) \to (D, T)$ via expand.

When enabled, the classifier filters `compute_boundaries` to only use discontinuities with $e_d > 0.5$.

#### Input/Output Format

**Input** (dictionary):
- `discontinuities`: $(D, 3)$ - initial shock features $[x_0, \rho_L, \rho_R]$
- `disc_mask`: $(D,)$ - validity mask
- `t_coords`: $(1, n_t, n_x)$ - query times
- `x_coords`: $(1, n_t, n_x)$ - query positions

**Output** (dictionary):
- `positions`: $(D, T)$ - shock positions
- `output_grid`: $(1, n_t, n_x)$ - predicted density grid
- `existence`: $(D, T)$ - shock/rarefaction probability (only when `classifier=True`, constant across $T$)

#### Default Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 32 | Hidden dimension for all networks |
| `num_frequencies_t` | 8 | Fourier frequencies for time |
| `num_frequencies_x` | 8 | Fourier frequencies for space |
| `num_disc_frequencies` | 8 | Fourier frequencies for discontinuity positions |
| `num_disc_layers` | 2 | MLP layers in discontinuity encoder |
| `num_time_layers` | 2 | MLP layers in time encoder |
| `num_coord_layers` | 2 | MLP layers in coordinate encoder |
| `num_interaction_layers` | 2 | Self-attention layers for cross-disc interaction |
| `num_traj_cross_layers` | 2 | Cross-attention layers in trajectory decoder |
| `num_density_cross_layers` | 2 | Cross-attention layers in density decoder |
| `num_attention_heads` | 4 | Attention heads |
| `dropout` | 0.0 | Dropout rate |

#### Factory Functions

- `build_traj_transformer(args)`: Standard version (`with_traj=True`, `classifier=False`)
- `build_classifier_traj_transformer(args)`: With classifier (`classifier=True`)
- `build_no_traj_transformer(args)`: Without trajectory prediction (`with_traj=False`)
- `build_classifier_all_traj_transformer(args)`: Classifier + all-boundaries density decoding (`classifier=True`, `all_boundaries=True`)
- `build_biased_classifier_traj_transformer(args)`: Classifier + characteristic attention bias (`classifier=True`, `characteristic_bias=True`)

#### All-Boundaries Mode (`all_boundaries=True`) — DynamicDensityDecoder

When `all_boundaries=True`, the density decoder uses `DynamicDensityDecoder` with **time-varying boundary tokens** instead of static disc embeddings or Fourier-encoded left/right positions:

1. **Dynamic boundary tokens**: For each time step $t$, boundary token $d$ combines wave properties with predicted position: $\mathbf{b}_d(t) = \mathbf{e}_d + \text{proj}(\text{Fourier}(x_d(t)))$, where $\mathbf{e}_d$ is the discontinuity embedding and $x_d(t)$ is the predicted trajectory position
2. **Soft existence weighting**: Tokens are weighted by $\hat{p}_d \cdot m_d$ (classifier probability times validity mask) — fully differentiable, no hard threshold
3. **Per-time-step cross-attention**: Spatial queries $\text{MLP}(\text{Fourier}(t) \| \text{Fourier}(x))$ at each time step cross-attend to the $D$ dynamic boundary tokens at that time step. Batched as $(B \cdot T, n_x, H)$ queries and $(B \cdot T, D, H)$ keys/values for efficiency
4. **Density head**: Output projection maps enriched queries to density values

This design is fully differentiable (no `compute_boundaries` or hard thresholds), enabling end-to-end gradient flow from density loss through existence predictions to the classifier. The attention mechanism naturally learns spatial locality — queries attend primarily to nearby boundary tokens.

#### Characteristic Bias Mode (`characteristic_bias=True`) — BiasedClassifierTrajTransformer

When `characteristic_bias=True`, the density decoder uses `BiasedCrossDecoderLayer` (from `biased_cross_attention.py`) instead of `CrossDecoderLayer`, adding a physics-informed attention bias based on backward characteristic propagation from each discontinuity.

##### Discontinuity Characteristic Bias

For each query $(t, x)$ and discontinuity $d$ at position $x_d$ with states $(\rho_L, \rho_R)$:

1. **Characteristic speeds**: $\lambda_L = f'(\rho_L)$, $\lambda_R = f'(\rho_R)$
2. **Influence zone**: $[x_d + v_{min} \cdot t,\ x_d + v_{max} \cdot t]$ where $v_{min} = \min(\lambda_L, \lambda_R)$, $v_{max} = \max(\lambda_L, \lambda_R)$
3. **Distance outside zone**: $d_{outside} = \text{ReLU}(z_{left} - x) + \text{ReLU}(x - z_{right})$
4. **Bias**: $\text{bias}(t, x, d) = -|\alpha| \cdot d_{outside}^2$

where $\alpha$ is a learnable scale parameter (initialized at 5.0).

- **bias = 0** when query $(t, x)$ falls inside discontinuity $d$'s influence zone (full attention)
- **bias << 0** when query is far outside (suppressed attention)
- The model can **override** the bias via learned $Q \cdot K^T$ scores when physics alone is insufficient

##### Collision-Time Damping

The bias fades per-discontinuity after its influence zone collides with adjacent discontinuities' zones:

**Right collision time** (disc $d$ with disc $d+1$):
$$t_{coll,R} = \frac{x_{d+1} - x_d}{(v_{max,d} - v_{min,d+1})^+}$$

**Left collision time** (disc $d-1$ with disc $d$):
$$t_{coll,L} = \frac{x_d - x_{d-1}}{(v_{max,d-1} - v_{min,d})^+}$$

$$t_{coll,d} = \min(t_{coll,L}, t_{coll,R})$$

**Damping factor**:
$$\gamma(t, d) = \sigma\left(|\beta| \cdot (t_{coll,d} - t)\right)$$

**Final bias**: $\text{bias}(t, x, d) = \text{bias}_{raw}(t, x, d) \cdot \gamma(t, d)$

##### Why This Improves Resolution Generalization

At training resolution, standard cross-attention learns which discontinuity embeddings to attend to for each coordinate region. At higher resolutions, the query grid becomes denser with interpolated coordinate values. Without physics bias, the attention patterns must generalize purely from learned Fourier features. With characteristic bias, the backward characteristic foot correctly identifies relevant discontinuities at **any** resolution, providing a resolution-invariant inductive bias.

Uses only `flux.derivative()` — works for any flux function.

---

### CharNO

**Location**: `models/charno.py`

Characteristic Neural Operator that mirrors the **Lax-Hopf variational formula**. Instead of predicting trajectories then assembling a grid, CharNO directly answers: for each query $(t, x)$, which initial segment controls the solution (Lax-Hopf selection), and what value does it contribute?

#### Pseudocode

```
xs, ks, pieces_mask → SegmentPhysicsEncoder → seg_emb: (B, K, H)
seg_emb → SelfAttention(EncoderLayer × L) → contextualized seg_emb: (B, K, H)

(t_coords, x_coords, xs, ks) → CharacteristicFeatureComputer → char_feat: (B, Q, K, H_char)

cat(seg_emb_expanded, char_feat) → ScoreMLP → scores: (B, Q, K)
softmax(-scores / τ) → weights: (B, Q, K)

cat(seg_emb_expanded, char_feat) → ValueMLP → sigmoid → local_rho: (B, Q, K)

Σ_k weights_k · local_rho_k → output_grid: (B, 1, nt, nx)
```

#### Sub-Components

##### SegmentPhysicsEncoder

**Location**: `models/base/characteristic_features.py`

Encodes IC segments with physics-augmented features. For each constant piece $k$ with value $\rho_k$ on $[x_k, x_{k+1})$:

| Feature | Formula | Physical meaning |
|---------|---------|-----------------|
| $x_{center}$ | $(x_k + x_{k+1})/2$ | Segment center |
| $w$ | $x_{k+1} - x_k$ | Segment width |
| $\rho_k$ | $\text{ks}[k]$ | Density value |
| $\lambda_k$ | $f'(\rho_k)$ | Characteristic speed |
| $f_k$ | $f(\rho_k)$ | Flux value |

Content features (width, density, characteristic speed, flux, cumulative mass) are projected to $H$ via a linear layer, then **rotary positional embedding (RoPE)** injects $x_{center}$ as position by rotating consecutive dimension pairs. A post-RoPE MLP refines the position-aware representation → $(B, K, H)$.

##### CharacteristicFeatureComputer

**Location**: `models/base/characteristic_features.py`

Computes characteristic-relative coordinates for each (query, segment) pair:

| Feature | Formula | Physical meaning |
|---------|---------|-----------------|
| $\xi_k$ | $(x - x_{c,k}) / \max(t, \varepsilon)$ | Self-similarity variable |
| $\Delta_k$ | $x - x_{c,k} - f'(\rho_k) \cdot t$ | Characteristic offset |
| $d_{L,k}$ | $x - (x_k + f'(\rho_k) \cdot t)$ | Distance to left characteristic boundary |
| $d_{R,k}$ | $x - (x_{k+1} + f'(\rho_k) \cdot t)$ | Distance to right characteristic boundary |
| $t$ | $t$ | Time coordinate |

Each feature is Fourier-encoded and projected via MLP → $(B, Q, K, H_{char})$.

##### Flux Interface

**Location**: `models/base/flux.py`

Pluggable flux function for computing physics features:

```python
class Flux(nn.Module):
    forward(rho) → f(ρ)           # flux value
    derivative(rho) → f'(ρ)        # characteristic speed
    shock_speed(ρ_L, ρ_R) → s      # Rankine-Hugoniot speed
```

Implementations: `GreenshieldsFlux` ($f = \rho(1-\rho)$), `TriangularFlux` ($f = \min(v_f \rho, w(1-\rho))$).

##### Score Network (Lax-Hopf Selection)

MLP mapping $[\text{seg\_emb}; \text{char\_feat}]$ → scalar score per (query, segment) pair.

**Selection weights** (softmin = softmax of negated scores):
$$w_k(t,x) = \frac{\exp(-s_k / \tau)}{\sum_j \exp(-s_j / \tau)}$$

Learnable temperature $\tau$ stored as $\log \tau$. As $\tau \to 0$, weights approach the exact Lax-Hopf argmin.

##### Value Network (Local Solution)

MLP mapping $[\text{seg\_emb}; \text{char\_feat}]$ → density per (query, segment) pair, passed through sigmoid → $[0, 1]$.

##### Output Assembly

$$\rho(t, x) = \sum_k w_k(t,x) \cdot v_k(t,x)$$

#### Input/Output Format

**Input** (dictionary):
- `xs`: $(B, K+1)$ — breakpoint positions
- `ks`: $(B, K)$ — piece values
- `pieces_mask`: $(B, K)$ — validity mask
- `t_coords`: $(B, 1, n_t, n_x)$ — time coordinates
- `x_coords`: $(B, 1, n_t, n_x)$ — space coordinates

**Output** (dictionary):
- `output_grid`: $(B, 1, n_t, n_x)$ — predicted density grid
- `selection_weights`: $(B, n_t, n_x, K)$ — segment selection weights (interpretable)
- `local_rho`: $(B, n_t, n_x, K)$ — per-segment density predictions before weighting
- `temperature`: scalar — current softmin temperature $\tau = \exp(\log\tau)$

#### Default Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 64 | Segment embedding dimension |
| `char_hidden_dim` | 32 | Characteristic feature dimension |
| `num_frequencies` | 8 | Fourier bands for segment encoder |
| `num_char_frequencies` | 8 | Fourier bands for characteristic features |
| `num_seg_mlp_layers` | 2 | MLP depth in segment encoder |
| `num_self_attn_layers` | 2 | Self-attention layers |
| `num_char_mlp_layers` | 2 | MLP depth in char feature computer |
| `num_score_layers` | 2 | MLP depth in score network |
| `num_local_layers` | 2 | MLP depth in value network |
| `num_heads` | 4 | Self-attention heads |
| `initial_temperature` | 1.0 | Softmin temperature |

#### Factory Functions

- `build_charno(args)`: Creates CharNO with configuration from args dict

---

### WaveNO

**Location**: `models/waveno.py`

Wavefront Neural Operator that replaces CharNO's softmin selection with **physics-biased cross-attention**. Instead of scoring and selecting a winning segment, spatial queries discover relevant segment information via cross-attention with a characteristic-distance attention bias (analogous to ALiBi in NLP, but using wave propagation geometry).

**v2: Breakpoint Evolution** — Adds breakpoint trajectory prediction to provide each spatial query with local boundary context `(x_left, x_right)`, making the model invariant to the total number of IC segments (K-invariant). This enables generalization from 4-piece to 10+ piece initial conditions.

#### Pseudocode (with predict_trajectories=True)

```
xs, ks, pieces_mask → SegmentPhysicsEncoder → seg_emb: (K, H)
seg_emb → SelfAttention(EncoderLayer × L) → contextualized seg_emb: (K, H)

seg_emb, t_unique → TimeConditioner (FiLM) → seg_emb_t: (nt, K, H)
seg_emb_t → CrossSegmentAttention × L_cs → seg_emb_t: (nt, K, H)

# Breakpoint evolution (NEW):
seg_emb, disc_mask, t_unique → BreakpointEvolution → positions: (D, nt)
positions, x_coords, disc_mask → compute_boundaries → x_left, x_right: (nt, nx)

# Query encoding with local boundary context:
t_coords, x_coords, x_left, x_right → Fourier(t) || Fourier(x) || Fourier(x_L) || Fourier(x_R)
→ MLP → query_emb: (nt, nx, H)

(t, x, xs, ks, flux) → backward char foot → bias: (nt, nx, K)

# Per-time-step cross-attention (batched as B*nt):
query (B*nt, nx, H), keys/values (B*nt, K, H), bias (B*nt*heads, nx, K)
→ BiasedCrossAttn × N_cross → query: (B*nt, nx, H)

query → DensityHead(MLP) → clamp[0,1] → output_grid: (1, nt, nx)
```

#### Sub-Components

##### BiasedCrossDecoderLayer

**Location**: `models/base/biased_cross_attention.py`

Identical to `CrossDecoderLayer` but passes `attn_mask` to `nn.MultiheadAttention`, enabling additive physics-informed bias on attention logits.

$$\text{attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + \text{bias}\right) V$$

##### Characteristic Attention Bias

**Location**: `models/base/biased_cross_attention.py`

For each query $(t, x)$ and segment $k$, computes the backward characteristic foot and measures distance outside segment $k$'s interval:

$$y^* = x - f'(\rho_k) \cdot t$$
$$d_{outside} = \text{ReLU}(x_k - y^*) + \text{ReLU}(y^* - x_{k+1})$$
$$\text{bias}_{raw}(t, x, k) = -|\alpha| \cdot d_{outside}^2$$

where $\alpha$ is a learnable scale parameter (initialized at 10.0).

- **bias = 0** when $y^*$ falls inside segment $k$ (query is in $k$'s characteristic cone → full attention)
- **bias << 0** when $y^*$ is far outside (query far from $k$'s influence → suppressed attention)
- The model can **override** the bias via learned $Q \cdot K^T$ scores when physics alone is insufficient

Uses only `flux.derivative()` — works for any flux function.

##### Collision-Time Damping

The raw characteristic bias assumes independent wave propagation, which is correct before waves interact but misleading after collisions. With more IC segments, collisions happen earlier and the bias is wrong for a larger fraction of the domain. Collision-time damping fades the bias per-segment after its estimated collision time.

**Collision time estimation** for segment $k$:

$$t_{coll,k} = \min\left(\frac{w_{k-1}}{|\lambda_{k-1} - \lambda_k|}, \frac{w_k}{|\lambda_k - \lambda_{k+1}|}\right)$$

where $w_k = x_{k+1} - x_k$ is the segment width and $\lambda_k = f'(\rho_k)$ is the characteristic speed. Edge segments self-pad (use their own width/speed), giving large $t_{coll}$ so their bias stays strong.

**Damping factor**:

$$\gamma(t, k) = \sigma\left(|\beta| \cdot (t_{coll,k} - t)\right)$$

where $\beta$ is a learnable sharpness parameter (initialized at 5.0). This gives $\gamma \approx 1$ before collision (full bias) and $\gamma \approx 0$ after collision (learned attention takes over).

**Final bias**:

$$\text{bias}(t, x, k) = \text{bias}_{raw}(t, x, k) \cdot \gamma(t, k)$$

##### BreakpointEvolution

**Location**: `models/base/breakpoint_evolution.py`

Predicts how IC breakpoints evolve over time. Creates breakpoint embeddings from adjacent segment pairs, then uses `TrajectoryDecoderTransformer` (from `traj_transformer.py`) for cross-attention trajectory decoding.

```
seg_emb: (K, H) → pad to (K+1, H)
left_seg = seg_padded[:K], right_seg = seg_padded[1:K+1]  → concat → (D, 2H)
→ bp_encoder(Linear(2H→H) → GELU → Linear(H→H)) → bp_emb: (D, H)
bp_emb *= disc_mask

t_unique: (nt,) → TimeEncoder → time_emb: (nt, H)

(bp_emb as K/V, time_emb as Q) → TrajectoryDecoderTransformer
  → CrossDecoderLayer × L → time_enriched: (nt, H)
  → disc_emb + time_enriched → PositionHead → positions: (D, nt) ∈ [0, 1]
```

The predicted positions are then used by `compute_boundaries` (from `traj_deeponet.py`) to extract per-query local boundary positions `(x_left, x_right)`.

##### Reused from CharNO

- **SegmentPhysicsEncoder**: Physics-augmented segment encoding with RoPE (x_center via rotation; width, $\rho$, $\lambda$, $f$, $N_k$ as content)
- **TimeConditioner**: FiLM-based time modulation of segment embeddings
- **CrossSegmentAttention**: Lightweight self-attention over K segments per timestep
- **EncoderLayer**: Standard transformer self-attention for initial segment interaction

#### Input/Output Format

**Input** (dictionary):
- `xs`: $(B, K+1)$ — breakpoint positions
- `ks`: $(B, K)$ — piece values
- `pieces_mask`: $(B, K)$ — validity mask
- `t_coords`: $(B, 1, n_t, n_x)$ — time coordinates
- `x_coords`: $(B, 1, n_t, n_x)$ — space coordinates

**Output** (dictionary):
- `output_grid`: $(B, 1, n_t, n_x)$ — predicted density grid
- `characteristic_bias`: $(B, n_t, n_x, K)$ — physics-informed bias (diagnostic)
- `positions`: $(B, D, n_t)$ — predicted breakpoint positions (only when `predict_trajectories=True`)

#### Default Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 64 | All embedding dimensions |
| `num_freq_t` | 8 | Fourier bands for time in query encoder |
| `num_freq_x` | 8 | Fourier bands for space in query encoder |
| `num_seg_frequencies` | 8 | Fourier bands for segment encoder |
| `num_seg_mlp_layers` | 2 | MLP depth in segment encoder |
| `num_self_attn_layers` | 2 | Self-attention over segments |
| `num_cross_layers` | 2 | Biased cross-attention layers (queries → segments) |
| `num_heads` | 4 | Attention heads (both self and cross) |
| `num_cross_segment_layers` | 1 | Cross-segment attention per timestep |
| `time_condition` | True | FiLM time conditioning |
| `initial_bias_scale` | 5.0 | Initial characteristic bias scale |
| `initial_damping_sharpness` | 5.0 | Collision-time damping sharpness (learnable) |
| `predict_trajectories` | True | Enable breakpoint evolution + local boundary features |
| `num_traj_cross_layers` | 2 | Cross-attention layers in breakpoint trajectory decoder |
| `num_time_layers` | 2 | MLP layers in breakpoint time encoder |
| `num_freq_bound` | 8 | Fourier bands for boundary features in query encoder |

#### Factory Functions

- `build_waveno(args)`: Creates WaveNO with configuration from args dict

---

### WaveFrontModel

**Location**: `models/wavefront_model.py`

A learned Riemann solver that explicitly constructs solutions by predicting wave characteristics (shock vs rarefaction, speeds) for each discontinuity, handling wave interactions iteratively, and reconstructing the full density grid from the resulting wave pattern. No attention mechanisms — each discontinuity is processed independently by a shared encoder + 3 heads.

#### Pseudocode

```
discontinuities: (D, 3)  →  DiscontinuityEncoder(Fourier + MLP)  →  emb: (D, H)
emb  →  ClassifierHead(MLP → Sigmoid)    →  is_shock: (D,)
emb  →  ShockHead(MLP)                   →  shock_speed: (D,)
emb  →  RarefactionHead(MLP)             →  [speed1, delta]: (D, 2)

Per disc d:
  shock branch:  1 wave at shock_speed, jump = (rho_R - rho_L) * is_shock
  rarefaction branch: N sub-waves, speeds linspace(speed1, speed1 + softplus(delta)),
                      each jump = (rho_R - rho_L) * (1 - is_shock) / N

→ Wave buffer: (W_max,) containing [origin_x, origin_t, speed, jump, active, type]

Collision loop (max_interaction_rounds times):
  Sort active waves by position → find earliest pairwise collision
  At collision: create new disc (x_coll, rho_L_outer, rho_R_outer)
  Re-encode with same encoder+heads → spawn new waves in buffer
  Soft-deactivate colliding waves

Grid reconstruction (fully vectorized over B, W, nt, nx):
  density(t, x) = ks[0] + Σ_w jump_w · σ((x - pos_w(t)) / σ) · active_w
  where pos_w(t) = origin_x_w + speed_w · (t - origin_t_w)
→ output_grid: (1, nt, nx)
```

#### Sub-Components

##### DiscontinuityEncoder (shared)

Reuses `models/base/feature_encoders.py:DiscontinuityEncoder`. Fourier features on x coordinate + MLP. Each discontinuity processed independently. Same encoder used for both initial and spawned discontinuities (weight sharing).

##### Classifier Head

$$P(\text{shock})_d = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot h_d + b_1) + b_2) \cdot m_d$$

Binary per-discontinuity: shock (1) vs rarefaction (0).

##### Shock Head

$$s_d = W_2 \cdot \text{GELU}(W_1 \cdot h_d + b_1) + b_2$$

Predicts shock speed (dx/dt) per discontinuity. Unbounded output.

##### Rarefaction Head

$$[\text{speed}_1, \delta]_d = W_2 \cdot \text{GELU}(W_1 \cdot h_d + b_1) + b_2$$

Speed range: $[\text{speed}_1, \text{speed}_1 + \text{softplus}(\delta)]$. $N$ sub-waves evenly spaced within this range.

##### Wave Buffer

Pre-allocated tensor of size $W_{max} = D \cdot (N + 1) + R \cdot (N + 1)$ where $R$ = `max_interaction_rounds`.

Each wave stores: `(origin_x, origin_t, speed, jump, active, type)`.

##### Collision Processing

For each round:
1. Sort active waves by spatial position at $t_{ref} = T/2$
2. Compute pairwise collision time for adjacent converging waves:
   $$t_{coll} = \frac{x_j - x_i + s_i \cdot t_{o,i} - s_j \cdot t_{o,j}}{s_i - s_j}$$
3. Find earliest valid collision (both active, converging, within domain)
4. At collision: spawn new wave(s) by re-encoding $(x_{coll}, \rho_{L,outer}, \rho_{R,outer})$
5. Soft-deactivate colliding waves: $\text{active}_w \leftarrow \text{active}_w \cdot (1 - \mathbb{1}_{colliding})$

##### Grid Reconstruction (Jump-based)

$$\rho(t, x) = \rho_0 + \sum_{w} j_w \cdot \sigma\left(\frac{x - p_w(t)}{\sigma}\right) \cdot a_w$$

where:
- $\rho_0 = \text{ks}[0]$ is the leftmost piece value (base density)
- $j_w$ is the density jump at wave $w$
- $p_w(t) = x_{o,w} + s_w \cdot (t - t_{o,w})$ is the wave position at time $t$
- $a_w$ is the soft activity of wave $w$
- $\sigma$ is the sigmoid sharpness parameter

Fully vectorized: no loops over spatial or temporal dimensions.

#### Soft Blending for Differentiability

During training, both shock and rarefaction waves are active for each discontinuity, weighted by classifier probability:
- Shock wave jump: $(\\rho_R - \\rho_L) \cdot P(\text{shock})$
- Rarefaction sub-wave jumps: $(\\rho_R - \\rho_L) \cdot (1 - P(\text{shock})) / N$

This ensures gradients flow through both branches regardless of the classifier output.

#### Input/Output Format

**Input** (dictionary):
- `discontinuities`: $(D, 3)$ — $[x_0, \\rho_L, \\rho_R]$ per discontinuity
- `disc_mask`: $(D,)$ — validity mask
- `t_coords`: $(1, n_t, n_x)$ — time coordinates
- `x_coords`: $(1, n_t, n_x)$ — space coordinates
- `ks`: $(K,)$ — piece values (used for base density $\\rho_0 = ks[0]$)

**Output** (dictionary):
- `output_grid`: $(1, n_t, n_x)$ — predicted density grid
- `wave_origins_x`: $(W,)$ — wave origin x-positions (for plotting)
- `wave_origins_t`: $(W,)$ — wave origin times (for plotting)
- `wave_speeds`: $(W,)$ — wave speeds (for plotting)
- `wave_active`: $(W,)$ — wave activity masks (for plotting)
- `wave_types`: $(W,)$ — 0=shock, 1=rarefaction, 2=spawned (for plotting)

#### Default Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 64 | Encoder and head hidden dimension |
| `num_disc_frequencies` | 8 | Fourier features for position encoding |
| `num_disc_layers` | 2 | MLP depth in discontinuity encoder |
| `rarefaction_angles` | 5 | N sub-waves per rarefaction fan |
| `max_interaction_rounds` | 5 | Bounded collision processing iterations |
| `sigma` | 0.01 | Sigmoid sharpness for reconstruction |
| `dropout` | 0.05 | Dropout rate in encoder |

#### Factory Functions

- `build_wavefront_model(args)`: Creates WaveFrontModel with configuration from args dict

---

### LatentDiffusionDeepONet

**Location**: `models/latent_diffusion_deeponet.py`

Generative model for hyperbolic PDE solutions. Trains a VAE on coarse-grid solutions to learn a latent space, then trains a flow matching denoiser conditioned on the IC. The DeepONet decoder is resolution-invariant.

#### Pseudocode

```
Phase 1 (VAE training):
  target_grid (1, nt, nx) → VAEEncoder → (mean, logvar) → reparameterize → z (latent_dim,)
  z → DeepONetDecoder(z, t_coords, x_coords) → output_grid (1, nt, nx)
  Loss: MSE(output_grid, target) + beta * KL(mean, logvar)

Phase 2 (Flow matching training, encoder/decoder frozen):
  target_grid → VAEEncoder → z (mean only, no grad)
  (xs, ks, pieces_mask) → ConditionEncoder → c (condition_dim,)
  noise ~ N(0,I), t ~ U(0,1)
  z_t = (1-t)*noise + t*z
  (z_t, t, c) → FlowMatchingDenoiser → predicted_velocity (latent_dim,)
  Loss: MSE(predicted_velocity, z - noise)

Inference:
  (xs, ks, pieces_mask) → ConditionEncoder → c
  noise ~ N(0,I) → HeunODESolver(denoiser, noise, c, num_steps) → z
  z → DeepONetDecoder(z, t_coords, x_coords) → output_grid (1, nt, nx)
```

#### Sub-Components

**VAEEncoder** (`models/base/vae_encoder.py`):
- 2D conv encoder: Conv2d [1→32→64→128], stride 2, GELU+BatchNorm
- AdaptiveAvgPool2d(1) → flatten → two linear heads for mean and logvar
- `reparameterize(mean, logvar)`: z = mean + exp(0.5 * logvar) * eps

**DeepONetDecoder** (`models/base/deeponet_decoder.py`):
- Branch net: MLP (latent_dim → hidden → num_basis) with GELU+LayerNorm
- Trunk net: FourierFeatures for t and x, concatenate, MLP → (num_basis,)
- Output: sum_p(branch_p * trunk_p) + bias, reshaped to (1, nt, nx)
- Resolution-invariant: trunk evaluates at any (t, x) query points

**ConditionEncoder** (`models/base/flow_matching.py`):
- Input: concatenate (xs, ks, pieces_mask) → flat vector
- MLP → condition vector (condition_dim,)

**FlowMatchingDenoiser** (`models/base/flow_matching.py`):
- Sinusoidal time embedding → MLP → time_embed
- Input projection: concat(z_t, time_embed, condition) → hidden
- ResidualBlock layers → output projection → predicted velocity

**HeunODESolver** (`models/base/flow_matching.py`):
- Heun's method (2nd order) from t=0 to t=1
- At each step: Euler predictor + trapezoidal corrector

#### Input/Output Format

| Key | Phase 1 Output | Phase 2 Output | Inference Output |
|-----|---------------|----------------|-----------------|
| `output_grid` | (1, nt, nx) | (1, nt, nx) detached | (1, nt, nx) |
| `z_mean` | (latent_dim,) | — | — |
| `z_logvar` | (latent_dim,) | — | — |
| `predicted_velocity` | — | (latent_dim,) | — |
| `target_velocity` | — | (latent_dim,) | — |

#### Default Configuration

| Parameter | Default | CLI Flag |
|-----------|---------|----------|
| `latent_dim` | 32 | `--ld_latent_dim` |
| `num_basis` | 64 | `--ld_num_basis` |
| `condition_dim` | 64 | `--ld_condition_dim` |
| `num_ode_steps` | 100 | `--ld_num_ode_steps` |
| `beta` | 0.01 | `--ld_beta` |
| `beta_warmup_epochs` | 10 | `--ld_beta_warmup` |
| `phase1_epochs` | 2/3 of --epochs | `--ld_phase1_epochs` |
| `phase2_epochs` | 1/3 of --epochs | `--ld_phase2_epochs` |

#### Factory Functions

- `build_ld_deeponet(args)`: Creates LatentDiffusionDeepONet from config dict
- Two-phase training via `train_model_two_phase()` in `train.py`

---

### NeuralFVSolver

**Location**: `models/neural_fv_solver.py`

A learned finite volume time-marching scheme for scalar conservation laws. Instead of predicting the full $(T, X)$ solution at once, it learns a single-step update operator (a learned Riemann solver) and rolls it forward in time. Each step gathers local stencil features, passes them through a shared flux MLP, and performs an Euler update. This respects the causal, local, finite-speed-of-propagation structure of hyperbolic PDEs by design.

#### Components

**DifferentiableShockProximity**: Detects shocks at cell interfaces via the Lax entropy condition ($\lambda_L > s > \lambda_R$) and computes an exponential proximity field $p_i = \exp(-d_{\min,i} / \sigma)$ where $d_{\min,i}$ is the distance from cell $i$ to the nearest shock interface. Output is detached (no gradient flow).

**FluxNetwork**: Pointwise MLP implemented as `Conv1d(kernel_size=1)` layers. Shared across all cells (translation invariant). Final layer initialized to zero for small initial updates.

```
features: (3*(2k+1)+1, nx) → Conv1d(in, H, 1) → [GELU → Dropout → Conv1d(H, H, 1)] × (L-1) → GELU → Conv1d(H, 1, 1) → update: (1, nx)
```

**NeuralFVSolver**: Main model with autoregressive rollout.

```
grid_input: (1, nt, nx), dt: scalar, dx: scalar

state = grid_input[:, 0, :]                              # (1, nx) — IC

For t = 0..rollout_steps-1:
  padded = replicate_pad(state, k)                        # (1, nx+2k)
  stencil = unfold(padded, 2k+1)                          # (2k+1, nx)
  char_speeds = flux.derivative(stencil)                  # (2k+1, nx)
  prox = DifferentiableShockProximity(state, dx)           # (1, nx), detached
  stencil_prox = replicate_pad+unfold(prox)               # (2k+1, nx)
  features = cat[stencil, char_speeds, stencil_prox, dt]  # (3*(2k+1)+1, nx)
  update = FluxNetwork(features)                          # (1, nx)
  state = clamp(state + (dt/dx) * update, 0, 1)

  # Training only:
  state += N(0, noise_std)                                # pushforward noise
  state = teacher_force(state, GT[t+1])                   # stochastic replacement

# Pad remaining steps with detached last state (curriculum)
→ output_grid: (1, nt, nx)
```

#### Input Features (per cell)

| Feature | Count | Description |
|---------|-------|-------------|
| Stencil values | $2k+1$ | Cell values in local neighborhood |
| Characteristic speeds | $2k+1$ | $f'(\rho)$ at each stencil position |
| Shock proximity | $2k+1$ | Proximity field at stencil positions (detached) |
| Time step | $1$ | $\Delta t$ broadcast |
| **Total** | $3(2k+1)+1$ | Default $k=3$: 22 features |

#### Training Schedule

- **Curriculum**: Rollout steps ramp linearly from 1 to $n_t - 1$ over `curriculum_fraction` of training
- **Pushforward noise**: $\mathcal{N}(0, \sigma)$ added after each step, $\sigma$ decays linearly to 0 over `noise_decay_fraction` of training
- **Teacher forcing**: Optional stochastic replacement of predicted state with GT

#### Default Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stencil_k` | 3 | Stencil half-width ($2k+1 = 7$ cells) |
| `flux_hidden_dim` | 64 | FluxNetwork hidden channels |
| `flux_n_layers` | 3 | FluxNetwork depth |
| `proximity_sigma` | 0.05 | Shock proximity decay scale |
| `curriculum_fraction` | 0.5 | Epochs to reach full rollout |
| `initial_noise_std` | 0.01 | Initial pushforward noise |
| `noise_decay_fraction` | 0.75 | Epochs for noise decay |

Uses `ToGridNoCoords` transform, `mse_wasserstein` loss preset, `grid_residual` plot preset.

---

## Losses

### Unified Loss Interface

**Location**: `losses/base.py`

All losses inherit from `BaseLoss` and follow a unified interface:

```python
def forward(
    self,
    input_dict: dict[str, torch.Tensor],
    output_dict: dict[str, torch.Tensor],
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
```

**Arguments**:
- `input_dict`: Dictionary of input tensors (discontinuities, coords, masks, etc.)
- `output_dict`: Dictionary of model outputs (positions, existence, grids, etc.)
- `target`: Ground truth tensor (typically the target grid)

**Returns**:
- `loss`: Scalar loss tensor
- `components`: Dictionary of named loss component values as floats

---

### Flux Functions

**Location**: `losses/flux.py`

Centralized flux functions for LWR traffic flow with Greenshields flux.

#### greenshields_flux

$$f(\rho) = \rho (1 - \rho)$$

#### greenshields_flux_derivative

$$f'(\rho) = 1 - 2\rho$$

#### compute_shock_speed

From Rankine-Hugoniot condition:

$$s = \frac{f(\rho_R) - f(\rho_L)}{\rho_R - \rho_L} = 1 - \rho_L - \rho_R$$

---

### Individual Losses

#### MSELoss

**Location**: `losses/mse.py`

Mean squared error loss for grid predictions.

**Formula**:
$$\mathcal{L}_{MSE} = \frac{1}{n_t \cdot n_x} \sum_{t,x} \left( \rho_{pred}(t, x) - \rho_{target}(t, x) \right)^2$$

**Required outputs**: `output_grid`

**Components returned**: `{"mse": float}`

---

#### ICLoss

**Location**: `losses/ic.py`

Initial condition loss - penalizes deviation from target at $t=0$.

**Formula**:
$$\mathcal{L}_{IC} = \frac{1}{n_x} \sum_{x} \left( \rho_{pred}(0, x) - \rho_{target}(0, x) \right)^2$$

**Required outputs**: `output_grid`

**Components returned**: `{"ic": float}`

---

#### TrajectoryConsistencyLoss

**Location**: `losses/trajectory_consistency.py`

Enforces that predicted trajectories match the analytical Rankine-Hugoniot solution.

**Analytical trajectory**:
$$x_{analytical}(t) = x_0 + s \cdot t = x_0 + (1 - \rho_L - \rho_R) \cdot t$$

**Loss formula**:
$$\mathcal{L}_{traj} = \frac{1}{|\mathcal{V}|} \sum_{(b,d,t) \in \mathcal{V}} \left( x_{pred}^{(b,d)}(t) - x_{analytical}^{(b,d)}(t) \right)^2$$

where $\mathcal{V}$ is the set of valid (batch, discontinuity, time) indices weighted by `disc_mask`.

**Required inputs**: `discontinuities`, `t_coords`, `disc_mask`
**Required outputs**: `positions`

**Components returned**: `{"trajectory": float}`

---

#### BoundaryLoss

**Location**: `losses/boundary.py`

Penalizes shocks that exist outside the domain $[0, 1]$.

**Formula**:
$$\mathcal{L}_{bound} = \frac{1}{N} \sum_{b,d,t} \mathbb{1}_{outside}(x_{pred}^{(b,d,t)}) \cdot \left(e^{(b,d,t)}\right)^2$$

**Configuration**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `domain_min` | 0.0 | Minimum domain boundary |
| `domain_max` | 1.0 | Maximum domain boundary |

**Required inputs**: `disc_mask`
**Required outputs**: `positions`, `existence`

**Components returned**: `{"boundary": float}`

---

#### CollisionLoss

**Location**: `losses/collision.py`

Prevents simultaneous existence of overlapping shocks (encourages shock merging).

**Formula**:
$$\mathcal{L}_{coll} = \frac{1}{N_{pairs}} \sum_{i < j} \mathbb{1}_{colliding}(x_i, x_j) \cdot e_i \cdot e_j$$

where $\mathbb{1}_{colliding}(x_i, x_j) = 1$ if $|x_i - x_j| < \epsilon_{collision}$.

**Configuration**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `collision_threshold` | 0.02 | Distance threshold for collision detection |

**Required inputs**: `disc_mask`
**Required outputs**: `positions`, `existence`

**Components returned**: `{"collision": float}`

---

#### ICAnchoringLoss

**Location**: `losses/existence_regularization.py`

IC anchoring constraint that enforces predicted trajectories to start at the correct IC discontinuity positions. Optionally weighted by existence probability when the model predicts existence.

**Formula** (with existence):
$$\mathcal{L}_{anchor} = \frac{1}{N} \sum_{b,d} e^{(b,d)}_0 \cdot \left( x^{(b,d)}_{pred,0} - x^{(b,d)}_{IC} \right)^2$$

**Formula** (without existence):
$$\mathcal{L}_{anchor} = \frac{1}{N} \sum_{b,d} \left( x^{(b,d)}_{pred,0} - x^{(b,d)}_{IC} \right)^2$$

**Configuration**: No parameters.

**Required inputs**: `discontinuities`, `disc_mask`
**Required outputs**: `positions`
**Optional outputs**: `existence` (weights loss by existence probability at t=0)

**Components returned**: `{"ic_anchoring": float}`

---

#### SupervisedTrajectoryLoss

**Location**: `losses/supervised_trajectory.py`

Supervised loss for trajectory prediction when ground truth is available.

**Formula**:
$$\mathcal{L}_{sup} = w_{pos} \cdot \mathcal{L}_{pos} + w_{exist} \cdot \mathcal{L}_{exist}$$

where:
$$\mathcal{L}_{pos} = \frac{1}{|\mathcal{V}|} \sum_{(b,d,t) \in \mathcal{V}} \left( x_{pred} - x_{target} \right)^2$$
$$\mathcal{L}_{exist} = \frac{1}{|\mathcal{V}|} \sum_{(b,d,t) \in \mathcal{V}} BCE(e_{pred}, e_{target})$$

**Configuration**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `position_weight` | 1.0 | Weight for position MSE |
| `existence_weight` | 1.0 | Weight for existence BCE |

**Required inputs**: `disc_mask`, `target_positions`, `target_existence`
**Required outputs**: `positions`, `existence`

**Components returned**: `{"position": float, "existence": float}`

---

#### PDEResidualLoss

**Location**: `losses/pde_residual.py`

Physics-informed loss enforcing the conservation law on the predicted grid over all interior points (no shock masking).

**Physical equation**:
$$\frac{\partial \rho}{\partial t} + \frac{\partial f(\rho)}{\partial x} = 0$$

**Residual computation** (central finite differences):
$$R(t, x) = \frac{\rho(t+\Delta t, x) - \rho(t-\Delta t, x)}{2\Delta t} + \frac{f(\rho(t, x+\Delta x)) - f(\rho(t, x-\Delta x))}{2\Delta x}$$

**Loss formula**:
$$\mathcal{L}_{PDE} = \frac{1}{|\mathcal{I}|} \sum_{(t,x) \in \mathcal{I}} R(t, x)^2$$

**Optional IC loss** (if `ic_weight > 0`):
$$\mathcal{L}_{total} = \mathcal{L}_{PDE} + w_{IC} \cdot \mathcal{L}_{IC}$$

**Configuration**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `dt` | 0.004 | Time step size |
| `dx` | 0.02 | Spatial step size |
| `ic_weight` | 0.0 | Weight for IC loss (0 = disabled) |

**Required inputs**: None
**Required outputs**: `output_grid`

**Components returned**: `{"pde_residual": float, "ic": float (if enabled), "total": float}`

---

#### PDEShockResidualLoss

**Location**: `losses/pde_residual.py`

PDE residual loss computed on the **ground truth** grid, weighted by distance to the nearest predicted shock. The GT residual is non-zero at actual shocks. Each cell's squared residual is multiplied by its distance to the nearest active predicted discontinuity, providing a smooth gradient signal that rewards the model for moving predictions toward actual shocks.

**Distance weighting**:
$$d_{min}(t, x) = \min_{d \in \mathcal{A}} |x - x_d(t)|$$

where $\mathcal{A} = \{d : e_d(t) > 0.5 \text{ and } m_d = 1\}$ is the set of active predicted shocks.

**Loss formula**:
$$\mathcal{L}_{PDE\text{-}shock} = \frac{1}{|\mathcal{I}|} \sum_{(t,x) \in \mathcal{I}} R_{GT}(t, x)^2 \cdot d_{min}(t, x)$$

**Configuration**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `dt` | 0.004 | Time step size |
| `dx` | 0.02 | Spatial step size |

**Required inputs**: `x_coords`, `disc_mask`
**Required outputs**: `positions`
**Optional outputs**: `existence`

**Components returned**: `{"pde_shock_residual": float, "total": float}`

---

#### RHResidualLoss

**Location**: `losses/rh_residual.py`

Rankine-Hugoniot residual loss computed from sampled densities. Verifies that the predicted shock velocity satisfies the RH condition using density values sampled from different sources.

**Shock velocity** (from predicted positions):
$$\dot{x}_d(t) = \frac{x_d(t + \Delta t) - x_d(t - \Delta t)}{2 \Delta t}$$

**Rankine-Hugoniot residual**:
$$R_{RH}^{(d)}(t) = \dot{x}_d(t) \cdot (\rho^+_d - \rho^-_d) - (f(\rho^+_d) - f(\rho^-_d))$$

**Loss formula**:
$$\mathcal{L}_{RH} = \frac{1}{|\mathcal{V}|} \sum_{(d,t) \in \mathcal{V}} e_d(t) \cdot \left( R_{RH}^{(d)}(t) \right)^2$$

**Sampling Modes**:

| Mode | Density Source | Use Case |
|------|----------------|----------|
| `per_region` | Sample $\rho^-$ from `region_densities[d]`, $\rho^+$ from `region_densities[d+1]` | Training HybridDeepONet. Gets unblended density values. |
| `pred` | Sample both from `output_grid` | Testing RH on assembled (blended) prediction. |
| `gt` | Sample both from `target` grid | Testing if predicted trajectories match GT physics. |

**Configuration**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `dt` | 0.004 | Time step size for velocity computation |
| `epsilon` | 0.01 | Offset for density sampling near shocks |
| `mode` | `"per_region"` | Sampling mode: `"per_region"`, `"pred"`, or `"gt"` |

**Components returned**: `{"rh_residual": float}`

---

#### AccelerationLoss

**Location**: `losses/acceleration.py`

Penalizes low existence predictions where the ground truth grid shows high temporal acceleration, indicating a shock should be present.

**Acceleration computation** (central finite differences):
$$a(t, x) = \frac{\rho(t + \Delta t, x) - 2\rho(t, x) + \rho(t - \Delta t, x)}{\Delta t^2}$$

##### Original Term

For each predicted trajectory position $x_d(t)$, sample the maximum absolute acceleration in a spatial window:
$$a_{near}^{(d)}(t) = \max_{|x - x_d(t)| < \epsilon} |a(t, x)|$$

**Loss**: penalize low existence where acceleration is high:
$$\mathcal{L}_{accel} = \frac{1}{N} \sum_{(b,d,t) \in \mathcal{H}} \left(1 - e^{(b,d,t)}\right)^2 \cdot m_{b,d}$$

where $\mathcal{H} = \{(b,d,t) : |a_{near}^{(d)}(t)| > \tau\}$.

##### Missed Shock Term

Scans the entire domain for high-acceleration points not covered by any nearby prediction with high existence:

$$\text{coverage}(b, t, x) = \max_d \left[ e^{(b,d,t)} \cdot m_{b,d} \cdot \mathbf{1}(|x - x_d(t)| < \delta) \right]$$

$$\mathcal{L}_{missed} = \frac{1}{M} \sum_{(b,t,x) \in \mathcal{H}_{domain}} \left(1 - \text{coverage}(b, t, x)\right)^2$$

**Combined**: $\mathcal{L}_{total} = \mathcal{L}_{accel} + w_{missed} \cdot \mathcal{L}_{missed}$

**Configuration**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `dt` | 0.004 | Time step size for acceleration computation |
| `accel_threshold` | 1.0 | Threshold for "high" acceleration |
| `epsilon` | 0.02 | Spatial window for sampling near trajectories |
| `missed_shock_weight` | 0.0 | Weight for missed shock loss (0 = disabled) |
| `missed_shock_buffer` | None | Buffer for coverage computation (defaults to `epsilon`) |

**Required inputs**: `x_coords`, `disc_mask`
**Required outputs**: `positions`, `existence`

**Components returned**: `{"acceleration": float, "missed_shock": float (if enabled)}`

---

#### WassersteinLoss

**Location**: `losses/wasserstein.py`

Wasserstein-1 (Earth Mover's Distance) loss for discontinuous solutions. The mathematically correct metric for conservation law solutions — penalizes mislocated shocks linearly in displacement.

**Formula** (per time slice):
$$W_1(t) = \sum_x \left| \sum_{x'=0}^{x} (\rho_{pred}(t, x') - \rho_{target}(t, x')) \cdot \Delta x \right| \cdot \Delta x$$

$$\mathcal{L}_{W1} = \frac{1}{n_t} \sum_t W_1(t)$$

This is the $L^1$ norm of the antiderivative of the error — equivalent to the Sobolev $W^{-1,1}$ norm.

**Configuration**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `dx` | None | Spatial grid spacing (None → 1.0) |

**Required outputs**: `output_grid`

**Components returned**: `{"wasserstein": float}`

---

#### ConservationLoss

**Location**: `losses/conservation.py`

Enforces mass conservation — the total mass $\int \rho \, dx$ should be approximately constant over time.

**Formula**:
$$\mathcal{L}_{cons} = \text{Var}_t\left(\sum_x \rho_{pred}(t, x) \cdot \Delta x\right)$$

**Configuration**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `dx` | None | Spatial grid spacing (None → 1.0) |

**Required outputs**: `output_grid`

**Components returned**: `{"conservation": float}`

---

#### RegularizeTrajLoss

**Location**: `losses/regularize_traj.py`

Penalizes erratic trajectory behavior by limiting large spatial jumps between consecutive timesteps.

**Loss formula**:
$$\mathcal{L}_{reg} = \frac{1}{N} \sum_{b,d,t} \max(0, |x_d(t+1) - x_d(t)| - \Delta_{max})^2 \cdot e_{min}^{(b,d,t)} \cdot m_{b,d}$$

where $e_{min}^{(b,d,t)} = \min(e^{(b,d,t)}, e^{(b,d,t+1)})$ if existence is available.

**Configuration**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_step` | 0.05 | Maximum allowed position change per timestep |

**Required inputs**: `disc_mask`
**Required outputs**: `positions`
**Optional outputs**: `existence` (weights by min existence of consecutive pair)

**Components returned**: `{"regularize_traj": float}`

---

#### VAEReconstructionLoss

**Location**: `losses/vae_reconstruction.py`

**Formula**:

$$\mathcal{L}_{\text{VAE}} = \text{MSE}(\hat{u}, u) + \beta(e) \cdot D_{\text{KL}}(q(z|u) \| p(z))$$

where $\beta(e) = \beta_{\max} \cdot \min(e / E_{\text{warmup}}, 1)$ (linear warmup over $E_{\text{warmup}}$ epochs).

KL divergence: $D_{\text{KL}} = -\frac{1}{2} \sum (1 + \log\sigma^2 - \mu^2 - \sigma^2)$

**Configuration**: `beta` (default 0.01), `beta_warmup_epochs` (default 10)

**Required outputs**: `output_grid`, `z_mean`, `z_logvar`

**Components returned**: `{"recon_mse": float, "kl": float, "effective_beta": float}`

---

#### FlowMatchingLoss

**Location**: `losses/flow_matching.py`

**Formula**:

$$\mathcal{L}_{\text{FM}} = \text{MSE}(v_\theta(z_t, t, c), z - \epsilon)$$

where $z_t = (1-t)\epsilon + tz$ is the OT interpolation and $v^* = z - \epsilon$ is the target velocity.

**Required outputs**: `predicted_velocity`, `target_velocity`

**Components returned**: `{"flow_matching_mse": float}`

---

#### CellAverageMSELoss

**Location**: `losses/cell_avg_mse.py`

Cell-average MSE loss for finite-volume-consistent training. When used with `CellSamplingTransform`, the model predicts density at $k$ random query points per FV cell. This loss reshapes and averages those predictions back to cell level before computing MSE against the cell-averaged ground truth from the FV solver.

**Formula**:

Given model output $\hat{\rho}(t, x_{i,j})$ at $k$ query points $x_{i,j} \sim \mathcal{U}[i \cdot \Delta x, (i+1) \cdot \Delta x)$ within cell $i$:

$$\bar{\rho}_{pred}(t, i) = \frac{1}{k} \sum_{j=1}^{k} \hat{\rho}(t, x_{i,j})$$

$$\mathcal{L}_{cell\_avg\_mse} = \frac{1}{n_t \cdot n_x} \sum_{t,i} \left( \bar{\rho}_{pred}(t, i) - \rho_{FV}(t, i) \right)^2$$

Falls back to standard MSE when `cell_sampling_k` is not present in `input_dict`.

**Required inputs** (optional): `cell_sampling_k`, `original_nx` (set by `CellSamplingTransform`)
**Required outputs**: `output_grid`

**Components returned**: `{"cell_avg_mse": float}`

---

### CombinedLoss

**Location**: `loss.py`

Combines multiple loss functions with configurable weights.

```python
class CombinedLoss(BaseLoss):
    def __init__(
        self,
        losses: dict[str, tuple[nn.Module, float]] | list[tuple[nn.Module, float]],
    ):
```

**Usage**:
```python
# From dict
loss = CombinedLoss({
    "mse": (MSELoss(), 1.0),
    "ic": (ICLoss(), 10.0),
})

# From preset
loss = CombinedLoss.from_preset("shock_net")

# From custom config
loss = CombinedLoss.from_config([
    ("trajectory", 1.0),
    ("boundary", 0.5),
])
```

**Forward pass**:
$$\mathcal{L}_{total} = \sum_{i} w_i \cdot \mathcal{L}_i$$

**Components returned**: Each individual loss name maps to its value, plus `"total"`.

---

### Loss Presets

**Location**: `loss.py`

Pre-configured loss combinations for common use cases. Model-to-preset mapping is in `train.py` (`MODEL_LOSS_PRESET` dict) — when `--loss mse` (default), the preset is auto-selected per model.

#### mse Preset

Simple grid MSE only. Default for most models.

| Loss | Weight |
|------|--------|
| `mse` | 1.0 |

#### pde_shocks Preset

For models supervised with PDE shock residual.

| Loss | Weight | Kwargs |
|------|--------|--------|
| `mse` | 1.0 | — |
| `pde_shock_residual` | 1.0 | — |
| `rh_residual` | 1.0 | `mode="gt"` |

$$\mathcal{L} = \mathcal{L}_{MSE} + \mathcal{L}_{PDE\text{-}shock} + \mathcal{L}_{RH}$$

#### cell_avg_mse Preset

Cell-average MSE for finite-volume-consistent training.

| Loss | Weight |
|------|--------|
| `cell_avg_mse` | 1.0 |

#### traj_regularized Preset

For trajectory-predicting models (e.g. TrajTransformer). Auto-selected via `MODEL_LOSS_PRESET`.

| Loss | Weight |
|------|--------|
| `mse` | 1.0 |
| `ic_anchoring` | 0.1 |
| `boundary` | 1.0 |
| `regularize_traj` | 0.1 |

$$\mathcal{L} = \mathcal{L}_{MSE} + 0.1 \cdot \mathcal{L}_{anchor} + \mathcal{L}_{bound} + 0.1 \cdot \mathcal{L}_{reg}$$

#### cvae Preset

For CVAEDeepONet. Auto-selected via `MODEL_LOSS_PRESET`.

| Loss | Weight | Kwargs |
|------|--------|--------|
| `mse` | 1.0 | — |
| `kl_divergence` | 1.0 | `free_bits=0.01` |

$$\mathcal{L} = \mathcal{L}_{MSE} + \mathcal{L}_{KL}$$

#### mse_wasserstein Preset

For NeuralFVSolver. Combines grid MSE with Wasserstein distance for sharper shocks. Auto-selected via `MODEL_LOSS_PRESET`.

| Loss | Weight |
|------|--------|
| `mse` | 1.0 |
| `wasserstein` | 0.1 |

$$\mathcal{L} = \mathcal{L}_{MSE} + 0.1 \cdot \mathcal{L}_{W_1}$$

#### mse_shock Preset

MSE on non-shock cells only. Excludes shock regions detected via Lax entropy condition on GT.

| Loss | Weight |
|------|--------|
| `mse_shock` | 1.0 |

#### Using Presets

```python
from loss import get_loss

# Use preset directly
loss = get_loss("mse")

# Use preset with custom kwargs for individual losses
loss = get_loss("pde_shocks", loss_kwargs={
    "pde_shock_residual": {"dt": 0.004, "dx": 0.02},
    "rh_residual": {"dt": 0.004},
})
```

---

## Summary

| **Model** | **Input** | **Output** | **Purpose** |
|-----------|-----------|-----------|-------------|
| ShockTrajectoryNet | Discontinuities $(D, 3)$ | Positions, Existence $(D, T)$ | Pure trajectory prediction |
| HybridDeepONet | Discontinuities + coordinates | Trajectories + Full grid | Combined trajectory + solution |
| TrajDeepONet | Discontinuities + coordinates | Positions + Full grid | Boundary-conditioned single trunk (bilinear) |
| ClassifierTrajDeepONet | Discontinuities + coordinates | Positions + Existence + Full grid | TrajDeepONet with shock/rarefaction classifier |
| NoTrajDeepONet | Discontinuities + coordinates | Full grid | TrajDeepONet without trajectory prediction |
| TrajTransformer | Discontinuities + coordinates | Positions + Full grid | Cross-attention variant of TrajDeepONet |
| ClassifierTrajTransformer | Discontinuities + coordinates | Positions + Existence + Full grid | TrajTransformer with shock/rarefaction classifier |
| ClassifierAllTrajTransformer | Discontinuities + coordinates | Positions + Existence + Full grid | Classifier variant with dynamic boundary tokens (time-varying cross-attention) |
| BiasedClassifierTrajTransformer | Discontinuities + coordinates | Positions + Existence + Full grid | Classifier variant with characteristic attention bias for resolution generalization |
| DeepONet | Grid tensor $(3, n_t, n_x)$ | Full grid | Classic baseline (branch-trunk dot product) |
| FNO | Grid tensor $(3, n_t, n_x)$ | Full grid | Fourier Neural Operator baseline |
| EncoderDecoder | Grid tensor $(3, n_t, n_x)$ | Full grid | Transformer encoder + axial attention decoder |
| EncoderDecoderCross | Grid tensor $(3, n_t, n_x)$ | Full grid | Transformer encoder + cross-attention decoder |
| CharNO | IC segments (xs, ks) + coordinates | Full grid + selection weights | Characteristic neural operator (Lax-Hopf softmin) |
| WaveNO | IC segments (xs, ks) + coordinates | Full grid + characteristic bias + positions | Wavefront neural operator (characteristic-biased cross-attention + breakpoint evolution) |
| WaveNOBase | IC segments (xs, ks) + coordinates | Full grid + characteristic bias | WaveNO without trajectory prediction (no breakpoint evolution, no boundary features) |
| WaveNODisc | Discontinuities (x, rho_L, rho_R) + coordinates | Full grid + characteristic bias + positions | WaveNO variant with discontinuity tokens instead of segments |
| CTTSeg | IC segments (xs, ks) + coordinates | Positions + Existence + Full grid | CTT with segment tokens + BreakpointEvolution instead of discontinuity tokens |
| TransformerSeg | IC segments (xs, ks) + coordinates | Full grid | Segment-based encoding + cross-attention density decoder, no trajectory prediction |
| WaveFrontModel | Discontinuities (x, rho_L, rho_R) + coordinates | Full grid + wave pattern | Learned Riemann solver with analytical wave reconstruction |
| LDDeepONet | Piecewise IC (xs, ks) + target grid (training) | Full grid (generative) | VAE + flow matching with resolution-invariant DeepONet decoder |
| NeuralFVSolver | Grid IC $(1, n_t, n_x)$ | Full grid | Learned FV time-marching with stencil features + shock proximity |

| **Loss** | **Location** | **Key Physics** | **Use Case** |
|----------|--------------|-----------------|--------------|
| MSELoss | `losses/mse.py` | Grid supervision | Grid matching |
| ICLoss | `losses/ic.py` | Initial condition | IC accuracy |
| TrajectoryConsistencyLoss | `losses/trajectory_consistency.py` | Analytical RH trajectories | Trajectory models |
| BoundaryLoss | `losses/boundary.py` | Domain constraints | All trajectory models |
| CollisionLoss | `losses/collision.py` | Shock merging | Multi-shock models |
| ICAnchoringLoss | `losses/existence_regularization.py` | Anchor trajectories to IC positions | All trajectory models |
| SupervisedTrajectoryLoss | `losses/supervised_trajectory.py` | Direct supervision | When GT available |
| PDEResidualLoss | `losses/pde_residual.py` | Conservation in smooth regions | Grid models |
| PDEShockResidualLoss | `losses/pde_residual.py` | GT residual penalizing unpredicted shocks | Trajectory models |
| RHResidualLoss | `losses/rh_residual.py` | RH at shocks (from densities) | Hybrid models |
| AccelerationLoss | `losses/acceleration.py` | High acceleration = shock | Existence supervision |
| RegularizeTrajLoss | `losses/regularize_traj.py` | Smooth trajectories | Penalize erratic jumps |
| WassersteinLoss | `losses/wasserstein.py` | $W_1$ distance (sharp shocks) | CharNO, discontinuous solutions |
| ConservationLoss | `losses/conservation.py` | Mass conservation | CharNO, physics regularization |
| VAEReconstructionLoss | `losses/vae_reconstruction.py` | MSE + KL with beta warmup | LDDeepONet phase 1 |
| FlowMatchingLoss | `losses/flow_matching.py` | OT velocity matching | LDDeepONet phase 2 |

### Key Design Principles

1. **DeepONet Architecture**: Branch-trunk factorization enables learning solution operators with variable-length inputs (variable shock counts)

2. **Cross-Attention Alternative**: TrajTransformer replaces bilinear fusion with cross-attention, allowing richer discontinuity-aware feature aggregation without branch pooling

3. **Soft Region Boundaries**: Differentiable sigmoid boundaries in GridAssembler enable end-to-end gradient flow

4. **Multi-Scale Physics**:
   - Shock scale: Rankine-Hugoniot residuals
   - Smooth scale: PDE conservation
   - Global scale: Grid MSE supervision

5. **Fourier Features**: Exponentially-spaced sinusoidal encoding enables MLPs to learn high-frequency shock dynamics (used in TimeEncoder, DiscontinuityEncoder, SpaceTimeEncoder, and FourierTokenizer)

6. **Independent Discontinuity Encoding**: Each discontinuity is encoded independently via Fourier + MLP, with optional cross-discontinuity self-attention for interaction

7. **Modular Loss Design**: One file per loss enables easy composition and testing

---

## Ablation Models

Six ablation models isolate which architectural component drives the performance gap between WaveNO and ClassifierTrajTransformer.

### WaveNO Ablations (fixing step-generalization weakness)

| Model | Flag | Change |
|-------|------|--------|
| `WaveNOBase` | `predict_trajectories=False` | Removes stages 3-4 (breakpoint evolution + boundary computation); query encoder uses only (t, x) without boundary features. Grid-only output, no trajectory prediction. Ablation baseline. |
| `WaveNOCls` | `with_classifier=True` | Adds classifier head on breakpoint embeddings to filter breakpoints in `compute_boundaries` |
| `WaveNOLocal` | `local_features=False` | Removes cumulative mass $N_k$ from `SegmentPhysicsEncoder` (3 scalar features instead of 4) |
| `WaveNOIndepTraj` | `independent_traj=True` | Bypasses `BreakpointEvolution`; encodes raw discontinuities via `DiscontinuityEncoder` for trajectory prediction |
| `WaveNODisc` | `use_discontinuities=True` | Uses discontinuities as tokens instead of segments: `DiscontinuityPhysicsEncoder` for encoding, disc-based characteristic bias, trajectory directly from disc embeddings |
| `ShockAwareWaveNO` | `predict_proximity=True` | Adds a second MLP head (proximity head) that predicts a sigmoid-activated shock proximity field from the same cross-attention features as the density head |

#### ShockAwareWaveNO Architecture

WaveNO with a dual output: density solution and shock proximity. The proximity head is an MLP identical to the density head but with sigmoid activation instead of clamp. Both heads operate on the same fused cross-attention features `q`, forcing the shared representation to be shock-aware.

```
# Same as WaveNO up to cross-attention output q: (nt, nx, H)

q → DensityHead(MLP) → clamp[0,1] → output_grid: (1, nt, nx)
q → ProximityHead(MLP) → sigmoid → shock_proximity: (1, nt, nx)
```

Components:
- **DensityHead**: `Linear(H, H) → ReLU → Dropout → Linear(H, 1)` — initialized near zero (bias=0.5)
- **ProximityHead**: `Linear(H, H) → ReLU → Dropout → Linear(H, 1)` — initialized at zero

Uses `shock_proximity` loss preset (MSE + proximity MSE) and `traj_residual` plot preset. Uses native dict input (no transform).

### CTT Ablations (fixing resolution weakness)

| Model | Flag | Change |
|-------|------|--------|
| `CTTBiased` | `characteristic_bias=True` | Adds characteristic attention bias to density cross-attention (alias for `BiasedClassifierTrajTransformer`) |
| `CTTSegPhysics` | `segment_physics=True` | Enriches disc encoder input with $\lambda_L$, $\lambda_R$, $s$ (6 features instead of 3) |
| `CTTFiLM` | `film_time=True` | Applies FiLM time conditioning to disc embeddings, producing per-timestep keys for density decoding |
| `CTTSeg` | `use_segments=True` | Replaces discontinuity tokens with segment tokens (SegmentPhysicsEncoder + BreakpointEvolution), like WaveNO's representation |

#### CTTSeg Architecture

Replaces the discontinuity-based input representation with WaveNO's segment-based representation while keeping the CTT architecture for trajectory decoding and density prediction.

```
xs, ks, pieces_mask → SegmentPhysicsEncoder → seg_emb: (K, H)
seg_emb → SelfAttention(EncoderLayer × L) → contextualized seg_emb: (K, H)

seg_emb, disc_mask, t_unique → BreakpointEvolution(return_embeddings=True)
  → positions: (D, nt), bp_emb: (D, H)

bp_emb → ClassifierHead(MLP → Sigmoid) → existence: (D,)

positions, x_coords, effective_mask → compute_boundaries → x_left, x_right: (nt, nx)

(t, x, x_left, x_right) → FourierEncode → CoordMLP → coord_emb: (nt*nx, H)
(coord_emb as Q, seg_emb as K/V) → CrossDecoderLayer × L → DensityHead → (nt, nx)
→ output_grid: (1, nt, nx)
```

**Key difference from CTT**: Encodes K piecewise-constant segments (center, width, density, characteristic speed, flux, cumulative mass) instead of D discontinuity points (x, rho_L, rho_R). Trajectories are derived from adjacent segment pairs via `BreakpointEvolution` rather than from `TimeEncoder` + `TrajectoryDecoderTransformer`. The density decoder cross-attends to segment embeddings (K tokens) with `pieces_mask`.

#### TransformerSeg Architecture

Combines CTTSeg's segment-based input encoding with NoTrajTransformer's grid-only output. No trajectory prediction, no classifier, no boundary conditioning.

```
xs, ks, pieces_mask → SegmentPhysicsEncoder → seg_emb: (K, H)
seg_emb → SelfAttention(EncoderLayer × L) → contextualized seg_emb: (K, H)

(t, x) → FourierEncode → CoordMLP → coord_emb: (nt*nx, H)
(coord_emb as Q, seg_emb as K/V) → CrossDecoderLayer × L → DensityHead → (nt, nx)
→ output_grid: (1, nt, nx)
```

**Key difference from CTTSeg**: Removes BreakpointEvolution (trajectory prediction), classifier head, and boundary conditioning. The density decoder uses `with_boundaries=False`, encoding only `(t, x)` coordinates instead of `(t, x, x_left, x_right)`.

---

#### ShockAwareDeepONet Architecture

Dual-head DeepONet with a shared trunk (SpaceTimeEncoder) and two branch heads: one for the solution field and one for shock proximity. The shared trunk forces the model to learn shock-aware basis functions.

```
grid_input: (3, nt, nx) [ic_masked, t_coords, x_coords]

ic = grid_input[0, 0, :]                       # (nx,)
ic_pooled = AdaptiveAvgPool1d(ic, train_nx)     # resolution invariance
ic_emb = ICEncoder(ic_pooled)                   # (hidden_dim,)

sol_coeffs = SolutionBranch(ic_emb)             # (latent_dim,)
prox_coeffs = ProximityBranch(ic_emb)           # (latent_dim,)

trunk_out = SpaceTimeEncoder(t_coords, x_coords)  # (nt, nx, latent_dim)

output_grid = einsum(sol_coeffs, trunk_out) + bias_sol        # (1, nt, nx)
shock_proximity = sigmoid(einsum(prox_coeffs, trunk_out) + bias_prox)  # (1, nt, nx)
```

Components:
- **ICEncoder**: `Linear(nx, H) → GELU → [Linear(H, H) → GELU] × (L-1)` — shared IC MLP
- **SolutionBranch**: `[Linear(H, H) → GELU] × (L-1) → Linear(H, latent)` — solution coefficients
- **ProximityBranch**: `[Linear(H, H) → GELU] × (L-1) → Linear(H, latent)` — proximity coefficients
- **Trunk**: `SpaceTimeEncoder(hidden_dim, latent_dim)` — Fourier-encoded coordinate MLP

Default hyperparameters: hidden_dim=128, latent_dim=64, num_ic_layers=3, num_branch_layers=2, num_trunk_layers=3.

Uses `ToGridInput` transform. Resolution invariant via SpaceTimeEncoder + adaptive IC pooling.

---

#### ShockProximityLoss

MSE on the shock proximity field:

$$\mathcal{L}_{\text{prox}} = \text{MSE}(\hat{p}, p)$$

where:
- $\hat{p}$ = predicted shock proximity (`output_dict["shock_proximity"]`)
- $p$ = ground truth shock proximity (`input_dict["shock_proximity"]`)

Ground truth proximity is precomputed from the Lax entropy condition:
1. Detect shocks at each cell interface via $\lambda_L > s > \lambda_R$ where $\lambda = 1 - 2\rho$ and $s = 1 - \rho_L - \rho_R$
2. Filter the binary shock mask with connected component analysis (`scipy.ndimage.label`), removing components with fewer than `min_component_size` cells (default: 5, configurable via `--min_component_size`)
3. For each cell, compute minimum distance to any shock interface
4. $p = \exp(-d_{\min} / \sigma)$ where $\sigma$ = `proximity_sigma` (default: 0.05)

Used via the `shock_proximity` preset: `mse` (weight 1.0) + `shock_proximity` (weight 0.1).

#### EntropyConditionLoss

Uses the Lax entropy condition on the GT grid as a threshold-free shock detector. Penalizes missed shocks (entropy-detected shocks far from predictions) and false positives (predictions far from shocks).

Shock detection:
1. For each cell interface: $\lambda_L = 1 - 2\rho_j$, $\lambda_R = 1 - 2\rho_{j+1}$, $s = 1 - \rho_j - \rho_{j+1}$
2. Interface is a shock if $\lambda_L > s > \lambda_R$ (Lax entropy condition)
3. Small isolated components (< `min_component_size` cells) are removed via connected component filtering (`scipy.ndimage.label`)

Loss:
$$\mathcal{L} = \mathcal{L}_{\text{miss}} + w_{\text{fp}} \cdot \mathcal{L}_{\text{fp}}$$

- $\mathcal{L}_{\text{miss}}$: jump-weighted distance from entropy-detected shocks to nearest active prediction
- $\mathcal{L}_{\text{fp}}$: existence-weighted distance from predictions to nearest entropy-detected shock

Parameters: `dx` (spatial step), `fp_weight` (false-positive weight, default: 1.0), `min_component_size` (default: 5, 0 to disable).

#### MSEShockLoss

MSE computed only on non-shock cells. Shocks are detected on the GT grid using the Lax entropy condition, then expanded from interface masks to cell masks. This focuses learning on smooth regions without penalizing sharp shock approximations.

Shock detection (same as EntropyConditionLoss):
1. For each cell interface: $\lambda_L = 1 - 2\rho_j$, $\lambda_R = 1 - 2\rho_{j+1}$, $s = 1 - \rho_j - \rho_{j+1}$
2. Interface is a shock if $\lambda_L > s > \lambda_R$ (Lax entropy condition)
3. Small isolated components (< `min_component_size` cells) are removed via connected component filtering

Cell mask expansion: a cell $j$ is marked as shock if either adjacent interface ($j-1$ or $j$) is a shock.

Loss:
$$\mathcal{L} = \frac{1}{|\mathcal{S}^c|} \sum_{i \in \mathcal{S}^c} (\hat{u}_i - u_i)^2$$

where $\mathcal{S}^c$ is the set of non-shock cells and $|\mathcal{S}^c|$ is its cardinality. Returns 0 if all cells are shock.

Parameters: `dx` (spatial step, kept for interface compatibility), `min_component_size` (default: 5, 0 to disable).
