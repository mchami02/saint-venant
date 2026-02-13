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

#### All-Boundaries Mode (`all_boundaries=True`) — DynamicDensityDecoder

When `all_boundaries=True`, the density decoder uses `DynamicDensityDecoder` with **time-varying boundary tokens** instead of static disc embeddings or Fourier-encoded left/right positions:

1. **Dynamic boundary tokens**: For each time step $t$, boundary token $d$ combines wave properties with predicted position: $\mathbf{b}_d(t) = \mathbf{e}_d + \text{proj}(\text{Fourier}(x_d(t)))$, where $\mathbf{e}_d$ is the discontinuity embedding and $x_d(t)$ is the predicted trajectory position
2. **Soft existence weighting**: Tokens are weighted by $\hat{p}_d \cdot m_d$ (classifier probability times validity mask) — fully differentiable, no hard threshold
3. **Per-time-step cross-attention**: Spatial queries $\text{MLP}(\text{Fourier}(t) \| \text{Fourier}(x))$ at each time step cross-attend to the $D$ dynamic boundary tokens at that time step. Batched as $(B \cdot T, n_x, H)$ queries and $(B \cdot T, D, H)$ keys/values for efficiency
4. **Density head**: Output projection maps enriched queries to density values

This design is fully differentiable (no `compute_boundaries` or hard thresholds), enabling end-to-end gradient flow from density loss through existence predictions to the classifier. The attention mechanism naturally learns spatial locality — queries attend primarily to nearby boundary tokens.

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

Pre-configured loss combinations for common use cases.

#### shock_net Preset

For trajectory-only models (ShockNet).

| Loss | Weight | Kwargs |
|------|--------|--------|
| `boundary` | 1.0 | — |
| `acceleration` | 1.0 | `missed_shock_weight=1.0` |
| `ic_anchoring` | 0.1 | — |

$$\mathcal{L} = \mathcal{L}_{bound} + \mathcal{L}_{accel} + 0.1 \cdot \mathcal{L}_{anchor}$$

#### hybrid Preset

For HybridDeepONet (trajectory + grid prediction).

| Loss | Weight | Kwargs |
|------|--------|--------|
| `mse` | 1.0 | — |
| `rh_residual` | 1.0 | — |
| `pde_residual` | 0.1 | — |
| `ic` | 10.0 | — |
| `ic_anchoring` | 0.01 | — |

$$\mathcal{L} = \mathcal{L}_{MSE} + \mathcal{L}_{RH} + 0.1 \cdot \mathcal{L}_{PDE} + 10 \cdot \mathcal{L}_{IC} + 0.01 \cdot \mathcal{L}_{anchor}$$

#### traj_net Preset

For TrajDeepONet and NoTrajDeepONet.

| Loss | Weight |
|------|--------|
| `mse` | 1.0 |
| `ic_anchoring` | 0.1 |
| `boundary` | 1.0 |
| `regularize_traj` | 0.1 |

$$\mathcal{L} = \mathcal{L}_{MSE} + 0.1 \cdot \mathcal{L}_{anchor} + \mathcal{L}_{bound} + 0.1 \cdot \mathcal{L}_{reg}$$

#### classifier_traj_net Preset

For ClassifierTrajDeepONet (TrajDeepONet with existence classifier).

| Loss | Weight | Kwargs |
|------|--------|--------|
| `mse` | 1.0 | — |
| `ic_anchoring` | 0.1 | — |
| `boundary` | 1.0 | — |
| `regularize_traj` | 0.1 | — |
| `acceleration` | 1.0 | `missed_shock_weight=1.0` |

$$\mathcal{L} = \mathcal{L}_{MSE} + 0.1 \cdot \mathcal{L}_{anchor} + \mathcal{L}_{bound} + 0.1 \cdot \mathcal{L}_{reg} + \mathcal{L}_{accel}$$

#### traj_transformer Preset

For TrajTransformer.

| Loss | Weight |
|------|--------|
| `mse` | 1.0 |
| `ic_anchoring` | 0.1 |
| `boundary` | 1.0 |
| `regularize_traj` | 0.1 |

$$\mathcal{L} = \mathcal{L}_{MSE} + 0.1 \cdot \mathcal{L}_{anchor} + \mathcal{L}_{bound} + 0.1 \cdot \mathcal{L}_{reg}$$

#### classifier_traj_transformer Preset

For ClassifierTrajTransformer.

| Loss | Weight | Kwargs |
|------|--------|--------|
| `mse` | 1.0 | — |
| `ic_anchoring` | 0.1 | — |
| `boundary` | 1.0 | — |
| `regularize_traj` | 0.1 | — |
| `acceleration` | 1.0 | `missed_shock_weight=1.0` |

$$\mathcal{L} = \mathcal{L}_{MSE} + 0.1 \cdot \mathcal{L}_{anchor} + \mathcal{L}_{bound} + 0.1 \cdot \mathcal{L}_{reg} + \mathcal{L}_{accel}$$

#### classifier_all_traj_transformer Preset

For ClassifierAllTrajTransformer. Same losses as `classifier_traj_transformer`.

| Loss | Weight | Kwargs |
|------|--------|--------|
| `mse` | 1.0 | — |
| `ic_anchoring` | 0.1 | — |
| `boundary` | 1.0 | — |
| `regularize_traj` | 0.1 | — |
| `acceleration` | 1.0 | `missed_shock_weight=1.0` |

$$\mathcal{L} = \mathcal{L}_{MSE} + 0.1 \cdot \mathcal{L}_{anchor} + \mathcal{L}_{bound} + 0.1 \cdot \mathcal{L}_{reg} + \mathcal{L}_{accel}$$

#### pde_shocks Preset

For models supervised with PDE shock residual.

| Loss | Weight |
|------|--------|
| `mse` | 1.0 |
| `pde_shock_residual` | 1.0 |

$$\mathcal{L} = \mathcal{L}_{MSE} + \mathcal{L}_{PDE\text{-}shock}$$

#### mse Preset

Simple grid MSE only.

| Loss | Weight |
|------|--------|
| `mse` | 1.0 |

#### Using Presets

```python
from loss import get_loss

# Use preset directly
loss = get_loss("shock_net")

# Use preset with custom kwargs for individual losses
loss = get_loss("hybrid", loss_kwargs={
    "pde_residual": {"dt": 0.004, "dx": 0.02},
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
| DeepONet | Grid tensor $(3, n_t, n_x)$ | Full grid | Classic baseline (branch-trunk dot product) |
| FNO | Grid tensor $(3, n_t, n_x)$ | Full grid | Fourier Neural Operator baseline |
| EncoderDecoder | Grid tensor $(3, n_t, n_x)$ | Full grid | Transformer encoder + axial attention decoder |
| EncoderDecoderCross | Grid tensor $(3, n_t, n_x)$ | Full grid | Transformer encoder + cross-attention decoder |

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
