# Wavefront Learning Architecture

This document describes the neural network architectures and loss functions used in the wavefront learning module for predicting shock trajectories and solutions in LWR traffic flow equations.

## Table of Contents

1. [Models Overview](#models-overview)
   - [Directory Structure](#directory-structure)
   - [Base Components](#base-components)
2. [Main Models](#main-models)
   - [ShockTrajectoryNet](#shocktrajectorynet)
   - [HybridDeepONet](#hybriddeepopnet)
   - [TrajDeepONet](#trajdeeponet)
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
     - [RHResidualLoss](#rhresidualloss)
     - [AccelerationLoss](#accelerationloss)
   - [CombinedLoss](#combinedloss)
   - [Loss Presets](#loss-presets)

---

## Models Overview

### Directory Structure

The models package is organized with main models at the top level and reusable building blocks in a `base/` subdirectory:

```
models/
├── __init__.py              # Exports main models and BaseWavefrontModel
├── shock_trajectory_net.py  # ShockTrajectoryNet + build_shock_net()
├── hybrid_deeponet.py       # HybridDeepONet + build_hybrid_deeponet()
├── traj_deeponet.py         # TrajDeepONet + build_traj_deeponet()
└── base/                    # Reusable building blocks
    ├── __init__.py          # Re-exports all base components
    ├── base_model.py        # BaseWavefrontModel (abstract base class)
    ├── encoders.py          # FourierFeatures, TimeEncoder, DiscontinuityEncoder, SpaceTimeEncoder
    ├── decoders.py          # TrajectoryDecoder
    ├── blocks.py            # ResidualBlock
    ├── regions.py           # RegionTrunk, RegionTrunkSet
    └── assemblers.py        # GridAssembler
```

### Base Components

All base components are located in `models/base/` and can be imported from there:

```python
from wavefront_learning.models.base import (
    FourierFeatures,
    TimeEncoder,
    DiscontinuityEncoder,
    SpaceTimeEncoder,
    TrajectoryDecoder,
    ResidualBlock,
    RegionTrunk,
    RegionTrunkSet,
    GridAssembler,
    BaseWavefrontModel,
)
```

| File | Components | Description |
|------|------------|-------------|
| `base_model.py` | `BaseWavefrontModel` | Abstract base class for wavefront models |
| `encoders.py` | `FourierFeatures`, `TimeEncoder`, `DiscontinuityEncoder`, `SpaceTimeEncoder` | Input encoding modules |
| `decoders.py` | `TrajectoryDecoder` | Decodes trajectories from branch/trunk embeddings |
| `blocks.py` | `ResidualBlock` | Residual MLP block with LayerNorm |
| `regions.py` | `RegionTrunk`, `RegionTrunkSet` | Density prediction for inter-shock regions |
| `assemblers.py` | `GridAssembler` | Assembles grid from region predictions with soft boundaries |

#### Dependency Graph

```
base/blocks.py        (no deps - leaf)
base/assemblers.py    (no deps - leaf)
base/base_model.py    (no deps - leaf)
base/encoders.py      (self-contained - FourierFeatures used by SpaceTimeEncoder)
base/decoders.py      → imports blocks.ResidualBlock
base/regions.py       → imports blocks.ResidualBlock
base/__init__.py      → imports all above
     ↓
shock_trajectory_net.py  → imports from base
hybrid_deeponet.py       → imports from base
traj_deeponet.py         → imports from base
     ↓
models/__init__.py       → imports main models
```

---

## Main Models

### ShockTrajectoryNet

**Location**: `models/shock_trajectory_net.py`

A DeepONet-style architecture for predicting shock (discontinuity) trajectories in LWR traffic flow using unsupervised physics-based training.

#### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     ShockTrajectoryNet                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Discontinuities (B, D, 3)      Query Times (B, T)             │
│         │                              │                        │
│         ▼                              ▼                        │
│  ┌──────────────────┐         ┌───────────────────┐            │
│  │ DiscontinuityEncoder │     │   TimeEncoder     │            │
│  │    (Branch Net)      │     │   (Trunk Net)     │            │
│  └──────────────────┘         └───────────────────┘            │
│         │                              │                        │
│         │  (B, D, hidden)              │  (B, T, hidden)       │
│         └──────────────┬───────────────┘                        │
│                        │                                        │
│                        ▼                                        │
│               ┌─────────────────┐                              │
│               │ TrajectoryDecoder │                            │
│               └─────────────────┘                              │
│                        │                                        │
│           ┌───────────┴───────────┐                            │
│           ▼                       ▼                            │
│    Positions (B, D, T)    Existence (B, D, T)                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Sub-Components

##### FourierFeatures

**Location**: `models/base/encoders.py`

Positional encoding for scalar inputs using sinusoidal features.

**Formula**:
$$\gamma(x) = \left[ \sin(2^0 \pi x), \cos(2^0 \pi x), \sin(2^1 \pi x), \cos(2^1 \pi x), \ldots, \sin(2^{L-1} \pi x), \cos(2^{L-1} \pi x) \right]$$

where $L$ = `num_frequencies` (default: 32).

**Output dimension**: $2L$ (or $2L + 1$ if including original input)

##### DiscontinuityEncoder (Branch Network)

**Location**: `models/base/encoders.py`

Encodes discontinuities using Fourier features + MLP. Each discontinuity is processed independently (no cross-attention).

**Architecture**:
```
Input: (B, D, 3) where each discontinuity = [x_position, rho_L, rho_R]
       │
       ├── x_position ──► FourierFeatures ──► (B, D, 2*num_freq + 1)
       │
       └── [rho_L, rho_R] ────────────────► (B, D, 2)
       │
       └── concatenate ───────────────────► (B, D, 2*num_freq + 3)
               │
               ▼
       MLP: [Linear + GELU + LayerNorm] × num_layers
               │
               ▼
       Apply mask × output
               │
               ▼
       Output: (B, D, output_dim)
```

**Default configuration**:
- `hidden_dim`: 128
- `output_dim`: 128
- `num_frequencies`: 16
- `num_layers`: 3

##### TimeEncoder (Trunk Network)

**Location**: `models/base/encoders.py`

Encodes query times for trajectory prediction.

**Architecture**:
```
Input: t ∈ (B, T)
       │
       ▼
Fourier Features: γ(t) → (B, T, 2L+1)
       │
       ▼
MLP: [Linear + GELU + LayerNorm] × num_layers
       │
       ▼
Output: (B, T, output_dim)
```

**Default configuration**:
- `hidden_dim`: 128
- `output_dim`: 128
- `num_frequencies`: 32
- `num_layers`: 3

##### TrajectoryDecoder

**Location**: `models/base/decoders.py`

Predicts shock positions and existence probabilities from branch and trunk embeddings.

**Architecture**:
```
branch_emb: (B, D, branch_dim)
trunk_emb: (B, T, trunk_dim)
       │
       ▼
Bilinear Fusion: W_bilinear(branch ⊗ trunk) → (B, D, T, hidden_dim)
       +
Skip Paths: Linear(branch) + Linear(trunk)
       │
       ▼
LayerNorm
       │
       ▼
Residual Blocks × num_res_blocks
       │
       ├──────────────────────┐
       ▼                      ▼
Position Head:           Existence Head:
Linear → GELU → Linear   Linear → GELU → Linear → Sigmoid
       │                      │
       ▼                      ▼
positions: (B, D, T)     existence: (B, D, T) ∈ [0, 1]
```

**Bilinear fusion formula**:
$$h_{d,t} = W_{bilinear}(b_d \otimes e_t) + W_{branch}(b_d) + W_{trunk}(e_t)$$

where $b_d$ is the branch embedding for discontinuity $d$ and $e_t$ is the trunk embedding for time $t$.

#### Input/Output Format

**Input** (dictionary):
- `discontinuities`: $(B, D, 3)$ - initial shock features $[x_0, \rho_L, \rho_R]$
- `disc_mask`: $(B, D)$ - validity mask for discontinuities
- `t_coords`: $(B, 1, n_t, n_x)$ - query times

**Output** (dictionary):
- `positions`: $(B, D, T)$ - predicted x-coordinates of each shock at each time
- `existence`: $(B, D, T)$ - probability $\in [0, 1]$ that each shock exists

---

### HybridDeepONet

**Location**: `models/hybrid_deeponet.py`

Combined model that predicts both shock trajectories AND full solution grids by assembling per-region density predictions.

#### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         HybridDeepONet                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Discontinuities (B, D, 3)     t_coords (B, 1, nt, nx)                 │
│         │                      x_coords (B, 1, nt, nx)                 │
│         ▼                              │                               │
│  ┌──────────────────┐                  │                               │
│  │DiscontinuityEncoder│                │                               │
│  │   (Shared Branch)  │                │                               │
│  └──────────────────┘                  │                               │
│         │                              │                               │
│         ├─────────────────┬────────────┤                               │
│         │                 │            │                               │
│         ▼                 │            ▼                               │
│  ┌─────────────┐          │     ┌─────────────────┐                    │
│  │ TimeEncoder │          │     │SpaceTimeEncoder │                    │
│  └─────────────┘          │     └─────────────────┘                    │
│         │                 │            │                               │
│         ▼                 │            │                               │
│  ┌─────────────────┐      │            │                               │
│  │TrajectoryDecoder│      │            │                               │
│  └─────────────────┘      │            │                               │
│         │                 │            │                               │
│    positions,             │            │                               │
│    existence              │            │                               │
│         │                 ▼            ▼                               │
│         │          ┌──────────────────────┐                            │
│         │          │    RegionTrunkSet    │                            │
│         │          │  (K region trunks)   │                            │
│         │          └──────────────────────┘                            │
│         │                 │                                            │
│         │                 ▼                                            │
│         │          region_densities (B, K, nt, nx)                     │
│         │                 │                                            │
│         └────────────────►│                                            │
│                           ▼                                            │
│                    ┌─────────────┐                                     │
│                    │GridAssembler│                                     │
│                    └─────────────┘                                     │
│                           │                                            │
│                           ▼                                            │
│                    output_grid (B, 1, nt, nx)                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Sub-Components

##### SpaceTimeEncoder

**Location**: `models/base/encoders.py`

Encodes $(t, x)$ coordinate pairs for region density prediction.

**Architecture**:
```
t_coords: (B, nt, nx)
x_coords: (B, nt, nx)
       │
       ▼
Separate Fourier encoding:
  γ_t(t) with L_t frequencies
  γ_x(x) with L_x frequencies
       │
       ▼
Concatenate: [γ_t(t), γ_x(x)] → (B, nt, nx, 2L_t + 2L_x + 2)
       │
       ▼
MLP: [Linear + GELU + LayerNorm] × num_layers
       │
       ▼
Output: (B, nt, nx, output_dim)
```

**Default configuration**:
- `hidden_dim`: 128
- `output_dim`: 128
- `num_frequencies_t`: 16
- `num_frequencies_x`: 16
- `num_layers`: 3

##### RegionTrunk

**Location**: `models/base/regions.py`

Predicts density values for a single region between shocks.

**Architecture**:
```
branch_emb: (B, branch_dim)         coord_emb: (B, nt, nx, coord_dim)
       │                                   │
       ▼                                   │
Expand: (B, nt, nx, branch_dim)           │
       │                                   │
       └──────────────┬────────────────────┘
                      │
                      ▼
Bilinear Fusion + Skip Paths
                      │
                      ▼
Residual Blocks × num_res_blocks
                      │
                      ▼
Density Head: Linear → GELU → Linear → Sigmoid
                      │
                      ▼
Output: (B, nt, nx) ∈ [0, 1]
```

##### RegionTrunkSet

**Location**: `models/base/regions.py`

Set of $K = \text{max\_discontinuities} + 1$ region trunks, one for each region.

**Output**: $(B, K, n_t, n_x)$ - stacked per-region density predictions

##### GridAssembler

**Location**: `models/base/assemblers.py`

Assembles the final solution grid from per-region predictions using soft sigmoid boundaries.

**Region Assignment**:

For a domain with $D$ discontinuities, there are $K = D + 1$ regions:
- Region 0: Left of the first shock
- Region $k$ (for $1 \leq k < D$): Between shock $k-1$ and shock $k$
- Region $D$: Right of the last shock

**Soft Boundary Computation**:

For each shock $d$ at position $x_d(t)$:
$$\text{left\_of\_shock}_d(t, x) = \sigma\left(\frac{x_d(t) - x}{\sigma_{soft}}\right) \cdot e_d(t) \cdot m_d$$

where:
- $\sigma(\cdot)$ is the sigmoid function
- $\sigma_{soft} = 0.02$ is the softness parameter
- $e_d(t)$ is the existence probability
- $m_d$ is the discontinuity mask

**Region Weights**:
$$w_0(t, x) = \text{left\_of\_shock}_0(t, x)$$
$$w_k(t, x) = (1 - \text{left\_of\_shock}_{k-1}(t, x)) \cdot \text{left\_of\_shock}_k(t, x) \quad \text{for } 1 \leq k < K-1$$
$$w_{K-1}(t, x) = 1 - \text{left\_of\_shock}_{D-1}(t, x)$$

**Final Grid Assembly**:
$$\rho(t, x) = \sum_{k=0}^{K-1} w_k(t, x) \cdot \rho_k(t, x)$$

where $\rho_k$ is the density prediction from region trunk $k$.

#### Input/Output Format

**Input** (dictionary):
- `discontinuities`: $(B, D, 3)$ - initial shock features
- `disc_mask`: $(B, D)$ - validity mask
- `t_coords`: $(B, 1, n_t, n_x)$ - query times
- `x_coords`: $(B, 1, n_t, n_x)$ - query positions

**Output** (dictionary):
- `positions`: $(B, D, T)$ - shock positions
- `existence`: $(B, D, T)$ - shock existence probabilities
- `output_grid`: $(B, 1, n_t, n_x)$ - assembled solution
- `region_densities`: $(B, K, n_t, n_x)$ - per-region predictions
- `region_weights`: $(B, K, n_t, n_x)$ - soft region assignments

---

### TrajDeepONet

**Location**: `models/traj_deeponet.py`

Trajectory-conditioned DeepONet that predicts shock trajectories and uses them to condition a single density trunk. Key simplifications over HybridDeepONet:
- **No existence head**: all input discontinuities persist through time
- **Single trunk**: one network conditioned on boundary positions instead of K separate region trunks
- **No GridAssembler**: the trunk directly outputs density

#### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          TrajDeepONet                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Discontinuities (B, D, 3)     t_coords (B, 1, nt, nx)              │
│         │                      x_coords (B, 1, nt, nx)              │
│         ▼                              │                             │
│  ┌──────────────────┐                  │                             │
│  │DiscontinuityEncoder│                │                             │
│  │     (Branch)       │                │                             │
│  └──────────────────┘                  │                             │
│         │                              │                             │
│         ├───── per-disc (B,D,h) ───┐   │                             │
│         │                          │   │                             │
│         ▼ pooled (B,h)             │   │                             │
│         │                          │   │                             │
│         │    ┌─────────────┐       │   │                             │
│         │    │ TimeEncoder │       │   │                             │
│         │    └─────────────┘       │   │                             │
│         │           │              │   │                             │
│         │           ▼              │   │                             │
│         │    ┌───────────────┐     │   │                             │
│         │    │PositionDecoder│◄────┘   │                             │
│         │    │ (no existence)│         │                             │
│         │    └───────────────┘         │                             │
│         │           │                  │                             │
│         │    positions (B,D,T)         │                             │
│         │           │                  │                             │
│         │           ▼                  │                             │
│         │    ┌────────────────┐        │                             │
│         │    │compute_boundaries│◄──────┘                             │
│         │    └────────────────┘                                      │
│         │      │            │                                        │
│         │   x_left       x_right                                     │
│         │   (B,nt,nx)    (B,nt,nx)                                   │
│         │      │            │                                        │
│         ▼      ▼            ▼                                        │
│  ┌──────────────────────────────────┐                                │
│  │   BoundaryConditionedTrunk      │                                │
│  │   Input: branch + (t, x,       │                                │
│  │          x_left, x_right)       │                                │
│  └──────────────────────────────────┘                                │
│                    │                                                 │
│                    ▼                                                 │
│             output_grid (B, 1, nt, nx)                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

#### Sub-Components

##### PositionDecoder

**Location**: `models/traj_deeponet.py`

Simplified trajectory decoder that only predicts positions (no existence head).

**Architecture**:
```
branch_emb: (B, D, branch_dim)
trunk_emb: (B, T, trunk_dim)
       │
       ▼
Bilinear Fusion: W_bilinear(branch ⊗ trunk) → (B, D, T, hidden_dim)
       +
Skip Paths: Linear(branch) + Linear(trunk)
       │
       ▼
LayerNorm → Residual Blocks × num_res_blocks
       │
       ▼
Position Head: Linear → GELU → Linear → clamp[0, 1]
       │
       ▼
positions: (B, D, T)
```

##### compute_boundaries

**Location**: `models/traj_deeponet.py`

Computes left and right boundary discontinuity positions for each grid point.

For each spatial point $x$ at time $t$ with $D$ discontinuities at positions $\{x_d(t)\}$:

$$x_{left}(t, x) = \max_{d : x_d(t) \leq x,\ m_d = 1} x_d(t) \quad \text{(or 0 if none)}$$

$$x_{right}(t, x) = \min_{d : x_d(t) > x,\ m_d = 1} x_d(t) \quad \text{(or 1 if none)}$$

**Input/Output**:
- Input: `positions` $(B, D, n_t)$, `x_coords` $(B, n_t, n_x)$, `disc_mask` $(B, D)$
- Output: `left_bound` $(B, n_t, n_x)$, `right_bound` $(B, n_t, n_x)$

##### BoundaryConditionedTrunk

**Location**: `models/traj_deeponet.py`

Single trunk that predicts density conditioned on boundary positions.

**Architecture**:
```
Inputs: t (B,nt,nx), x (B,nt,nx), x_left (B,nt,nx), x_right (B,nt,nx)
       │
       ▼
Fourier encode (shared spatial encoder for x, x_left, x_right):
  γ_t(t) with L_t frequencies
  γ_x(x), γ_x(x_left), γ_x(x_right) with L_x frequencies
       │
       ▼
Concatenate: [γ_t, γ_x, γ_x_left, γ_x_right]
       │
       ▼
Coord MLP: [Linear + GELU + LayerNorm] × num_layers → coord_emb
       │
       └──────────────────┐
                          ▼
branch_emb (B, h) ──► Bilinear Fusion + Skip Paths
                          │
                          ▼
                    LayerNorm → Residual Blocks × num_res_blocks
                          │
                          ▼
                    Density Head: Linear → GELU → Linear → Sigmoid
                          │
                          ▼
                    Output: (B, nt, nx) ∈ [0, 1]
```

#### Input/Output Format

**Input** (dictionary):
- `discontinuities`: $(B, D, 3)$ - initial shock features $[x_0, \rho_L, \rho_R]$
- `disc_mask`: $(B, D)$ - validity mask
- `t_coords`: $(B, 1, n_t, n_x)$ - query times
- `x_coords`: $(B, 1, n_t, n_x)$ - query positions

**Output** (dictionary):
- `positions`: $(B, D, T)$ - shock positions (all discontinuities persist)
- `output_grid`: $(B, 1, n_t, n_x)$ - predicted density grid

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

```python
def greenshields_flux(rho: torch.Tensor) -> torch.Tensor
```

#### greenshields_flux_derivative

$$f'(\rho) = 1 - 2\rho$$

```python
def greenshields_flux_derivative(rho: torch.Tensor) -> torch.Tensor
```

#### compute_shock_speed

From Rankine-Hugoniot condition:

$$s = \frac{f(\rho_R) - f(\rho_L)}{\rho_R - \rho_L} = 1 - \rho_L - \rho_R$$

```python
def compute_shock_speed(rho_L: torch.Tensor, rho_R: torch.Tensor) -> torch.Tensor
```

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

where:
- $\mathbb{1}_{outside}(x) = 1$ if $x < 0$ or $x > 1$, else $0$
- $e^{(b,d,t)}$ is the existence probability

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

where:
- $e^{(b,d)}_0$ = existence probability at t=0 for discontinuity d in batch b (if available)
- $x^{(b,d)}_{pred,0}$ = predicted position at t=0
- $x^{(b,d)}_{IC}$ = IC discontinuity position from `discontinuities[:, :, 0]`
- N = number of valid discontinuities (sum of `disc_mask`)

**Interpretation**:
- Without existence: all valid discontinuities are penalized equally for position errors at t=0
- With existence at t=0 high (~1): full penalty for position errors
- With existence at t=0 low (~0): position errors are ignored

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

Physics-informed loss enforcing the conservation law in smooth regions (away from shocks).

**Physical equation**:
$$\frac{\partial \rho}{\partial t} + \frac{\partial f(\rho)}{\partial x} = 0$$

**Residual computation** (central finite differences):
$$R(t, x) = \frac{\rho(t+\Delta t, x) - \rho(t-\Delta t, x)}{2\Delta t} + \frac{f(\rho(t, x+\Delta x)) - f(\rho(t, x-\Delta x))}{2\Delta x}$$

**Shock masking**:
Points within `shock_buffer` distance from predicted shocks are excluded:
$$\text{mask}(t, x) = \prod_{d=1}^{D} \left[ 1 - \mathbb{1}_{|x - x_d(t)| < \delta} \cdot \mathbb{1}_{e_d(t) > 0.5} \cdot m_d \right]$$

**Loss formula**:
$$\mathcal{L}_{PDE} = \frac{1}{|\mathcal{I}|} \sum_{(t,x) \in \mathcal{I}} R(t, x)^2 \cdot \text{mask}(t, x)$$

**Optional IC loss** (if `ic_weight > 0`):
$$\mathcal{L}_{total} = \mathcal{L}_{PDE} + w_{IC} \cdot \mathcal{L}_{IC}$$

**Configuration**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `dt` | 0.004 | Time step size |
| `dx` | 0.02 | Spatial step size |
| `shock_buffer` | 0.05 | Buffer around shocks for masking |
| `ic_weight` | 0.0 | Weight for IC loss (0 = disabled) |

**Required inputs**: `x_coords`, `disc_mask`
**Required outputs**: `output_grid`, `positions`, `existence`

**Components returned**: `{"pde_residual": float, "ic": float (if enabled), "total": float}`

---

#### RHResidualLoss

**Location**: `losses/rh_residual.py`

Rankine-Hugoniot residual loss computed from sampled densities. Verifies that the predicted shock velocity satisfies the RH condition using density values sampled from different sources.

**Shock velocity** (from predicted positions):
$$\dot{x}_d(t) = \frac{x_d(t + \Delta t) - x_d(t - \Delta t)}{2 \Delta t}$$

**Density sampling at shocks**:
- $\rho^-_d(t)$: density at $x = x_d(t) - \epsilon$ (left of shock)
- $\rho^+_d(t)$: density at $x = x_d(t) + \epsilon$ (right of shock)

**Rankine-Hugoniot residual**:
$$R_{RH}^{(d)}(t) = \dot{x}_d(t) \cdot (\rho^+_d - \rho^-_d) - (f(\rho^+_d) - f(\rho^-_d))$$

This should be zero if the shock velocity satisfies the RH condition.

**Loss formula**:
$$\mathcal{L}_{RH} = \frac{1}{|\mathcal{V}|} \sum_{(d,t) \in \mathcal{V}} e_d(t) \cdot \left( R_{RH}^{(d)}(t) \right)^2$$

**Sampling Modes**:

| Mode | Density Source | Use Case |
|------|----------------|----------|
| `per_region` | Sample $\rho^-$ from `region_densities[d]`, $\rho^+$ from `region_densities[d+1]` | Training HybridDeepONet. Gets unblended density values. |
| `pred` | Sample both from `output_grid` | Testing RH on assembled (blended) prediction. |
| `gt` | Sample both from `target` grid | Testing if predicted trajectories match GT physics. |

**Why `per_region` for training**: The `output_grid` is assembled using soft sigmoid boundaries that blend region densities near shocks. Sampling from `output_grid` would give blended values, not the true left/right densities. `per_region` mode samples directly from each region's prediction before blending.

**Configuration**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `dt` | 0.004 | Time step size for velocity computation |
| `epsilon` | 0.01 | Offset for density sampling near shocks |
| `mode` | `"per_region"` | Sampling mode: `"per_region"`, `"pred"`, or `"gt"` |

**Required inputs**: `x_coords`, `disc_mask`
**Required outputs** (depends on mode):
- `per_region`: `positions`, `existence`, `region_densities`
- `pred`: `positions`, `existence`, `output_grid`
- `gt`: `positions`, `existence` (uses `target` argument)

**Components returned**: `{"rh_residual": float}`

---

#### AccelerationLoss

**Location**: `losses/acceleration.py`

Penalizes low existence predictions where the ground truth grid shows high temporal acceleration, indicating a shock should be present. This loss helps the model learn to predict high existence at shock locations even without explicit trajectory supervision.

The loss has two components:
1. **Original term**: Samples acceleration at predicted trajectory positions and penalizes low existence where acceleration is high.
2. **Missed shock term** (optional): Scans the entire domain for high-acceleration points and penalizes those not covered by any nearby prediction with high existence.

**Acceleration computation** (central finite differences):
$$a(t, x) = \frac{\rho(t + \Delta t, x) - 2\rho(t, x) + \rho(t - \Delta t, x)}{\Delta t^2}$$

This is computed for interior time points only (indices 1 to $n_t - 2$).

##### Original Term

**Acceleration sampling near trajectories**:

For each predicted trajectory position $x_d(t)$, sample the maximum absolute acceleration in a spatial window:
$$a_{near}^{(d)}(t) = \max_{|x - x_d(t)| < \epsilon} |a(t, x)|$$

In practice, samples are taken at $x - \epsilon$, $x$, and $x + \epsilon$.

**Loss formula**:
$$\mathcal{L}_{accel} = \frac{1}{N} \sum_{(b,d,t) \in \mathcal{H}} \left(1 - e^{(b,d,t)}\right)^2 \cdot m_{b,d}$$

where:
- $\mathcal{H} = \{(b,d,t) : |a_{near}^{(d)}(t)| > \tau\}$ is the set of high-acceleration points
- $e^{(b,d,t)}$ = existence probability
- $\tau$ = acceleration threshold (configurable)
- $m_{b,d}$ = discontinuity validity mask
- $N = |\mathcal{H}|$ = count of high-acceleration points

##### Missed Shock Term

**Problem solved**: The original term only samples acceleration at **predicted** trajectory positions. If the model predicts positions away from actual shocks, those shocks go undetected because the loss never "looks" there.

**Coverage computation**:
$$\text{coverage}(b, t, x) = \max_d \left[ e^{(b,d,t)} \cdot m_{b,d} \cdot \mathbf{1}(|x - x_d(t)| < \delta) \right]$$

where $\delta$ is the `missed_shock_buffer` parameter.

**Loss formula**:
$$\mathcal{L}_{missed} = \frac{1}{M} \sum_{(b,t,x) \in \mathcal{H}_{domain}} \left(1 - \text{coverage}(b, t, x)\right)^2$$

where:
- $\mathcal{H}_{domain} = \{(b,t,x) : |a(t,x)| > \tau\}$ is the set of high-acceleration points in the entire domain
- $M = |\mathcal{H}_{domain}|$ = count of high-acceleration points

**Combined loss**:
$$\mathcal{L}_{total} = \mathcal{L}_{accel} + w_{missed} \cdot \mathcal{L}_{missed}$$

**Interpretation**:
- High acceleration in the ground truth indicates a shock (rapid density change)
- The original term: If the model predicts low existence ($e \approx 0$) at predicted locations with high acceleration, the loss is high
- The missed shock term: If there are high-acceleration points in the domain that are not covered by any predicted trajectory with high existence, the loss is high
- Regions with low acceleration (smooth solution) do not contribute to either term

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

| Loss | Weight |
|------|--------|
| `trajectory` | 1.0 |
| `boundary` | 1.0 |
| `collision` | 0.5 |
| `existence_reg` | 0.1 |

**Total loss**:
$$\mathcal{L} = \mathcal{L}_{traj} + \mathcal{L}_{bound} + 0.5 \cdot \mathcal{L}_{coll} + 0.1 \cdot \mathcal{L}_{reg}$$

#### hybrid Preset

For HybridDeepONet (trajectory + grid prediction).

| Loss | Weight |
|------|--------|
| `mse` | 1.0 |
| `rh_residual` | 1.0 |
| `pde_residual` | 0.1 |
| `ic` | 10.0 |
| `existence_reg` | 0.01 |

**Total loss**:
$$\mathcal{L} = \mathcal{L}_{MSE} + \mathcal{L}_{RH} + 0.1 \cdot \mathcal{L}_{PDE} + 10 \cdot \mathcal{L}_{IC} + 0.01 \cdot \mathcal{L}_{reg}$$

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

# Backwards compatibility: "rankine_hugoniot" maps to "shock_net"
loss = get_loss("rankine_hugoniot")  # Same as get_loss("shock_net")
```

---

## Summary

| **Model** | **Input** | **Output** | **Purpose** |
|-----------|-----------|-----------|-------------|
| ShockTrajectoryNet | Discontinuities $(B, D, 3)$ | Positions, Existence $(B, D, T)$ | Pure trajectory prediction |
| HybridDeepONet | Discontinuities + coordinates | Trajectories + Full grid | Combined trajectory + solution |
| TrajDeepONet | Discontinuities + coordinates | Positions + Full grid | Boundary-conditioned single trunk |

| **Loss** | **Location** | **Key Physics** | **Use Case** |
|----------|--------------|-----------------|--------------|
| MSELoss | `losses/mse.py` | Grid supervision | Grid matching |
| ICLoss | `losses/ic.py` | Initial condition | IC accuracy |
| TrajectoryConsistencyLoss | `losses/trajectory_consistency.py` | Analytical RH trajectories | Trajectory models |
| BoundaryLoss | `losses/boundary.py` | Domain constraints | All models |
| CollisionLoss | `losses/collision.py` | Shock merging | Multi-shock models |
| ICAnchoringLoss | `losses/existence_regularization.py` | Anchor trajectories to IC positions | All models |
| SupervisedTrajectoryLoss | `losses/supervised_trajectory.py` | Direct supervision | When GT available |
| PDEResidualLoss | `losses/pde_residual.py` | Conservation in smooth regions | Grid models |
| RHResidualLoss | `losses/rh_residual.py` | RH at shocks (from densities) | Hybrid models |
| AccelerationLoss | `losses/acceleration.py` | High acceleration = shock | Existence supervision |

### Key Design Principles

1. **DeepONet Architecture**: Branch-trunk factorization enables learning solution operators with variable-length inputs (variable shock counts)

2. **Soft Region Boundaries**: Differentiable sigmoid boundaries in GridAssembler enable end-to-end gradient flow

3. **Multi-Scale Physics**:
   - Shock scale: Rankine-Hugoniot residuals
   - Smooth scale: PDE conservation
   - Global scale: Grid MSE supervision

4. **Fourier Features**: Exponentially-spaced sinusoidal encoding enables MLPs to learn high-frequency shock dynamics (used in TimeEncoder, DiscontinuityEncoder, and SpaceTimeEncoder)

5. **Independent Discontinuity Encoding**: Each discontinuity is encoded independently via Fourier + MLP, enabling efficient parallel processing

6. **Modular Loss Design**: One file per loss enables easy composition and testing
