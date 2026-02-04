# Wavefront Learning Architecture

This document describes the neural network architectures and loss functions used in the wavefront learning module for predicting shock trajectories and solutions in LWR traffic flow equations.

## Table of Contents

1. [Models](#models)
   - [ShockTrajectoryNet](#shocktrajectorynet)
   - [HybridDeepONet](#hybriddeepopnet)
2. [Losses](#losses)
   - [RankineHugoniotLoss](#rankinehugoniotloss)
   - [PDEResidualLoss](#pderesidualloss)
   - [HybridDeepONetLoss](#hybriddeepopnetloss)

---

## Models

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

Positional encoding for scalar inputs using sinusoidal features.

**Formula**:
$$\gamma(x) = \left[ \sin(2^0 \pi x), \cos(2^0 \pi x), \sin(2^1 \pi x), \cos(2^1 \pi x), \ldots, \sin(2^{L-1} \pi x), \cos(2^{L-1} \pi x) \right]$$

where $L$ = `num_frequencies` (default: 32).

**Output dimension**: $2L$ (or $2L + 1$ if including original input)

##### DiscontinuityEncoder (Branch Network)

Encodes variable-length discontinuity sequences using transformer self-attention.

**Architecture**:
```
Input: (B, D, 3) where each discontinuity = [x_position, rho_L, rho_R]
       │
       ▼
Linear(3 → hidden_dim) + GELU + LayerNorm
       │
       ▼
TransformerEncoder(num_layers, num_heads, dropout)
       │
       ▼
Linear(hidden_dim → output_dim)
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
- `num_heads`: 4
- `num_layers`: 2
- `dropout`: 0.1

##### TimeEncoder (Trunk Network)

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

**Location**: `models/region_trunk.py`

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

**Location**: `models/region_trunk.py`

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

Set of $K = \text{max\_discontinuities} + 1$ region trunks, one for each region.

**Output**: $(B, K, n_t, n_x)$ - stacked per-region density predictions

##### GridAssembler

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

## Losses

### RankineHugoniotLoss

**Location**: `losses/rankine_hugoniot.py`

Unsupervised physics-based loss for shock trajectory prediction using Rankine-Hugoniot jump conditions.

#### Physical Background

For the LWR traffic flow equation with Greenshields flux:
$$f(\rho) = \rho(1 - \rho)$$

The **Rankine-Hugoniot condition** states that the shock speed is:
$$s = \frac{f(\rho_R) - f(\rho_L)}{\rho_R - \rho_L}$$

For Greenshields flux, this simplifies to:
$$s = \frac{\rho_R(1-\rho_R) - \rho_L(1-\rho_L)}{\rho_R - \rho_L} = 1 - \rho_L - \rho_R$$

The **analytical trajectory** is therefore:
$$x(t) = x_0 + s \cdot t = x_0 + (1 - \rho_L - \rho_R) \cdot t$$

#### Loss Components

##### 1. Trajectory Consistency Loss

Enforces that predicted trajectories match the analytical Rankine-Hugoniot solution.

$$\mathcal{L}_{traj} = \frac{1}{|\mathcal{V}|} \sum_{(b,d,t) \in \mathcal{V}} \left( x_{pred}^{(b,d)}(t) - x_{analytical}^{(b,d)}(t) \right)^2$$

where:
- $\mathcal{V}$ is the set of valid (batch, discontinuity, time) indices
- $x_{analytical}^{(b,d)}(t) = x_0^{(b,d)} + (1 - \rho_L^{(b,d)} - \rho_R^{(b,d)}) \cdot t$

##### 2. Boundary Loss

Penalizes shocks that exist outside the domain $[0, 1]$.

$$\mathcal{L}_{bound} = \frac{1}{N} \sum_{b,d,t} \mathbb{1}_{outside}(x_{pred}^{(b,d,t)}) \cdot \left(e^{(b,d,t)}\right)^2$$

where:
- $\mathbb{1}_{outside}(x) = 1$ if $x < 0$ or $x > 1$, else $0$
- $e^{(b,d,t)}$ is the existence probability

##### 3. Collision Loss

Prevents simultaneous existence of overlapping shocks.

$$\mathcal{L}_{coll} = \frac{1}{N_{pairs}} \sum_{i < j} \mathbb{1}_{colliding}(x_i, x_j) \cdot e_i \cdot e_j$$

where:
- $\mathbb{1}_{colliding}(x_i, x_j) = 1$ if $|x_i - x_j| < \epsilon_{collision}$ (default: 0.02)

##### 4. Existence Regularization

Prevents existence from collapsing to trivial solutions.

$$\mathcal{L}_{reg} = \left( \bar{e} - 0.5 \right)^2$$

where $\bar{e}$ is the mean existence probability across all valid shocks.

#### Combined Loss

$$\mathcal{L}_{total} = w_{traj} \cdot \mathcal{L}_{traj} + w_{bound} \cdot \mathcal{L}_{bound} + w_{coll} \cdot \mathcal{L}_{coll} + w_{reg} \cdot \mathcal{L}_{reg}$$

**Default weights**:
- $w_{traj} = 1.0$
- $w_{bound} = 1.0$
- $w_{coll} = 0.5$
- $w_{reg} = 0.1$

---

### PDEResidualLoss

**Location**: `losses/pde_residual.py`

Physics-informed loss enforcing the conservation law in smooth regions (away from shocks).

#### Physical Background

In smooth regions, the LWR equation must satisfy:
$$\frac{\partial \rho}{\partial t} + \frac{\partial f(\rho)}{\partial x} = 0$$

where $f(\rho) = \rho(1 - \rho)$.

#### PDE Residual Computation

Using central finite differences on interior points:

$$\frac{\partial \rho}{\partial t} \approx \frac{\rho(t+\Delta t, x) - \rho(t-\Delta t, x)}{2 \Delta t}$$

$$\frac{\partial f}{\partial x} \approx \frac{f(\rho(t, x+\Delta x)) - f(\rho(t, x-\Delta x))}{2 \Delta x}$$

**Residual**:
$$R(t, x) = \frac{\partial \rho}{\partial t} + \frac{\partial f}{\partial x}$$

#### Shock Masking

The PDE residual is only valid in smooth regions. Near shocks, we apply a mask:

$$\text{mask}(t, x) = \prod_{d=1}^{D} \left[ 1 - \mathbb{1}_{|x - x_d(t)| < \delta} \cdot e_d(t) \cdot m_d \right]$$

where $\delta = 0.05$ is the shock buffer distance.

#### Loss Formula

$$\mathcal{L}_{PDE} = \frac{1}{|\mathcal{I}|} \sum_{(t,x) \in \mathcal{I}} R(t, x)^2 \cdot \text{mask}(t, x)$$

where $\mathcal{I}$ is the set of interior grid points.

**Default parameters**:
- $\Delta t = 0.004$
- $\Delta x = 0.02$
- $\delta = 0.05$

---

### HybridDeepONetLoss

**Location**: `losses/hybrid_loss.py`

Combined multi-scale loss for training HybridDeepONet with supervised grid matching and physics constraints.

#### Loss Components

##### 1. Grid MSE Loss

Supervised loss comparing predicted grid to target.

$$\mathcal{L}_{grid} = \frac{1}{n_t \cdot n_x} \sum_{t,x} \left( \rho_{pred}(t, x) - \rho_{target}(t, x) \right)^2$$

##### 2. Rankine-Hugoniot Residual Loss

Enforces Rankine-Hugoniot conditions using the model's own predictions (not analytical trajectories).

**Shock velocity** (from predicted positions):
$$\dot{x}_d(t) = \frac{x_d(t + \Delta t) - x_d(t - \Delta t)}{2 \Delta t}$$

**Density sampling at shocks**:
- $\rho^-_d(t) = \rho_{region_d}(t, x_d(t) - \epsilon)$ (left of shock)
- $\rho^+_d(t) = \rho_{region_{d+1}}(t, x_d(t) + \epsilon)$ (right of shock)

where $\epsilon = 0.01$.

**Rankine-Hugoniot residual**:
$$R_{RH}^{(d)}(t) = \dot{x}_d(t) \cdot (\rho^+_d - \rho^-_d) - (f(\rho^+_d) - f(\rho^-_d))$$

**Loss**:
$$\mathcal{L}_{RH} = \frac{1}{|\mathcal{V}|} \sum_{(d,t) \in \mathcal{V}} e_d(t) \cdot \left( R_{RH}^{(d)}(t) \right)^2$$

##### 3. Smooth Region Loss (Configurable)

The loss for smooth regions can be configured via the `smooth_loss_type` parameter:

###### Mode 1: `pde_residual` (default) - Unsupervised Physics

Same as [PDEResidualLoss](#pderesidualloss), applied to the output grid. Enforces the conservation law in regions away from shocks:

$$\mathcal{L}_{smooth} = \frac{1}{|\mathcal{S}|} \sum_{(t,x) \in \mathcal{S}} \left( \frac{\partial \rho}{\partial t} + \frac{\partial f(\rho)}{\partial x} \right)^2$$

where $\mathcal{S}$ is the set of smooth region points (masked by `create_shock_mask()`).

###### Mode 2: `supervised` - Supervised MSE in Smooth Regions

Computes MSE between prediction and target only in smooth regions, excluding points near shocks:

$$\mathcal{L}_{smooth} = \frac{1}{|\mathcal{S}|} \sum_{(t,x) \in \mathcal{S}} \left( \rho_{pred}(t, x) - \rho_{target}(t, x) \right)^2$$

The smooth region mask is computed using `create_shock_mask()` from `pde_residual.py`, which identifies points that are at least `shock_buffer` distance away from all predicted shock positions.

**Usage**:
```bash
# Default: PDE residual (unsupervised physics)
python train.py --model HybridDeepONet --loss hybrid

# Supervised MSE in smooth regions
python train.py --model HybridDeepONet --loss hybrid --smooth_loss_type supervised
```

##### 4. Existence Regularization

Same as in [RankineHugoniotLoss](#rankinehugoniotloss):
$$\mathcal{L}_{reg} = \left( \bar{e} - 0.5 \right)^2$$

#### Combined Loss

$$\mathcal{L}_{total} = w_{grid} \cdot \mathcal{L}_{grid} + w_{RH} \cdot \mathcal{L}_{RH} + w_{smooth} \cdot \mathcal{L}_{smooth} + w_{reg} \cdot \mathcal{L}_{reg}$$

**Default weights**:
- $w_{grid} = 1.0$ (primary supervised term)
- $w_{RH} = 1.0$ (physics at shocks)
- $w_{smooth} = 0.1$ (smooth region loss - PDE or supervised)
- $w_{reg} = 0.01$ (weak regularization)

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid_weight` | float | 1.0 | Weight for full grid MSE |
| `rh_weight` | float | 1.0 | Weight for RH residual |
| `smooth_weight` | float | 0.1 | Weight for smooth region loss |
| `reg_weight` | float | 0.01 | Weight for existence regularization |
| `ic_weight` | float | 10.0 | Weight for IC loss (within PDE loss) |
| `smooth_loss_type` | str | "pde_residual" | "pde_residual" or "supervised" |
| `dt` | float | 0.004 | Time step size |
| `dx` | float | 0.02 | Spatial step size |
| `shock_buffer` | float | 0.05 | Buffer around shocks for masking |
| `epsilon` | float | 0.01 | Offset for density sampling in RH |

---

## Summary

| **Model** | **Input** | **Output** | **Purpose** |
|-----------|-----------|-----------|-------------|
| ShockTrajectoryNet | Discontinuities $(B, D, 3)$ | Positions, Existence $(B, D, T)$ | Pure trajectory prediction |
| HybridDeepONet | Discontinuities + coordinates | Trajectories + Full grid | Combined trajectory + solution |

| **Loss** | **Type** | **Key Physics** |
|----------|----------|-----------------|
| RankineHugoniotLoss | Unsupervised | Analytical RH trajectories |
| PDEResidualLoss | Physics-informed | Conservation in smooth regions |
| HybridDeepONetLoss | Semi-supervised | Multi-scale: RH + PDE + supervision |

### Key Design Principles

1. **DeepONet Architecture**: Branch-trunk factorization enables learning solution operators with variable-length inputs (variable shock counts)

2. **Soft Region Boundaries**: Differentiable sigmoid boundaries in GridAssembler enable end-to-end gradient flow

3. **Multi-Scale Physics**:
   - Shock scale: Rankine-Hugoniot residuals
   - Smooth scale: PDE conservation
   - Global scale: Grid MSE supervision

4. **Fourier Features**: Exponentially-spaced sinusoidal encoding enables MLPs to learn high-frequency shock dynamics

5. **Transformer Branch**: Self-attention handles interactions between variable numbers of shocks
