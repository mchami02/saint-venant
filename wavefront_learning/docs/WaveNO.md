# WaveNO: Wavefront Neural Operator

A detailed analysis of the WaveNO architecture, its physics-informed design, and why it is particularly suited for hyperbolic conservation laws.

## Table of Contents

1. [Overview](#overview)
2. [The Problem: Learning Hyperbolic PDEs](#the-problem-learning-hyperbolic-pdes)
3. [Architecture](#architecture)
4. [Why WaveNO Is a Physics-Informed Neural Network](#why-waveno-is-a-physics-informed-neural-network)
5. [Strengths](#strengths)
6. [Comparison to Baselines](#comparison-to-baselines)

---

## Overview

WaveNO (Wavefront Neural Operator) is a neural operator architecture for learning the solution operator of scalar hyperbolic conservation laws of the form:

$$\frac{\partial \rho}{\partial t} + \frac{\partial f(\rho)}{\partial x} = 0$$

where $\rho(t, x)$ is the conserved quantity (e.g., traffic density) and $f(\rho)$ is the flux function (e.g., Greenshields flux $f(\rho) = \rho(1-\rho)$).

Given a piecewise-constant initial condition described by breakpoint positions $\{x_k\}_{k=0}^{K}$ and segment values $\{\rho_k\}_{k=1}^{K}$, WaveNO maps this compact representation directly to the full space-time solution grid $\rho(t, x) \in \mathbb{R}^{n_t \times n_x}$.

The key insight is that the solution to a hyperbolic conservation law is fully determined by the **wavefronts** (shocks and rarefactions) emanating from each initial discontinuity. Rather than tracking these wavefronts explicitly or selecting a "winning" segment via soft-argmax, WaveNO lets spatial query points **attend to all segments** via cross-attention, guided by a physics-informed attention bias derived from characteristic propagation.

---

## The Problem: Learning Hyperbolic PDEs

Hyperbolic conservation laws pose unique challenges for neural operators that do not arise with parabolic or elliptic PDEs:

### 1. Discontinuous Solutions (Shocks)

Unlike diffusive PDEs where solutions are smooth, hyperbolic PDEs develop **discontinuities in finite time** even from smooth initial data. The classical solution ceases to exist, and one must work with weak solutions satisfying the Rankine-Hugoniot jump condition:

$$s[\rho] = [f(\rho)] \quad \Longleftrightarrow \quad s = \frac{f(\rho_R) - f(\rho_L)}{\rho_R - \rho_L}$$

where $s$ is the shock speed and $\rho_L, \rho_R$ are the left and right states. Standard neural operators (FNO, DeepONet) that learn smooth mappings struggle with these sharp jumps — they either smear the discontinuity (Gibbs-like ringing in Fourier space) or fail to locate it correctly.

### 2. Finite Propagation Speed

Information in hyperbolic PDEs travels at **finite speed** along characteristics $\frac{dx}{dt} = f'(\rho)$. This means the solution at a point $(t, x)$ depends only on a finite **domain of dependence** determined by the characteristic speeds of the initial data. Standard architectures that process the entire spatial domain globally (e.g., Fourier convolution in FNO) ignore this locality structure.

### 3. Wave Interactions

Waves emanating from different discontinuities eventually **collide**, producing new waves whose behavior cannot be predicted from any single discontinuity in isolation. A model must capture both the independent propagation phase (before collision) and the interaction phase (after collision).

### 4. Entropy Condition

Among all weak solutions satisfying Rankine-Hugoniot, only the **entropy solution** is physically admissible. The Lax entropy condition requires that characteristics compress into the shock (information flows into the discontinuity, not out of it). A model that merely interpolates between states can produce non-physical "expansion shocks."

WaveNO addresses each of these challenges through its architecture design, as described below.

---

## Architecture

WaveNO is an 8-stage pipeline. The stages are:

### Stage 1: Segment Encoding

Each IC segment $k$ on interval $[x_k, x_{k+1}]$ is encoded with **physics-augmented features**:

- **Geometric**: center position $\bar{x}_k = (x_k + x_{k+1})/2$ and width $w_k = x_{k+1} - x_k$, both Fourier-encoded
- **Physical**: density $\rho_k$, characteristic speed $\lambda_k = f'(\rho_k)$, flux value $f_k = f(\rho_k)$
- **Global context**: cumulative mass $N_k = \sum_{j < k} \rho_j \cdot w_j$ (provides awareness of total mass distribution)

These features are projected through an MLP to produce segment embeddings $\mathbf{s}_k \in \mathbb{R}^H$.

### Stage 2: Segment Self-Attention

Segment embeddings pass through $L$ transformer encoder layers (self-attention + FFN), allowing segments to exchange information about the global IC structure. After this stage, each segment embedding "knows" about its neighbors — critical for predicting wave interactions.

### Stage 3: FiLM Time Conditioning

A **Feature-wise Linear Modulation** (FiLM) layer generates per-timestep scale and bias from Fourier-encoded time values:

$$\mathbf{s}_k^{(t)} = \gamma(t) \odot \mathbf{s}_k + \beta(t)$$

This produces time-evolved segment embeddings $\mathbf{s}_k^{(t)} \in \mathbb{R}^{n_t \times K \times H}$, followed by optional cross-segment attention for per-timestep segment refinement.

### Stage 4: Breakpoint Evolution

A `BreakpointEvolution` module predicts how IC breakpoints move over time:

1. **Breakpoint embeddings**: adjacent segment pairs $(\mathbf{s}_k, \mathbf{s}_{k+1})$ are concatenated and projected to create one embedding per discontinuity
2. **Cross-attention trajectory decoding**: time embeddings (queries) attend to breakpoint embeddings (keys/values) via a transformer decoder
3. **Position head**: outputs predicted breakpoint positions $p_d(t) \in [0, 1]$ for each discontinuity $d$ at each time $t$

The predicted trajectories are used by `compute_boundaries` to extract, for each query point $(t, x)$, the positions of the nearest left and right breakpoints $(x_L, x_R)$. This **local boundary context** is the mechanism that makes WaveNO invariant to the total number of IC segments (K-invariant).

### Stage 5: Query Encoding

Each query point $(t, x)$ is encoded using Fourier features of four quantities:

$$\mathbf{q}(t, x) = \text{MLP}\Big(\gamma(t) \,\|\, \gamma(x) \,\|\, \gamma(x_L) \,\|\, \gamma(x_R)\Big)$$

where $\gamma(\cdot)$ denotes Fourier positional encoding and $\|$ denotes concatenation. The boundary features $x_L, x_R$ tell each query "where am I relative to the nearest discontinuities?" — a resolution-invariant local coordinate.

### Stage 6: Characteristic Attention Bias

This is the core physics-informed component. For each query $(t, x)$ and segment $k$, WaveNO computes a **backward characteristic foot**:

$$y^* = x - f'(\rho_k) \cdot t$$

This is the point on the initial condition that would influence $(t, x)$ if segment $k$'s characteristic speed persisted. The distance of $y^*$ outside segment $k$'s interval determines the bias:

$$d_{\text{outside}} = \text{ReLU}(x_k - y^*) + \text{ReLU}(y^* - x_{k+1})$$

$$\text{bias}_{\text{raw}}(t, x, k) = -|\alpha| \cdot d_{\text{outside}}^2$$

where $\alpha$ is a learnable scale parameter (initialized at 5.0). The semantics are:

- **$\text{bias} = 0$** when $y^*$ falls inside segment $k$: the query is in $k$'s characteristic cone, so $k$ should receive full attention
- **$\text{bias} \ll 0$** when $y^*$ is far outside: $k$'s wave has not reached this query point, so attention is suppressed

#### Collision-Time Damping

The raw bias assumes independent wave propagation, which is only valid **before** waves interact. After collisions, the characteristic geometry changes and the raw bias becomes misleading. WaveNO handles this via a **collision-time damping** factor:

$$t_{\text{coll},k} = \min\left(\frac{w_{k-1}}{|\lambda_{k-1} - \lambda_k|},\; \frac{w_k}{|\lambda_k - \lambda_{k+1}|}\right)$$

$$\gamma(t, k) = \sigma\big(|\beta| \cdot (t_{\text{coll},k} - t)\big)$$

$$\text{bias}(t, x, k) = \text{bias}_{\text{raw}}(t, x, k) \cdot \gamma(t, k)$$

where $\beta$ is a learnable sharpness parameter. Before collision ($t < t_{\text{coll},k}$), $\gamma \approx 1$ and the physics bias is fully active. After collision ($t > t_{\text{coll},k}$), $\gamma \approx 0$ and the learned $QK^T$ attention scores take over. This two-phase design gives the model **exact physics guidance when available** and **learned flexibility when physics alone is insufficient**.

### Stage 7: Biased Cross-Attention

Spatial queries attend to time-evolved segment tokens via multi-head cross-attention with the additive physics bias:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + \text{bias}\right) V$$

This is analogous to ALiBi (Attention with Linear Biases) in NLP, but using wave propagation geometry instead of linear position offsets. The bias acts as a **prior**, not a hard constraint — the learned $QK^T$ scores can override it when necessary. Multiple biased cross-attention layers are stacked (default: 2).

### Stage 8: Density Head

The attended query embeddings are projected to scalar density values:

$$\hat{\rho}(t, x) = \text{clamp}\big(\text{MLP}(\mathbf{q}(t, x)),\; 0,\; 1\big)$$

The last layer is initialized near zero with bias 0.5 (neutral density), ensuring stable early training.

---

## Why WaveNO Is a Physics-Informed Neural Network

WaveNO qualifies as a PINN not through a PDE residual loss (though such losses are available), but through deep **structural** incorporation of the governing physics into the network architecture itself. This is a stronger form of physics-informedness than soft penalty terms, because the inductive biases are always active regardless of loss weighting.

### 1. Characteristic Propagation as Attention Bias

The most direct physics encoding. For a scalar conservation law $\partial_t \rho + \partial_x f(\rho) = 0$, the solution along characteristic curves $dx/dt = f'(\rho)$ is constant. WaveNO encodes this via the backward characteristic foot $y^* = x - f'(\rho_k) \cdot t$, which determines which IC segment "should" influence each query point. This is not a learned heuristic — it is the **exact method of characteristics** for the independent-wave regime, embedded as an attention prior.

### 2. Flux-Aware Feature Encoding

The `SegmentPhysicsEncoder` computes $f'(\rho_k)$ (characteristic speed) and $f(\rho_k)$ (flux value) for each segment using the pluggable `Flux` interface. These are **derived physical quantities**, not raw features. The model receives the same information a classical solver would use to determine wave behavior.

### 3. Rankine-Hugoniot via Breakpoint Evolution

The breakpoint trajectory prediction module learns to predict shock positions $p_d(t)$ — the same quantity governed by the Rankine-Hugoniot condition. While the trajectories are learned rather than analytically computed, the architecture (cross-attention from time queries to discontinuity embeddings) mirrors the physical structure: each discontinuity evolves according to its local states, and the time query asks "where is this discontinuity at time $t$?"

### 4. Collision-Time Damping Encodes Wave Interaction Physics

The analytical collision time $t_{\text{coll},k} = w_k / |\Delta\lambda|$ is derived from the geometry of characteristic convergence. It tells the model **when** its characteristic-based prior becomes invalid — precisely the moment waves interact and the simple method of characteristics breaks down. This is a physics-informed transition from analytical guidance to learned behavior.

### 5. Cumulative Mass as a Conservation Encoding

The cumulative mass feature $N_k = \sum_{j < k} \rho_j w_j$ encodes the total mass to the left of each segment. Since the PDE is a **conservation** law, total mass is an invariant. Providing this feature helps the model respect conservation without an explicit constraint.

### 6. Pluggable Flux Interface

WaveNO accesses physics through an abstract `Flux` class requiring only `f(\rho)$, $f'(\rho)$, and $s(\rho_L, \rho_R)`. This means the entire physics-informed machinery (characteristic bias, collision times, segment features) works for **any** scalar conservation law — Greenshields, triangular, or otherwise — by swapping the flux. The physics is not hard-coded to a specific equation.

### Structural vs. Loss-Based Physics

Most PINNs enforce physics via a **soft loss penalty**: $\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda \cdot \mathcal{L}_{\text{PDE}}$. This approach has well-documented failure modes for hyperbolic PDEs (the PDE residual is undefined at shocks, gradient pathologies from competing loss terms, sensitivity to $\lambda$).

WaveNO instead encodes physics **structurally**:
- The attention bias physically restricts which segments can influence which query points
- Collision-time damping physically transitions from analytical to learned behavior
- Flux derivatives physically determine propagation speeds

These structural priors are active unconditionally — they do not compete with a data loss, do not require tuning a penalty weight, and do not break down at discontinuities. The model can be trained with pure MSE loss because the architecture already "knows" the wave structure.

---

## Strengths

### 1. Resolution Generalization

WaveNO's representation is **resolution-invariant** by construction:

- **Fourier positional encodings** have fixed output dimension regardless of the grid density — adding more query points at test time simply produces more Fourier vectors, not a different-dimensional input
- **Characteristic bias** is computed from continuous $(t, x)$ values and characteristic speeds, not grid indices
- **Local boundary features** $(x_L, x_R)$ are continuous positions extracted from predicted trajectories

The model trained at $n_x = 50$ can be evaluated at $n_x = 100$ without any architectural change. The same attention weights, biases, and density head apply at arbitrary resolution.

### 2. K-Invariance (Generalization Across IC Complexity)

Through breakpoint evolution and local boundary context, WaveNO generalizes across ICs with different numbers of segments. A query at $(t, x)$ encodes "where are my nearest boundaries?" rather than "how many boundaries exist globally." This means a model trained on 4-piece ICs can handle 10+ piece ICs at test time — a critical capability for real-world problems where IC complexity varies.

### 3. Continuous Gradient Flow

Previous approaches (CharNO) used softmin-based segment selection, which concentrates gradients on the "winning" segment and starves others. WaveNO's cross-attention allows **all segments to contribute** with smooth, well-conditioned gradients. The physics bias acts as a prior guiding attention, not a hard gate cutting gradient flow. This eliminates the need for temperature scheduling or auxiliary selection supervision.

### 4. Graceful Degradation After Wave Collision

The collision-time damping mechanism gives WaveNO a principled two-phase behavior:

- **Before collision** ($t < t_{\text{coll}}$): the characteristic bias provides exact physics guidance, and the model benefits from strong inductive bias
- **After collision** ($t > t_{\text{coll}}$): the bias fades, and the learned $QK^T$ attention takes over to model complex post-interaction dynamics

This is strictly better than having no damping (bias becomes misleading after collision, hurting generalization to longer time horizons) or having no bias at all (no physics guidance for the easy independent-propagation phase).

### 5. Interpretability

The characteristic bias tensor $\text{bias}(t, x, k) \in \mathbb{R}^{B \times n_t \times n_x \times K}$ is a diagnostic output that can be visualized directly. It shows, for each point in the space-time domain, which IC segments the model considers relevant — a direct window into the model's "reasoning" about wave propagation. This is far more interpretable than the learned features of an FNO or vanilla DeepONet.

### 6. Modularity and Extensibility

WaveNO is composed of independently meaningful components:

| Component | Physical role | Can be swapped/ablated |
|-----------|--------------|----------------------|
| `Flux` | Defines the conservation law | Any flux function |
| `SegmentPhysicsEncoder` | IC representation | With/without cumulative mass |
| `BreakpointEvolution` | Shock tracking | Enable/disable (WaveNO vs WaveNOBase) |
| Characteristic bias | Wave propagation prior | Adjustable scale, damping |
| `CollisionTimeHead` | Post-interaction transition | Analytical or learned |
| Density head | Solution reconstruction | Can add proximity head (ShockAwareWaveNO) |

This modularity enables controlled ablation studies (WaveNOBase, WaveNOLocal, WaveNODisc, etc.) and extension to new physics without rewriting the core architecture.

---

## Comparison to Baselines

### FNO (Fourier Neural Operator)

FNO applies spectral convolutions in Fourier space, which is natural for smooth periodic solutions but fundamentally mismatched with shocks. Discontinuities require infinitely many Fourier modes to represent exactly, and truncation produces Gibbs oscillations. FNO also processes the entire spatial domain globally in each layer, ignoring the finite propagation speed of hyperbolic PDEs. WaveNO's attention mechanism is inherently local (guided by characteristic cones) and operates on the compact IC representation rather than discretized grids.

### DeepONet

DeepONet learns a branch-trunk factorization of the solution operator. It is expressive but has no physics priors — the branch network must learn wave propagation, shock formation, and entropy selection purely from data. WaveNO provides these as architectural inductive biases, reducing the learning burden and improving data efficiency.

### CharNO (Characteristic Neural Operator)

CharNO, WaveNO's predecessor, uses the same IC representation and characteristic-speed features but selects the "winning" segment via a softmin operation. This creates gradient concentration issues (only the selected segment receives strong gradients), requires temperature scheduling, and the selection becomes unreliable after wave collisions. WaveNO replaces softmin with cross-attention + physics bias, providing continuous gradients, no temperature tuning, and collision-time damping.

### WaveFrontModel (Explicit Wave Tracking)

The WaveFrontModel takes the opposite extreme — it explicitly constructs waves (shock/rarefaction), handles collisions via hand-crafted rules, and reconstructs the grid analytically. This is highly accurate when it works but brittle: adding new wave interaction types requires manual implementation, and the straight-through estimator for shock/rarefaction classification can be unstable. WaveNO achieves comparable accuracy with a fully differentiable, learned architecture that naturally extends to more complex scenarios.

### Summary Table

| Property | FNO | DeepONet | CharNO | WaveFrontModel | **WaveNO** |
|----------|-----|----------|--------|----------------|------------|
| Shock handling | Gibbs oscillations | Learned (no prior) | Softmin selection | Explicit construction | Physics-biased attention |
| Physics encoding | None | None | Characteristic features | Full analytical | Characteristic bias + flux features |
| Gradient flow | Smooth | Smooth | Concentrated (softmin) | STE (noisy) | Smooth (cross-attention) |
| Resolution generalization | Fixed grid | Fixed grid | Fourier + softmin | Resolution-free | Fourier + local boundaries |
| K-invariance | N/A (grid input) | N/A (grid input) | No | Yes (per-disc) | Yes (local boundaries) |
| Post-collision behavior | Global (no distinction) | Global (no distinction) | Bias becomes wrong | Explicit collision handling | Damped bias + learned attention |
| Interpretability | Low (spectral) | Low (latent) | Medium (selection weights) | High (explicit waves) | High (attention bias maps) |
| Extensibility | New architecture needed | New architecture needed | New features needed | New collision rules needed | Swap `Flux` interface |
