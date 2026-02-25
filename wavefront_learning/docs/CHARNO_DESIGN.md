# CharNO: Characteristic Neural Operator — Design Document

## 1. Problem Statement

We solve the 1D scalar conservation law

$$\frac{\partial \rho}{\partial t} + \frac{\partial f(\rho)}{\partial x} = 0, \quad x \in [0, 1],\; t \in [0, T]$$

with piecewise constant initial conditions $\rho(0, x) = \rho_k$ for $x \in [x_k, x_{k+1})$, and a given flux function $f$ (default: Greenshields $f(\rho) = \rho(1-\rho)$).

The solution consists of three types of structures:
- **Constant regions** where $\rho = \rho_k$ (one of the initial values)
- **Shock waves** — moving discontinuities where characteristics converge
- **Rarefaction fans** — smooth expansion waves where characteristics diverge

The ground truth is computed by the **Lax-Hopf exact solver**, which exploits a variational characterization of the entropy solution.

---

## 2. Why Existing Architectures Fall Short

### 2.1 Grid-based models (FNO, DeepONet, EncoderDecoder)

These models treat the PDE solution as a generic function mapping and ignore the hyperbolic structure entirely.

**FNO** operates in Fourier space, which is fundamentally ill-suited for discontinuities — the Fourier transform of a step function decays as $1/k$ (Gibbs phenomenon), requiring many modes to approximate and producing oscillatory artifacts near shocks. Truncating modes (as FNO does) necessarily smooths discontinuities.

**DeepONet** (branch-trunk dot product) lacks the capacity to represent sharp transitions because the dot product of smooth branch/trunk outputs is itself smooth. It has no mechanism to "select" between different solution regimes.

**EncoderDecoder** (transformer on discretized grids) processes the solution as a sequence of spatial tokens. While attention can in principle capture long-range dependencies, the model has no inductive bias about characteristic propagation and must learn the entire wave structure from data.

**Common failure mode**: All grid-based models minimize MSE, which penalizes a shock shifted by $\delta$ with error $O(|\Delta\rho|^2 \cdot \delta)$. The optimal MSE-minimizing approximation to a step function is a smooth sigmoid, not a sharp jump — so MSE training actively encourages smeared shocks.

### 2.2 Trajectory-based models (ShockNet, HybridDeepONet, TrajDeepONet)

These models introduce wave-awareness by explicitly predicting shock trajectories, then using them to construct the density field.

**Progress**: They correctly identify that the solution is piecewise smooth, separated by moving discontinuities. The trajectory prediction step encodes the Rankine-Hugoniot kinematics.

**Limitations**:

1. **Two-stage pipeline**: Predict trajectories first, then predict density conditioned on trajectories. Errors in trajectory prediction propagate to density prediction. The two objectives can conflict during training (a trajectory that's good for one loss may be bad for the other).

2. **Rarefaction handling is ad hoc**: Rarefaction fans are not trajectories — they are smooth regions bounded by characteristics. The trajectory models either ignore rarefactions entirely (ShockNet) or add a binary classifier to distinguish shocks from rarefactions (ClassifierTrajTransformer). Neither approach captures the smooth self-similar structure of rarefactions.

3. **Wave interactions are not modeled directly**: When two shocks merge, the trajectory models must learn this implicitly through the self-attention over discontinuity embeddings. But the trajectory prediction still produces separate curves for each initial discontinuity, with an existence probability to handle merging. This is a workaround, not a principled solution.

4. **Many auxiliary losses needed**: The trajectory models require 4-5 loss terms (MSE + IC anchoring + boundary + trajectory regularization + acceleration) to constrain the trajectory predictions. Each loss has hyperparameters that need tuning. This complexity is a signal that the architecture isn't well-matched to the problem.

### 2.3 The TrajTransformer family

The most advanced current models (TrajTransformer, ClassifierAllTrajTransformer) use cross-attention from spacetime queries to discontinuity embeddings. This is conceptually closer to what CharNO does, but still operates through the trajectory prediction bottleneck.

**DynamicDensityDecoder** (ClassifierAllTrajTransformer) is the closest precursor: it creates time-varying boundary tokens from discontinuity embeddings + predicted positions, and spatial queries cross-attend to these tokens. This captures the idea that the solution at (t,x) is determined by the wave structure at time t.

**What's still missing**: The cross-attention queries are raw Fourier-encoded coordinates (t, x) with no characteristic structure. The model must learn from data that the relevant geometric quantity is not the raw position, but the position *relative to the characteristic cone* of each wave. This is learnable in principle, but wastes model capacity on rediscovering known physics.

---

## 3. The Lax-Hopf Insight

For scalar conservation laws with convex or concave flux, the entropy solution has an exact variational characterization — the **Lax-Hopf formula**:

$$\rho(t, x) = -R'\left(\frac{x - y^*}{t}\right)$$

where $R$ is the Legendre transform of the flux, and $y^*$ minimizes the **value function** $M(t, x, y)$ over all initial spatial origins $y$:

$$y^* = \arg\min_y M(t, x, y)$$

For piecewise constant initial conditions with $K$ segments, this minimization reduces to comparing $K$ candidate values — one per segment. The segment that achieves the minimum "wins" and determines the solution at $(t, x)$.

**This means the solution at every point is a selection problem**: choose the correct initial segment, then compute the local value.

### 3.1 What the value function encodes

For segment $k$ with value $\rho_k$ on $[a_k, b_k]$, the Lax-Hopf value function $M_k(t, x)$ depends on:
- The cumulative initial data $N_k = \int_0^{b_k} \rho_0(y) dy$ (global information from all segments up to $k$)
- The Legendre transform $R\left(\frac{x - y}{t}\right)$ evaluated at the boundaries of segment $k$ (local characteristic geometry)

The first factor is why **self-attention over segments** is needed: the selection depends on the cumulative context, not just on local segment properties. The second factor is why **characteristic coordinates** are the right features: they encode the geometric relationship between the query point and the segment's characteristic fan.

### 3.2 Why this is a natural fit for attention

The Lax-Hopf argmin is mathematically analogous to the softmax-weighted attention:

| Lax-Hopf | CharNO |
|----------|--------|
| $K$ candidate segments | $K$ key-value pairs |
| Value function $M_k(t,x)$ | Score network output $s_k(t,x)$ |
| $\arg\min_k M_k$ | $\text{softmax}(-s_k / \tau)$ |
| Local solution from winning segment | Value network output $v_k(t,x)$ |
| Hard selection (exact) | Soft selection (differentiable) |

As the temperature $\tau \to 0$, the soft selection converges to the exact Lax-Hopf argmin. The softmax provides a differentiable relaxation that enables gradient-based training.

---

## 4. Architecture Design

### 4.1 Overview

CharNO has five stages that mirror the Lax-Hopf solution process:

```
IC segments → [Segment Encoder + Self-Attention] → contextualized segment embeddings
                                                        ↓
Query (t,x) → [Characteristic Feature Computer] → per-(query, segment) features
                                                        ↓
                                [Score Network] → softmin weights   (which segment wins?)
                                [Value Network] → local solutions   (what density value?)
                                                        ↓
                                Weighted sum → ρ(t,x)
```

### 4.2 Pluggable Flux Interface

**Design choice**: The flux function $f(\rho)$ is a parameter, not hardcoded.

**Justification**: The architecture's physics-informed features (characteristic speeds, shock speeds) depend on $f$. By abstracting the flux into a callable interface, the same architecture works for:
- Greenshields: $f(\rho) = \rho(1-\rho)$, $f'(\rho) = 1 - 2\rho$
- Triangular: $f(\rho) = \min(v_f \rho,\; w(1-\rho))$
- Any smooth concave/convex flux

The interface is minimal:

```
Flux.forward(rho)     → f(ρ)           # flux value
Flux.derivative(rho)  → f'(ρ)          # characteristic speed
Flux.shock_speed(ρ_L, ρ_R) → s         # Rankine-Hugoniot speed = [f]/[ρ]
```

The existing `losses/flux.py` already provides standalone functions (`greenshields_flux`, `greenshields_flux_derivative`, `compute_shock_speed`). The new Flux class generalizes these into a swappable object.

**Why not learn the flux?** The flux is known in our problem. Hardcoding it as an analytical formula provides exact physics features at zero parameter cost. If the flux were unknown, the same architecture could replace the analytical Flux with a learned MLP — the rest of the pipeline is unchanged.

### 4.3 Stage 1: Segment Encoder

**Input**: Piecewise constant IC described by breakpoints $\{x_k\}_{k=0}^K$ and values $\{\rho_k\}_{k=1}^K$.

**Design choice**: Operate on **constant segments** (pieces), not discontinuities.

**Justification**: The Lax-Hopf formula minimizes over *segments*, not over discontinuities. Each segment is a candidate "source" for the solution at any query point. Discontinuities are emergent — they appear at the boundaries between the domains of influence of adjacent segments. This is the opposite of the trajectory-based models, which start from discontinuities and work outward.

For each segment $k$, we compute physics-augmented features:

| Feature | Formula | Why included |
|---------|---------|--------------|
| $x_{center,k}$ | $(x_k + x_{k+1})/2$ | Spatial location of the segment |
| $w_k$ | $x_{k+1} - x_k$ | Segment extent (wider segments have more influence) |
| $\rho_k$ | $\text{ks}[k]$ | The density value (directly determines the solution in constant regions) |
| $\lambda_k$ | $f'(\rho_k)$ | Characteristic speed — determines how far and in which direction the segment's influence propagates |
| $f_k$ | $f(\rho_k)$ | Flux value — appears in the Rankine-Hugoniot condition at segment boundaries |

**Why Fourier features on spatial quantities?** The segment center and width determine the characteristic cone geometry. FourierFeatures with multiple octaves allow the MLP to represent both broad influence regions (low frequencies) and sharp transitions (high frequencies). This is the standard approach in the codebase (used by DiscontinuityEncoder, TimeEncoder, SpaceTimeEncoder).

**Why self-attention?** The Lax-Hopf value function for segment $k$ depends on the cumulative initial data $N_k = \sum_{j \leq k} \rho_j \cdot w_j$. This is inherently a *global* quantity that requires information from all preceding segments. Self-attention naturally captures this: each segment can attend to all others, enabling the embeddings to encode cumulative and contextual information.

Additionally, self-attention captures **interaction potential**: segments whose characteristic fans overlap will generate wave interactions. Self-attention allows the model to precompute which segments will interact, encoding this information into the embeddings before any query is processed.

**Number of layers**: 2 self-attention layers. The IC has at most ~10 segments (max_steps=4 gives up to 4 pieces), so the sequence is very short. Two layers suffice for full contextual mixing — with 4 tokens, 2 layers of attention can propagate information across the entire sequence.

**Reuses**: `FourierFeatures` from `models/base/feature_encoders.py`, `EncoderLayer` from `models/base/transformer_encoder.py`.

### 4.4 Stage 2: Characteristic Feature Computation

**Design choice**: For each query point $(t, x)$ and each segment $k$, compute **characteristic-relative coordinates** — analytical features that encode the geometric relationship in the space of characteristics.

**Justification**: This is the core inductive bias of CharNO. Instead of giving the model raw $(t, x)$ coordinates and asking it to learn the characteristic structure from data, we compute the physically meaningful coordinates directly.

#### Feature 1: Similarity variable $\xi_k = (x - x_{center,k}) / \max(t, \varepsilon)$

**Why**: Riemann problem solutions are **self-similar** — they depend on $x/t$, not on $x$ and $t$ independently. For a rarefaction fan emanating from position $x_0$, the solution is:

$$\rho(t, x) = (f')^{-1}\left(\frac{x - x_0}{t}\right)$$

For Greenshields: $\rho = (1 - (x-x_0)/t) / 2$.

By providing $\xi_k$ directly, the value network can learn the rarefaction solution as a simple function of a single variable, rather than discovering the self-similar structure from $(t, x)$ pairs.

**The $\max(t, \varepsilon)$ regularization** prevents division by zero at $t = 0$. At $t = 0$, the solution is just the initial condition — no characteristic propagation has occurred. The regularization is harmless because $\xi_k$ is most informative for $t > 0$.

#### Feature 2: Characteristic shift $\Delta_k = x - x_{center,k} - f'(\rho_k) \cdot t$

**Why**: A point $(t, x)$ lies on the characteristic emanating from segment $k$'s center if $\Delta_k = 0$. The sign and magnitude of $\Delta_k$ tell the model whether the query is to the left or right of the central characteristic, and by how much. This is the most direct encoding of "is this point in segment $k$'s domain of influence?"

For a constant region with no interactions, $\Delta_k \approx 0$ means the query is controlled by segment $k$. As $|\Delta_k|$ increases, the query is more likely controlled by a different segment.

#### Features 3-4: Distance to characteristic cone boundaries

$$d_{left,k} = x - (x_k + f'(\rho_k) \cdot t), \quad d_{right,k} = x - (x_{k+1} + f'(\rho_k) \cdot t)$$

**Why**: These measure how far the query is from the boundaries of the **characteristic fan** of segment $k$ — the region reachable from segment $k$ via characteristics traveling at speed $f'(\rho_k)$.

- If $d_{left,k} > 0$ and $d_{right,k} < 0$: the query is *inside* the characteristic fan → likely controlled by segment $k$
- If $d_{left,k} < 0$: the query is to the left of the fan → unlikely to be controlled
- If $d_{right,k} > 0$: the query is to the right → unlikely to be controlled

These features enable the score network to learn the domain-of-influence structure without discovering characteristic speeds from data.

**Important subtlety**: These distances use the characteristic speed $f'(\rho_k)$ of segment $k$, not the shock speed between adjacent segments. This is intentional — the characteristic fan is bounded by characteristics at speed $f'(\rho_k)$, while shocks travel at different speeds determined by the Rankine-Hugoniot condition. The self-attention in Stage 1 encodes the shock information into the segment embeddings, and the score network combines both sources.

#### Feature 5: Time $t$

**Why**: Wave interactions depend on time — shocks may merge, rarefactions may weaken. Including $t$ directly allows the model to learn time-dependent behavior. While $t$ is also implicitly encoded in the other features (which all involve $t$), providing it explicitly as a separate feature makes the time dependence easier to disentangle.

#### Processing

All 5 features are encoded via FourierFeatures (multi-octave sinusoidal encoding) and projected through a small MLP. The output has shape $(B, Q, K, H_{char})$ where $Q = n_t \cdot n_x$ is the number of query points and $K$ is the number of segments.

**Memory analysis**: For a typical grid ($n_t = 250$, $n_x = 50$, $K = 4$, $H_{char} = 32$, $B = 8$): the intermediate tensor is $8 \times 12500 \times 4 \times 32 \times 4$ bytes $\approx 50$ MB. This is well within GPU memory limits and comparable to other model intermediates.

### 4.5 Stage 3: Score Network (Lax-Hopf Selection)

**Design choice**: A small MLP that takes concatenated [segment embedding, characteristic features] and outputs a scalar score per (query, segment) pair, followed by a softmin (= softmax of negated scores) to produce selection weights.

**Justification**: The Lax-Hopf formula determines the solution by minimizing the value function $M_k(t, x)$ over segments. The score network learns an approximation to $M_k$:

$$s_k(t, x) = \text{ScoreMLP}([z_k; \phi_k(t, x)])$$

$$w_k(t, x) = \frac{\exp(-s_k / \tau)}{\sum_j \exp(-s_j / \tau)}$$

where $z_k$ is the contextualized segment embedding (from Stage 1) and $\phi_k(t, x)$ is the characteristic feature vector (from Stage 2).

**Why softmin instead of softmax?** The Lax-Hopf formula selects the segment with the *minimum* value function. Softmin($s$) = softmax($-s$) naturally implements this: the segment with the lowest score gets the highest weight.

**Learnable temperature $\tau$**: Stored as $\log \tau$ (exponentiated at forward time) to ensure $\tau > 0$.

- **Early training** ($\tau \approx 1$): Soft weights allow gradient flow to all segments, enabling the score network to calibrate
- **Late training** ($\tau \to 0$): Weights approach one-hot, recovering the exact Lax-Hopf selection

The model learns the optimal temperature jointly with the rest of the parameters. In practice, we expect $\tau$ to decrease during training as the score network becomes more confident.

**Why not use standard cross-attention?** Standard cross-attention computes $\text{softmax}(QK^T / \sqrt{d})$, where $Q$ depends only on the query and $K$ depends only on the key. But the Lax-Hopf value function $M_k(t, x)$ depends on *both* the query $(t, x)$ and the segment $k$ in a coupled way — specifically through the characteristic coordinates. The score network with concatenated inputs captures this coupling directly, while factored $QK^T$ attention cannot.

An alternative would be to add the characteristic features as an additive bias to the attention scores (like ALiBi or relative positional encoding). The MLP approach is more expressive but slightly more expensive. Given that $K$ is small ($\leq 10$), the cost is negligible.

### 4.6 Stage 4: Value Network (Local Solution)

**Design choice**: A small MLP that takes the same concatenated [segment embedding, characteristic features] and outputs a density value per (query, segment) pair, passed through sigmoid to constrain to $[0, 1]$.

**Justification**: Even after selecting the correct segment, the solution value is not always trivial:

- **In constant regions**: $\rho = \rho_k$ (the segment value). The value network should learn to output $\rho_k$ regardless of the query point. The segment embedding already contains $\rho_k$, so this is straightforward.

- **In rarefaction fans**: $\rho = (f')^{-1}(\xi_k)$ where $\xi_k$ is the similarity variable. For Greenshields: $\rho = (1 - \xi_k) / 2$. The value network should learn this as a function of $\xi_k$, which is provided directly in the characteristic features.

- **At shock locations**: The solution is discontinuous. On a discrete grid, the query falls on one side or the other — the score network's sharp selection handles this. The value network just needs to output the correct side's density.

**Why sigmoid output?** The density $\rho$ for LWR traffic flow is bounded in $[0, 1]$. The sigmoid constraint ensures physically valid outputs. This is consistent with the existing codebase (HybridDeepONet's RegionTrunk, TrajTransformer's density head all use clamping or sigmoid).

**Why share the input representation with the score network?** Both networks need the same information: which segment is this, and what's the geometric relationship to the query. Sharing the concatenated representation avoids redundant computation. The two MLPs specialize on different tasks: the score network learns "how likely is this segment to control this point" while the value network learns "what's the density if it does."

### 4.7 Stage 5: Output Assembly

$$\rho(t, x) = \sum_k w_k(t, x) \cdot v_k(t, x)$$

This is the differentiable Lax-Hopf approximation. The output is reshaped to $(B, 1, n_t, n_x)$ for compatibility with the existing evaluation pipeline.

**Selection weights for interpretability**: The weights $w_k(t, x)$ are returned as an auxiliary output. When visualized as a heatmap over the $(t, x)$ plane (with color indicating the dominant segment), they reveal the domain of influence of each initial segment — directly showing the characteristic structure and wave interactions. This is a unique feature of CharNO that no other model in the codebase provides.

---

## 5. What CharNO Does Not Do (and Why)

### 5.1 No explicit trajectory prediction

**Why not**: Trajectories are an intermediate representation that adds complexity without fundamental benefit. In CharNO, shock locations emerge naturally as the *boundaries between regions of different dominant segments* — the locus of points where $w_k \approx w_{k+1}$. These can be extracted from the selection weights if needed (e.g., for visualization), but they don't need to be predicted as a separate task.

**What this eliminates**: The trajectory prediction head, the existence classifier, the IC anchoring loss, the boundary loss, the trajectory regularization loss, and the compute_boundaries function. This is a significant simplification — CharNO needs only 2-3 loss terms instead of 4-5.

### 5.2 No per-region trunk networks

**Why not**: HybridDeepONet uses $K$ separate RegionTrunk networks (one per inter-shock region). This scales linearly with the number of regions and requires the GridAssembler for soft boundary blending. CharNO replaces this with a single value network applied to all (query, segment) pairs — the "per-region specialization" is handled by the segment embeddings and characteristic features, not by separate networks.

### 5.3 No grid-level operations (convolutions, spectral transforms)

**Why not**: CharNO operates per-query-point with attention to segments. It never convolves or transforms the spatial grid as a whole. This makes it naturally **resolution-invariant** — the same model works at any $(n_t, n_x)$ without retraining. Grid-level operators (FNO's spectral convolution, CNN refinement layers) are tied to a specific grid resolution.

---

## 6. Loss Design

### 6.1 The Problem with MSE for Discontinuous Solutions

Consider the true solution with a shock at position $x_0$ with left state $\rho_L$ and right state $\rho_R$:

$$\rho_{true}(x) = \begin{cases} \rho_L & x < x_0 \\ \rho_R & x \geq x_0 \end{cases}$$

And a predicted solution with the shock shifted by $\delta$:

$$\rho_{pred}(x) = \begin{cases} \rho_L & x < x_0 + \delta \\ \rho_R & x \geq x_0 + \delta \end{cases}$$

**MSE**: The error is $(\rho_R - \rho_L)^2$ on $\delta / \Delta x$ grid cells, giving $\text{MSE} \propto |\Delta\rho|^2 \cdot \delta / L$.

**Problem**: The gradient of MSE with respect to the shock position is proportional to $|\Delta\rho|^2$, which is *independent of $\delta$*. MSE provides the same gradient whether the shock is off by 1 cell or by 100 cells — it has no sense of "distance to the correct location." Moreover, the MSE-optimal smooth approximation to a step function is a sigmoid, not a sharp step. So MSE training actively discourages sharp shocks.

### 6.2 Wasserstein-1 Loss

The **Wasserstein-1 distance** (also called the Earth Mover's Distance or Kantorovich-Rubinstein distance) measures the minimum "cost of transport" to transform one distribution into another. For 1D functions, it has an exact closed-form that is trivial to compute.

**Definition**: For functions $f, g$ on $[0, L]$:

$$W_1(f, g) = \int_0^L \left| \int_0^x (f(y) - g(y))\, dy \right| dx = \int_0^L |F(x) - G(x)|\, dx$$

where $F, G$ are the antiderivatives (cumulative sums) of $f, g$.

**This is the $L^1$ norm of the antiderivative of the error** — equivalently, the Sobolev $W^{-1,1}$ norm (also called the flat norm or negative Sobolev norm).

**Discrete implementation**:

```python
diff = pred - target                          # (B, nt, nx)
cumulative = torch.cumsum(diff, dim=-1) * dx  # antiderivative along x
w1 = torch.mean(torch.abs(cumulative) * dx)   # L1 of antiderivative, averaged over t
```

**Computation**: $O(B \cdot n_t \cdot n_x)$ — same as MSE. Trivially differentiable via `torch.cumsum` and `torch.abs`.

#### Why W1 is the right metric for conservation law solutions

**Property 1: Linear shock sensitivity.** For a shock shifted by $\delta$, the cumulative error is a triangle of height $|\Delta\rho| \cdot \delta \cdot \Delta x$ and width $\delta$. The W1 loss is $O(|\Delta\rho| \cdot \delta^2 \cdot \Delta x^2 / L)$ — **quadratic in $\delta$**, providing stronger gradient signal for large displacements than MSE (which is linear in $\delta$).

Actually, let's be more precise. The cumulative difference is:
- Zero for $x < x_0$
- Linearly increasing (or decreasing) from 0 to $(\rho_L - \rho_R) \cdot \delta \cdot \Delta x$ for $x_0 \leq x < x_0 + \delta$
- Constant at $(\rho_L - \rho_R) \cdot \delta \cdot \Delta x$ for $x \geq x_0 + \delta$

So $W_1 \propto |\Delta\rho| \cdot \delta \cdot (L - \delta/2) \cdot \Delta x$. For small $\delta$, this is approximately $|\Delta\rho| \cdot \delta \cdot L \cdot \Delta x$, which is **linear in both $\delta$ and $|\Delta\rho|$** — compared to MSE's $|\Delta\rho|^2 \cdot \delta$.

The key advantage: W1 provides a gradient proportional to $|\Delta\rho| \cdot L$ for the shock location, which is a *directional* signal — it points toward the correct position. MSE's gradient is local (only non-zero at the error cells), while W1's gradient propagates through the entire cumulative sum.

**Property 2: No Gibbs phenomenon.** Because W1 doesn't penalize pointwise errors at shocks as harshly as MSE, the optimal W1 approximation to a step function is a step function (possibly shifted), not a smoothed sigmoid. This encourages the model to learn sharp shocks.

**Property 3: Mathematical foundation.** The stability of entropy solutions of conservation laws is naturally measured in $W^{-1,1}$, not in $L^2$. The classical result (e.g., Kružkov, Lucier) is that for scalar conservation laws:

$$\|\rho(t, \cdot) - \tilde{\rho}(t, \cdot)\|_{W^{-1,1}} \leq C \cdot \|\rho_0 - \tilde{\rho}_0\|_{W^{-1,1}}$$

This means W1 error is preserved by the PDE evolution — unlike $L^2$ error, which can grow. Training with the metric that matches the PDE's natural stability is fundamentally sound.

### 6.3 Conservation Loss

The conservation law $\partial_t \rho + \partial_x f(\rho) = 0$ implies that for solutions on $[0, L]$:

$$\frac{d}{dt} \int_0^L \rho(t, x)\, dx = -[f(\rho)]_0^L = f(\rho(t, 0)) - f(\rho(t, L))$$

For the common case where the boundary values are constant (outflow BCs), the right-hand side is constant, so the total mass changes linearly. For periodic BCs, mass is exactly conserved.

**Implementation**: Penalize the variance of total mass over time:

$$\mathcal{L}_{conservation} = \text{Var}_t\left(\sum_x \rho(t, x) \cdot \Delta x\right)$$

This is a **global constraint** that complements the local W1 and MSE losses. It prevents the model from creating or destroying mass, which is the most fundamental physical requirement of a conservation law.

**Cost**: $O(B \cdot n_t \cdot n_x)$ for the sum, plus $O(B \cdot n_t)$ for the variance. Trivially differentiable.

### 6.4 Combined Loss Preset

$$\mathcal{L} = \underbrace{\mathcal{L}_{MSE}}_{\text{smooth accuracy}} + \lambda_W \cdot \underbrace{\mathcal{L}_{W1}}_{\text{sharp shocks}} + \lambda_C \cdot \underbrace{\mathcal{L}_{conservation}}_{\text{mass conservation}}$$

with $\lambda_W = 0.5$, $\lambda_C = 0.1$.

**Why all three?**
- **MSE** ($\lambda = 1.0$): Provides strong gradient signal in smooth regions. Well-understood, stable training.
- **W1** ($\lambda = 0.5$): Provides correct geometric signal for shock placement. Encourages sharp discontinuities.
- **Conservation** ($\lambda = 0.1$): Lightweight global constraint. Small weight because it's a regularizer, not the primary objective.

**Why not W1 only?** Pure W1 might have weaker gradients in smooth regions (where the cumulative error is small). MSE provides stronger pointwise signal there. The combination gives the best of both.

**Why not PDE residual loss?** The PDE residual $|\partial_t \rho + \partial_x f(\rho)|^2$ is not well-defined at shocks (the PDE holds only in the distributional sense at discontinuities). Existing approaches mask out shock cells, but this creates a chicken-and-egg problem (need to know where shocks are to mask them). The conservation loss provides the physical constraint more robustly.

### 6.5 What CharNO's loss eliminates

Compared to existing model presets, CharNO's loss is drastically simpler:

| Existing losses (e.g. classifier_all_traj_transformer) | CharNO |
|---|---|
| MSE (1.0) | MSE (1.0) |
| IC anchoring (0.1) | *not needed* — no trajectory prediction |
| Boundary (1.0) | *not needed* — no trajectory prediction |
| Regularize trajectory (0.1) | *not needed* — no trajectory prediction |
| Acceleration (1.0) | *not needed* — no existence classifier |
| — | W1 (0.5) — **new** |
| — | Conservation (0.1) — **new** |
| **5 losses, 5 hyperparameters** | **3 losses, 2 hyperparameters** |

The two new losses (W1, Conservation) are physics-motivated and have clear mathematical justifications. The eliminated losses were all workarounds for the trajectory prediction pipeline.

---

## 7. Comparison with Related Work

### 7.1 RiemannONets (Peyvan et al., 2024)

RiemannONets use a modified DeepONet with two-stage training (first extract a basis from the trunk, then train the branch) for Riemann problems in compressible flow.

**Similarities**: Both operate on Riemann problem structure, both are neural operators.

**Differences**: RiemannONets use grid-based input (discretized IC), standard DeepONet architecture, and $L^2$ loss. They don't use characteristic coordinates and don't exploit the Lax-Hopf variational structure. CharNO's segment-based input, characteristic features, and softmin selection are all novel with respect to RiemannONets.

### 7.2 Transolver (Wu et al., ICML 2024)

Transolver introduces Physics-Attention, which decomposes the spatial domain into learnable "slices" where points with similar physical states share a token.

**Similarities**: Both decompose the domain into physically meaningful regions and use attention for aggregation.

**Differences**: Transolver's slices are *learned* from the data, while CharNO's segments come from the known IC structure. Transolver operates on discretized grids and uses standard coordinates; CharNO uses segment-based input with characteristic coordinates. Transolver is a general PDE solver; CharNO is specialized for hyperbolic conservation laws with much stronger inductive bias.

### 7.3 GoRINNs (2025)

Godunov-Riemann Informed Neural Networks combine shallow neural networks with Godunov-type finite volume schemes.

**Similarities**: Both embed Riemann problem structure into the learning framework.

**Differences**: GoRINNs learn numerical fluxes within a traditional finite volume loop, requiring sequential time stepping. CharNO predicts the full space-time solution in one forward pass.

### 7.4 CLINN (2025)

Conservation Law Informed Neural Network integrates conservation features (implicit solution, boundedness, jump conditions) into the loss function.

**Similarities**: Both enforce conservation law structure.

**Differences**: CLINN adds physics constraints to the loss of a standard PINN. CharNO embeds the physics into the *architecture* (characteristic coordinates, segment selection) and uses a simpler loss. Architecture-level inductive bias is generally stronger than loss-level regularization.

---

## 8. Properties and Expected Behavior

### 8.1 Resolution invariance

CharNO queries at arbitrary $(t, x)$ points. The model has no convolutional layers, no fixed grid structure, and no resolution-dependent parameters. The same trained model can be evaluated on a $50 \times 250$ grid, a $100 \times 500$ grid, or on scattered query points.

### 8.2 Interpretability

The selection weights $w_k(t, x)$ provide a direct window into the model's reasoning:

- **Before training**: Weights should be approximately uniform (all segments equally likely)
- **After training on simple ICs** (1 discontinuity): Weights should be near-binary, with a sharp transition at the shock trajectory
- **After training on complex ICs**: Weight boundaries should trace out the wave structure — shock trajectories where weights transition, rarefaction fan boundaries where weights smoothly blend

A *failed training* would show uniform or random weights. A *successful training* shows weights matching the exact Lax-Hopf domain-of-influence structure.

### 8.3 Parameter efficiency

Estimated parameter count with defaults ($H = 64$, $H_{char} = 32$):

| Component | Parameters |
|-----------|-----------|
| Segment encoder (MLP + FourierFeatures) | ~10K |
| Self-attention (2 layers, 4 heads) | ~33K |
| Characteristic feature MLP | ~8K |
| Score MLP | ~12K |
| Value MLP | ~12K |
| Temperature | 1 |
| **Total** | **~75K** |

This is comparable to TrajTransformer at hidden_dim=32 (~50-80K) and much smaller than FNO (~200K) or EncoderDecoder (~150K).

### 8.4 Computational complexity

| Operation | Cost |
|-----------|------|
| Segment encoding | $O(K^2 \cdot L_{self} \cdot H)$ |
| Characteristic features | $O(Q \cdot K \cdot H_{char})$ |
| Score + value networks | $O(Q \cdot K \cdot (H + H_{char}))$ |
| Softmin + weighted sum | $O(Q \cdot K)$ |
| **Total** | $O(Q \cdot K \cdot H)$ where $Q = n_t \cdot n_x$ |

With $K \leq 10$ and $H = 64$, this is very fast — linear in the number of query points with a small constant factor. No spectral transforms, no graph construction, no ODE integration.

---

## 9. Potential Risks and Mitigations

### 9.1 Selection collapse

**Risk**: All weights converge to $1/K$ (uniform), meaning the model ignores the score network and outputs the average of all local solutions.

**Mitigation**: (a) Initialize the score network with small weights so initial scores are near-zero, giving approximately uniform weights — this is the correct starting point. (b) The MSE loss should naturally drive the weights toward correct selection, as the average-of-all-segments prediction has high MSE. (c) If needed, add an entropy regularization: $\mathcal{L}_{entropy} = -\sum_k w_k \log w_k$ (penalize high-entropy = uniform weights).

### 9.2 Rarefaction fans

**Risk**: The value network fails to learn the smooth rarefaction structure.

**Mitigation**: The similarity variable $\xi_k$ is provided directly in the characteristic features. For Greenshields, the rarefaction solution is $\rho = (1 - \xi_k)/2$ — a simple linear function of $\xi_k$. A 2-layer MLP should learn this easily. If needed, add an explicit "rarefaction hint" feature: the inverse characteristic speed $(f')^{-1}(\xi_k)$ evaluated analytically.

### 9.3 Wave interactions at late times

**Risk**: When shocks merge (two shocks collide and become one), the domain-of-influence boundaries change discontinuously. The smooth softmin may not capture this.

**Mitigation**: (a) The self-attention in Stage 1 encodes interaction potential between segments. (b) The characteristic features naturally encode the geometry of interaction — when two characteristic cones overlap at time $t$, the features reflect this. (c) The softmin with low temperature $\tau$ can represent sharp transitions. (d) If interactions prove challenging, increase the self-attention depth (from 2 to 3-4 layers) to allow more interaction modeling.

### 9.4 Gradient flow through softmin at low $\tau$

**Risk**: As $\tau \to 0$, gradients through the softmin vanish (the argmin is not differentiable).

**Mitigation**: Use $\log \tau$ parameterization and optionally clamp $\tau \geq \tau_{min}$ (e.g., $\tau_{min} = 0.01$). In practice, the temperature should settle at a moderate value (not zero) because the training data includes rarefaction fans that benefit from soft selection.

---

## 10. Extensions (Not Implemented)

### 10.1 Systems of conservation laws

For 2×2 systems (e.g., shallow water $\partial_t [h, hu]^T + \partial_x [hu, hu^2 + gh^2/2]^T = 0$):
- Two characteristic families with speeds $u \pm \sqrt{gh}$
- Segment features would include both characteristic speeds
- Characteristic features would have separate coordinates for each family
- The score/value networks would be multi-headed (one per family)

### 10.2 Non-convex fluxes

For fluxes with inflection points (e.g., Buckley-Leverett), the solution can include compound waves (shock-rarefaction-shock). The architecture handles this naturally — the value network can learn arbitrary density profiles within each segment's domain, not just constants or simple rarefactions. The score network determines the domain boundaries.

### 10.3 Unknown flux

Replace the analytical Flux class with a learnable MLP: $f_\theta(\rho)$, $f'_\theta(\rho)$. The characteristic features become learned rather than analytical, but the architecture structure is unchanged. This requires more training data since the model must simultaneously learn the flux and the solution structure.
