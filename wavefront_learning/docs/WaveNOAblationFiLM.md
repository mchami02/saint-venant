# WaveNO Ablation: FiLM Time Conditioning

A complete specification of the `WaveNOAblationFiLM` variant — a controlled ablation of WaveNO that isolates the contribution of FiLM (Feature-wise Linear Modulation) time conditioning on top of the characteristic-biased attention baseline.

## Table of Contents

1. [Position in the Ablation Ladder](#position-in-the-ablation-ladder)
2. [Problem Statement](#problem-statement)
3. [Architecture Overview](#architecture-overview)
4. [Stage-by-Stage Specification](#stage-by-stage-specification)
5. [Deep Dive: FiLM Time Conditioning](#deep-dive-film-time-conditioning)
6. [Why FiLM Works for Hyperbolic Conservation Laws](#why-film-works-for-hyperbolic-conservation-laws)
7. [Comparison with Adjacent Ablations](#comparison-with-adjacent-ablations)
8. [Default Configuration](#default-configuration)

---

## Position in the Ablation Ladder

WaveNOMinimal is a stripped-down version of WaveNO designed for controlled ablation experiments. It removes the trajectory prediction, boundary extraction, and boundary Fourier features of the full WaveNO, isolating the core attention-based density prediction pipeline. Four boolean flags toggle individual components:

| Ablation name | `use_char_bias` | `use_damping` | `use_film` | `use_cross_seg_attn` |
|---|---|---|---|---|
| `WaveNOAblation` (bare) | off | off | off | off |
| `WaveNOAblationBias` | **on** | off | off | off |
| `WaveNOAblationDamp` | **on** | **on** | off | off |
| **`WaveNOAblationFiLM`** | **on** | **on** | **on** | off |
| `WaveNOAblationCrossAttn` | **on** | **on** | off | **on** |
| `WaveNOAblationFull` | **on** | **on** | **on** | **on** |

`WaveNOAblationFiLM` is the first variant where segment embeddings become **time-dependent**. All prior variants use static segment embeddings that are identical across all timesteps — time dependence enters only through the query encoding and the characteristic bias. FiLM adds a direct mechanism for the segment representations themselves to evolve with time.

---

## Problem Statement

We solve the scalar hyperbolic conservation law:

$$\frac{\partial \rho}{\partial t} + \frac{\partial f(\rho)}{\partial x} = 0, \qquad \rho(0, x) = \rho_0(x)$$

where $\rho(t, x) \in [0, 1]$ is the conserved quantity (e.g., traffic density) and $f(\rho)$ is the flux function. The initial condition $\rho_0$ is piecewise constant:

$$\rho_0(x) = \rho_k \quad \text{for } x \in [x_k, x_{k+1}), \quad k = 1, \ldots, K$$

The model receives the compact IC representation — breakpoint positions $\{x_k\}_{k=0}^K$ and segment values $\{\rho_k\}_{k=1}^K$ — and predicts the full space-time density grid $\hat{\rho}(t, x) \in \mathbb{R}^{n_t \times n_x}$.

The default flux is Greenshields: $f(\rho) = \rho(1 - \rho)$, with derivative $f'(\rho) = 1 - 2\rho$ and shock speed $s(\rho_L, \rho_R) = 1 - \rho_L - \rho_R$.

---

## Architecture Overview

WaveNOAblationFiLM is a 6-stage pipeline:

```
Input: xs (K+1,), ks (K,), pieces_mask (K,), t_coords (nt, nx), x_coords (nt, nx)

Stage 1: Segment Encoding
    xs, ks → SegmentPhysicsEncoder → seg_emb (K, H)
    seg_emb → [EncoderLayer × L_self] → seg_emb (K, H)

Stage 2: Query Encoding
    t_coords, x_coords → FourierFeatures → MLP → query_emb (nt, nx, H)

Stage 3: FiLM Time Conditioning  ← THE KEY ADDITION
    seg_emb (K, H) + t_unique (nt,) → TimeConditioner → kv (nt, K, H)

Stage 4: Characteristic Attention Bias
    t_coords, x_coords, xs, ks → compute_characteristic_bias → bias (nt, nx, K)
    bias *= collision_time_damping(t, t_coll)

Stage 5: Biased Cross-Attention
    q = query_emb (nt*nx per timestep → reshaped to (nt, nx, H) → (nt, nx, H))
    kv = time-conditioned segment embeddings (nt, K, H)
    Per timestep: q(nx, H) attends to kv(K, H) with bias(nx, K)
    → [BiasedCrossDecoderLayer × L_cross] → attended (nt, nx, H)

Stage 6: Density Head
    attended → MLP → clamp(0, 1) → output_grid (1, nt, nx)
```

---

## Stage-by-Stage Specification

### Stage 1: Segment Encoding + Self-Attention

**Component**: `SegmentPhysicsEncoder` + `EncoderLayer` stack

**Purpose**: Encode each IC segment as a physics-rich embedding and let segments exchange information about the global IC structure.

**Input**:
- `xs`: $(K+1)$ breakpoint positions in $[0, 1]$
- `ks`: $(K)$ segment density values in $[0, 1]$
- `pieces_mask`: $(K)$ binary validity mask

**Physics feature computation** (per segment $k$):

| Feature | Formula | Meaning |
|---------|---------|---------|
| $\bar{x}_k$ | $(x_k + x_{k+1}) / 2$ | Segment center position |
| $w_k$ | $x_{k+1} - x_k$ | Segment width |
| $\rho_k$ | `ks[k]` | Segment density value |
| $\lambda_k$ | $f'(\rho_k)$ | Characteristic speed |
| $f_k$ | $f(\rho_k)$ | Flux value |
| $N_k$ | $\sum_{j < k} \rho_j \cdot w_j$ / total mass | Normalized cumulative mass |

The spatial features ($\bar{x}_k$, $w_k$) are Fourier-encoded via `FourierFeatures`:

$$\gamma(v) = \left[v, \sin(2\pi \cdot 1 \cdot v), \cos(2\pi \cdot 1 \cdot v), \ldots, \sin(2\pi \cdot F \cdot v), \cos(2\pi \cdot F \cdot v)\right]$$

producing a vector of dimension $2F + 1$ per scalar (default $F = 8$, so 17 per scalar).

All features are concatenated and projected through a 2-layer MLP (Linear → GELU → Dropout → LayerNorm → Linear):

$$\mathbf{s}_k = \text{MLP}\Big(\gamma(\bar{x}_k) \;\|\; \gamma(w_k) \;\|\; [\rho_k, \lambda_k, f_k, N_k]\Big) \in \mathbb{R}^H$$

**Self-attention**: The $K$ segment embeddings then pass through $L_{\text{self}}$ transformer encoder layers (default 2). Each layer is post-norm:

$$\mathbf{S}' = \text{LayerNorm}\left(\mathbf{S} + \text{MHA}(\mathbf{S}, \mathbf{S}, \mathbf{S})\right)$$
$$\mathbf{S}'' = \text{LayerNorm}\left(\mathbf{S}' + \text{FFN}(\mathbf{S}')\right)$$

where FFN is Linear$(H \to 4H)$ → GELU → Linear$(4H \to H)$.

After self-attention, each segment embedding "knows" about its neighbors — it encodes not just local properties but inter-segment relationships (density jumps, relative positions, potential collisions). Padded segments are re-zeroed after self-attention.

**Output**: `seg_emb` $(K, H)$ — static, time-independent segment embeddings.

---

### Stage 2: Query Encoding

**Component**: `FourierFeatures` (time) + `FourierFeatures` (space) + MLP

**Purpose**: Encode each space-time query point $(t, x)$ as a fixed-size embedding.

**Computation**:

$$\mathbf{q}(t, x) = \text{MLP}\Big(\gamma_t(t) \;\|\; \gamma_x(x)\Big) \in \mathbb{R}^H$$

where:
- $\gamma_t$ uses $F_t$ Fourier frequencies (default 8) → output dim $2F_t + 1 = 17$
- $\gamma_x$ uses $F_x$ Fourier frequencies (default 8) → output dim $2F_x + 1 = 17$
- MLP: Linear$(34 \to H)$ → ReLU → Dropout → Linear$(H \to H)$

All $n_t \times n_x$ query points are encoded in parallel (flattened, processed, reshaped).

**Note**: Unlike the full WaveNO, this variant does **not** include boundary features ($x_L$, $x_R$) in the query encoding because there is no trajectory prediction module to provide them. The query knows its absolute $(t, x)$ position but not its position relative to nearby discontinuities.

**Output**: `query_emb` $(n_t, n_x, H)$

---

### Stage 3: FiLM Time Conditioning

**Component**: `TimeConditioner`

**Purpose**: Transform static segment embeddings into time-dependent key-value tokens for cross-attention. This is the distinguishing feature of this ablation variant.

**The core idea**: A segment with density $\rho_k = 0.3$ has characteristic speed $\lambda_k = 0.4$ and its wavefront moves rightward. At $t = 0$, the segment's "influence region" is $[x_k, x_{k+1}]$. At $t = 0.5$, that region has shifted and possibly been modified by wave interactions. The static embedding $\mathbf{s}_k$ captures the segment's identity but not its time-evolving state. FiLM modulates $\mathbf{s}_k$ per timestep to encode this evolution.

**Architecture of `TimeConditioner`**:

```
Input: seg_emb (K, H), t_unique (nt,)

1. Fourier-encode time: t_unique → γ_t(t) → (nt, F_t)       where F_t = 2·num_freq+1
2. Expand for (time, segment) pairs:
     t_enc  → (nt, K, F_t)    (broadcast over segments)
     seg_emb → (nt, K, H)     (broadcast over timesteps)
3. Concatenate: [t_enc, seg_emb] → (nt, K, F_t + H)
4. FiLM network (MLP):
     Linear(F_t + H, H) → GELU → Dropout → Linear(H, 2H)
     → film_params (nt, K, 2H)
5. Split: γ(t,k), β(t,k) = chunk(film_params)   each (nt, K, H)
6. Modulate: kv(t,k) = γ(t,k) ⊙ seg_emb(k) + β(t,k)

Output: kv (nt, K, H)
```

**Critical initialization**: The final linear layer of `film_net` is initialized to:
- Weights: all zeros
- Bias: first $H$ values (γ) set to 1.0, last $H$ values (β) set to 0.0

This ensures that at initialization:
$$\gamma(t, k) = \mathbf{1}, \quad \beta(t, k) = \mathbf{0} \quad \Longrightarrow \quad \mathbf{kv}(t, k) = \mathbf{s}_k$$

The model starts **exactly equivalent** to the non-FiLM variant and gradually learns time modulation during training. This identity initialization is essential — it prevents the randomly initialized FiLM layer from destroying the pre-existing segment representations early in training.

**What γ and β learn**: The FiLM network receives both the Fourier-encoded time and the segment embedding as input. This means:
- $\gamma$ and $\beta$ are **segment-specific**: different segments get different modulations at the same time
- $\gamma$ and $\beta$ are **time-dependent**: the same segment gets different modulations at different times
- The conditioning is **content-aware**: modulation depends on what the segment represents (its density, speed, position, width, and contextual information from self-attention), not just its index

**Dimensionality**: FiLM operates on the $n_t \times K$ grid of (timestep, segment) pairs, which is much smaller than the full $n_t \times n_x$ query grid. For typical values ($n_t = 250$, $K = 4$, $n_x = 50$), this is 1,000 pairs vs. 12,500 query points — a 12.5x reduction.

**Output**: `kv` $(n_t, K, H)$ — time-conditioned segment embeddings that serve as keys and values for cross-attention in Stage 5.

---

### Stage 4: Characteristic Attention Bias

**Component**: `compute_characteristic_bias` function

**Purpose**: Compute a physics-informed additive bias for the cross-attention in Stage 5 that encodes which IC segments should influence which space-time query points, based on characteristic wave propagation.

**Backward characteristic foot**: For each query $(t, x)$ and segment $k$ with characteristic speed $\lambda_k = f'(\rho_k)$, trace backward along the characteristic:

$$y_k^* = x - \lambda_k \cdot t$$

This is the point on the $t = 0$ axis that would be connected to $(t, x)$ by segment $k$'s characteristic line. If $y_k^*$ falls inside segment $k$'s interval $[x_k, x_{k+1}]$, then segment $k$ is the correct "source" for this query point (in the independent-wave regime).

**Distance outside interval**:

$$d_k = \text{ReLU}(x_k - y_k^*) + \text{ReLU}(y_k^* - x_{k+1})$$

This is zero when $y_k^* \in [x_k, x_{k+1}]$ and positive otherwise.

**Raw bias**:

$$\text{bias}_{\text{raw}}(t, x, k) = -|\alpha| \cdot d_k^2$$

where $\alpha$ is a learnable scalar parameter (initialized at 5.0). The quadratic penalty creates a smooth, differentiable suppression:
- **Inside characteristic cone** ($d_k = 0$): bias = 0, full attention allowed
- **Outside characteristic cone** ($d_k > 0$): large negative bias, attention suppressed
- The quadratic (not linear) decay means the suppression grows rapidly with distance, creating a sharp but smooth boundary

**Collision-time damping**: The backward characteristic foot assumes each segment's wave propagates independently forever. In reality, waves collide at a finite time $t_{\text{coll},k}$, after which the characteristic geometry is no longer valid.

The analytical collision time for segment $k$ is:

$$t_{\text{coll},k} = \min\left(\frac{w_{k-1}}{|\lambda_{k-1} - \lambda_k|}, \;\frac{w_k}{|\lambda_k - \lambda_{k+1}|}\right)$$

where $w_j = x_{j+1} - x_j$ is the width of the neighboring segment whose characteristics converge with $k$'s. Edge segments use their own width as a fallback. Speed differences are clamped to $\geq 10^{-3}$ for numerical stability.

The damping factor is a learnable sigmoid:

$$\gamma_{\text{damp}}(t, k) = \sigma\Big(|\beta_{\text{damp}}| \cdot (t_{\text{coll},k} - t)\Big)$$

where $\beta_{\text{damp}}$ is a learnable sharpness parameter (initialized at 5.0). This gives:
- $t \ll t_{\text{coll},k}$: $\gamma_{\text{damp}} \approx 1$ → full bias active (physics is valid)
- $t \gg t_{\text{coll},k}$: $\gamma_{\text{damp}} \approx 0$ → bias fades (learned attention takes over)

**Final bias**:

$$\text{bias}(t, x, k) = \text{bias}_{\text{raw}}(t, x, k) \cdot \gamma_{\text{damp}}(t, k)$$

Padded segments receive a mask value of $-10^9$ to completely zero them out in softmax.

**Output**: `char_bias` $(n_t, n_x, K)$

---

### Stage 5: Biased Cross-Attention

**Component**: `BiasedCrossDecoderLayer` stack (default: 2 layers)

**Purpose**: Let each spatial query "read" from the time-conditioned segment tokens, with the characteristic bias guiding which segments to attend to.

**Reshaping for per-timestep attention**: The computation is organized per-timestep to keep the attention matrix small:
- Queries: `query_emb` $(n_t, n_x, H)$ → reshaped to $(n_t, n_x, H)$, treated as $n_t$ independent batches of $n_x$ queries
- Keys/Values: `kv` $(n_t, K, H)$ — already per-timestep from FiLM
- Attention mask: `char_bias` $(n_t, n_x, K)$ — expanded to $(n_t \cdot n_{\text{heads}}, n_x, K)$ for multi-head attention

**Each `BiasedCrossDecoderLayer`** is pre-norm with residual connections:

$$\mathbf{x}' = \mathbf{x} + \text{Dropout}\Big(\text{MHA}\big(\text{LN}(\mathbf{x}),\; \text{LN}(\mathbf{z}),\; \text{LN}(\mathbf{z}),\; \text{mask}=\text{bias}\big)\Big)$$
$$\mathbf{x}'' = \mathbf{x}' + \text{Dropout}\Big(\text{FFN}\big(\text{LN}(\mathbf{x}')\big)\Big)$$

where $\mathbf{x}$ are query embeddings $(n_x, H)$, $\mathbf{z}$ are segment KV tokens $(K, H)$, and the attention mechanism computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_{\text{head}}}} + \text{bias}\right) V$$

The additive bias shifts the attention logits before softmax. This is a **soft** prior:
- The learned $QK^T$ scores can override the bias when the data requires it
- But the bias provides a strong default that aligns with physics, reducing the learning burden

**Why FiLM matters here**: Without FiLM (the `WaveNOAblationDamp` variant), the KV tokens are the same static segment embeddings at every timestep. The only time-varying signal in the cross-attention is the query encoding (which knows $t$) and the attention bias (which encodes characteristic geometry). With FiLM, the KV tokens themselves carry time-specific information — the segment's "state" at time $t$ — giving the attention mechanism richer key-value representations to read from.

**Output**: `attended` $(n_t, n_x, H)$

---

### Stage 6: Density Head

**Component**: 2-layer MLP with clamping

**Architecture**: Linear$(H \to H)$ → ReLU → Dropout → Linear$(H \to 1)$

**Initialization**: The final linear layer has:
- Weights: initialized to zero
- Bias: initialized to 0.5

This means the initial prediction is a constant $\hat{\rho} = 0.5$ everywhere — a neutral midpoint. During training, the model learns to deviate from this baseline. This initialization prevents exploding or collapsing predictions in early training.

**Output clamping**: $\hat{\rho}(t, x) = \text{clamp}(\text{MLP}(\mathbf{q}), 0, 1)$

The clamp enforces the physical constraint that density lies in $[0, 1]$, which is a hard constraint for the LWR traffic model (density is a fraction of maximum capacity).

**Final output**: `output_grid` $(1, n_t, n_x)$

---

## Deep Dive: FiLM Time Conditioning

### What Is FiLM?

FiLM (Feature-wise Linear Modulation), introduced by Perez et al. (2018), is a conditioning mechanism that modulates neural network features via a learned affine transformation:

$$\text{FiLM}(\mathbf{h} \mid z) = \gamma(z) \odot \mathbf{h} + \beta(z)$$

where $\mathbf{h}$ is a feature vector, $z$ is a conditioning input, and $\gamma(z)$, $\beta(z)$ are scale and shift vectors produced by a small conditioning network.

### FiLM vs. Alternatives

There are several ways to inject time information into segment embeddings. Here is why FiLM was chosen:

**Concatenation** ($[\mathbf{s}_k \| \gamma_t(t)]$ → MLP): Additive only. The network must learn multiplicative interactions (feature scaling, gating) implicitly through nonlinearities. This is less parameter-efficient and harder to optimize for the kind of scaling behavior physics requires.

**Additive bias** ($\mathbf{s}_k + \text{MLP}(\gamma_t(t))$): Cannot scale features — only shifts them. A segment whose influence should diminish over time (because its wave has been absorbed) cannot be "turned off" by a shift alone.

**Hypernetwork** (generate all weights from $t$): Extremely expensive (weights scale as $H^2$ per layer). Overfits easily with small datasets. Unnecessary when the modulation is affine.

**Cross-attention** ($\mathbf{s}_k$ attends to time embeddings): Works but introduces $O(K \cdot n_t)$ attention computations with a heavyweight attention mechanism (Q, K, V projections + softmax) for what is fundamentally a 1D conditioning signal. Overkill.

**FiLM**: Two parameters per feature per conditioning input (scale + shift). Lightweight but expressive: can amplify, suppress, invert, or shift any feature dimension. The multiplicative $\gamma$ term is critical — it enables gating behavior that purely additive methods cannot achieve.

### How FiLM Is Applied in This Architecture

The conditioning network takes **both** the Fourier-encoded time **and** the segment embedding as input:

$$[\gamma(t, k), \; \beta(t, k)] = \text{MLP}\Big(\gamma_t(t) \;\|\; \mathbf{s}_k\Big)$$

This is a **content-dependent** FiLM: the modulation depends not just on time but on which segment is being modulated. Different segments receive different scale and shift vectors at the same timestep. This is essential because:

- A segment with $\rho_k = 0.8$ (high density, leftward characteristic) should evolve differently than one with $\rho_k = 0.2$ (low density, rightward characteristic)
- A wide segment ($w_k = 0.3$) remains influential for longer than a narrow one ($w_k = 0.05$) before collisions occur
- A segment that has already interacted (as encoded in its self-attention-enriched embedding) should be modulated differently than an isolated one

### Information Flow Through FiLM

The following diagram traces what information is available at each stage:

```
IC (xs, ks)
    │
    ├── SegmentPhysicsEncoder ──► seg_emb (K, H)
    │     Contains: position, width, ρ, λ=f'(ρ), f(ρ), cumulative mass
    │
    ├── Self-Attention ──► seg_emb (K, H)
    │     Now also encodes: neighbor relationships, density jumps, relative positions
    │
    ├── FiLM(seg_emb, t) ──► kv (nt, K, H)
    │     Each segment embedding is modulated per timestep:
    │     - γ(t,k) can amplify features relevant at time t
    │     - γ(t,k) can suppress features irrelevant at time t
    │     - β(t,k) can shift the representation to encode time-dependent state
    │
    └── Cross-Attention(query, kv, bias) ──► density prediction
          Query: knows (t, x) via Fourier encoding
          KV: knows segment identity + time-evolved state via FiLM
          Bias: knows characteristic geometry via physics computation
```

### The Identity Initialization Guarantee

The FiLM network's final layer is initialized so that $\gamma = \mathbf{1}$ and $\beta = \mathbf{0}$:

```python
nn.init.zeros_(self.film_net[-1].weight)
nn.init.zeros_(self.film_net[-1].bias)
with torch.no_grad():
    self.film_net[-1].bias[:hidden_dim] = 1.0  # gamma = 1
```

This means:
1. At initialization, `TimeConditioner` is an identity function: $\text{kv}(t, k) = \mathbf{s}_k$ for all $t$
2. The model starts exactly equivalent to the `WaveNOAblationDamp` variant (no FiLM)
3. Training gradually introduces time dependence only where it improves the loss
4. If time conditioning is not helpful, the model can keep $\gamma \approx 1, \beta \approx 0$ — it never hurts

This is a strict improvement over the non-FiLM baseline: the hypothesis class of `WaveNOAblationFiLM` is a superset of `WaveNOAblationDamp`.

---

## Why FiLM Works for Hyperbolic Conservation Laws

### 1. Physical systems are parameterized by time

In hyperbolic conservation laws, the solution structure evolves qualitatively over time:

- **Early time** ($t \ll t_{\text{coll}}$): segments propagate independently. Each segment's influence region moves at characteristic speed $\lambda_k = f'(\rho_k)$. The solution is a simple translation of the IC.
- **Collision time** ($t \approx t_{\text{coll}}$): neighboring wavefronts interact. Some segments merge (shock absorption), some split (rarefaction fan widening). The number of "effective" segments changes.
- **Late time** ($t \gg t_{\text{coll}}$): the solution has reorganized into a new wave structure that bears little resemblance to the original IC segmentation.

FiLM captures this by modulating segment representations differently at each time:
- At early $t$: $\gamma \approx 1, \beta \approx 0$ (segment identity is preserved — it hasn't changed yet)
- At collision $t$: $\gamma$ may scale down features of absorbed segments, $\beta$ may shift representations to encode the merged state
- At late $t$: large-magnitude modulations can effectively rewrite segment representations to reflect the post-interaction structure

### 2. Affine modulation matches linearized PDE behavior

For small time increments $\delta t$, the evolution of a segment's influence on a query point is approximately affine in the segment's features. Consider a segment at position $\bar{x}_k$ with speed $\lambda_k$:

- Its center moves to $\bar{x}_k + \lambda_k \delta t$ (linear in $\lambda_k$)
- Its flux contribution scales as $f(\rho_k) \cdot \delta t$ (linear in $f_k$)
- Its width shrinks or grows linearly with the difference of boundary speeds

FiLM's affine form $\gamma(t) \odot \mathbf{s}_k + \beta(t)$ directly matches these linearized dynamics. The network doesn't need to learn a complex nonlinear evolution — the affine structure is built in.

### 3. Multiplicative gating enables segment "death"

When a shock overtakes a segment (a faster shock from the left catches and absorbs a slower shock on the right), that segment effectively ceases to exist in the solution. In the cross-attention, this segment should receive zero attention weight.

The characteristic bias partially handles this via collision-time damping (the bias fades, so the learned $QK^T$ must learn to ignore the segment). But FiLM provides a more direct mechanism: $\gamma(t, k) \to \mathbf{0}$ can zero out the segment's embedding entirely at the KV level, before cross-attention even computes scores. This is a stronger signal than relying solely on learned attention scores to ignore a segment.

### 4. Content-dependent conditioning captures heterogeneous evolution

Different segments evolve at different rates — a narrow segment between two converging shocks is absorbed quickly, while a wide segment in a rarefaction zone persists for a long time. Because FiLM conditions on both $t$ and $\mathbf{s}_k$, it can learn segment-specific evolution rates. A global (segment-independent) time conditioning would apply the same modulation to all segments, which is physically incorrect.

### 5. Complementarity with the characteristic bias

The characteristic bias (Stage 4) encodes **where** each segment's influence reaches at time $t$ — it modulates the spatial attention pattern. FiLM encodes **what** each segment represents at time $t$ — it modulates the content of the attended features. These are complementary:

| Mechanism | What it controls | How |
|-----------|-----------------|-----|
| Characteristic bias | Which segments a query attends to (spatial routing) | Additive bias on attention logits |
| FiLM | What information the attended segments carry (content evolution) | Affine modulation of segment embeddings |

Without FiLM, the cross-attention can route queries to the correct segment but can only read the segment's static initial-condition features. With FiLM, the cross-attention reads a time-evolved representation that encodes the segment's current state.

### 6. The collision-time gap

The collision-time damping mechanism transitions the characteristic bias from "fully active" to "fully off" around $t_{\text{coll}}$. After this transition, the model has no physics guidance — it relies entirely on learned $QK^T$ scores. This is the hardest regime for the model.

FiLM helps precisely here: by providing time-evolved segment representations, it gives the learned attention mechanism better features to work with in the post-collision regime. The $QK^T$ scores are computed from time-modulated KV tokens, so they implicitly encode temporal state even when the explicit physics bias has faded.

---

## Comparison with Adjacent Ablations

### vs. WaveNOAblationDamp (predecessor: no FiLM)

The only difference is the absence of FiLM. In `WaveNOAblationDamp`:
- KV tokens for cross-attention are the static segment embeddings, broadcast identically to every timestep
- Time information enters only through query encoding and the characteristic bias
- After collision-time damping fades the bias, the model has no time-dependent signal in the KV branch

Adding FiLM gives the KV branch its own time dependence, which is expected to improve:
- Post-collision prediction quality (the KV tokens carry temporal state)
- Smooth region accuracy (FiLM can encode the gradual evolution of rarefaction fans)
- Generalization to longer time horizons (the model can learn extrapolation of the FiLM modulation)

### vs. WaveNOAblationCrossAttn (sibling: cross-segment attention instead of FiLM)

`WaveNOAblationCrossAttn` uses `CrossSegmentAttention` — a lightweight self-attention over the $K$ segment dimension, applied per-timestep after broadcasting the static embeddings. This is a different mechanism for introducing time-dependent segment interactions:

| | FiLM | CrossSegmentAttention |
|---|---|---|
| Time dependence | Continuous (Fourier-encoded $t$ → MLP → γ, β) | Discrete (applied identically per timestep; no explicit time input) |
| Segment interaction | Implicit (through self-attention-enriched input) | Explicit (per-timestep self-attention) |
| Parameter cost | Linear $(F_t + H) \cdot 2H$ | Quadratic $O(H^2)$ for attention projections |
| Expressivity | Affine modulation per feature | Full self-attention (any pairwise interaction) |

FiLM is lighter and introduces continuous time dependence. Cross-segment attention is heavier but can model pairwise segment interactions that change between timesteps (e.g., two segments that were independent at $t=0$ but are now interacting).

### vs. WaveNOAblationFull (successor: FiLM + CrossSegmentAttention)

`WaveNOAblationFull` combines both FiLM and cross-segment attention. The pipeline becomes:

1. Static segment embeddings $(K, H)$
2. FiLM modulation → time-dependent embeddings $(n_t, K, H)$
3. Cross-segment attention → per-timestep segment interaction $(n_t, K, H)$
4. Cross-attention with characteristic bias → density prediction

This is the most expressive ablation variant. It tests whether FiLM and cross-segment attention provide complementary benefits (FiLM for continuous time modulation, cross-segment attention for dynamic inter-segment relationships).

### vs. Full WaveNO

The full WaveNO adds three more components on top of the ablation:
- **Breakpoint evolution** (trajectory prediction via cross-attention decoder)
- **Boundary extraction** (local $x_L, x_R$ from predicted trajectories)
- **Boundary Fourier features** ($\gamma(x_L), \gamma(x_R)$ added to query encoding)

These components provide the model with explicit local boundary context — each query knows where the nearest discontinuities are, not just its absolute $(t, x)$ position. This is critical for K-invariance and resolution generalization.

---

## Default Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `hidden_dim` | 64 | All embedding dimensions |
| `num_freq_t` | 8 | Fourier bands for time in query encoder |
| `num_freq_x` | 8 | Fourier bands for space in query encoder |
| `num_seg_frequencies` | 8 | Fourier bands for segment encoder + FiLM time encoder |
| `num_seg_mlp_layers` | 2 | MLP depth in segment encoder |
| `num_self_attn_layers` | 2 | Self-attention layers for segment interaction |
| `num_cross_layers` | 2 | Biased cross-attention layers (queries → segments) |
| `num_heads` | 4 | Attention heads (self and cross) |
| `initial_bias_scale` | 5.0 | Initial learnable $\alpha$ for characteristic bias |
| `initial_damping_sharpness` | 5.0 | Initial learnable $\beta_{\text{damp}}$ for collision-time sigmoid |
| `local_features` | True | Include cumulative mass $N_k$ in segment encoder |
| `dropout` | 0.05 | Dropout rate throughout |
| `use_char_bias` | True | Characteristic attention bias enabled |
| `use_damping` | True | Collision-time damping enabled |
| `use_film` | True | FiLM time conditioning enabled |
| `use_cross_seg_attn` | False | Cross-segment attention disabled |

**Factory function**: `build_waveno_ablation_film(args)` in `waveno_minimal.py`

**Model registration**: Registered as `"WaveNOAblationFiLM"` in the model registry.

**Loss**: Uses the default `mse` preset (no model-specific override).

**Training command**:
```bash
cd wavefront_learning
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python train.py \
    --model WaveNOAblationFiLM \
    --loss mse \
    --epochs 100 \
    --no_wandb
```
