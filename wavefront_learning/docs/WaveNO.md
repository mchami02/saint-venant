# WaveNO: Wavefront Neural Operator

WaveNO learns the solution operator of scalar hyperbolic conservation laws:

$$\frac{\partial \rho}{\partial t} + \frac{\partial f(\rho)}{\partial x} = 0$$

Given a piecewise-constant initial condition (breakpoint positions $\{x_k\}_{k=0}^{K}$ and segment values $\{\rho_k\}_{k=1}^{K}$), WaveNO maps this compact representation to the full space-time solution grid $\rho(t, x) \in \mathbb{R}^{n_t \times n_x}$.

The central idea: the solution is determined by wavefronts (shocks and rarefactions) emanating from each IC discontinuity. Rather than processing a discretized grid globally, WaveNO lets each spatial query point attend to the IC segments that physically influence it, guided by a physics-derived attention bias that encodes wave propagation geometry.

## Architecture

WaveNO is a 6-stage pipeline. Each stage corresponds to a distinct module.

### 1. SegmentPhysicsEncoder

**Goal**: Represent each IC segment as a rich embedding that captures both its geometry and its physical role in the solution.

A piecewise-constant IC is a collection of $K$ segments, each with a constant density $\rho_k$ on an interval $[x_k, x_{k+1}]$. A naive representation would use just these raw values, but the solution depends on *derived* physical quantities — characteristic speeds determine how information propagates, flux values appear in the Rankine-Hugoniot condition, and mass distribution constrains what the solution can look like globally.

For each segment $k$, the encoder computes:

| Feature | Formula | Why it matters |
|---------|---------|----------------|
| Center position | $\bar{x}_k = (x_k + x_{k+1})/2$ | Locates the segment in space |
| Width | $w_k = x_{k+1} - x_k$ | Determines how long before neighboring waves collide |
| Density | $\rho_k$ | The conserved quantity value |
| Characteristic speed | $\lambda_k = f'(\rho_k)$ | Speed at which information propagates within this segment |
| Flux value | $f_k = f(\rho_k)$ | Appears in the Rankine-Hugoniot shock speed formula |
| Cumulative mass | $N_k = \sum_{j < k} \rho_j w_j / M_{\text{total}}$ | How much of the total conserved quantity lies to the left |

Spatial features ($\bar{x}_k$, $w_k$) are optionally Fourier-encoded. All features are concatenated and projected through a 2-layer MLP:

$$\mathbf{s}_k = \text{MLP}\big(\bar{x}_k \,\|\, w_k \,\|\, \rho_k \,\|\, \lambda_k \,\|\, f_k \,\|\, N_k\big) \in \mathbb{R}^H$$

The physics features ($\lambda_k$, $f_k$) are computed via a pluggable `Flux` interface, so the encoder works for any scalar conservation law (Greenshields, triangular, etc.) by swapping the flux function.

**Output**: segment embeddings $(B, K, H)$.

### 2. Segment Self-Attention

**Goal**: Let segments exchange information so each embedding reflects not just its own properties but the global IC structure.

The solution at any point depends on how *all* waves interact — a shock from one discontinuity may collide with a rarefaction from another. For the model to predict these interactions, each segment must be aware of its neighbors' properties (their speeds, densities, distances).

The segment embeddings pass through $L$ transformer encoder layers (default $L = 2$), each consisting of multi-head self-attention followed by a feed-forward network, with pre-norm residual connections. A key padding mask ensures padded (invalid) segments do not participate.

After self-attention, each segment embedding $\mathbf{s}_k$ encodes not just segment $k$'s local properties but contextual information about the entire IC — analogous to how a transformer encoder in NLP lets each token "see" the full sentence.

**Output**: contextualized segment embeddings $(B, K, H)$.

### 3. FiLM Time Conditioning (`TimeConditioner`)

**Goal**: Evolve the static segment embeddings over time, so that keys/values in the cross-attention stage reflect *when* the query is asking, not just *where*.

The IC defines the initial wave structure, but as time progresses, waves move, interact, and change the solution. A static segment embedding cannot capture this — the "meaning" of segment $k$ at $t = 0$ (just a constant region) is very different from its meaning at $t = 0.5$ (its wave has propagated and possibly collided).

FiLM (Feature-wise Linear Modulation) addresses this by computing per-timestep affine transformations of the segment embeddings:

$$\gamma(t, k),\; \beta(t, k) = \text{MLP}\big(\text{Fourier}(t) \,\|\, \mathbf{s}_k\big)$$

$$\mathbf{s}_k^{(t)} = \gamma(t, k) \odot \mathbf{s}_k + \beta(t, k)$$

The FiLM MLP takes both the Fourier-encoded time and the segment embedding as input, so the modulation is *segment-specific* — different segments can evolve differently over time (e.g., a segment behind a fast shock changes meaning faster than one in a quiescent region).

The layer is initialized near identity ($\gamma = 1$, $\beta = 0$), so training starts from the static baseline and gradually learns time dependence. This is computed at the $n_t$ unique timesteps (not all $n_t \times n_x$ query points), since segment evolution is a property of time alone.

**Output**: time-conditioned segment embeddings $(B, n_t, K, H)$, reshaped to $(B \cdot n_t, K, H)$ for per-timestep cross-attention.

### 4. LWR Attention Bias (`LWRBias`)

**Goal**: Tell the cross-attention module which segments *should* influence each query point, based on the physics of wave propagation.

In a hyperbolic PDE, the solution at $(t, x)$ depends only on the IC segments whose waves have reached that point. A segment far away, whose wave has not yet arrived, should have zero influence. Without this prior, the model would need to learn the entire wave propagation geometry from data alone.

The LWR bias encodes this by solving the local Riemann problem at each IC interface and computing how far each segment's influence extends at each query time.

**Step 1: Wave classification.** At each interface between segments $k$ and $k+1$, the module compares characteristic speeds:

- $\lambda_k > \lambda_{k+1}$: characteristics converge $\Rightarrow$ **shock** (single discontinuity moving at the Rankine-Hugoniot speed)
- $\lambda_k \leq \lambda_{k+1}$: characteristics diverge $\Rightarrow$ **rarefaction** (smooth fan connecting the two states)

**Step 2: Boundary speed selection.** The wave type determines how each segment's influence zone boundary moves:

- **Shock**: the boundary between the two segments is a single trajectory at the shock speed $s = (f(\rho_{k+1}) - f(\rho_k)) / (\rho_{k+1} - \rho_k)$. Both segments are sharply cut off at this line.
- **Rarefaction**: the fan occupies a wedge between speeds $\lambda_k$ and $\lambda_{k+1}$. Inside the fan, *both* segments have influence (the solution smoothly transitions). So each segment is only penalized beyond the *far* edge of the fan — segment $k$ (on the left) is penalized past $\lambda_{k+1} \cdot t$, and segment $k+1$ (on the right) is penalized past $\lambda_k \cdot t$.

**Step 3: One-sided distance penalty.** For each query $(t, x)$ and each interface at position $x_d$:

$$\text{penalty}_{\text{left}}(t, x, k) = \text{ReLU}\big(x - (x_d + v_R \cdot t)\big)$$

$$\text{penalty}_{\text{right}}(t, x, k+1) = \text{ReLU}\big((x_d + v_L \cdot t) - x\big)$$

where $v_L, v_R$ are the boundary speeds from Step 2. The ReLU ensures the penalty is zero when the query is inside the influence zone and grows linearly with distance outside it.

**Step 4: Accumulation.** Each segment $k$ accumulates penalties from both its left interface (with segment $k-1$) and its right interface (with segment $k+1$):

$$\text{bias}(t, x, k) = -\text{penalty}_{\text{right interface}} - \text{penalty}_{\text{left interface}}$$

The result is zero where the segment has physical influence and increasingly negative elsewhere. Padded segments get $-10^9$ (effectively masked out).

**Output**: bias tensor $(B, n_t, n_x, K)$, broadcast across attention heads.

### 5. Biased Cross-Attention (`BiasedCrossDecoderLayer`)

**Goal**: For each spatial query point, aggregate information from the relevant IC segments — weighted by both learned affinity and physics-derived relevance.

This is where the model assembles its prediction. Each query point $(t, x)$ needs to "decide" what density value it should have. The answer depends on which IC segments influence that point and how they combine. The cross-attention mechanism handles this by letting each query attend to all segments, with the LWR bias steering attention toward the physically relevant ones.

Queries are encoded from raw $(t, x)$ coordinates via a 2-layer MLP:

$$\mathbf{q}(t, x) = \text{MLP}(t \,\|\, x) \in \mathbb{R}^H$$

The attention computation adds the physics bias to the standard scaled dot-product scores:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + \text{bias}\right) V$$

The bias acts as a **soft prior**: it makes it easy for the model to attend to the right segments (those with bias $= 0$) and hard to attend to the wrong ones (those with large negative bias), but the learned $QK^T$ scores can override it when the physics prior is insufficient — for example, after wave collisions when the initial wave structure has been disrupted.

The cross-attention operates per-timestep: at each time $t_i$, the $n_x$ query embeddings attend to the $K$ time-conditioned segment embeddings from Stage 3. Two layers are stacked (default), each with pre-norm residual connections and a feed-forward sublayer.

**Output**: attended query embeddings $(B \cdot n_t, n_x, H)$.

### 6. Density Head

**Goal**: Convert the contextual query embeddings into scalar density predictions.

After cross-attention, each query embedding encodes "which segments are relevant here and what are their time-evolved properties." The density head is a simple 2-layer MLP that projects this to a scalar:

$$\hat{\rho}(t, x) = \text{clamp}\big(\text{Linear}(\text{ReLU}(\text{Linear}(\mathbf{q}))) ,\; 0,\; 1\big)$$

The final layer is initialized with zero weights and bias $0.5$, so the model starts by predicting a constant neutral density everywhere. This prevents large initial losses and ensures stable early training — the model gradually learns to deviate from $0.5$ as it picks up signal from the cross-attention layers.

The output is clamped to $[0, 1]$ since density is a physical quantity bounded by the domain.

**Output**: predicted solution grid $(B, 1, n_t, n_x)$.
