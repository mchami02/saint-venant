# Deep Analysis: ClassifierAllTrajTransformer + `pde_shocks`

## Context

The user's [eval.sh](eval.sh) runs all trajectory models with `--loss pde_shocks`. This analysis dissects both the model and the loss independently, then their combination, identifying fundamental weaknesses and proposing concrete improvements.

The `pde_shocks` preset is:
```python
"pde_shocks": [("mse", 1.0), ("pde_shock_residual", 1.0)]
```

---

# Part I: Model Weaknesses (ClassifierAllTrajTransformer)

These issues exist regardless of which loss is used.

## M1. Time-Independent Existence — Cannot Model Shock Evolution

The classifier outputs `existence: (B, D)` — a single scalar per discontinuity, constant across all time steps. In the forward pass ([traj_transformer.py:565](models/traj_transformer.py#L565)):
```python
"existence": existence.unsqueeze(-1).expand_as(positions)  # (B, D) → (B, D, T)
```

This is a **fake time dimension** — just repetition, not prediction. The model cannot represent:
- A shock that weakens and disappears mid-simulation
- Two shocks that merge (one should have existence → 0 after collision)
- A discontinuity whose shock/rarefaction nature changes due to interactions

For LWR with Greenshields flux, this is partially acceptable (a discontinuity's type is determined at t=0 by the Lax entropy condition: shock if ρ_L < ρ_R, rarefaction otherwise). But even then, shocks can exit the domain or collide, requiring time-dependent existence.

## M2. No Temporal Context in Density Decoder

The `DynamicDensityDecoder` reshapes queries and keys/values for per-time-step batched attention ([traj_transformer.py:347-350](models/traj_transformer.py#L347-L350)):
```python
q = coord_emb.reshape(B * nt, nx, H)          # (B*nt, nx, H)
kv = dynamic_emb.permute(0, 2, 1, 3).reshape(B * T, D, H)  # (B*T, D, H)
```

Each time step's spatial queries attend **only** to that time step's boundary tokens. There is no mechanism for density at time `t` to know about `t-1` or `t+1`. This means:
- The density at each time step is predicted independently
- No temporal smoothness is enforced architecturally
- The model cannot learn patterns like "density propagates at characteristic speed" from the architecture alone — it must learn this entirely from data through the coordinate Fourier encoding

This is a deliberate design trade-off (per-time-step batching is efficient), but it removes useful inductive bias for PDEs where solutions evolve temporally.

## M3. Addition-Based Fusion Loses Information

Dynamic boundary tokens are formed by addition ([traj_transformer.py:339](models/traj_transformer.py#L339)):
```python
dynamic_emb = disc_exp + pos_enc  # (B, D, T, H)
```

Addition is the weakest fusion operation. Compare:
- **Addition**: `a + b` — no interaction terms, information mixed in same channels
- **Concatenation + projection**: `W[a; b]` — separate channels, learned mixing
- **Bilinear**: `aᵀWb + Ua + Vb` — captures pairwise interactions

For TrajDeepONet, bilinear fusion is used for branch-trunk combining. For the transformer, this fusion determines how the density decoder knows "what wave properties exist at position x_d(t)" — losing interaction terms means the attention mechanism must compensate, adding learning burden.

## M4. Position Clamping Creates Gradient Dead Zones

Trajectory positions are hard-clamped ([traj_transformer.py:103](models/traj_transformer.py#L103)):
```python
positions = torch.clamp(positions, 0.0, 1.0)
```

When a position hits the boundary (0 or 1), `∂clamp/∂input = 0` — gradients are killed. Early in training when predictions are noisy, many positions will saturate at domain boundaries and receive no gradient signal. A smooth alternative (sigmoid rescaling, soft clamp) would maintain gradient flow everywhere.

## M5. One-Way Information Flow — No Feedback Loops

The forward pass is strictly feedforward:
```
disc_emb → classifier → existence (one-way, no density feedback)
disc_emb → traj_decoder → positions (one-way, no density feedback)
disc_emb + positions + existence → density_decoder → density (terminal)
```

The trajectory decoder never learns "my position prediction is wrong because the density came out poorly." The only feedback path is through backpropagation of the loss, which is indirect and noisy (see gradient path analysis below). An iterative refinement scheme or joint loss could address this.

## M6. Classifier Operates on Incomplete Information

The classifier sees only `branch_emb` — the discontinuity embeddings after self-attention ([traj_transformer.py:538-540](models/traj_transformer.py#L538-L540)). It does not see:
- The predicted trajectory (where the shock actually moves)
- The predicted density (whether the density field actually shows a discontinuity)
- The time coordinates (whether the shock persists or disappears)

This makes the classifier a purely local prediction from initial conditions, with no ability to incorporate trajectory quality or density consistency.

---

# Part II: Loss Weaknesses (`pde_shocks`)

These issues exist regardless of which model is used.

## L1. PDE Residual Computed on Ground Truth — Zero Gradient to Predictions

**The most fundamental flaw.** In [pde_residual.py:177](losses/pde_residual.py#L177):
```python
residual = compute_pde_residual(gt_density, self.dt, self.dx)
```

The conservation law `∂ρ/∂t + ∂f(ρ)/∂x = 0` is evaluated on the **ground truth** grid. Since GT is a constant w.r.t. model parameters, `residual` is a fixed tensor that provides **zero gradient to `output_grid`**.

**Why this is bad ML practice:** A physics-informed loss should constrain the *learned function*. Computing PDE residual on GT is just measuring how discontinuous the data is — a fixed quantity the model cannot influence. It's a diagnostic, not a training signal.

**What it actually does:** Uses the fixed GT residual as a "shock indicator map" to weight how trajectory positions are penalized. This is a valid idea for trajectory supervision, but calling it "PDE residual loss" is misleading — it's really a "position-to-shock-alignment loss."

## L2. `eps = 1.0` Flattens the Entire Loss Landscape

In [pde_residual.py:201](losses/pde_residual.py#L201):
```python
eps = 1.0
min_score = (dist / (combined.unsqueeze(-1) + eps)).min(dim=1).values
```

With domain [0, 1]:
- `dist` ranges from 0 to ~1.0
- `combined = disc_mask * existence` ranges from 0 to 1.0
- Denominator `combined + 1.0` ranges from 1.0 to 2.0

The loss barely distinguishes between a shock with existence=0 and existence=1 (factor of 2x at most). Similarly for positions — moving a position from far (dist=0.5) to near (dist=0.01) a shock changes the loss by less than 50%. This produces negligibly small gradients for both `positions` and `existence`.

A value like `eps = 0.01` would give denominator range [0.01, 1.01] — a 100x variation that creates meaningful gradients.

## L3. `min()` Operations Create Winner-Takes-All Dynamics

Both loss terms use hard `min()`:
```python
min_score = (dist / (combined + eps)).min(dim=1).values   # min over D discontinuities
fp_score = (dist / (residual² + eps)).min(dim=-1).values  # min over nx spatial points
```

`min()` has a gradient of 1 for the argmin element and 0 for all others. This means:
- Only the **nearest** discontinuity to each cell gets gradient from miss loss
- Only the **nearest** high-residual cell to each shock gets gradient from FP loss
- All other discontinuities/cells get zero gradient per step

This is unstable early in training when the "nearest" assignment changes frequently. A `softmin` (log-sum-exp) would distribute gradients across multiple candidates.

## L4. Competing Miss vs. FP Objectives at Initialization

- **Miss loss**: wants positions to move toward high-residual cells (actual shocks)
- **FP loss**: wants to suppress existence for positions far from high-residual cells

At init, positions are random → most are in smooth regions → FP loss dominates → pushes existence toward 0 → miss loss loses signal (since `combined → 0`). This is a well-known problem: false positive suppression is destructive before the detector has learned anything useful.

## L5. Central Differences at Shocks Produce Extreme Values

For Greenshields flux, the PDE residual near a shock is approximately:
```
residual ≈ Δρ / (2·dt)  where Δρ is the shock strength
```

With `dt = 0.004` and typical `Δρ ≈ 0.5`: `residual ≈ 62.5`. Squaring: `residual² ≈ 3906`. These extreme values dominate the loss and can destabilize training through gradient explosion. The distance weighting is supposed to down-weight these cells, but with `eps = 1.0`, the weighting is far too weak.

## L6. No Supervision for Existence or Trajectory Regularity

`pde_shocks` provides:
- `output_grid`: supervised by MSE (strong)
- `positions`: only PDEShockResidualLoss (weak, as shown)
- `existence`: only PDEShockResidualLoss (near-zero signal)

There is NO:
- Existence supervision (no `AccelerationLoss`)
- Trajectory smoothness (no `RegularizeTrajLoss`)
- Domain boundary enforcement (no `BoundaryLoss`)
- Initial condition anchoring (no `ICAnchoringLoss`)

For any model with multiple output heads, each head needs adequate supervision. Two of three heads are essentially untrained.

---

# Part III: Combined Weaknesses (Model + Loss)

When `ClassifierAllTrajTransformer` meets `pde_shocks`, the individual weaknesses compound.

## C1. Complete Gradient Path Analysis

| Output | Gradient sources with `pde_shocks` | Signal strength |
|--------|-------------------------------------|-----------------|
| `output_grid` | MSE (direct L2 on grid) | **Strong** |
| `positions` | PDEShockResidual miss/FP (weak: eps=1.0, min(), GT residual) + MSE→density_decoder→attention→pos_enc→positions (6+ transforms deep) | **Very weak** |
| `existence` | PDEShockResidual via `combined` (near-zero: eps=1.0 flattening) + MSE→density_decoder→attention→effective_weight→existence (7+ transforms deep) | **Near zero** |

Compare with `classifier_all_traj_transformer` preset:

| Output | Gradient sources | Signal strength |
|--------|-----------------|-----------------|
| `output_grid` | MSE | **Strong** |
| `positions` | BoundaryLoss (direct) + RegularizeTraj (direct) + IC anchoring (direct) + AccelerationLoss (clear indirect) | **Strong** |
| `existence` | AccelerationLoss: `(1-e)² · 1(|accel| > τ)` (direct) | **Strong** |

## C2. Chicken-and-Egg Deadlock

1. Positions are random at init → far from actual shocks
2. FP loss pushes existence → 0 (positions look like false positives)
3. `DynamicDensityDecoder` soft weighting: `effective_weight = existence * disc_mask` → all boundary tokens get weight ≈ 0
4. Cross-attention keys/values are near-zero → attention outputs are near-zero → density decoder falls back to coordinate MLP only (ignoring boundary tokens entirely)
5. Miss loss gradient vanishes (`combined ≈ 0` → all `min_score` values converge)
6. **Deadlock**: the model learns to predict density purely from coordinates, ignoring the trajectory/existence branches entirely

The `classifier_all_traj_transformer` preset breaks this because `AccelerationLoss` provides existence supervision **independent of positions** — it identifies shocks from GT acceleration, not from model predictions.

## C3. The Model Degenerates to NoTrajTransformer

When existence → 0 and positions are ignored (per C2), the `DynamicDensityDecoder` becomes:
```python
# dynamic_emb * effective_weight → ~0 (all boundary tokens suppressed)
# Cross-attention with near-zero KV → output ≈ 0
# q remains = coord_emb (unchanged by attention)
# density = density_head(coord_emb) → depends only on (t, x) coordinates
```

This is functionally equivalent to `NoTrajTransformer` (which uses no trajectories at all). The model "solves" `pde_shocks` by learning a coordinate-to-density mapping via MSE alone, wasting the entire trajectory/classifier architecture.

---

# Part IV: Concrete Improvement Directions

## Direction 1: Use the Correct Loss Preset (Minimal Change)

```bash
# eval.sh — fix per-model loss assignments:
uv run train.py --model TrajTransformer --loss traj_transformer
uv run train.py --model ClassifierTrajTransformer --loss classifier_traj_transformer
uv run train.py --model ClassifierAllTrajTransformer --loss classifier_all_traj_transformer
uv run train.py --model NoTrajTransformer --loss no_traj_transformer
```

Each model has a dedicated preset in [loss.py](loss.py) that provides proper gradient coverage. `pde_shocks` is only appropriate for simpler models or as an auxiliary loss on top of proper supervision.

## Direction 2: Create a Physics-Informed Preset (Moderate Change)

Add to [loss.py](loss.py):
```python
"physics_classifier_all_traj_transformer": [
    ("mse", 1.0),                                         # Grid supervision
    ("pde_residual", 0.1),                                 # PDE on PREDICTIONS (not GT)
    ("ic_anchoring", 0.1),                                 # Anchor to IC
    ("boundary", 1.0),                                     # Domain bounds
    ("regularize_traj", 0.1),                              # Smooth trajectories
    ("acceleration", 1.0, {"missed_shock_weight": 1.0}),   # Existence supervision
],
```

Key: uses `PDEResidualLoss` (residual on *predictions*) instead of `PDEShockResidualLoss` (residual on GT). This actually enforces physics on the learned solution.

## Direction 3: Fix Model Architecture Weaknesses (Significant Change)

### 3a. Time-dependent existence
Replace per-discontinuity scalar with per-timestep prediction:
```python
# After trajectory prediction, classify per-timestep
traj_features = positions_encoded  # (B, D, T, H)
existence = self.classifier_head(
    disc_emb.unsqueeze(2) + traj_features
).squeeze(-1)  # (B, D, T)
```

### 3b. Smooth position clamping
Replace hard clamp with sigmoid rescaling:
```python
positions = torch.sigmoid(raw_positions)  # Smooth, always in (0, 1), never zero gradient
```

### 3c. Physics-informed trajectory initialization
Initialize trajectory positions using analytical Rankine-Hugoniot speed:
```python
# x_d(t) = x_d(0) + s * t, where s = 1 - ρ_L - ρ_R
s = 1.0 - discontinuities[:, :, 1] - discontinuities[:, :, 2]  # (B, D)
analytical_positions = discontinuities[:, :, 0:1] + s.unsqueeze(-1) * query_times.unsqueeze(1)
positions = analytical_positions + learned_correction  # Learn residual only
```

This gives the model a strong initialization, reducing the learning burden.

## Direction 4: Fundamentally Different Approach (Major Change)

Instead of predicting trajectories and using them for density conditioning, train an **implicit neural representation** that directly maps `(discontinuities, t, x) → ρ(t, x)` with physics-informed regularization:

- Remove trajectory decoder and classifier entirely
- Use `NoTrajTransformer` architecture (cross-attention from coordinates to discontinuity embeddings)
- Train with `MSE + PDEResidualLoss(on predictions) + ICLoss`
- Let the model learn shock structure implicitly through the PDE constraint

This is simpler, has fewer failure modes, and the PDE residual provides genuine physics grounding. The trade-off: no explicit shock tracking, but the density field itself captures all the physics.

---

## Recommended Approach

**Start with Direction 1** (fix eval.sh). If you want physics-informed training, **add Direction 2**. For longer-term improvement, consider **Direction 3a** (time-dependent existence) and **3c** (physics-informed initialization) as they are the highest-impact model improvements with moderate implementation effort.

### Files to Modify

| Direction | Files | Effort |
|-----------|-------|--------|
| 1 | `wavefront_learning/eval.sh` | Trivial |
| 2 | `wavefront_learning/loss.py` | Small (add preset) |
| 3a | `wavefront_learning/models/traj_transformer.py` | Medium (classifier refactor) |
| 3b | `wavefront_learning/models/traj_transformer.py` | Small (one-line change) |
| 3c | `wavefront_learning/models/traj_transformer.py` | Medium (trajectory decoder refactor) |
| 4 | `wavefront_learning/loss.py`, training script | Large (new training paradigm) |

### Verification

For any change, run:
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python train.py \
    --model ClassifierAllTrajTransformer \
    --loss <chosen_preset> \
    --epochs 1 --n_samples 100 --no_wandb
```
to verify sanity check passes (all 4 steps, all parameters receive gradients).
