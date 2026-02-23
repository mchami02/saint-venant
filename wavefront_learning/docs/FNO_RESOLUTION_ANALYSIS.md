# Why FNO Fails on High-Resolution Grids

## Summary

The FNO produces garbage results on higher-resolution grids (2x, 3x, etc.) due to **spectral bandwidth halving** — a fundamental limitation of fixed-mode-count spectral convolutions. This is a model architecture issue, not a data or pipeline bug.

## Root Cause: Fixed Fourier Modes + Changing Grid Size

The FNO uses a fixed number of Fourier modes (`n_modes_t=32, n_modes_x=16`) regardless of input grid size. The `SpectralConv` in neuralop works as follows:

1. FFT transforms input `(B, C, nt, nx)` to frequency domain
2. Only the first `n_modes` frequency indices are kept per dimension
3. Learned weight tensor of shape `(C_in, C_out, 32, 16)` multiplies these modes
4. Result is zero-padded back to full spectrum, then IFFT'd

When the grid size doubles, the same mode indices map to **different physical frequencies**:

- Mode index `k=15` at training (nt=250): physical frequency `15/250 = 0.06`
- Mode index `k=15` at 2x test (nt=500): physical frequency `15/500 = 0.03`

The learned weight for mode 15 was optimized for frequency 0.06 but is applied to frequency 0.03. This mismatch occurs across ALL modes simultaneously.

### Bandwidth reduction table (defaults: nx=50, nt=250)

| Resolution | Grid size  | Time bandwidth       | Space bandwidth     |
|------------|------------|----------------------|---------------------|
| 1x (train) | 250 × 50  | ±15/250 = **±0.060** | 15/50 = **0.300**   |
| 2x         | 500 × 100 | ±15/500 = ±0.030     | 15/100 = 0.150      |
| 3x         | 750 × 150 | ±15/750 = ±0.020     | 15/150 = 0.100      |
| 5x         | 1250 × 250| ±15/1250 = ±0.012    | 15/250 = 0.060      |

At 2x resolution the FNO represents **half** the physical frequency range it was trained on. For solutions with shocks (high-frequency content from discontinuities), this is devastating.

## What Was Ruled Out

**FFT normalization**: neuralop defaults to `fft_norm='forward'` (confirmed in `.venv/.../neuralop/layers/spectral_convolution.py:283`), which divides by N. Magnitude scaling is resolution-invariant — the issue is frequency mismatch, not magnitude.

**Coordinate distribution shift**: The `wavefront_learning` pipeline produces coordinate ranges ≈ [0, 1] at all resolutions because `eval_res` doubles nx/nt while halving dx/dt:
- Training: t ∈ [0, 249×0.004] = [0, 0.996], x ∈ [0, 49×0.02] = [0, 0.98]
- 2x test: t ∈ [0, 499×0.002] = [0, 0.998], x ∈ [0, 99×0.01] = [0, 0.99]

**Test design**: The high-res test (`testing/test_results.py:147-218`) correctly preserves the physical domain by scaling both grid dimensions and spacing together.

**Transform creation**: `ToGridInputTransform` is recreated with the correct high-res `grid_config` for each test resolution (`data/__init__.py:158`).

## Aggravating Factor: IC Masking Pattern

The FNO input channel 0 contains the IC at t=0 then -1 for all subsequent timesteps (`transforms.py:69-70`). This sharp transition creates Gibbs-phenomenon spectral artifacts whose signature changes with grid size. The FNO learns to compensate for these artifacts at training resolution; at a different resolution the compensation is wrong.

## Why Coordinate-Based Models (TrajDeepONet) Don't Have This Problem

TrajDeepONet and TrajTransformer use Fourier feature encoding of continuous (t, x) coordinates, evaluated point-by-point. There is no FFT, no spectral truncation, and no resolution-dependent behavior. The same network evaluates at arbitrary (t, x) regardless of grid density. Resolution generalization is inherent to the architecture.

## DeepONet Baseline (for reference)

The DeepONet baseline would **crash** (not just garbage) at different resolutions due to `nn.Linear(nx, hidden_dim)` in the branch network (`deeponet.py:47`) and hardcoded `reshape(B, 1, self.nt, self.nx)` at line 97.

## Possible Mitigations

| Approach | Description | Trade-off |
|----------|-------------|-----------|
| Interpolate to training res | Downsample input → FNO → upsample output | Simple but not true super-resolution |
| Scale modes with resolution | Proportionally increase n_modes at test time, interpolate weight tensor | Principled but complex |
| Use coordinate-based models | TrajDeepONet / TrajTransformer are inherently resolution-independent | Strongest scientific finding for thesis |
