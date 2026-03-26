# CLAUDE.md - Wavefront Learning Module

## Module Purpose

Neural network-based learning of wavefront dynamics for 1D conservation laws (LWR traffic flow / shallow water equations). Unlike `operator_learning/` which uses discretized ICs, this module uses **exact discontinuity points** (breakpoints `xs` and piece values `ks`) as input. Models output shock trajectories and/or full space-time solution grids `(nt, nx)`.

## Runnable Scripts

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python train.py --model WaveNO --epochs 100
uv run python test.py --model_path wavefront_model.pth  # requires trained model
```

## Adding New Models / Losses

**Model**: implement `build_*` in `models/`, register in `model.py` `MODELS`, add transform/loss/plot presets in `configs/presets.yaml`.

**Loss**: inherit `BaseLoss` in `losses/`, implement `forward(input_dict, output_dict, target) -> (loss, components)`, register in `loss.py` `LOSSES` + `losses/__init__.py`. Composite presets go in `configs/presets.yaml` under `loss_presets`.

## Testing / Verification

Use `train.py` with `WaveNO` and minimal settings — it calls `test_model()` at the end, exercising the full pipeline:
```bash
uv run python train.py --model WaveNO --epochs 0 --no_wandb --n_samples 20
```
Do **not** rely on `--model_path` for quick verification.

## Platform Rules

- **Never modify code to work around macOS/MPS limitations.** If a model does not run on MPS, run the sanity check on CPU instead by passing `--force_cpu`. Do not add MPS workarounds to the codebase.

## Code Style Guidelines

- **File size limit**: Keep files under 600 lines; split into logical modules if exceeded.
- **Model file placement**: Only standalone models (with `build_*` factory, registered in `MODELS`) go in `models/`. Sub-modules and shared components go in `models/base/`.
- **No for loops over space/time in forward passes**: All operations must be fully vectorized. Iterating over layers in a `ModuleList` is fine.
- **Minimal analytical contribution after ICs**: Use the `Flux` interface for IC-level physics features, but do NOT hard-code flux-specific analytical solutions (Legendre transforms, explicit rarefaction profiles). Post-IC structure must be learned.

## LaTeX Writing

- **Refer to the code.** When writing or editing thesis content that describes the implementation (architecture, losses, data pipeline, etc.), read the relevant source files to ensure accuracy. Do not describe code from memory alone.
- **Never cite from memory.** Always do a web search to verify a paper exists, confirm its content, and find the correct BibTeX entry before citing.
- **Use commas, not em-dashes** (`---`) in text.
- **Iterate on schemas independently.** When writing TikZ/LaTeX diagrams or schemas, always render them in a standalone file first, analyze the result, and iterate until it looks correct before adding to the main document.
- **Verify citations compile.** After adding or modifying citations, run the full build (pdflatex + bibtex/biber) and check the output for unresolved references (`?`). Fix any missing `.bib` entries or mismatched citation keys before considering the task done.
