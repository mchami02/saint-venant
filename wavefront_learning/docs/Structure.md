# Wavefront Learning - Directory Structure

```
wavefront_learning/
├── .env                          # Environment variables (WANDB_API_KEY, HF_TOKEN)
├── ARCHITECTURE.md               # Model and loss architecture documentation
├── CHARNO_DESIGN.md              # CharNO design document with mathematical justifications
├── CLAUDE.md                     # Claude Code guidance for this module
├── Structure.md                  # This file
├── train.py                      # Main training script
├── test.py                       # Testing/evaluation CLI entry point
├── eval.sh                       # Shell script for batch evaluation
├── data/
│   ├── __init__.py               # WavefrontDataset, collate_wavefront_batch, get_wavefront_datasets
│   ├── data_loading.py           # HuggingFace upload/download for grid caching
│   ├── data_processing.py        # Grid generation, discontinuity extraction, preprocessing
│   ├── transforms.py             # Input transforms and TRANSFORMS registry
│   └── visualize_extraction.ipynb # Notebook: IC grid with extracted discontinuity positions
├── model.py                      # Model factory and registry
├── loss.py                       # Loss factory, CombinedLoss, presets
├── logger.py                     # Weights & Biases logging utilities
├── metrics.py                    # Evaluation metrics (MSE, MAE, rel_l2, max_error)
├── plotter.py                    # Plotting factory with preset system
├── models/
│   ├── __init__.py               # Re-exports all model classes and builders
│   ├── shock_trajectory_net.py   # ShockTrajectoryNet (DeepONet-like trajectory model)
│   ├── hybrid_deeponet.py        # HybridDeepONet (trajectory + grid prediction)
│   ├── traj_deeponet.py          # TrajDeepONet, ClassifierTrajDeepONet, NoTrajDeepONet
│   ├── traj_transformer.py       # TrajTransformer, ClassifierTrajTransformer, ClassifierAllTrajTransformer (disc-based variants)
│   ├── ctt_seg.py               # CTTSeg: standalone segment-based ClassifierTrajTransformer
│   ├── transformer_seg.py       # TransformerSeg: segment-based transformer without trajectory prediction
│   ├── deeponet.py               # Classic DeepONet baseline
│   ├── fno_wrapper.py            # FNO wrapper (neuralop FNO with dict interface)
│   ├── encoder_decoder.py        # Transformer encoder-decoder (axial/cross variants)
│   ├── charno.py                 # CharNO: Characteristic Neural Operator (Lax-Hopf softmin)
│   ├── waveno.py                 # WaveNO: Wavefront Neural Operator (characteristic-biased cross-attention)
│   ├── wavefront_model.py        # WaveFrontModel: Learned Riemann solver with analytical wave reconstruction
│   └── base/
│       ├── __init__.py           # Re-exports all base components
│       ├── base_model.py         # BaseWavefrontModel abstract class
│       ├── blocks.py             # ResidualBlock
│       ├── feature_encoders.py   # FourierFeatures, TimeEncoder, DiscontinuityEncoder, SpaceTimeEncoder
│       ├── boundaries.py         # compute_boundaries (left/right boundary extraction)
│       ├── decoders.py           # TrajectoryDecoder, TrajectoryDecoderTransformer, DensityDecoderTransformer
│       ├── regions.py            # RegionTrunk, RegionTrunkSet
│       ├── assemblers.py         # GridAssembler (soft region boundaries)
│       ├── transformer_encoder.py # Tokenizer, EncoderLayer, Encoder
│       ├── axial_decoder.py      # FourierTokenizer, AxialAttention, AxialDecoder
│       ├── cross_decoder.py      # CrossDecoderLayer, CrossDecoder (Nadaraya-Watson)
│       ├── shock_gnn.py          # GatedMPNNLayer, ShockGNN (optional, needs torch_geometric)
│       ├── flux.py               # Flux interface, GreenshieldsFlux, TriangularFlux
│       ├── characteristic_features.py  # SegmentPhysicsEncoder, DiscontinuityPhysicsEncoder, CharacteristicFeatureComputer
│       ├── biased_cross_attention.py   # BiasedCrossDecoderLayer, compute_characteristic_bias, compute_discontinuity_characteristic_bias
│       └── breakpoint_evolution.py     # BreakpointEvolution (adjacent segment pairs → trajectory positions)
├── losses/
│   ├── __init__.py               # Re-exports all loss classes and flux utilities
│   ├── base.py                   # BaseLoss abstract class
│   ├── flux.py                   # greenshields_flux, greenshields_flux_derivative, compute_shock_speed
│   ├── mse.py                    # MSELoss
│   ├── ic.py                     # ICLoss
│   ├── trajectory_consistency.py # TrajectoryConsistencyLoss
│   ├── boundary.py               # BoundaryLoss
│   ├── collision.py              # CollisionLoss
│   ├── existence_regularization.py # ICAnchoringLoss
│   ├── supervised_trajectory.py  # SupervisedTrajectoryLoss
│   ├── pde_residual.py           # PDEResidualLoss, PDEShockResidualLoss
│   ├── rh_residual.py            # RHResidualLoss
│   ├── acceleration.py           # AccelerationLoss
│   ├── regularize_traj.py        # RegularizeTrajLoss
│   ├── wasserstein.py            # WassersteinLoss (W1 / Earth Mover's Distance)
│   ├── conservation.py           # ConservationLoss (mass conservation regularizer)
│   ├── cell_avg_mse.py           # CellAverageMSELoss (FV-consistent cell-average MSE)
│   └── visualize_losses.ipynb    # Jupyter notebook for loss visualization
├── plotting/
│   ├── __init__.py               # Re-exports all plotting functions
│   ├── base.py                   # Common utilities (save_figure, _get_extent, _plot_heatmap, etc.)
│   ├── grid_plots.py             # Grid comparison plots (plot_ground_truth, plot_pred, etc.)
│   ├── trajectory_plots.py       # Core trajectory visualization functions
│   ├── wandb_trajectory_plots.py # Trajectory plots for PLOTS registry (W&B-compatible)
│   ├── hybrid_plots.py           # HybridDeepONet-specific visualization
│   ├── charno_plots.py           # CharNO diagnostic visualization (selection weights, entropy, etc.)
│   └── wavefront_plots.py        # WaveFrontModel visualization (wave pattern overlay on GT grid)
├── testing/
│   ├── __init__.py               # Re-exports all test functions
│   ├── test_running.py           # Sanity checks, profiling, inference testing
│   └── test_results.py           # Evaluation metrics, sample collection, high-res testing
├── artifacts/                    # Saved model checkpoints (versioned)
├── wandb/                        # W&B run logs (gitignored)
├── ClassifierTrajDeepONet.pth    # Saved model checkpoint
├── wavefront_model.pth           # Saved model checkpoint
└── tmp_wavefront_grids.npz       # Temporary cached grids
```

## File Details

### Entry Points

- **train.py** — Training loop with early stopping, LR scheduling, periodic plotting.
  - `parse_args()`, `train_step()`, `detach_output()`
- **test.py** — Loads a trained model and runs evaluation.
  - `parse_args()`, `main()`
- **eval.sh** — Shell script for batch evaluation runs.

### Data Pipeline (`data/`)

- **\_\_init\_\_.py** — WavefrontDataset and public API for data loading.
  - `WavefrontDataset`, `collate_wavefront_batch()`, `get_wavefront_datasets()`
  - Re-exports `TRANSFORMS`, all transform classes, `get_wavefront_data`
- **data_loading.py** — Upload/download grids to HuggingFace for caching.
  - `upload_grids()`, `download_grids()`
- **data_processing.py** — Grid generation and IC preprocessing.
  - `PiecewiseRandom` (IC class)
  - `get_nfv_dataset()`, `extract_discontinuities_from_grid()`, `extract_ic_representation_from_grid()`, `preprocess_wavefront_data()`, `get_wavefront_data()`
- **transforms.py** — Input representation transforms.
  - `FlattenDiscontinuitiesTransform`, `ToGridInputTransform`, `DiscretizeICTransform`, `CellSamplingTransform`
  - `TRANSFORMS` registry

### Factories & Registries

- **model.py** — Model creation and persistence.
  - `get_model()`, `load_model()`, `save_model()`
  - `MODELS` registry, `MODEL_TRANSFORM` registry
- **loss.py** — Loss creation with preset combinations.
  - `CombinedLoss`, `get_loss()`, `create_loss_from_args()`
  - `LOSSES` registry, `LOSS_PRESETS` registry
- **plotter.py** — Plotting facade with preset system.
  - `plot()`
  - `PLOTS` registry, `PLOT_PRESETS` registry

### Utilities

- **logger.py** — Weights & Biases wrapper.
  - `WandbLogger` (log_metrics, log_image, log_figure, log_model, download_model, etc.)
  - `init_logger()`, `log_values()`
- **metrics.py** — Shared evaluation metrics.
  - `compute_metrics()`, `extract_grid_prediction()`, `can_compute_grid_metrics()`

### Models (`models/`)

- **shock_trajectory_net.py** — DeepONet-like trajectory prediction.
  - `ShockTrajectoryNet`, `build_shock_net()`
- **hybrid_deeponet.py** — Combined trajectory + grid prediction.
  - `HybridDeepONet`, `build_hybrid_deeponet()`
- **traj_deeponet.py** — Trajectory-conditioned single trunk variants.
  - `TrajDeepONet`, `PositionDecoder`, `BoundaryConditionedTrunk`
  - `build_traj_deeponet()`, `build_classifier_traj_deeponet()`, `build_no_traj_deeponet()`
- **traj_transformer.py** — Cross-attention trajectory decoder variants (disc-based).
  - `TrajTransformer`, `DynamicDensityDecoder`
  - `build_traj_transformer()`, `build_classifier_traj_transformer()`, `build_classifier_all_traj_transformer()`, `build_no_traj_transformer()`, `build_biased_classifier_traj_transformer()`, `build_ctt_biased()`, `build_ctt_seg_physics()`, `build_ctt_film()`
- **ctt_seg.py** — Standalone segment-based ClassifierTrajTransformer.
  - `CTTSeg`, `build_ctt_seg()`
- **transformer_seg.py** — Segment-based transformer without trajectory prediction.
  - `TransformerSeg`, `build_transformer_seg()`
- **deeponet.py** — Classic DeepONet baseline.
  - `DeepONet`, `build_deeponet()`
- **fno_wrapper.py** — Wraps neuralop FNO with dict interface.
  - `FNOWrapper`, `build_fno()`
- **encoder_decoder.py** — Transformer encoder-decoder.
  - `EncoderDecoder`, `build_encoder_decoder()`, `build_encoder_decoder_cross()`
- **charno.py** — Characteristic Neural Operator (Lax-Hopf softmin selection).
  - `CharNO`, `build_charno()`
- **waveno.py** — Wavefront Neural Operator (characteristic-biased cross-attention).
  - `WaveNO`, `build_waveno()`, `build_waveno_cls()`, `build_waveno_local()`, `build_waveno_indep_traj()`, `build_waveno_disc()`
- **wavefront_model.py** — Learned Riemann solver with analytical wave reconstruction.
  - `WaveFrontModel`, `build_wavefront_model()`

### Model Base Components (`models/base/`)

- **base_model.py** — `BaseWavefrontModel` (abstract base with `count_parameters()`)
- **blocks.py** — `ResidualBlock`
- **feature_encoders.py** — `FourierFeatures`, `TimeEncoder`, `DiscontinuityEncoder`, `SpaceTimeEncoder`
- **boundaries.py** — `compute_boundaries()` (left/right boundary extraction from trajectory positions)
- **decoders.py** — `TrajectoryDecoder`, `TrajectoryDecoderTransformer`, `DensityDecoderTransformer`
- **regions.py** — `RegionTrunk`, `RegionTrunkSet`
- **assemblers.py** — `GridAssembler`
- **transformer_encoder.py** — `Tokenizer`, `EncoderLayer`, `Encoder`
- **axial_decoder.py** — `FourierTokenizer`, `AxialAttention`, `AxialDecoderLayer`, `AxialDecoder`
- **cross_decoder.py** — `CrossDecoderLayer`, `CrossDecoder`
- **shock_gnn.py** — `GatedMPNNLayer`, `ShockGNN` (optional, requires torch_geometric)
- **flux.py** — `Flux`, `GreenshieldsFlux`, `TriangularFlux`, `DEFAULT_FLUX`
- **characteristic_features.py** — `SegmentPhysicsEncoder`, `DiscontinuityPhysicsEncoder`, `CharacteristicFeatureComputer`
- **biased_cross_attention.py** — `BiasedCrossDecoderLayer`, `compute_characteristic_bias`, `compute_discontinuity_characteristic_bias`
- **breakpoint_evolution.py** — `BreakpointEvolution` (predicts breakpoint positions from adjacent segment pairs via cross-attention)

### Losses (`losses/`)

All losses inherit from `BaseLoss` with interface: `forward(input_dict, output_dict, target) -> (loss, components)`

- **base.py** — `BaseLoss` abstract class
- **flux.py** — `greenshields_flux()`, `greenshields_flux_derivative()`, `compute_shock_speed()`
- **mse.py** — `MSELoss` (grid MSE)
- **ic.py** — `ICLoss` (initial condition matching at t=0)
- **trajectory_consistency.py** — `TrajectoryConsistencyLoss` (analytical RH trajectory matching)
- **boundary.py** — `BoundaryLoss` (penalize shocks outside domain)
- **collision.py** — `CollisionLoss` (penalize colliding shocks)
- **existence_regularization.py** — `ICAnchoringLoss` (anchor trajectories to IC positions)
- **supervised_trajectory.py** — `SupervisedTrajectoryLoss` (supervised when GT available)
- **pde_residual.py** — `PDEResidualLoss`, `PDEShockResidualLoss`
- **rh_residual.py** — `RHResidualLoss` (Rankine-Hugoniot from sampled densities); also `compute_shock_velocity()`, `sample_density_from_grid()`
- **acceleration.py** — `AccelerationLoss` (shock detection via acceleration); also `compute_acceleration()`
- **regularize_traj.py** — `RegularizeTrajLoss` (penalize erratic trajectory jumps)
- **wasserstein.py** — `WassersteinLoss` (W1 / Earth Mover's Distance for sharp shocks)
- **conservation.py** — `ConservationLoss` (mass conservation regularizer)
- **cell_avg_mse.py** — `CellAverageMSELoss` (cell-average MSE for FV-consistent training with `CellSamplingTransform`)
- **visualize_losses.ipynb** — Jupyter notebook for visualizing loss components

### Plotting (`plotting/`)

- **base.py** — `save_figure()`, `_get_extent()`, `_get_colors()`, `_plot_heatmap()`, `_create_comparison_animation()`, `_log_figure()`
- **grid_plots.py** — `plot_prediction_comparison()`, `plot_error_map()`, `plot_comparison()`, `plot_grid_comparison()`, `plot_ground_truth()`, `plot_pred()`
- **trajectory_plots.py** — `plot_shock_trajectories()`, `plot_existence_heatmap()`, `plot_trajectory_on_grid()`, `plot_wavefront_trajectory()`, `plot_loss_curves()`, `plot_sample_predictions()`
- **wandb_trajectory_plots.py** — `plot_grid_with_trajectory_existence()`, `plot_grid_with_acceleration()`, `plot_trajectory_vs_analytical()`, `plot_existence()`, `plot_gt_traj()`
- **hybrid_plots.py** — `plot_prediction_with_trajectory_existence()`, `plot_mse_error()`, `plot_region_weights()`, `plot_pred_traj()`, `plot_hybrid_predictions()`
- **charno_plots.py** — `plot_selection_weights()`, `plot_winning_segment()`, `plot_selection_entropy()`, `plot_local_densities()`, `plot_charno_decomposition()`
- **wavefront_plots.py** — `plot_wave_pattern()` (wave lines overlaid on GT grid heatmap)

### Testing (`testing/`)

- **test_running.py** — `run_sanity_check()`, `run_profiler()`, `eval_inference_time()`, `run_inference()`
- **test_results.py** — `eval_model()`, `eval_steps()`, `eval_high_res()`, `collect_samples()`, `test_model()`
