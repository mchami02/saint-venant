"""Pure-data preset mappings for wavefront learning.

This module contains ONLY string-to-string/list mappings with ZERO imports
from the wavefront_learning package. This makes it a safe leaf node in the
dependency graph (no circular import risk).
"""

# ── Model → loss preset (auto-selected when --loss is default "mse") ──
# Models not listed here default to "mse".
MODEL_LOSS_PRESET: dict[str, str] = {
    "CVAEDeepONet": "cvae",
    "TrajTransformer": "traj_regularized",
    "ShockAwareDeepONet": "shock_proximity",
    "ShockAwareWaveNO": "shock_proximity",
    "AutoregressiveWaveNO": "traj_regularized",
    "NeuralFVSolver": "mse_wasserstein",
}

# ── Model → plot preset (auto-selected when --plot is None) ──
# Models not listed here default to "grid_residual".
MODEL_PLOT_PRESET: dict[str, str] = {
    "TrajDeepONet": "traj_residual",
    "TrajTransformer": "traj_residual",
    "WaveNO": "traj_residual",
    "WaveNOLocal": "traj_residual",
    "WaveNOIndepTraj": "traj_residual",
    "WaveNODisc": "traj_residual",
    "ClassifierTrajTransformer": "traj_existence",
    "ClassifierAllTrajTransformer": "traj_existence",
    "BiasedClassifierTrajTransformer": "traj_existence",
    "WaveNOCls": "traj_existence",
    "CTTBiased": "traj_existence",
    "CTTSegPhysics": "traj_existence",
    "CTTFiLM": "traj_existence",
    "CTTSeg": "traj_existence",
    "CharNO": "charno",
    "TransformerSeg": "grid_minimal",
    "WaveFrontModel": "wavefront",
    "CVAEDeepONet": "cvae",
    "ECARZ": "ecarz",
    "ShockAwareDeepONet": "shock_proximity",
    "ShockAwareWaveNO": "traj_residual",
    "AutoregressiveWaveNO": "traj_residual",
    "NeuralFVSolver": "grid_residual",
}

# ── Model → input transform ──
# String key into TRANSFORMS registry in data/transforms.py.
# Models not listed here default to None (no transform).
MODEL_TRANSFORM: dict[str, str | None] = {
    "FNO": "ToGridNoCoords",
    "AutoregressiveFNO": "ToGridNoCoords",
    "AutoregressiveRealFNO": "ToGridNoCoords",
    "AutoregressiveWaveNO": "ToGridNoCoords",
    "DeepONet": "ToGridInput",
    "EncoderDecoder": "ToGridInput",
    "EncoderDecoderCross": "ToGridInput",
    "ECARZ": "ToGridInput",
    "TrajTransformer": None,
    "ClassifierTrajTransformer": None,
    "ClassifierAllTrajTransformer": None,
    "BiasedClassifierTrajTransformer": None,
    "NoTrajTransformer": None,
    "CharNO": None,
    "WaveNO": None,
    "WaveNOCls": None,
    "WaveNOLocal": None,
    "WaveNOIndepTraj": None,
    "WaveNODisc": None,
    "ShockAwareWaveNO": None,
    "CTTBiased": None,
    "CTTSegPhysics": None,
    "CTTFiLM": None,
    "CTTSeg": None,
    "TransformerSeg": None,
    "WaveFrontModel": None,
    "LDDeepONet": "LDDeepONet",
    "CVAEDeepONet": "ToGridInput",
    "ShockAwareDeepONet": "ToGridInput",
    "NeuralFVSolver": "ToGridNoCoords",
}

# ── Loss presets ──
# Each preset is a list of (loss_name, weight) or (loss_name, weight, kwargs) tuples.
LOSS_PRESETS: dict[str, list[tuple[str, float] | tuple[str, float, dict]]] = {
    "mse": [
        ("mse", 1.0),
    ],
    "pde_shocks": [
        ("mse", 1.0),
        ("pde_shock_residual", 1.0),
    ],
    "cell_avg_mse": [
        ("cell_avg_mse", 1.0),
    ],
    "traj_regularized": [
        ("mse", 1.0),
        ("ic_anchoring", 0.1),
        ("boundary", 1.0),
        ("regularize_traj", 0.1),
    ],
    "cvae": [
        ("mse", 1.0),
        ("kl_divergence", 1.0, {"free_bits": 0.01}),
    ],
    "shock_proximity": [
        ("mse", 1.0),
        ("shock_proximity", 0.1),
    ],
    "mse_wasserstein": [
        ("mse", 1.0),
        ("wasserstein", 0.1),
    ],
}

# ── Plot presets ──
# Each preset lists which plots to generate.
PLOT_PRESETS: dict[str, list[str]] = {
    "traj_residual": [
        "ground_truth",
        "gt_traj",
        "pred_traj",
        "pred",
        "mse_error",
        "pde_residual",
    ],
    "grid_residual": [
        "ground_truth",
        "pred",
        "mse_error",
        "pde_residual",
    ],
    "traj_existence": [
        "ground_truth",
        "gt_traj",
        "pred_traj",
        "pred",
        "mse_error",
        "existence",
    ],
    "grid_minimal": [
        "ground_truth",
        "pred",
        "mse_error",
    ],
    "charno": [
        "ground_truth",
        "pred",
        "mse_error",
        "pde_residual",
        "charno_decomposition",
        "selection_weights",
        "winning_segment",
        "selection_entropy",
        "local_densities",
    ],
    "wavefront": [
        "ground_truth",
        "wave_pattern",
        "pred",
        "mse_error",
    ],
    "cvae": [
        "ground_truth",
        "pred",
        "mse_error",
        "cvae_uncertainty",
        "cvae_samples",
    ],
    "ecarz": [
        "arz_ground_truth",
        "arz_pred",
        "arz_mse_error",
    ],
    "shock_proximity": [
        "ground_truth",
        "pred",
        "mse_error",
        "shock_proximity",
    ],
}
