"""Preset mappings loaded from presets.yaml.

All existing imports (e.g. ``from configs.presets import LOSS_PRESETS``)
continue to work unchanged.
"""

from configs.loader import load_presets

_presets = load_presets()

MODEL_LOSS_PRESET: dict[str, str] = _presets["model_loss_preset"]
MODEL_PLOT_PRESET: dict[str, str] = _presets["model_plot_preset"]
MODEL_TRANSFORM: dict[str, str | None] = _presets["model_transform"]
LOSS_PRESETS: dict[str, list[tuple[str, float] | tuple[str, float, dict]]] = _presets[
    "loss_presets"
]
PLOT_PRESETS: dict[str, list[str]] = _presets["plot_presets"]
