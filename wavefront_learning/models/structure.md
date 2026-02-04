# wavefront_learning/models Structure

This directory contains neural network models for wavefront prediction.

## Directory Layout

```
models/
├── __init__.py              # Exports main models and BaseWavefrontModel
├── shock_trajectory_net.py  # ShockTrajectoryNet + build_shock_net()
├── hybrid_deeponet.py       # HybridDeepONet + build_hybrid_deeponet()
├── structure.md             # This file
└── base/                    # Submodules and building blocks
    ├── __init__.py          # Re-exports all base components
    ├── base_model.py        # BaseWavefrontModel (abstract base class)
    ├── encoders.py          # FourierFeatures, TimeEncoder, DiscontinuityEncoder, SpaceTimeEncoder
    ├── decoders.py          # TrajectoryDecoder
    ├── blocks.py            # ResidualBlock
    ├── regions.py           # RegionTrunk, RegionTrunkSet
    └── assemblers.py        # GridAssembler
```

## Main Models

| File | Model | Description |
|------|-------|-------------|
| `shock_trajectory_net.py` | `ShockTrajectoryNet` | DeepONet-like model for shock trajectory prediction |
| `hybrid_deeponet.py` | `HybridDeepONet` | Combined trajectory and region prediction |

## Building Blocks (base/)

| File | Components | Description |
|------|------------|-------------|
| `base_model.py` | `BaseWavefrontModel` | Abstract base class for wavefront models |
| `encoders.py` | `FourierFeatures`, `TimeEncoder`, `DiscontinuityEncoder`, `SpaceTimeEncoder` | Input encoding modules |
| `decoders.py` | `TrajectoryDecoder` | Decodes trajectory from branch/trunk embeddings |
| `blocks.py` | `ResidualBlock` | Residual MLP block with LayerNorm |
| `regions.py` | `RegionTrunk`, `RegionTrunkSet` | Density prediction for inter-shock regions |
| `assemblers.py` | `GridAssembler` | Assembles grid from region predictions with soft boundaries |

## Dependency Graph

```
base/blocks.py        (no deps - leaf)
base/assemblers.py    (no deps - leaf)
base/base_model.py    (no deps - leaf)
base/encoders.py      (self-contained - FourierFeatures used by SpaceTimeEncoder)
base/decoders.py      → imports blocks.ResidualBlock
base/regions.py       → imports blocks.ResidualBlock
base/__init__.py      → imports all above
     ↓
shock_trajectory_net.py  → imports from base
hybrid_deeponet.py       → imports from base
     ↓
models/__init__.py       → imports main models
```

## Usage

```python
# Import main models
from wavefront_learning.models import ShockTrajectoryNet, HybridDeepONet

# Import builder functions
from wavefront_learning.models import build_shock_net, build_hybrid_deeponet

# Import base components directly if needed
from wavefront_learning.models.base import FourierFeatures, ResidualBlock
```
