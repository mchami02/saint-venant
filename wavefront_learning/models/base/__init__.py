"""Base components and building blocks for wavefront learning models.

This module contains reusable components used by main models:
- Encoders: FourierFeatures, TimeEncoder, DiscontinuityEncoder, SpaceTimeEncoder
- Decoders: TrajectoryDecoder
- Blocks: ResidualBlock
- Regions: RegionTrunk, RegionTrunkSet
- Assemblers: GridAssembler
- Base class: BaseWavefrontModel
- Encoder-Decoder components: Encoder, AxialDecoder, CrossDecoder
"""

from .assemblers import GridAssembler
from .axial_decoder import AxialDecoder
from .base_model import BaseWavefrontModel
from .biased_cross_attention import BiasedCrossDecoderLayer, compute_characteristic_bias
from .breakpoint_evolution import BreakpointEvolution
from .blocks import ResidualBlock
from .characteristic_features import (
    CharacteristicFeatureComputer,
    DiscontinuityPhysicsEncoder,
    SegmentPhysicsEncoder,
    TimeConditioner,
)
from .cross_decoder import CrossDecoder
from .decoders import TrajectoryDecoder
from .feature_encoders import (
    DiscontinuityEncoder,
    FourierFeatures,
    SpaceTimeEncoder,
    TimeEncoder,
)
from .flux import DEFAULT_FLUX, Flux, GreenshieldsFlux, TriangularFlux
from .regions import RegionTrunk, RegionTrunkSet
from .transformer_encoder import Encoder

__all__ = [
    # Base class
    "BaseWavefrontModel",
    # Biased cross-attention
    "BiasedCrossDecoderLayer",
    "compute_characteristic_bias",
    # Breakpoint evolution
    "BreakpointEvolution",
    # Encoders
    "FourierFeatures",
    "TimeEncoder",
    "DiscontinuityEncoder",
    "SpaceTimeEncoder",
    # Decoders
    "TrajectoryDecoder",
    # Blocks
    "ResidualBlock",
    # Regions
    "RegionTrunk",
    "RegionTrunkSet",
    # Assemblers
    "GridAssembler",
    # Encoder-Decoder components
    "Encoder",
    "AxialDecoder",
    "CrossDecoder",
    # Characteristic features & flux
    "Flux",
    "GreenshieldsFlux",
    "TriangularFlux",
    "DEFAULT_FLUX",
    "SegmentPhysicsEncoder",
    "DiscontinuityPhysicsEncoder",
    "CharacteristicFeatureComputer",
    "TimeConditioner",
]
