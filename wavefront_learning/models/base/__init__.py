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
from .biased_cross_attention import (
    BiasedCrossDecoderLayer,
    CollisionTimeHead,
    compute_characteristic_bias,
)
from .blocks import ResidualBlock
from .boundaries import compute_boundaries
from .breakpoint_evolution import BreakpointEvolution
from .characteristic_features import (
    CharacteristicFeatureComputer,
    DiscontinuityPhysicsEncoder,
    SegmentPhysicsEncoder,
    TimeConditioner,
)
from .cross_decoder import CrossDecoder
from .decoders import (
    DensityDecoderTransformer,
    TrajectoryDecoder,
    TrajectoryDecoderTransformer,
)
from .deeponet_decoder import DeepONetDecoder
from .feature_encoders import (
    DiscontinuityEncoder,
    FourierFeatures,
    SpaceTimeEncoder,
    TimeEncoder,
)
from .flow_matching import ConditionEncoder, FlowMatchingDenoiser, HeunODESolver
from .collision_processor import process_collisions
from .flux import DEFAULT_FLUX, Flux, GreenshieldsFlux, TriangularFlux
from .regions import RegionTrunk, RegionTrunkSet
from .transformer_encoder import Encoder
from .vae_encoder import VAEEncoder
from .wave_builder import build_initial_waves
from .wave_reconstructor import reconstruct_grid

__all__ = [
    # Base class
    "BaseWavefrontModel",
    # Biased cross-attention
    "BiasedCrossDecoderLayer",
    "CollisionTimeHead",
    "compute_characteristic_bias",
    # Boundaries
    "compute_boundaries",
    # Breakpoint evolution
    "BreakpointEvolution",
    # Encoders
    "FourierFeatures",
    "TimeEncoder",
    "DiscontinuityEncoder",
    "SpaceTimeEncoder",
    # Decoders
    "TrajectoryDecoder",
    "TrajectoryDecoderTransformer",
    "DensityDecoderTransformer",
    # Blocks
    "ResidualBlock",
    # Latent diffusion components
    "VAEEncoder",
    "DeepONetDecoder",
    "ConditionEncoder",
    "FlowMatchingDenoiser",
    "HeunODESolver",
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
    # Wave construction & reconstruction
    "build_initial_waves",
    "reconstruct_grid",
    "process_collisions",
]
