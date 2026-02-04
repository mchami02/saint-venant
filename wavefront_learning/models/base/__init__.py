"""Base components and building blocks for wavefront learning models.

This module contains reusable components used by main models:
- Encoders: FourierFeatures, TimeEncoder, DiscontinuityEncoder, SpaceTimeEncoder
- Decoders: TrajectoryDecoder
- Blocks: ResidualBlock
- Regions: RegionTrunk, RegionTrunkSet
- Assemblers: GridAssembler
- Base class: BaseWavefrontModel
"""

from .assemblers import GridAssembler
from .base_model import BaseWavefrontModel
from .blocks import ResidualBlock
from .decoders import TrajectoryDecoder
from .encoders import (
    DiscontinuityEncoder,
    FourierFeatures,
    SpaceTimeEncoder,
    TimeEncoder,
)
from .regions import RegionTrunk, RegionTrunkSet

__all__ = [
    # Base class
    "BaseWavefrontModel",
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
]
