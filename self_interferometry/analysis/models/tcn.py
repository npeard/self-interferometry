#!/usr/bin/env python3

from dataclasses import dataclass

from .utcn import UTCN, UTCNConfig


@dataclass
class TCNConfig(UTCNConfig):
    """Configuration for Temporal Convolutional Network.

    TCN is a specialized case of UTCN with no horizontal skip connections and
    exponentially increasing dilations. It implements a causal temporal convolutional
    architecture for sequence modeling.

    Args:
        dilation_base: Base for exponential dilation sequence (default: 2).
            Dilations will be [1, base, base^2, base^3, ...] for each layer
        temporal_dilations: Computed automatically based on dilation_base
        horizontal_skips_map: Always None for TCN (no skip connections)
        horizontal_skip: Always None for TCN (no skip connections)

    All other parameters inherited from UTCNConfig:
        sequence_length, in_channels, activation, layer_norm, kernel_size,
        temporal_channels
    """

    temporal_dilations: list[int] = None
    horizontal_skips_map: dict[int, int] | None = None
    horizontal_skip: str | None = None
    dilation_base: int = 2

    def __post_init__(self):
        """Set TCN-specific parameters and validate."""
        # TCN has no horizontal skip connections
        self.horizontal_skip = None
        self.horizontal_skips_map = None

        # Generate exponential dilations based on dilation_base
        self.temporal_dilations = [
            self.dilation_base**i for i in range(len(self.temporal_channels))
        ]

        # TCN-specific validation (subset of UTCN validation)
        self.n_layers = len(self.temporal_channels)

        # Validate layer-wise configurations
        if len(self.temporal_dilations) != self.n_layers:
            raise ValueError(
                f'temporal_dilations length ({len(self.temporal_dilations)}) '
                f'must match temporal_channels length ({self.n_layers})'
            )

        # TCN only requires first dilation to be 1 (no last dilation constraint)
        if self.temporal_dilations[0] != 1:
            raise ValueError('First dilation should be 1')


class TCN(UTCN):
    """Temporal Convolutional Network with dilated convolutions.

    TCN is a special case of UTCN with no horizontal skip connections and
    exponential dilations. Inherits all functionality from UTCN but with
    TCN-specific configuration.
    """
