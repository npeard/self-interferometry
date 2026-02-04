#!/usr/bin/env python3

import logging
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .horizontal_skip import HorizontalSkip
from .temporal_block import TemporalBlock

logger = logging.getLogger(__name__)


@dataclass
class UTCNConfig:
    """Configuration for U-shaped Temporal Convolutional Network.

    A UTCN implements a U-Net style architecture with TemporalBlocks, featuring
    encoder-decoder structure with dilated convolutions and horizontal skip connections
    between corresponding layers.

    Args:
        sequence_length: Length of input sequence (time dimension)
        in_channels: Number of input channels (e.g., 3 for interferometer data)
        activation: Activation function name ('GELU', 'ReLU', etc.)
        layer_norm: Whether to use layer normalization in temporal blocks
        kernel_size: Convolution kernel size for all temporal blocks
        temporal_channels: Output channels for each layer in the network
        temporal_dilations: Dilation rates for each layer (must start and end with 1)
        horizontal_skips_map: Skip connection mapping between layers.
            None uses default U-Net symmetric skips, {} disables all skips
        horizontal_skip: Skip connection type ('linear', 'identity', or None).
            None disables horizontal skip connections entirely
    """

    sequence_length: int
    in_channels: int
    activation: str
    layer_norm: bool
    kernel_size: int
    temporal_channels: list[int]
    temporal_dilations: list[int]
    horizontal_skips_map: dict[int, int] | None
    horizontal_skip: str | None

    def __post_init__(self):
        """Validate UTCN configuration parameters and infer n_layers."""
        # Infer n_layers from temporal_channels length
        self.n_layers = len(self.temporal_channels)

        # Validate layer-wise configurations
        if len(self.temporal_dilations) != self.n_layers:
            raise ValueError(
                f'temporal_dilations length ({len(self.temporal_dilations)}) '
                f'must match temporal_channels length ({self.n_layers})'
            )

        # Validate dilation values
        if self.temporal_dilations[0] != 1:
            raise ValueError('First dilation should be 1')
        if self.temporal_dilations[-1] != 1:
            raise ValueError('Last dilation should be 1')

        # Handle horizontal skip configuration
        if self.horizontal_skip is None:
            # No horizontal skip connections
            self.horizontal_skips_map = {}
        elif self.horizontal_skip not in {'linear', 'identity'}:
            # Validate horizontal skip type
            raise ValueError(
                f'horizontal_skip must be "linear", "identity", or None, '
                f'got "{self.horizontal_skip}"'
            )


class UTCN(nn.Module):
    """U-shaped Temporal Convolutional Network with horizontal skip connections.

    This model implements a U-Net style architecture using TemporalBlocks from TCN.
    It includes encoder (downsampling path with increasing dilation), bottleneck,
    and decoder (upsampling path with decreasing dilation) with horizontal skip
    connections between corresponding encoder and decoder layers.
    """

    def __init__(self, config: UTCNConfig):
        super().__init__()
        self.sequence_length = config.sequence_length
        self.in_channels = config.in_channels
        self.n_layers = config.n_layers  # Inferred from utcn_out_channels length

        # Validation is now handled in UTCNConfig.__post_init__

        self.temporal_channels = config.temporal_channels
        self.temporal_dilations = config.temporal_dilations

        # Setup horizontal skip connections
        if config.horizontal_skips_map is None:
            # Default: symmetric U-Net skips
            self.horizontal_skips_map = {}
            for i in range(config.n_layers // 2):
                # Example: if n_layers = 5, then 4:0, 3:1
                self.horizontal_skips_map[config.n_layers - i - 1] = i
        else:
            # Use provided skip map, which may be empty (no horizontal skips)
            self.horizontal_skips_map = config.horizontal_skips_map

        # Lifting layer (input projection)
        self.lifting = nn.Conv1d(config.in_channels, config.temporal_channels[0], 1)

        # Create TemporalBlocks for each layer
        self.temporal_blocks = nn.ModuleList([])
        self.horizontal_skips = nn.ModuleDict({})

        for i in range(self.n_layers):
            # Determine input channels for this layer
            if i == 0:
                in_channels = config.temporal_channels[0]
            else:
                in_channels = config.temporal_channels[i - 1]

            # Add skip connection input channels if this layer receives a skip
            if (
                i in self.horizontal_skips_map.keys()  # noqa: SIM118
                and config.horizontal_skip is not None
            ):
                skip_source_idx = self.horizontal_skips_map[i]
                in_channels += config.temporal_channels[skip_source_idx]

            out_channels = config.temporal_channels[i]
            dilation = config.temporal_dilations[i]

            # Calculate padding to maintain sequence length
            padding = (config.kernel_size - 1) * dilation

            self.temporal_blocks.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    config.kernel_size,
                    dilation=dilation,
                    padding=padding,
                    activation=config.activation,
                    layer_norm=config.layer_norm,
                )
            )

            # Create horizontal skip connection if this layer provides a skip
            if (
                i in self.horizontal_skips_map.values()
                and config.horizontal_skip is not None
            ):
                self.horizontal_skips[str(i)] = HorizontalSkip(
                    config.temporal_channels[i], skip_type=config.horizontal_skip
                )

        # Projection layer (output projection)
        self.projection = nn.Conv1d(config.temporal_channels[-1], 1, 1)

        # Log model info
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f'Number of parameters in UTCN: {total_params:,}')

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the UTCN.

        Args:
            x: Input tensor of shape [batch_size, in_channels, sequence_length]

        Returns:
            Tensor of shape [batch_size, 1, sequence_length] containing
            predicted velocity
        """
        # Lifting: project input to initial feature dimension
        x = self.lifting(x)

        # Store skip connection outputs
        skip_outputs = {}

        # Process through all TemporalBlocks
        for layer_idx in range(self.n_layers):
            # Concatenate horizontal skip connection if applicable
            if layer_idx in self.horizontal_skips_map.keys():  # noqa: SIM118
                skip_source_idx = self.horizontal_skips_map[layer_idx]
                skip_val = skip_outputs[skip_source_idx]
                x = torch.cat([x, skip_val], dim=1)

            # Process through TemporalBlock
            x = self.temporal_blocks[layer_idx](x)

            # Store output for potential skip connection
            if layer_idx in self.horizontal_skips_map.values():
                skip_outputs[layer_idx] = self.horizontal_skips[str(layer_idx)](x)

        # Projection: map to output
        return self.projection(x)
