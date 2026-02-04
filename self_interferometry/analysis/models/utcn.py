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
    # Common parameters (required by training interface)
    sequence_length: int  # Length of input sequence
    in_channels: int  # Number of input channels
    activation: str
    layer_norm: bool

    # UTCN specific parameters
    kernel_size: int  # Kernel size for all layers
    utcn_out_channels: list[int]  # Output channels for each layer
    utcn_dilations: list[int]  # Dilation for each layer, remember to start with 1!
    horizontal_skips_map: dict[int, int] | None  # Skip connections between layers
    horizontal_skip: str  # Type of horizontal skip connection ('linear', 'identity')

    def __post_init__(self):
        """Validate UTCN configuration parameters and infer n_layers."""
        # Infer n_layers from utcn_out_channels length
        self.n_layers = len(self.utcn_out_channels)

        # Validate layer-wise configurations
        if len(self.utcn_dilations) != self.n_layers:
            raise ValueError(
                f'utcn_dilations length ({len(self.utcn_dilations)}) '
                f'must match utcn_out_channels length ({self.n_layers})'
            )

        # Validate dilation values
        if self.utcn_dilations[0] != 1:
            raise ValueError('First dilation should be 1')
        if self.utcn_dilations[-1] != 1:
            raise ValueError('Last dilation should be 1')

        # Validate horizontal skip type
        if self.horizontal_skip not in {'linear', 'identity'}:
            raise ValueError(
                f'horizontal_skip must be "linear" or "identity", got "{self.horizontal_skip}"'
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

        self.utcn_out_channels = config.utcn_out_channels
        self.utcn_dilations = config.utcn_dilations

        # Setup horizontal skip connections (default: symmetric U-Net skips)
        if config.horizontal_skips_map is None:
            self.horizontal_skips_map = {}
            for i in range(config.n_layers // 2):
                # Example: if n_layers = 5, then 4:0, 3:1
                self.horizontal_skips_map[config.n_layers - i - 1] = i
        else:
            self.horizontal_skips_map = config.horizontal_skips_map

        # Lifting layer (input projection)
        self.lifting = nn.Conv1d(config.in_channels, config.utcn_out_channels[0], 1)

        # Create TemporalBlocks for each layer
        self.temporal_blocks = nn.ModuleList([])
        self.horizontal_skips = nn.ModuleDict({})

        for i in range(self.n_layers):
            # Determine input channels for this layer
            if i == 0:
                in_channels = config.utcn_out_channels[0]
            else:
                in_channels = config.utcn_out_channels[i - 1]

            # Add skip connection input channels if this layer receives a skip
            if i in self.horizontal_skips_map.keys():  # noqa: SIM118
                skip_source_idx = self.horizontal_skips_map[i]
                in_channels += config.utcn_out_channels[skip_source_idx]

            out_channels = config.utcn_out_channels[i]
            dilation = config.utcn_dilations[i]

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
            if i in self.horizontal_skips_map.values():
                self.horizontal_skips[str(i)] = HorizontalSkip(
                    config.utcn_out_channels[i], skip_type=config.horizontal_skip
                )

        # Projection layer (output projection)
        self.projection = nn.Conv1d(config.utcn_out_channels[-1], 1, 1)

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
            if layer_idx in self.horizontal_skips_map.keys():
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
