#!/usr/bin/env python3

import logging
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from self_interferometry.analysis.tcn import TemporalBlock

logger = logging.getLogger(__name__)


@dataclass
class UTCNConfig:
    # Common parameters (required by training interface)
    input_size: int  # Length of input sequence
    output_size: int  # Length of output sequence (same as input for our case)
    in_channels: int  # Number of input channels
    activation: str
    norm: str  # 'layer' or 'batch'
    dropout: float

    # UTCN specific parameters
    kernel_size: int  # Kernel size for all layers
    n_layers: int  # Total number of TemporalBlock layers in the U-Net
    utcn_out_channels: list[int]  # Output channels for each layer
    utcn_dilations: list[int]  # Dilation for each layer
    horizontal_skips_map: dict[int, int] | None  # Skip connections between layers
    horizontal_skip: str  # Type of horizontal skip connection ('linear', 'identity')
    stride: int  # Stride for all layers


class HorizontalSkip(nn.Module):
    """Horizontal skip connection for U-Net architecture."""

    def __init__(self, n_channels: int, skip_type: str = 'linear'):
        super().__init__()
        self.skip_type = skip_type

        if skip_type == 'linear':
            self.skip = nn.Conv1d(n_channels, n_channels, 1)
        elif skip_type == 'identity':
            self.skip = nn.Identity()
        else:
            raise ValueError(f'Unknown skip type: {skip_type}')

    def forward(self, x: Tensor) -> Tensor:
        return self.skip(x)


class UTCN(nn.Module):
    """U-shaped Temporal Convolutional Network with horizontal skip connections.

    This model implements a U-Net style architecture using TemporalBlocks from TCN.
    It includes encoder (downsampling path with increasing dilation), bottleneck,
    and decoder (upsampling path with decreasing dilation) with horizontal skip
    connections between corresponding encoder and decoder layers.
    """

    def __init__(self, config: UTCNConfig):
        super().__init__()
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.in_channels = config.in_channels
        self.n_layers = config.n_layers
        self.stride = config.stride

        # Validate layer-wise configurations
        assert len(config.utcn_out_channels) == config.n_layers, (
            f'utcn_out_channels length ({len(config.utcn_out_channels)}) '
            f'must match n_layers ({config.n_layers})'
        )
        assert len(config.utcn_dilations) == config.n_layers, (
            f'utcn_dilations length ({len(config.utcn_dilations)}) '
            f'must match n_layers ({config.n_layers})'
        )

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
            if i in self.horizontal_skips_map.keys():
                skip_source_idx = self.horizontal_skips_map[i]
                in_channels += config.utcn_out_channels[skip_source_idx]

            out_channels = config.utcn_out_channels[i]
            dilation = config.utcn_dilations[i]

            # Calculate padding to maintain sequence length
            padding = (config.kernel_size - self.stride) * dilation

            self.temporal_blocks.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    config.kernel_size,
                    stride=self.stride,
                    dilation=dilation,
                    padding=padding,
                    dropout=config.dropout,
                    activation=config.activation,
                    norm=config.norm,
                    input_length=config.input_size,
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
