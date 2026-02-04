#!/usr/bin/env python3

from dataclasses import dataclass

from torch import Tensor, nn

from .temporal_block import TemporalBlock


@dataclass
class TCNConfig:
    # Common parameters (consistent across all model configs)
    sequence_length: int  # sequence length, always same for input and output
    in_channels: int  # Number of input channels
    activation: str
    layer_norm: bool  # use layer normalization or not, batch norm causes leakage

    # TCN specific parameters
    kernel_size: int  # Kernel size for all layers
    num_channels: list[int]  # Number of channels in each layer
    dilation_base: int  # Base for dilation


class TCN(nn.Module):
    """Temporal Convolutional Network with dilated convolutions.

    This model efficiently processes the entire sequence at once and maintains
    the temporal resolution, making it ideal for time series prediction tasks.
    """

    def __init__(self, config: TCNConfig):
        super().__init__()
        self.sequence_length = config.sequence_length
        self.in_channels = config.in_channels
        self.dilation_base = config.dilation_base

        layers = []
        num_levels = len(config.num_channels)
        for i in range(num_levels):
            dilation_size = self.dilation_base**i  # Exponentially increasing dilation
            in_channels = config.in_channels if i == 0 else config.num_channels[i - 1]
            out_channels = config.num_channels[i]

            # Calculate padding to maintain sequence length
            padding = (config.kernel_size - 1) * dilation_size

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    config.kernel_size,
                    dilation=dilation_size,
                    padding=padding,
                    activation=config.activation,
                    layer_norm=config.layer_norm,
                )
            )

        self.network = nn.Sequential(*layers)

        # Final 1x1 convolution to map to output size
        self.output_layer = nn.Conv1d(config.num_channels[-1], 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the TCN.

        Args:
            x: Input tensor of shape [batch_size, in_channels, sequence_length]

        Returns:
            Tensor of shape [batch_size, 1, sequence_length] containing
            predicted velocity
        """
        # Process through the TCN network
        features = self.network(x)

        # Map to output using 1x1 convolution and return
        # Return output with shape [batch_size, 1, sequence_length]
        return self.output_layer(features)
