#!/usr/bin/env python3

from dataclasses import dataclass

import torch
from torch import Tensor, nn

act_fn_by_name = {
    'LeakyReLU': nn.LeakyReLU(),
    'Tanh': nn.Tanh(),
    'ReLU': nn.ReLU(),
    'GELU': nn.GELU(),
}


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


class Transpose(nn.Module):
    """Simple transpose module used to swap channel and sequence dimensions
    in TemporalBlock.
    """

    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(self.dim0, self.dim1)


class TemporalBlock(nn.Module):
    """Single block of temporal convolutions with dilation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        padding: int,
        activation: str,
        layer_norm: bool,
    ):
        super().__init__()
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)  # Remove padding at the end

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)  # Remove padding at the end

        # Select normalization layer
        if layer_norm:
            # LayerNorm across channel dimension requires transpose
            self.norm = nn.Sequential(
                Transpose(1, 2),
                nn.LayerNorm(in_channels),
                Transpose(1, 2),
            )
        else:
            self.norm = nn.Identity()

        # Select activation function
        if activation in act_fn_by_name:
            self.activation_fn = act_fn_by_name[activation]
        else:
            raise ValueError(f'Unknown activation: {activation}')

        self.net = nn.Sequential(
            self.norm,
            self.conv1,
            self.chomp1,
            self.activation_fn,
            self.conv2,
            self.chomp2,
            self.activation_fn,
        )

        # 1x1 convolution for residual connection if input and output channels differ
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )
        self.activation = self.activation_fn

    def forward(self, x: Tensor) -> Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.activation(out + res)


class Chomp1d(nn.Module):
    """Remove padding at the end of the sequence."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: Tensor) -> Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


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
