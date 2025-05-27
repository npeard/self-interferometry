#!/usr/bin/env python3

from dataclasses import dataclass, field

from torch import Tensor, nn

act_fn_by_name = {'LeakyReLU': nn.LeakyReLU(), 'Tanh': nn.Tanh(), 'ReLU': nn.ReLU()}


@dataclass
class FCNConfig:
    """Configuration for the Fully Convolutional Network (FCN)."""

    # Common parameters (consistent across all model configs)
    input_size: int = 16384
    output_size: int = 16384
    in_channels: int = 1
    activation: str = 'LeakyReLU'
    dropout: float = 0.1

    # FCN specific parameters
    kernel_size: int = 7  # Kernel size for all layers
    num_channels: list[int] = field(
        default_factory=lambda: [16, 32, 64, 64]  # Same channel progression as TCN
    )

    # Final 1x1 convolution to map to output
    use_final_conv: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.activation not in act_fn_by_name:
            raise ValueError(
                f'Activation {self.activation} not supported. '
                f'Choose from {list(act_fn_by_name.keys())}'
            )


class ConvResidualBlock(nn.Module):
    """A convolutional block with residual connections."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float = 0.0,
        activation: str = 'LeakyReLU',
    ):
        super().__init__()

        # Calculate padding to maintain sequence length
        padding = (kernel_size - 1) // 2

        # Create the main convolutional layer
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        # Add activation function
        if activation in act_fn_by_name:
            self.activation = act_fn_by_name[activation]
        else:
            self.activation = nn.LeakyReLU()

        # Add dropout if specified
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # 1x1 convolution for residual connection if input and output channels differ
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        # Apply convolution
        out = self.conv(x)

        # Apply activation
        out = self.activation(out)

        # Apply dropout if specified
        if self.dropout is not None:
            out = self.dropout(out)

        # Apply residual connection
        if self.downsample is not None:
            residual = self.downsample(residual)

        return out + residual


class FCN(nn.Module):
    """Fully Convolutional Network for signal processing.

    This network is designed to be fully convolutional with residual connections,
    making it comparable to the TCN architecture while using standard convolutions
    instead of dilated convolutions.
    """

    def __init__(self, config: FCNConfig):
        super().__init__()
        self.config = config
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.in_channels = config.in_channels

        # Create layers list
        layers = []

        # Add initial normalization
        layers.append(nn.LayerNorm(config.input_size))

        # Create standard convolutional blocks with residual connections
        num_levels = len(config.num_channels)
        for i in range(num_levels):
            # Determine input and output channels
            in_ch = config.in_channels if i == 0 else config.num_channels[i - 1]
            out_ch = config.num_channels[i]

            # Add dropout only to later layers
            use_dropout = config.dropout if i >= num_levels // 2 else 0.0

            # Add the convolutional block with residual connection
            layers.append(
                ConvResidualBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=config.kernel_size,
                    dropout=use_dropout,
                    activation=config.activation,
                )
            )

        self.network = nn.Sequential(*layers)

        # Final 1x1 convolution to map to output size
        if config.use_final_conv:
            self.output_layer = nn.Conv1d(config.num_channels[-1], 1, 1)
        else:
            self.output_layer = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the FCN.

        Args:
            x: Input tensor of shape [batch_size, in_channels, sequence_length]

        Returns:
            Tensor of shape [batch_size, 1, sequence_length] containing
            predicted output
        """
        # Process through the network
        features = self.network(x)

        # Apply final 1x1 convolution if configured
        if self.output_layer is not None:
            out = self.output_layer(features)
            return out

        return features

    def get_output_size(self, input_size: int) -> int:
        """Calculate the output size for a given input size.

        With the dilated convolution approach and proper padding, the output size
        should be the same as the input size.

        Args:
            input_size: The length of the input sequence

        Returns:
            The length of the output sequence
        """
        # With proper padding in dilated convolutions, output size equals input size
        return input_size
