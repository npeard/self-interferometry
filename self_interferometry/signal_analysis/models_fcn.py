#!/usr/bin/env python3

from dataclasses import dataclass, field

from torch import Tensor, nn

act_fn_by_name = {'LeakyReLU': nn.LeakyReLU(), 'Tanh': nn.Tanh(), 'ReLU': nn.ReLU()}


@dataclass
class FCNConfig:
    """Configuration for the Fully Convolutional Network (FCN)."""

    input_size: int = 256
    output_size: int = 256
    activation: str = 'LeakyReLU'
    in_channels: int = 1
    dropout: float = 0.1
    window_stride: int = 128  # controls the window stride in the training loop

    # Convolutional block configurations
    # Each tuple contains (out_channels, kernel_size, stride, padding)
    conv_blocks: list[tuple[int, int, int, int]] = field(
        default_factory=lambda: [
            (16, 7, 1, 3),  # First conv block with padding to maintain size
            (32, 7, 1, 3),  # Second conv block with padding to maintain size
            (64, 7, 1, 3),  # Third conv block with padding to maintain size
            (64, 7, 1, 3),  # Fourth conv block with padding to maintain size
        ]
    )

    # Pooling configuration (set to 1 to maintain sequence length)
    pool_size: int = 1

    # Final 1x1 convolution to map to output
    use_final_conv: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if not self.conv_blocks:
            raise ValueError('At least one convolutional block must be specified')

        if self.activation not in act_fn_by_name:
            raise ValueError(
                f'Activation {self.activation} not supported. '
                f'Choose from {list(act_fn_by_name.keys())}'
            )


class ConvBlock(nn.Module):
    """A modular convolutional block with normalization, activation, and pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dropout: float = 0.0,
        activation: str = 'LeakyReLU',
        pool_size: int | None = None,
        use_norm: bool = True,
    ):
        super().__init__()

        layers = []

        # Add normalization if requested
        if use_norm:
            layers.append(nn.BatchNorm1d(in_channels))

        # Add convolutional layer
        layers.append(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        )

        # Add activation function
        if activation in act_fn_by_name:
            layers.append(act_fn_by_name[activation])

        # Add pooling if specified
        if pool_size is not None and pool_size > 1:
            layers.append(nn.MaxPool1d(pool_size))

        # Add dropout if specified
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class FCN(nn.Module):
    """Fully Convolutional Network for signal processing.

    This network is designed to be fully convolutional, with no fully connected
    layers, making it more suitable for processing signals of varying lengths.
    """

    def __init__(self, config: FCNConfig):
        super().__init__()
        self.config = config

        # Create sequential convolutional blocks
        conv_blocks = []
        in_channels = config.in_channels

        # Initial normalization of the input
        conv_blocks.append(nn.LayerNorm(config.input_size))

        # Add convolutional blocks
        for i, (out_channels, kernel_size, stride, padding) in enumerate(
            config.conv_blocks
        ):
            # Add dropout only to later layers
            use_dropout = config.dropout if i >= len(config.conv_blocks) // 2 else 0.0

            conv_blocks.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dropout=use_dropout,
                    activation=config.activation,
                    pool_size=config.pool_size,
                    use_norm=(
                        i > 0
                    ),  # Skip normalization for first conv after LayerNorm
                )
            )
            in_channels = out_channels

        self.feature_extractor = nn.Sequential(*conv_blocks)

        # Final 1x1 convolution to map to output size
        if config.use_final_conv:
            # Use 1x1 convolution to map to the desired number of output channels
            # while maintaining the sequence length
            self.output_layer = nn.Conv1d(
                in_channels, config.output_size, kernel_size=1
            )
        else:
            self.output_layer = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the FCN.

        Args:
            x: Input tensor of shape [batch_size, in_channels, sequence_length]

        Returns:
            Tensor of shape [batch_size, output_size, output_sequence_length]
        """
        # Extract features
        features = self.feature_extractor(x)

        # Apply final 1x1 convolution if configured
        if self.output_layer is not None:
            out = self.output_layer(features)
            return out

        return features

    def get_output_size(self, input_size: int) -> int:
        """Calculate the output size for a given input size.

        This is useful for determining the size of the output tensor
        for a given input size, which can vary based on the network
        configuration.

        Args:
            input_size: The length of the input sequence

        Returns:
            The length of the output sequence
        """
        size = input_size

        # Calculate size reduction through the network
        for out_channels, kernel_size, stride, padding in self.config.conv_blocks:
            # Conv1d size formula: L_out = (L_in + 2*padding - kernel_size) / stride + 1
            size = (size + 2 * padding - kernel_size) // stride + 1

            # MaxPool1d size formula: L_out = (L_in - pool_size) / pool_stride + 1
            # With default pool_stride = pool_size
            if self.config.pool_size > 1:
                size = (size - self.config.pool_size) // self.config.pool_size + 1

        # With the default configuration (padding=3, kernel_size=7, stride=1, pool_size=1),
        # the output size should be the same as the input size
        return size
