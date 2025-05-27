#!/usr/bin/env python3

from dataclasses import dataclass

from torch import Tensor, nn

act_fn_by_name = {'LeakyReLU': nn.LeakyReLU(), 'Tanh': nn.Tanh()}


@dataclass
class BarlandCNNConfig:
    input_size: int = 256
    output_size: int = 1
    activation: str = 'LeakyReLU'
    in_channels: int = 1
    dropout: float = 0.1
    window_stride: int = 128  # controls the window stride in the training loop


@dataclass
class TCNConfig:
    input_size: int = 16384  # Length of input sequence
    output_size: int = 16384  # Length of output sequence (same as input for our case)
    num_channels: list[int] = None  # Number of channels in each layer
    kernel_size: int = 7  # Kernel size for all layers
    dropout: float = 0.1
    activation: str = 'LeakyReLU'
    in_channels: int = 1  # Number of input channels

    def __post_init__(self):
        if self.num_channels is None:
            self.num_channels = [16, 32, 64, 64]  # Default channel configuration


class BarlandCNN(nn.Module):
    def __init__(self, config: BarlandCNNConfig):
        super().__init__()
        self.in_channels = config.in_channels
        self.conv_layers = nn.Sequential(
            nn.LayerNorm(config.input_size),
            nn.Conv1d(self.in_channels, 16, kernel_size=7),
            # Lout = 250, given L = 256
            act_fn_by_name[config.activation],
            nn.MaxPool1d(2),
            # Lout = 125, given L = 250
            nn.Conv1d(16, 32, kernel_size=7),
            # Lout = 119, given L = 125
            act_fn_by_name[config.activation],
            nn.MaxPool1d(2),
            # Lout = 59, given L = 119
            nn.Conv1d(32, 64, kernel_size=7),
            # Lout = 53, given L = 59
            act_fn_by_name[config.activation],
            nn.MaxPool1d(2),
            # Lout = 26, given L = 53
            nn.Dropout(config.dropout),
            nn.Conv1d(64, 64, kernel_size=7),
            # Lout = 20, given L = 26
            act_fn_by_name[config.activation],
            nn.MaxPool1d(2),
            # Lout = 10, given L = 20
        )
        self.fc_layers = nn.Sequential(
            # length of input = 64 filters * length of 10 left
            nn.Linear(640, 16),
            act_fn_by_name[config.activation],
            nn.Linear(16, config.output_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv_layers(x)  # expect out [128, 64, 10]
        out = out.view(out.size(0), 1, -1)  # expect out [128, 1, 640]
        return self.fc_layers(out)  # expect out [128, 1, 1]


class TemporalBlock(nn.Module):
    """Single block of temporal convolutions with dilation."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)  # Remove padding at the end
        self.activation = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)  # Remove padding at the end
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            nn.LayerNorm(16384),
            self.conv1,
            self.chomp1,
            self.activation,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.activation,
            self.dropout2,
        )

        # 1x1 convolution for residual connection if input and output channels differ
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.activation = nn.LeakyReLU()

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
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.in_channels = config.in_channels

        layers = []
        num_levels = len(config.num_channels)
        for i in range(num_levels):
            dilation_size = 2**i  # Exponentially increasing dilation
            in_channels = config.in_channels if i == 0 else config.num_channels[i - 1]
            out_channels = config.num_channels[i]

            # Calculate padding to maintain sequence length
            padding = (config.kernel_size - 1) * dilation_size

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    config.kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=config.dropout,
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

        # Map to output using 1x1 convolution
        out = self.output_layer(features)

        # Return output with shape [batch_size, 1, sequence_length]
        return out
