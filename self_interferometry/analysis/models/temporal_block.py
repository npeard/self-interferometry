#!/usr/bin/env python3

from torch import Tensor, nn

from .chomp1d import Chomp1d
from .transpose import Transpose

act_fn_by_name = {
    'LeakyReLU': nn.LeakyReLU(),
    'Tanh': nn.Tanh(),
    'ReLU': nn.ReLU(),
    'GELU': nn.GELU(),
}


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
                Transpose(1, 2), nn.LayerNorm(in_channels), Transpose(1, 2)
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
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.activation = self.activation_fn

    def forward(self, x: Tensor) -> Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.activation(out + res)
