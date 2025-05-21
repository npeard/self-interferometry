#!/usr/bin/env python3

from dataclasses import dataclass

from torch import Tensor, nn

act_fn_by_name = {'LeakyReLU': nn.LeakyReLU(), 'ReLU': nn.ReLU()}


@dataclass
class CNNConfig:
    input_size: int = 256
    output_size: int = 1
    activation: str = 'LeakyReLU'
    in_channels: int = 1


class CNN(nn.Module):
    def __init__(self, config: CNNConfig):
        super().__init__()
        self.in_channels = config.in_channels
        self.conv_layers = nn.Sequential(
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
            nn.Dropout(0.1),
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
