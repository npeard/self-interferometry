#!/usr/bin/env python3

from torch import Tensor, nn


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
