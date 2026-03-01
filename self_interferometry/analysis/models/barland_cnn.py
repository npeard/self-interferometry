#!/usr/bin/env python3

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

act_fn_by_name = {'LeakyReLU': nn.LeakyReLU(), 'Tanh': nn.Tanh()}


@dataclass
class BarlandCNNConfig:
    # Common parameters (consistent across all model configs)
    sequence_length: int  # length of each input window (256)
    in_channels: int
    activation: str
    dropout: float

    # BarlandCNN specific parameters
    window_stride: int  # controls the sliding window stride in the forward pass


class BarlandCNN(nn.Module):
    def __init__(self, config: BarlandCNNConfig):
        super().__init__()
        self.config = config
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
            nn.Dropout(config.dropout),
            nn.Conv1d(64, 64, kernel_size=7),
            # Lout = 20, given L = 26
            act_fn_by_name[config.activation],
            nn.MaxPool1d(2),
            # Lout = 10, given L = 20
        )
        self.fc_layers = nn.Sequential(
            # length of input = 64 filters * 10 remaining time steps
            nn.Linear(640, 16),
            act_fn_by_name[config.activation],
            nn.Linear(16, 1),
        )

    @property
    def total_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _forward_windows(self, x: Tensor) -> Tensor:
        """Run a batch of windows through the CNN.

        Args:
            x: [n_windows, channels, sequence_length]

        Returns:
            [n_windows, 1] predictions
        """
        out = self.conv_layers(x)  # [n_windows, 64, 10]
        out = out.view(out.size(0), 1, -1)  # [n_windows, 1, 640]
        return self.fc_layers(out).squeeze(-1)  # [n_windows, 1]

    def forward(self, x: Tensor) -> Tensor:
        """Slide a fixed context window over the full input sequence.

        All windows are extracted simultaneously via F.unfold and processed in a
        single batched forward pass. Overlapping window predictions are averaged
        using the overlap-add pattern (F.fold + count normalisation).

        Args:
            x: [batch, channels, signal_length]

        Returns:
            [batch, 1, signal_length]
        """
        batch_size, in_channels, signal_length = x.shape
        window_size = self.config.sequence_length
        window_stride = self.config.window_stride

        # Pad so every time step falls inside at least one centred window
        padding = window_size // 2
        padded_length = signal_length + 2 * padding
        padded = F.pad(x, (padding, padding), mode='constant', value=0)

        # Extract all strided windows at once.
        # Treat the 1D signal as a 1×L "image" for F.unfold.
        # Result: [batch, in_channels*window_size, num_windows]
        windows = F.unfold(
            padded.unsqueeze(2), kernel_size=(1, window_size), stride=(1, window_stride)
        )
        num_windows = windows.shape[2]

        # Reshape to [batch*num_windows, in_channels, window_size]
        windows = windows.permute(0, 2, 1).reshape(
            batch_size * num_windows, in_channels, window_size
        )

        # Single batched forward pass → [batch*num_windows, 1]
        preds = self._forward_windows(windows)

        # Reshape to [batch, 1, num_windows] for F.fold
        preds = preds.reshape(batch_size, 1, num_windows)

        # F.fold output_size must satisfy: num_windows == floor((fold_length - 1) / stride) + 1
        # so fold_length = (num_windows - 1) * stride + 1.
        # We fold into this length then trim back to signal_length.
        fold_length = (num_windows - 1) * window_stride + 1

        # F.fold accumulates (sums) each window prediction into its stride position.
        folded = F.fold(
            preds,
            output_size=(1, fold_length),
            kernel_size=(1, 1),
            stride=(1, window_stride),
        )  # [batch, 1, 1, fold_length]

        # Build a count tensor to average overlapping predictions.
        ones = torch.ones(batch_size, 1, num_windows, device=x.device, dtype=x.dtype)
        counts = F.fold(
            ones,
            output_size=(1, fold_length),
            kernel_size=(1, 1),
            stride=(1, window_stride),
        )  # [batch, 1, 1, fold_length]

        averaged = folded / counts.clamp(min=1)

        # Remove the height dim and trim to exact signal_length
        return averaged.squeeze(2)[:, :, :signal_length]  # [batch, 1, signal_length]
