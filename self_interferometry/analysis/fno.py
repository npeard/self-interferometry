#!/usr/bin/env python3

from dataclasses import dataclass

from neuralop.models import FNO
from torch import Tensor, nn


@dataclass
class FNOConfig:
    # Common parameters (required by training interface)
    input_size: int  # Used for validation but FNO processes full sequence
    output_size: int  # Used for validation but FNO processes full sequence
    in_channels: int  # Number of input channels

    # FNO specific parameters
    n_modes: tuple[int, ...]  # Number of modes for each dimension
    hidden_channels: int
    n_layers: int
    max_n_modes: int | None
    fno_block_precision: str
    use_mlp: bool
    mlp_dropout: float
    mlp_expansion: float
    non_linearity: str
    stabilizer: str | None
    norm: str | None
    preactivation: bool
    fno_skip: str
    mlp_skip: str
    separable: bool
    factorization: str | None
    rank: float
    joint_factorization: bool
    fixed_rank_modes: bool
    implementation: str
    decomposition_kwargs: dict | None


class FNO1d(nn.Module):
    """Fourier Neural Operator for 1D signals.

    This model uses the neuralop library's FNO implementation adapted for
    1D time series data with multi-channel inputs. The FNO processes signals
    in the Fourier domain, making it particularly effective for capturing
    global dependencies in time series data.
    """

    def __init__(self, config: FNOConfig):
        super().__init__()
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.in_channels = config.in_channels

        # Set up FNO parameters
        fno_kwargs = {
            'n_modes': config.n_modes,
            'hidden_channels': config.hidden_channels,
            'in_channels': config.in_channels,
            'out_channels': 1,  # Always output single channel for velocity
            'n_layers': config.n_layers,
            'max_n_modes': config.max_n_modes,
            'fno_block_precision': config.fno_block_precision,
            'use_mlp': config.use_mlp,
            'mlp_dropout': config.mlp_dropout,
            'mlp_expansion': config.mlp_expansion,
            'non_linearity': config.non_linearity,
            'stabilizer': config.stabilizer,
            'norm': config.norm,
            'preactivation': config.preactivation,
            'fno_skip': config.fno_skip,
            'mlp_skip': config.mlp_skip,
            'separable': config.separable,
            'factorization': config.factorization,
            'rank': config.rank,
            'joint_factorization': config.joint_factorization,
            'fixed_rank_modes': config.fixed_rank_modes,
            'implementation': config.implementation,
        }

        # Remove None values to use defaults
        fno_kwargs = {k: v for k, v in fno_kwargs.items() if v is not None}

        # Add decomposition kwargs if provided
        if config.decomposition_kwargs:
            fno_kwargs['decomposition_kwargs'] = config.decomposition_kwargs

        self.fno = FNO(**fno_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the FNO.

        Args:
            x: Input tensor of shape [batch_size, in_channels, sequence_length]

        Returns:
            Tensor of shape [batch_size, 1, sequence_length] containing
            predicted velocity
        """
        # FNO expects input of shape [batch_size, in_channels, *spatial_dims]
        # For 1D signals, this is [batch_size, in_channels, sequence_length]
        return self.fno(x)
