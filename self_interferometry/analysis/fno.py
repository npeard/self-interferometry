#!/usr/bin/env python3

import logging
from dataclasses import dataclass

from torch import Tensor, nn

# Conditional import for neuralop - requires torch >= 2.8
try:
    from neuralop.models import FNO
    from neuralop.utils import count_model_params

    NEURALOP_AVAILABLE = True
except ImportError:
    NEURALOP_AVAILABLE = False
    FNO = None  # Placeholder to avoid NameError

logger = logging.getLogger(__name__)

act_fn_by_name = {
    'LeakyReLU': nn.LeakyReLU(),
    'Tanh': nn.Tanh(),
    'ReLU': nn.ReLU(),
    'gelu': nn.GELU(),
}


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
    use_channel_mlp: bool
    channel_mlp_dropout: float
    channel_mlp_expansion: float
    non_linearity: str
    stabilizer: str | None
    norm: str | None
    preactivation: bool
    fno_skip: str
    channel_mlp_skip: str
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

        if not NEURALOP_AVAILABLE:
            raise ImportError(
                'FNO model requires the neuralop library, which requires PyTorch >= 2.8. '
                'Please upgrade PyTorch or use a different model (CNN/TCN).'
            )

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
            'use_channel_mlp': config.use_channel_mlp,
            'channel_mlp_dropout': config.channel_mlp_dropout,
            'channel_mlp_expansion': config.channel_mlp_expansion,
            'non_linearity': act_fn_by_name[config.non_linearity],
            'stabilizer': config.stabilizer,
            'norm': config.norm,
            'preactivation': config.preactivation,
            'fno_skip': config.fno_skip,
            'channel_mlp_skip': config.channel_mlp_skip,
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
        logger.info(f'Number of parameters in FNO: {count_model_params(self.fno)}')

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
