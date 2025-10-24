#!/usr/bin/env python3

import logging
from dataclasses import dataclass

from torch import Tensor, nn

# Conditional import for neuralop - requires torch >= 2.8
try:
    from neuralop.models import UNO
    from neuralop.utils import count_model_params

    NEURALOP_AVAILABLE = True
except ImportError:
    NEURALOP_AVAILABLE = False
    UNO = None  # Placeholder to avoid NameError

logger = logging.getLogger(__name__)

act_fn_by_name = {
    'LeakyReLU': nn.LeakyReLU(),
    'Tanh': nn.Tanh(),
    'ReLU': nn.ReLU(),
    'gelu': nn.GELU(),
}


@dataclass
class UNOConfig:
    # Common parameters (required by training interface)
    input_size: int  # Used for validation but UNO processes full sequence
    output_size: int  # Used for validation but UNO processes full sequence
    in_channels: int  # Number of input channels

    # UNO specific parameters
    hidden_channels: int
    lifting_channels: int
    projection_channels: int
    positional_embedding: str | None
    n_layers: int
    uno_out_channels: list[int]  # Output channels for each layer
    uno_n_modes: list[list[int]]  # Modes for each layer, per dimension
    uno_scalings: list[list[float]]  # Scaling factors for each layer
    horizontal_skips_map: dict[int, int] | None
    incremental_n_modes: tuple[int, ...] | None
    channel_mlp_dropout: float
    channel_mlp_expansion: float
    non_linearity: str
    norm: str | None
    preactivation: bool
    fno_skip: str
    horizontal_skip: str
    channel_mlp_skip: str
    separable: bool
    factorization: str | None
    rank: float
    fixed_rank_modes: bool
    implementation: str
    decomposition_kwargs: dict | None
    domain_padding: float | None
    domain_padding_mode: str


class UNO1d(nn.Module):
    """U-shaped Neural Operator for 1D signals.

    This model uses the neuralop library's UNO implementation adapted for
    1D time series data with multi-channel inputs. The UNO uses a U-shaped
    architecture with encoder-decoder structure and horizontal skip connections,
    processing signals in the Fourier domain for effective time series modeling.
    """

    def __init__(self, config: UNOConfig):
        super().__init__()

        if not NEURALOP_AVAILABLE:
            raise ImportError(
                'UNO model requires the neuralop library, which requires PyTorch >= 2.8. '
                'Please upgrade PyTorch or use a different model (CNN/TCN).'
            )

        self.input_size = config.input_size
        self.output_size = config.output_size
        self.in_channels = config.in_channels

        # Set up UNO parameters
        uno_kwargs = {
            'in_channels': config.in_channels,
            'out_channels': 1,  # Always output single channel for velocity
            'hidden_channels': config.hidden_channels,
            'lifting_channels': config.lifting_channels,
            'projection_channels': config.projection_channels,
            'positional_embedding': config.positional_embedding,
            'n_layers': config.n_layers,
            'uno_out_channels': config.uno_out_channels,
            'uno_n_modes': config.uno_n_modes,
            'uno_scalings': config.uno_scalings,
            'horizontal_skips_map': config.horizontal_skips_map,
            'incremental_n_modes': config.incremental_n_modes,
            'channel_mlp_dropout': config.channel_mlp_dropout,
            'channel_mlp_expansion': config.channel_mlp_expansion,
            'non_linearity': act_fn_by_name[config.non_linearity],
            'norm': config.norm,
            'preactivation': config.preactivation,
            'fno_skip': config.fno_skip,
            'horizontal_skip': config.horizontal_skip,
            'channel_mlp_skip': config.channel_mlp_skip,
            'separable': config.separable,
            'factorization': config.factorization,
            'rank': config.rank,
            'fixed_rank_modes': config.fixed_rank_modes,
            'implementation': config.implementation,
            'domain_padding': config.domain_padding,
            'domain_padding_mode': config.domain_padding_mode,
        }

        # Remove None values to use defaults
        uno_kwargs = {k: v for k, v in uno_kwargs.items() if v is not None}

        # Add decomposition kwargs if provided
        if config.decomposition_kwargs:
            uno_kwargs['decomposition_kwargs'] = config.decomposition_kwargs

        self.uno = UNO(**uno_kwargs)
        logger.info(f'Number of parameters in UNO: {count_model_params(self.uno)}')

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the UNO.

        Args:
            x: Input tensor of shape [batch_size, in_channels, sequence_length]

        Returns:
            Tensor of shape [batch_size, 1, sequence_length] containing
            predicted velocity
        """
        # UNO expects input of shape [batch_size, in_channels, *spatial_dims]
        # For 1D signals, this is [batch_size, in_channels, sequence_length]
        return self.uno(x)
