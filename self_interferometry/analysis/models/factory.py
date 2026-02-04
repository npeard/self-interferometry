"""Model factory for creating neural network models."""

import logging
from typing import Any

from torch import nn

from .barland_cnn import BarlandCNN, BarlandCNNConfig
from .stemtcan import StemTCAN, StemTCANConfig
from .tcn import TCN, TCNConfig
from .utcn import UTCN, UTCNConfig

# Conditional import for FNO and UNO - only available with torch >= 2.8
try:
    from .fno import NEURALOP_AVAILABLE, FNO1d, FNOConfig
    from .uno import UNO1d, UNOConfig
except ImportError:
    NEURALOP_AVAILABLE = False
    FNO1d = None
    FNOConfig = None
    UNO1d = None
    UNOConfig = None

logger = logging.getLogger(__name__)


def _create_barland_config(model_hparams: dict[str, Any]) -> BarlandCNNConfig:
    """Create BarlandCNNConfig from model configuration."""
    return BarlandCNNConfig(
        # Common parameters
        input_size=model_hparams['input_size'],
        output_size=model_hparams['output_size'],
        in_channels=model_hparams['in_channels'],
        activation=model_hparams['activation'],
        dropout=model_hparams['dropout'],
        # BarlandCNN specific parameters
        window_stride=model_hparams['window_stride'],
    )


def _create_tcn_config(model_hparams: dict[str, Any]) -> TCNConfig:
    """Create TCNConfig from model configuration."""
    return TCNConfig(
        # Common parameters
        sequence_length=model_hparams['input_size'],
        in_channels=model_hparams['in_channels'],
        activation=model_hparams['activation'],
        layer_norm=model_hparams['norm'],
        # TCN specific parameters
        kernel_size=model_hparams['kernel_size'],
        num_channels=model_hparams['num_channels'],
        dilation_base=model_hparams['dilation_base'],
    )


def _create_utcn_config(model_hparams: dict[str, Any]) -> UTCNConfig:
    """Create UTCNConfig from model configuration."""
    return UTCNConfig(
        # Common parameters
        sequence_length=model_hparams['input_size'],
        output_size=model_hparams['output_size'],
        in_channels=model_hparams['in_channels'],
        activation=model_hparams['activation'],
        layer_norm=model_hparams['norm'],
        dropout=model_hparams['dropout'],
        # UTCN specific parameters
        kernel_size=model_hparams['kernel_size'],
        n_layers=model_hparams['n_layers'],
        utcn_out_channels=model_hparams['utcn_out_channels'],
        utcn_dilations=model_hparams['utcn_dilations'],
        horizontal_skips_map=model_hparams['horizontal_skips_map'],
        horizontal_skip=model_hparams['horizontal_skip'],
        stride=model_hparams['stride'],
    )


def _create_stemtcan_config(model_hparams: dict[str, Any]) -> StemTCANConfig:
    """Create StemTCANConfig from model configuration."""
    return StemTCANConfig(
        # Common parameters
        input_size=model_hparams['input_size'],
        output_size=model_hparams['output_size'],
        in_channels=model_hparams['in_channels'],
        activation=model_hparams['activation'],
        norm=model_hparams['norm'],
        dropout=model_hparams['dropout'],
        # StemTCAN specific parameters
        kernel_size=model_hparams['kernel_size'],
        n_stem_blocks=model_hparams['n_stem_blocks'],
        n_post_attention_blocks=model_hparams['n_post_attention_blocks'],
        stem_out_channels=model_hparams['stem_out_channels'],
        post_attention_channels=model_hparams['post_attention_channels'],
        atten_len=model_hparams['atten_len'],
        atten_heads=model_hparams['atten_heads'],
        atten_chunk_size=model_hparams['atten_chunk_size'],
        dilation_base=model_hparams['dilation_base'],
        stride=model_hparams['stride'],
    )


def _create_fno_config(model_hparams: dict[str, Any]):
    """Create FNOConfig from model configuration."""
    if not NEURALOP_AVAILABLE:
        raise ImportError(
            'FNO model requires the neuralop library, which requires PyTorch >= 2.8. '
            'Please upgrade PyTorch or use a different model (CNN/TCN).'
        )
    return FNOConfig(
        # Common parameters (required by training interface)
        input_size=model_hparams['input_size'],
        output_size=model_hparams['output_size'],
        in_channels=model_hparams['in_channels'],
        # FNO specific parameters
        n_modes=(model_hparams['n_modes'],),  # Expects tuple of length 1
        hidden_channels=model_hparams['hidden_channels'],
        n_layers=model_hparams['n_layers'],
        max_n_modes=model_hparams['max_n_modes'],
        fno_block_precision=model_hparams['fno_block_precision'],
        use_channel_mlp=model_hparams['use_channel_mlp'],
        channel_mlp_dropout=model_hparams['channel_mlp_dropout'],
        channel_mlp_expansion=model_hparams['channel_mlp_expansion'],
        non_linearity=model_hparams['non_linearity'],
        stabilizer=model_hparams['stabilizer'],
        norm=model_hparams['norm'],
        preactivation=model_hparams['preactivation'],
        fno_skip=model_hparams['fno_skip'],
        channel_mlp_skip=model_hparams['channel_mlp_skip'],
        separable=model_hparams['separable'],
        factorization=model_hparams['factorization'],
        rank=model_hparams['rank'],
        joint_factorization=model_hparams['joint_factorization'],
        fixed_rank_modes=model_hparams['fixed_rank_modes'],
        implementation=model_hparams['implementation'],
        decomposition_kwargs=model_hparams['decomposition_kwargs'],
    )


def _create_uno_config(model_hparams: dict[str, Any]):
    """Create UNOConfig from model configuration."""
    if not NEURALOP_AVAILABLE:
        raise ImportError(
            'UNO model requires the neuralop library, which requires PyTorch >= 2.8. '
            'Please upgrade PyTorch or use a different model (CNN/TCN).'
        )
    return UNOConfig(
        # Common parameters (required by training interface)
        input_size=model_hparams['input_size'],
        output_size=model_hparams['output_size'],
        in_channels=model_hparams['in_channels'],
        # UNO specific parameters
        hidden_channels=model_hparams['hidden_channels'],
        lifting_channels=model_hparams['lifting_channels'],
        projection_channels=model_hparams['projection_channels'],
        positional_embedding=model_hparams['positional_embedding'],
        n_layers=model_hparams['n_layers'],
        uno_out_channels=model_hparams['uno_out_channels'],
        uno_n_modes=model_hparams['uno_n_modes'],
        uno_scalings=model_hparams['uno_scalings'],
        horizontal_skips_map=model_hparams['horizontal_skips_map'],
        incremental_n_modes=model_hparams['incremental_n_modes'],
        channel_mlp_dropout=model_hparams['channel_mlp_dropout'],
        channel_mlp_expansion=model_hparams['channel_mlp_expansion'],
        non_linearity=model_hparams['non_linearity'],
        norm=model_hparams['norm'],
        preactivation=model_hparams['preactivation'],
        fno_skip=model_hparams['fno_skip'],
        horizontal_skip=model_hparams['horizontal_skip'],
        channel_mlp_skip=model_hparams['channel_mlp_skip'],
        separable=model_hparams['separable'],
        factorization=model_hparams['factorization'],
        rank=model_hparams['rank'],
        fixed_rank_modes=model_hparams['fixed_rank_modes'],
        implementation=model_hparams['implementation'],
        decomposition_kwargs=model_hparams['decomposition_kwargs'],
        domain_padding=model_hparams['domain_padding'],
        domain_padding_mode=model_hparams['domain_padding_mode'],
    )


def create_model(model_hparams: dict[str, Any]) -> nn.Module | None:
    """Create model instance based on configuration.

    Args:
        model_hparams: Model hyperparameters dictionary containing 'type' key
                      and model-specific configuration parameters.

    Returns:
        A PyTorch model instance based on the configuration, or None for ensemble.

    Raises:
        ValueError: If unknown model type is specified.
        ImportError: If required dependencies are not available.
    """
    if model_hparams == 'ensemble':
        # This indicates that we are creating an ensemble of models
        # and no model will be created at this point.
        return None

    model_type = model_hparams['type']

    if model_type == 'Barland':
        logger.debug('Creating BarlandCNN model...')
        config = _create_barland_config(model_hparams)
        return BarlandCNN(config)
    elif model_type == 'TCN':
        logger.debug('Creating TCN model...')
        config = _create_tcn_config(model_hparams)
        return TCN(config)
    elif model_type == 'UTCN':
        logger.debug('Creating UTCN model...')
        config = _create_utcn_config(model_hparams)
        return UTCN(config)
    elif model_type == 'StemTCAN':
        logger.debug('Creating StemTCAN model...')
        config = _create_stemtcan_config(model_hparams)
        return StemTCAN(config)
    elif model_type == 'FNO':
        if not NEURALOP_AVAILABLE:
            raise ImportError(
                'FNO model requires the neuralop library, which requires PyTorch >= 2.8. '
                'Please upgrade PyTorch or use a different model (CNN/TCN).'
            )
        logger.debug('Creating FNO model...')
        config = _create_fno_config(model_hparams)
        return FNO1d(config)
    elif model_type == 'UNO':
        if not NEURALOP_AVAILABLE:
            raise ImportError(
                'UNO model requires the neuralop library, which requires PyTorch >= 2.8. '
                'Please upgrade PyTorch or use a different model (CNN/TCN).'
            )
        logger.debug('Creating UNO model...')
        config = _create_uno_config(model_hparams)
        return UNO1d(config)
    else:
        raise ValueError(f'Unknown model type: {model_type}')
