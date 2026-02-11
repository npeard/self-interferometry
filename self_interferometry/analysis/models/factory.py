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


def create_model(model_hparams: dict[str, Any]) -> nn.Module | None:
    """Create model instance based on configuration.

    Args:
        model_hparams: Model hyperparameters dictionary containing 'type' key
                      and model-specific configuration parameters.

    Returns:
        A PyTorch model instance based on the configuration.

    Raises:
        ValueError: If unknown model type is specified.
        ImportError: If required dependencies are not available.
    """
    model_type = model_hparams['type']
    # Remove 'type' from params since it's not part of any dataclass
    params = model_hparams.copy()
    params.pop('type', None)

    if model_type == 'Barland':
        logger.debug('Creating BarlandCNN model...')
        config = BarlandCNNConfig(**params)
        return BarlandCNN(config)
    elif model_type == 'TCN':
        logger.debug('Creating TCN model...')
        config = TCNConfig(**params)
        return TCN(config)
    elif model_type == 'UTCN':
        logger.debug('Creating UTCN model...')
        config = UTCNConfig(**params)
        return UTCN(config)
    elif model_type == 'StemTCAN':
        logger.debug('Creating StemTCAN model...')
        config = StemTCANConfig(**params)
        return StemTCAN(config)
    elif model_type == 'FNO':
        if not NEURALOP_AVAILABLE:
            raise ImportError(
                'FNO model requires the neuralop library, which requires PyTorch >= 2.8. '
                'Please upgrade PyTorch or use a different model (CNN/TCN).'
            )
        logger.debug('Creating FNO model...')
        # Special handling for n_modes tuple
        fno_params = params.copy()
        fno_params['n_modes'] = (fno_params['n_modes'],)
        config = FNOConfig(**fno_params)
        return FNO1d(config)
    elif model_type == 'UNO':
        if not NEURALOP_AVAILABLE:
            raise ImportError(
                'UNO model requires the neuralop library, which requires PyTorch >= 2.8. '
                'Please upgrade PyTorch or use a different model (CNN/TCN).'
            )
        logger.debug('Creating UNO model...')
        config = UNOConfig(**params)
        return UNO1d(config)
    else:
        raise ValueError(f'Unknown model type: {model_type}')
