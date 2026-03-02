"""Model factory for creating neural network models."""

import logging
from typing import Any

from torch import nn

from .barland_cnn import BarlandCNN, BarlandCNNConfig
from .scnn import SCNN, SCNNConfig
from .tcan import TCAN, TCANConfig
from .tcn import TCN, TCNConfig
from .utcn import UTCN, UTCNConfig

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
    elif model_type == 'SCNN':
        logger.debug('Creating SCNN model...')
        config = SCNNConfig(**params)
        return SCNN(config)
    elif model_type == 'TCAN':
        logger.debug('Creating TCAN model...')
        config = TCANConfig(**params)
        return TCAN(config)

    else:
        raise ValueError(f'Unknown model type: {model_type}')
