import argparse
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from analysis.generate_data import generate_dataset_from_rp
from analysis.training_interface import TrainingConfig, TrainingInterface


def setup_logging(verbosity: str) -> None:
    """Setup logging configuration based on verbosity level."""
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
    }

    level = level_map.get(verbosity.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train a new model or test from checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='./analysis/configs/tcn-config.yaml',
        help='Path to YAML config file. Required for training, optional for testing.',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to checkpoint file for evaluation. Requires --dataset argument.',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Path to dataset file for checkpoint evaluation. Required when using --checkpoint.',
    )
    parser.add_argument(
        '--acquire_dataset',
        action='store_true',
        help='Acquire real data from Red Pitaya for training, validation, and test '
        'datasets',
    )
    parser.add_argument(
        '--verbosity',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Set the logging verbosity level (default: INFO)',
    )
    # IP address is now handled by RedPitayaManager defaults
    return parser.parse_args()


def setup_random_seed(seed: int | None = None) -> int:
    """Set random seed for reproducibility using NumPy's default_rng.

    Args:
        seed: Optional seed value. If None, a random seed will be generated.

    Returns:
        The seed value used.
    """
    if seed is None:
        # Create a non-seeded RNG to generate a seed
        temp_rng = np.random.default_rng()
        # Generate a random seed between 0 and 2^32 - 1
        seed = temp_rng.integers(0, 2**32)

    # Set seeds for both random and numpy
    random.seed(seed)
    np.random.default_rng(seed)

    return seed


def main():
    args = parse_args()

    # Setup logging based on verbosity argument
    setup_logging(args.verbosity)
    logger = logging.getLogger(__name__)

    # For checkpoint evaluation mode
    if args.checkpoint:
        if not args.dataset:
            raise ValueError(
                '--dataset argument is required when using --checkpoint. '
                'Please specify the path to the dataset file.'
            )

        logger.info('Loading from checkpoint for evaluation...')
        logger.info(f'Checkpoint path: {args.checkpoint}')
        logger.info(f'Dataset path: {args.dataset}')

        # Create trainer with no config for checkpoint evaluation
        trainer = TrainingInterface(config=None)
        trainer.plot_predictions_from_checkpoint(
            checkpoint_path=args.checkpoint, dataset_path=args.dataset
        )
        return

    # For training mode, load config
    if not args.config:
        raise ValueError('Config file is required for training mode')

    config = TrainingConfig.from_yaml(args.config)

    # Set random seed from time
    seed = setup_random_seed(int(time.time()))
    logger.info(f'Using random seed: {seed}')

    # Convert single config to list for unified processing
    configs = config if isinstance(config, list) else [config]
    base_config = configs[0]  # Use first config for dataset generation

    # Load data config from YAML file if not using training config
    data_config = (
        base_config.data_config if hasattr(base_config, 'data_config') else None
    )

    if not data_config and args.acquire_dataset:
        # Load data config directly from YAML file
        with Path(args.config).open() as f:
            config_data = yaml.safe_load(f)
            if 'data' in config_data:
                data_config = config_data['data']
            else:
                raise ValueError(f'Could not find data configuration in {args.config}')

    # Acquire real data from Red Pitaya if requested
    if args.acquire_dataset:
        from acquisition.redpitaya.manager import RedPitayaManager

        logger.info('Acquiring datasets from Red Pitaya using default connection')
        # Create Red Pitaya Manager with default connection settings
        rp_manager = RedPitayaManager(
            ['rp-f0c04a.local', 'rp-f0c026.local'],
            blink_on_connect=True,
            data_save_path=data_config['data_dir'],
        )

        # Configure the Red Pitaya for acquisition
        try:
            # Configure the Red Pitaya for acquisition
            logger.info('Configuring Red Pitaya for data acquisition...')
            rp_manager.reset_all()

            # Extract dataset parameters
            num_samples = data_config['num_acquire_samples']
            dataset_filename = data_config['dataset_file']

            # Create single dataset using the new function
            generate_dataset_from_rp(
                rp_manager=rp_manager,
                output_dir=data_config['data_dir'],
                num_samples=num_samples,
                dataset_filename=dataset_filename,
            )
            logger.info('Dataset acquisition complete!')
        finally:
            # Ensure we close the connection to the Red Pitaya
            rp_manager.close_all()
            sys.exit()

    # Train with each configuration
    for idx, train_config in enumerate(configs):
        logger.info(f'Starting training run {idx + 1}/{len(configs)}')
        # Create trainer
        trainer = TrainingInterface(
            config=train_config,
            experiment_name=train_config.training_config['experiment_name'],
            checkpoint_dir=train_config.training_config['checkpoint_dir'],
        )

        # Start training
        trainer.train()
        trainer.test()
        # Close the wandb logger if it was configured
        if trainer.config.training_config['use_logging']:
            trainer.trainer.loggers[0].experiment.finish()


if __name__ == '__main__':
    # Training new model:
    # python main.py --config path/to/config.yaml

    # Evaluating from checkpoint:
    # python main.py --checkpoint path/to/checkpoint.ckpt --dataset path/to/dataset.h5

    # Acquiring real data from Red Pitaya:
    # python main.py --config path/to/config.yaml --acquire_dataset

    main()
