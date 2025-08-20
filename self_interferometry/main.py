import argparse
import random
import time
from pathlib import Path

import numpy as np
import yaml
from analysis.generate_data import generate_training_data_from_rp
from analysis.training import ModelTrainer, TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train a new model or test from checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='./analysis/configs/cnn-tcn-config.yaml',
        help='Path to YAML config file. Required for training, optional for testing.',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to checkpoint for testing. If provided, will run in test mode.',
    )
    parser.add_argument(
        '--acquire_dataset',
        action='store_true',
        help='Acquire real data from Red Pitaya for training, validation, and test '
        'datasets',
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

    # For testing mode (checkpoint provided), config is not necessary
    if args.checkpoint:
        print('Loading from checkpoint for quick plotting...')  # noqa: T201
        print(args.checkpoint)  # noqa: T201
        model_trainer = ModelTrainer(
            TrainingConfig({}, {}, {}, {}), experiment_name='checkpoint_eval'
        )
        model_trainer.plot_predictions_from_checkpoint(checkpoint_path=args.checkpoint)
        return

    # For training mode, load config
    if not args.config:
        raise ValueError('Config file is required for training mode')

    config = TrainingConfig.from_yaml(args.config)

    # Set random seed from time
    seed = setup_random_seed(int(time.time()))

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

        print('\nAcquiring datasets from Red Pitaya using default connection')  # noqa: T201
        # Create Red Pitaya Manager with default connection settings
        rp_manager = RedPitayaManager(
            ['rp-f0c04a.local', 'rp-f0c026.local'],
            blink_on_connect=True,
            data_save_path=data_config['data_dir'],
        )

        # Configure the Red Pitaya for acquisition
        try:
            # Configure the Red Pitaya for acquisition
            print('Configuring Red Pitaya for data acquisition...')  # noqa: T201
            rp_manager.reset_all()

            # Extract dataset parameters
            dataset_params = data_config['dataset_params']
            train_samples = dataset_params['train_samples']
            val_samples = dataset_params['val_samples']
            test_samples = dataset_params['test_samples']

            # Create datasets using the new function that accepts a RedPitayaManager
            generate_training_data_from_rp(
                rp_manager=rp_manager,
                output_dir=data_config['data_dir'],
                train_samples=train_samples,
                val_samples=val_samples,
                test_samples=test_samples,
            )
            print('Dataset acquisition complete!\n')  # noqa: T201
        finally:
            # Ensure we close the connection to the Red Pitaya
            rp_manager.close_all()

    # Train with each configuration
    for idx, train_config in enumerate(configs):
        print(f'\nStarting training run {idx + 1}/{len(configs)}')  # noqa: T201
        # Create trainer
        trainer = ModelTrainer(
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

    # Testing from checkpoint:
    # python main.py --checkpoint path/to/checkpoint.ckpt

    # Acquiring real data from Red Pitaya:
    # python main.py --config path/to/config.yaml --acquire_dataset

    main()
