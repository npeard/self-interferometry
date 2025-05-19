import argparse
import random
import time

import numpy as np
import yaml
from biphase_gpt.training import ModelTrainer, TrainingConfig

from signal_analysis.datasets import (
    create_train_val_test_datasets,
    create_train_val_test_datasets_from_rp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train a new model or test from checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='./signal_analysis/configs/model_config.yaml',
        help='Path to YAML config file. Required for training, optional for testing.',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to checkpoint for testing. If provided, will run in test mode.',
    )
    parser.add_argument(
        '--regenerate_datasets',
        action='store_true',
        help='Regenerate training, validation, and test datasets',
    )
    parser.add_argument(
        '--acquire_dataset',
        action='store_true',
        help='Acquire real data from Red Pitaya for training, validation, and test datasets',
    )
    # IP address is now handled by RedPitayaManager defaults
    return parser.parse_args()


def setup_random_seed(seed: int | None = None) -> int:
    """Set random seed for reproducibility."""
    if seed is None:
        # Generate a random seed between 0 and 2^32 - 1
        seed = random.randint(0, 2**32 - 1)

    random.seed(seed)
    np.random.seed(seed)
    return seed


def main():
    args = parse_args()

    # For testing mode (checkpoint provided), config is not necessary
    if args.checkpoint:
        print('Loading from checkpoint for quick plotting...')
        print(args.checkpoint)
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

    if not data_config and (args.regenerate_datasets or args.acquire_dataset):
        # Load data config directly from YAML file
        with open(args.config) as f:
            config_data = yaml.safe_load(f)
            if 'data' in config_data:
                data_config = config_data['data']
            else:
                raise ValueError(f'Could not find data configuration in {args.config}')

    # Regenerate datasets if requested
    if args.regenerate_datasets:
        print(f'\nRegenerating datasets with random seed: {seed}')
        create_train_val_test_datasets(
            output_dir=data_config['data_dir'], **data_config.get('dataset_params', {})
        )
        print('Dataset regeneration complete!\n')

    # Acquire real data from Red Pitaya if requested
    if args.acquire_dataset:
        from redpitaya.manager import RedPitayaManager

        print('\nAcquiring datasets from Red Pitaya using default connection')
        # Create Red Pitaya Manager with default connection settings
        rp_manager = RedPitayaManager(
            ['rp-f0c04a.local', 'rp-f0c026.local'],
            blink_on_connect=True,
            data_save_path=data_config['data_dir'],
        )

        # Configure the Red Pitaya for acquisition
        try:
            # Configure the Red Pitaya for acquisition
            print('Configuring Red Pitaya for data acquisition...')
            rp_manager.reset_all()

            # Extract dataset parameters
            dataset_params = data_config.get('dataset_params', {})
            train_samples = dataset_params.get('train_samples', 1000)
            val_samples = dataset_params.get('val_samples', 100)
            test_samples = dataset_params.get('test_samples', 100)

            # Create datasets using the new function that accepts a RedPitayaManager
            create_train_val_test_datasets_from_rp(
                rp_manager=rp_manager,
                output_dir=data_config['data_dir'],
                train_samples=train_samples,
                val_samples=val_samples,
                test_samples=test_samples,
                # Additional parameters for acquisition
                device_idx=0,
                delay_between_shots=0.5,
                timeout=5,
            )
            print('Dataset acquisition complete!\n')
        finally:
            # Ensure we close the connection to the Red Pitaya
            rp_manager.close_all()

    # # Train with each configuration
    # for idx, train_config in enumerate(configs):
    #     print(f'\nStarting training run {idx + 1}/{len(configs)}')
    #     # Create trainer
    #     trainer = ModelTrainer(
    #         config=train_config,
    #         experiment_name=train_config.training_config.get('experiment_name'),
    #         checkpoint_dir=train_config.training_config.get('checkpoint_dir'),
    #     )

    #     # Start training
    #     trainer.train()
    #     trainer.test()
    #     # Close the wandb logger if it was configured
    #     if trainer.config.training_config.get('use_logging', False):
    #         trainer.trainer.loggers[0].experiment.finish()


if __name__ == '__main__':
    # Training new model:
    # python main.py --config path/to/config.yaml

    # Testing from checkpoint:
    # python main.py --checkpoint path/to/checkpoint.ckpt

    # Fine-tuning from checkpoint:
    # python main.py --config path/to/config.yaml --checkpoint path/to/checkpoint.ckpt

    # Regenerating datasets:
    # python main.py --config path/to/config.yaml --regenerate_datasets

    main()
