#!/usr/bin/env python

import contextlib
import random
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Union

import lightning as L
import matplotlib.pyplot as plt
import torch
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from self_interferometry.signal_analysis.datasets import get_data_loaders
from self_interferometry.signal_analysis.lightning_config import Standard, Teacher


@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""

    model_config: dict[str, Any]
    training_config: dict[str, Any]
    data_config: dict[str, Any]
    loss_config: dict[str, Any]
    is_hyperparameter_search: bool = False
    search_space: dict[str, list[Any]] | None = None

    def __post_init__(self):
        """Set default values for running checkpoints. Check points do not
        need all config variables.
        """
        if self.model_config == {}:
            print('TrainingConfig in checkpoint mode...')  # noqa: T201
            self._set_checkpoint_defaults()
        else:
            print('TrainingConfig in training mode...')  # noqa: T201

    def _set_checkpoint_defaults(self):
        """Set default values for running checkpoints. Check points do not
        need all config variables.
        """
        # Training defaults
        self.training_config.setdefault('batch_size', 64)

        # Data defaults
        self.data_config.setdefault(
            'data_dir', './signal_analysis/data'
        )
        self.data_config.setdefault('train_file', 'train.h5')
        self.data_config.setdefault('val_file', 'val.h5')
        self.data_config.setdefault('test_file', 'test.h5')
        self.data_config.setdefault('num_workers', 4)

    @classmethod
    def from_yaml(
        cls, config_path: str
    ) -> Union['TrainingConfig', list['TrainingConfig']]:
        """Load configuration from YAML file.

        If the file is a hyperparameter search config, returns a list of configs.
        Otherwise, returns a single config.

        Args:
            config_path: Path to YAML configuration file
        """
        with Path(config_path).open() as f:
            config_dict = yaml.safe_load(f)

        # Check if this is a hyperparameter search config
        if (
            any(isinstance(v, list) for v in config_dict['model'].values())
            or any(isinstance(v, list) for v in config_dict['training'].values())
            or any(isinstance(v, list) for v in config_dict['loss'].values())
        ):
            return cls._create_search_configs(config_dict)

        return cls(
            model_config=config_dict['model'],
            training_config=config_dict['training'],
            loss_config=config_dict['loss'],
            data_config=config_dict['data'],
        )

    @classmethod
    def _create_search_configs(
        cls, config_dict: dict[str, Any]
    ) -> list['TrainingConfig']:
        """Create multiple configurations for hyperparameter search."""
        # Separate list and non-list parameters
        model_lists = {
            k: v for k, v in config_dict['model'].items() if isinstance(v, list)
        }
        model_fixed = {
            k: v for k, v in config_dict['model'].items() if not isinstance(v, list)
        }

        training_lists = {
            k: v for k, v in config_dict['training'].items() if isinstance(v, list)
        }
        training_fixed = {
            k: v for k, v in config_dict['training'].items() if not isinstance(v, list)
        }

        loss_lists = {
            k: v for k, v in config_dict['loss'].items() if isinstance(v, list)
        }
        loss_fixed = {
            k: v for k, v in config_dict['loss'].items() if not isinstance(v, list)
        }

        # Generate all combinations
        model_keys = list(model_lists.keys())
        model_values = list(model_lists.values())

        training_keys = list(training_lists.keys())
        training_values = list(training_lists.values())

        loss_keys = list(loss_lists.keys())
        loss_values = list(loss_lists.values())

        configs = []

        # Generate model combinations
        model_combinations = list(product(*model_values)) if model_values else [()]
        training_combinations = (
            list(product(*training_values)) if training_values else [()]
        )
        loss_combinations = list(product(*loss_values)) if loss_values else [()]

        for model_combo in model_combinations:
            model_config = model_fixed.copy()
            model_config.update(dict(zip(model_keys, model_combo, strict=False)))

            for training_combo in training_combinations:
                training_config = training_fixed.copy()
                training_config.update(
                    dict(zip(training_keys, training_combo, strict=False))
                )

                for loss_combo in loss_combinations:
                    loss_config = loss_fixed.copy()
                    loss_config.update(dict(zip(loss_keys, loss_combo, strict=False)))

                    configs.append(
                        cls(
                            model_config=model_config,
                            training_config=training_config,
                            loss_config=loss_config,
                            data_config=config_dict['data'],
                            is_hyperparameter_search=True,
                            search_space={
                                'model': model_lists,
                                'training': training_lists,
                                'loss': loss_lists,
                            },
                        )
                    )

        # Randomly shuffle configurations
        random.shuffle(configs)
        return configs


class ModelTrainer:
    """Main trainer class for managing model training."""

    def __init__(
        self,
        config: TrainingConfig,
        experiment_name: str | None = None,
        checkpoint_dir: str | None = None,
    ):
        """Args:
        config: Training configuration
        experiment_name: Name for logging and checkpointing
        checkpoint_dir: Directory for saving checkpoints.
        """
        self.config = config
        self.experiment_name = experiment_name or config.model_config['type']
        self.checkpoint_dir = checkpoint_dir or './checkpoints'

        # Create checkpoint directory
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        if experiment_name != 'checkpoint_eval':
            # Setup data
            self.setup_data()

            # Create Lightning module
            self.lightning_module = self.create_lightning_module()

            # Setup training
            self.trainer = self.setup_trainer()

        # Check what version of PyTorch is installed
        print(torch.__version__)  # noqa: T201

        # Check the current CUDA version being used
        print('CUDA Version: ', torch.version.cuda)  # noqa: T201

        if torch.version.cuda is not None:
            # Check if CUDA is available and if so, print the device name
            print('Device name:', torch.cuda.get_device_properties('cuda').name)  # noqa: T201

            # Check if FlashAttention is available
            print('FlashAttention available:', torch.backends.cuda.flash_sdp_enabled())  # noqa: T201

    def setup_data(self):
        """Setup data loaders."""
        # Convert data_dir to absolute path
        base_dir = Path(__file__).parent.parent  # Go up two levels from training.py

        def resolve_path(data_dir: str, filename: str | None = None) -> str:
            """Resolve path relative to project root, optionally joining with
            filename.
            """
            # Remove leading './' if present
            data_dir = str(data_dir).lstrip('./')
            abs_dir = base_dir / data_dir

            # Create directory if it doesn't exist
            Path(abs_dir).mkdir(parents=True, exist_ok=True)

            # If filename is provided, join it with the directory
            return str(abs_dir / filename) if filename else str(abs_dir)

        # Get absolute data directory
        data_dir = self.config.data_config['data_dir']

        # Resolve paths for data files
        # Use real data inputs for the student and standard model
        # Student model uses targets from teacher
        if self.config.model_config.get('role') in ['standard', 'student']:
            train_path = resolve_path(
                data_dir, self.config.data_config.get('train_file')
            )
            val_path = resolve_path(data_dir, self.config.data_config.get('val_file'))
            test_path = resolve_path(data_dir, self.config.data_config.get('test_file'))
        # Use simulated data inputs and targets for the teacher model
        elif self.config.model_config.get('role') == 'teacher':
            train_path = resolve_path(
                data_dir, self.config.data_config.get('pretrain_file')
            )
            # Use real data inputs for validation and testing
            # We need to understand how well the teacher will predict
            # targets given real inputs
            val_path = resolve_path(data_dir, self.config.data_config.get('val_file'))
            test_path = resolve_path(data_dir, self.config.data_config.get('test_file'))
        else:
            raise ValueError(
                f'Unknown model role: {self.config.model_config.get("role")}'
            )

        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            batch_size=self.config.training_config['batch_size'],
            num_workers=self.config.data_config['num_workers'],
        )

    def create_lightning_module(self) -> Standard:
        """Create lightning module based on model type."""
        if self.config.model_config.get('role') == 'standard':
            return Standard(
                model_hparams=self.config.model_config,
                optimizer_hparams={
                    'name': self.config.training_config.get('optimizer', 'Adam'),
                    # TODO: why is this loaded as a string?
                    'lr': eval(self.config.training_config.get('learning_rate', 5e-4)),
                },
                scheduler_hparams={
                    'T_max': self.config.training_config.get('T_max', 500),
                    'eta_min': self.config.training_config.get('eta_min', 0),
                },
                loss_hparams=self.config.loss_config,
            )
        elif self.config.model_config.get('role') == 'teacher':
            return Teacher(
                model_hparams=self.config.model_config,
                optimizer_hparams={
                    'name': self.config.training_config.get('optimizer', 'Adam'),
                    # TODO: why is this loaded as a string?
                    'lr': eval(self.config.training_config.get('learning_rate', 5e-4)),
                },
                scheduler_hparams={
                    'T_max': self.config.training_config.get('T_max', 500),
                    'eta_min': self.config.training_config.get('eta_min', 0),
                },
                loss_hparams=self.config.loss_config,
                interferometer_config=self.config.data_config.get(
                    'interferometer_config'
                ),
            )
        else:
            raise TypeError("Unknown model type, can't initialize Lightning.")

    def setup_trainer(self) -> L.Trainer:
        """Setup Lightning trainer with callbacks and loggers."""
        callbacks = []
        # Add WandB logger if configured
        if self.config.training_config.get('use_logging', False):
            loggers = [
                WandbLogger(
                    project=self.config.training_config.get(
                        'wandb_project', 'ml-template'
                    ),
                    name=self.experiment_name,
                    save_dir=self.checkpoint_dir,
                )
            ]
            callbacks.append(LearningRateMonitor())
            callbacks.append(
                ModelCheckpoint(
                    dirpath=Path(self.checkpoint_dir) / self.experiment_name,
                    filename=str(loggers[0].experiment.id) + '_{epoch}-{val_total_loss:.4f}',
                    monitor='val_total_loss',
                    mode='min',
                    save_top_k=1,
                )
            )
        else:
            loggers = []

        # Get accelerator and device settings from config
        accelerator = self.config.training_config.get('accelerator', 'auto')
        devices = self.config.training_config.get('devices', 1)

        # Convert devices to proper type if it's a string
        if isinstance(devices, str):
            with contextlib.suppress(ValueError):
                devices = int(devices)

        return L.Trainer(
            max_epochs=self.config.training_config['max_epochs'],
            callbacks=callbacks,
            logger=loggers,
            check_val_every_n_epoch=7,
            accelerator=accelerator,
            devices=devices,
        )

    def train(self):
        """Train the model."""
        self.trainer.fit(
            self.lightning_module,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader,
        )

        # if self.config.training_config.get('use_logging', False):
        #     self.trainer.loggers[0].experiment.finish()

    def test(self):
        """Test the model."""
        if hasattr(self, 'test_loader'):
            self.trainer.test(self.lightning_module, dataloaders=self.test_loader)

    def plot_predictions_from_checkpoint(self, checkpoint_path: str):
        """Plot predictions from a checkpoint.

        Creates a plot with up to 4 rows:
        1. Predicted velocities vs targets
        2-4. Input signals (up to 3 photodiode channels)

        Args:
            checkpoint_path: Path to the checkpoint file
        """
        # Load the model from checkpoint
        model = Standard.load_from_checkpoint(checkpoint_path)
        trainer = L.Trainer(accelerator='cpu', logger=[])

        # Figure out which role the model was trained in so we setup the correct data
        model_role = model.model_hparams.get('role')
        if model_role == 'standard':
            self.config.model_config['role'] = 'standard'
        elif model_role == 'teacher':
            self.config.model_config['role'] = 'teacher'
        else:
            raise ValueError(f'Unknown model role: {model_role}')

        # Setup data loaders
        self.setup_data()

        # Get predictions
        predictions = trainer.predict(model, self.test_loader)

        # Process the first batch of predictions
        # predictions[batch_idx] returns a tuple of (velocity_hat, velocity_target, 
        # displacement_hat, displacement_target, signals)
        batch_idx = 0
        velocity_hat = predictions[batch_idx][0].cpu().numpy()  # Predicted velocities
        velocity_target = predictions[batch_idx][1].cpu().numpy()  # Target velocities
        displacement_hat = predictions[batch_idx][2].cpu().numpy()  # Predicted displacement
        displacement_target = predictions[batch_idx][3].cpu().numpy()  # Target displacement
        signals = predictions[batch_idx][4].cpu().numpy()  # Input signals

        print(f"Signals shape: {signals.shape}")
        print(f"Velocity hat shape: {velocity_hat.shape}")
        print(f"Velocity target shape: {velocity_target.shape}")
        print(f"Displacement hat shape: {displacement_hat.shape}")
        print(f"Displacement target shape: {displacement_target.shape}")

        # Get the number of samples in the batch and number of PD channels
        batch_size, num_channels, signal_length = signals.shape

        # Plot for each sample in the batch
        for i in range(
            min(batch_size, 10)
        ):  # Limit to 10 samples to avoid too many plots
            # Create a figure with subplots - one for velocities and up to 3 for signals
            fig, axs = plt.subplots(1 + num_channels, 1, figsize=(10, 8), sharex=True)

            # Plot predicted vs target velocities on the primary y-axis
            axs[0].plot(velocity_target[i], label='Target Velocity', color='blue')
            axs[0].plot(
                velocity_hat[i], label='Predicted Velocity', color='red', linestyle='--'
            )
            axs[0].set_title('Velocity and Displacement Comparison')
            axs[0].set_ylabel('Velocity (μm/s)', color='blue')
            axs[0].tick_params(axis='y', labelcolor='blue')
            #axs[0].legend(loc='upper left')
            axs[0].grid(True, alpha=0.3)
            
            # Create a twin axis for displacement
            ax_twin = axs[0].twinx()
            ax_twin.plot(displacement_target[i], label='Target Displacement', color='green')
            ax_twin.plot(
                displacement_hat[i], label='Predicted Displacement', color='orange', linestyle='--'
            )
            ax_twin.set_ylabel('Displacement (μm)', color='green')
            ax_twin.tick_params(axis='y', labelcolor='green')
            ax_twin.legend(loc='upper right')
            
            # Ensure both legends are visible
            lines1, labels1 = axs[0].get_legend_handles_labels()
            lines2, labels2 = ax_twin.get_legend_handles_labels()
            ax_twin.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            # Plot each input signal channel
            for j in range(num_channels):
                channel_names = ['PD1 (RP1_CH2)', 'PD2 (RP2_CH1)', 'PD3 (RP2_CH2)']
                axs[j + 1].plot(signals[i, j, :], label=f'Channel {j + 1}')
                axs[j + 1].set_title(f'Input Signal - {channel_names[j]}')
                axs[j + 1].set_ylabel('Amplitude')
                axs[j + 1].grid(True)

            # Set x-axis label for the bottom subplot
            axs[-1].set_xlabel('Sample')

            # Add overall title
            plt.suptitle(f'Sample {i + 1} from Batch {batch_idx + 1}')
            plt.tight_layout()
            plt.show()