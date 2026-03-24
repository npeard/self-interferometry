#!/usr/bin/env python

import contextlib
import logging
import random
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Union

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from self_interferometry.analysis.datasets import get_data_loaders
from self_interferometry.analysis.lit_module import LitModule
from self_interferometry.analysis.synthetic_lit_module import (
    SyntheticIndexDataset,
    SyntheticLitModule,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""

    model_config: dict[str, Any]
    training_config: dict[str, Any]
    data_config: dict[str, Any]
    loss_config: dict[str, Any]
    synthetic_config: dict[str, Any] | None = None
    is_hyperparameter_search: bool = False
    search_space: dict[str, list[Any]] | None = None

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

        synthetic_config = config_dict.get('synthetic')

        # Check if this is a hyperparameter search config
        # Also check synthetic section for list-valued params (e.g. wavelengths_nm)
        has_search_params = (
            any(isinstance(v, list) for v in config_dict['model'].values())
            or any(isinstance(v, list) for v in config_dict['training'].values())
            or any(isinstance(v, list) for v in config_dict['loss'].values())
            or any(isinstance(v, list) for v in config_dict['data'].values())
            or (
                synthetic_config is not None
                and any(isinstance(v, list) for v in synthetic_config.values())
            )
        )
        if has_search_params:
            return cls._create_search_configs(config_dict)

        return cls(
            model_config=config_dict['model'],
            training_config=config_dict['training'],
            loss_config=config_dict['loss'],
            data_config=config_dict['data'],
            synthetic_config=synthetic_config,
        )

    @classmethod
    def _split_list_params(
        cls, section: dict[str, Any]
    ) -> tuple[dict[str, list], dict[str, Any]]:
        """Split a config section into list (search) and fixed parameters."""
        lists = {k: v for k, v in section.items() if isinstance(v, list)}
        fixed = {k: v for k, v in section.items() if not isinstance(v, list)}
        return lists, fixed

    @classmethod
    def _create_search_configs(
        cls, config_dict: dict[str, Any]
    ) -> list['TrainingConfig']:
        """Create multiple configurations for hyperparameter search."""
        # Separate list and non-list parameters for each section
        model_lists, model_fixed = cls._split_list_params(config_dict['model'])
        training_lists, training_fixed = cls._split_list_params(config_dict['training'])
        loss_lists, loss_fixed = cls._split_list_params(config_dict['loss'])
        data_lists, data_fixed = cls._split_list_params(config_dict['data'])

        # Handle synthetic section (optional)
        synthetic_raw = config_dict.get('synthetic')
        if synthetic_raw is not None:
            synthetic_lists, synthetic_fixed = cls._split_list_params(synthetic_raw)
        else:
            synthetic_lists, synthetic_fixed = {}, {}

        # Gather all sections for combination generation
        sections = [
            ('model', model_lists, model_fixed),
            ('training', training_lists, training_fixed),
            ('loss', loss_lists, loss_fixed),
            ('data', data_lists, data_fixed),
            ('synthetic', synthetic_lists, synthetic_fixed),
        ]

        # Build keys/values for each section
        all_keys = []
        all_values = []
        section_names = []
        for name, lists, _ in sections:
            for k, v in lists.items():
                all_keys.append((name, k))
                all_values.append(v)
                section_names.append(name)

        # Generate all combinations across all sections
        all_combinations = list(product(*all_values)) if all_values else [()]

        configs = []
        for combo in all_combinations:
            # Start with fixed values for each section
            section_configs = {
                'model': model_fixed.copy(),
                'training': training_fixed.copy(),
                'loss': loss_fixed.copy(),
                'data': data_fixed.copy(),
                'synthetic': synthetic_fixed.copy(),
            }

            # Apply the search values from this combination
            for (section_name, key), value in zip(all_keys, combo, strict=False):
                section_configs[section_name][key] = value

            # Build synthetic_config (None if no synthetic section)
            synthetic_config = (
                section_configs['synthetic'] if synthetic_raw is not None else None
            )

            configs.append(
                cls(
                    model_config=section_configs['model'],
                    training_config=section_configs['training'],
                    loss_config=section_configs['loss'],
                    data_config=section_configs['data'],
                    synthetic_config=synthetic_config,
                    is_hyperparameter_search=True,
                    search_space={
                        'model': model_lists,
                        'training': training_lists,
                        'loss': loss_lists,
                        'data': data_lists,
                        'synthetic': synthetic_lists,
                    },
                )
            )

        # Randomly shuffle configurations
        random.shuffle(configs)
        return configs


class TrainingInterface:
    """Main trainer class for managing model training."""

    CHECKPOINT_DIR = Path(__file__).parent / 'models' / 'checkpoints'

    def __init__(
        self, config: TrainingConfig | None = None, experiment_name: str | None = None
    ):
        """Initialize the training interface.

        Args:
            config: Training configuration (None for checkpoint evaluation mode)
            experiment_name: Name for logging and checkpointing
        """
        self.config = config
        self.checkpoint_dir = str(self.CHECKPOINT_DIR)

        # Only setup training components if config is provided
        if config is not None:
            self.experiment_name = experiment_name or config.model_config['type']

            # Create checkpoint directory
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

            # Setup data
            self.setup_data()

            # Create Lightning module
            self.lightning_module = self.create_lightning_module()

            # Setup training
            self.trainer = self.setup_trainer()

        # Check what version of PyTorch is installed
        logger.info(f'PyTorch version: {torch.__version__}')

        # Check the current CUDA version being used
        logger.info(f'CUDA version: {torch.version.cuda}')

        if torch.version.cuda is not None:
            # Check if CUDA is available and if so, print the device name
            logger.info(f'Device name: {torch.cuda.get_device_properties("cuda").name}')

            # Check if FlashAttention is available
            logger.info(
                f'FlashAttention available: {torch.backends.cuda.flash_sdp_enabled()}'
            )

    def _is_synthetic(self) -> bool:
        """Check if synthetic training mode is enabled."""
        return (
            self.config.synthetic_config is not None
            and self.config.synthetic_config.get('use_synthetic_training', False)
        )

    def setup_data(
        self,
        dataset_path: str | None = None,
        batch_size: int | None = None,
        split_ratios: tuple[int, int, int] | None = None,
        num_workers: int | None = None,
    ):
        """Setup data loaders.

        For synthetic training mode, creates dummy dataloaders that provide
        batch indices (actual data is generated on-device in the LitModule).
        For real data, loads HDF5 datasets as before.

        Args:
            dataset_path: Path to dataset file (if None, uses config)
            batch_size: Batch size (if None, uses config)
            split_ratios: Train/val/test split ratios (if None, uses config)
            num_workers: Number of data loader workers (if None, uses config)
        """
        if batch_size is None:
            batch_size = self.config.training_config['batch_size']

        # Synthetic training: create dummy dataloaders
        if self._is_synthetic():
            syn = self.config.synthetic_config
            steps = syn['steps_per_epoch']
            val_steps = syn.get('val_steps', steps // 5)

            train_ds = SyntheticIndexDataset(steps * batch_size)
            val_ds = SyntheticIndexDataset(val_steps * batch_size)

            self.train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True
            )
            self.val_loader = DataLoader(val_ds, batch_size=batch_size)
            self.test_loader = self.val_loader
            return

        # Real data: load HDF5 datasets
        base_dir = Path(__file__).parent.parent

        def resolve_path(data_dir: str, filename: str | None = None) -> str:
            """Resolve path relative to project root."""
            data_dir = str(data_dir).lstrip('./')
            abs_dir = base_dir / data_dir
            Path(abs_dir).mkdir(parents=True, exist_ok=True)
            return str(abs_dir / filename) if filename else str(abs_dir)

        if dataset_path is None:
            data_dir = self.config.data_config['data_dir']
            dataset_path = resolve_path(
                data_dir, self.config.data_config['dataset_file']
            )

        if split_ratios is None:
            split_ratios = self.config.data_config['split_ratios']

        if num_workers is None:
            num_workers = self.config.data_config['num_workers']

        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(
            dataset_path=dataset_path,
            split_ratios=split_ratios,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    def create_lightning_module(self) -> LitModule:
        """Create lightning module based on model type.

        Derives in_channels from data configuration:
        - Synthetic: len(wavelengths_nm)
        - Real data: num_pd_channels (default 3)
        """
        # Derive in_channels from data, not model config
        if self._is_synthetic():
            syn = self.config.synthetic_config
            wavelengths = syn['wavelengths_nm']
            self.config.model_config['in_channels'] = len(wavelengths)
        else:
            self.config.model_config['in_channels'] = self.config.data_config.get(
                'num_pd_channels', 3
            )

        # Common optimizer hyperparameters
        optimizer_hparams = {
            'name': self.config.training_config['optimizer'],
            # TODO: why is lr a string?
            'lr': eval(self.config.training_config['learning_rate']),
            'momentum': self.config.training_config['momentum'],
            'weight_decay': eval(self.config.training_config['weight_decay']),
        }

        # Common scheduler hyperparameters
        warmup_epochs = self.config.training_config['warmup_epochs']
        T_0 = self.config.training_config['T_0']
        T_mult = self.config.training_config['T_mult']

        scheduler_hparams = {
            'warmup_epochs': warmup_epochs,
            'T_0': T_0,
            'T_mult': T_mult,
            'eta_min': self.config.training_config['eta_min'],
        }

        common_kwargs = dict(
            model_hparams=self.config.model_config,
            optimizer_hparams=optimizer_hparams,
            scheduler_hparams=scheduler_hparams,
            loss_hparams=self.config.loss_config,
            training_hparams=self.config.training_config,
            data_hparams=self.config.data_config,
        )

        if self._is_synthetic():
            syn = self.config.synthetic_config
            return SyntheticLitModule(
                **common_kwargs,
                wavelengths_nm=syn['wavelengths_nm'],
                start_freq=syn.get('start_freq', 1.0),
                end_freq=syn.get('end_freq', 1000.0),
                steps_per_epoch=syn['steps_per_epoch'],
                max_displacement_um=syn.get('max_displacement_um', 5.0),
            )

        return LitModule(**common_kwargs)

    def setup_trainer(self) -> L.Trainer:
        """Setup Lightning trainer with callbacks and loggers."""
        callbacks = []
        # Add WandB logger if configured
        if self.config.training_config['use_logging']:
            loggers = [
                WandbLogger(
                    project=self.config.training_config['wandb_project'],
                    name=self.experiment_name,
                    save_dir=self.checkpoint_dir,
                )
            ]
            callbacks.append(LearningRateMonitor())
            callbacks.append(
                ModelCheckpoint(
                    dirpath=Path(self.checkpoint_dir) / self.experiment_name,
                    filename=str(loggers[0].experiment.id)
                    + '_{epoch}-{val_total_unweighted_loss:.4f}',
                    monitor=None,  #'val/total_unweighted_loss',
                    mode='min',
                    save_top_k=1,
                )
            )
        else:
            loggers = []

        # Get accelerator and device settings from config
        accelerator = self.config.training_config['accelerator']
        devices = self.config.training_config['devices']

        # Convert devices to proper type if it's a string
        if isinstance(devices, str):
            with contextlib.suppress(ValueError):
                devices = int(devices)

        return L.Trainer(
            max_epochs=self.config.training_config['max_epochs'],
            callbacks=callbacks,
            logger=loggers,
            check_val_every_n_epoch=5,
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

    def test(self):
        """Test the model using best checkpoint."""
        if hasattr(self, 'test_loader'):
            checkpoint_callback = None
            for callback in self.trainer.callbacks:
                if isinstance(callback, ModelCheckpoint):
                    checkpoint_callback = callback
                    break

            self.trainer.test(
                self.lightning_module,
                dataloaders=self.test_loader,
                ckpt_path='best' if checkpoint_callback else None,
            )
