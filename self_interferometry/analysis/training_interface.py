#!/usr/bin/env python

import contextlib
import logging
import random
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Union

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from matplotlib.collections import LineCollection
from torch.utils.data import DataLoader

from self_interferometry.analysis.datasets import get_data_loaders
from self_interferometry.analysis.generate_data import generate_synthetic_test_data
from self_interferometry.analysis.lit_module import LitModule
from self_interferometry.analysis.synthetic_lit_module import (
    DEFAULT_WAVELENGTHS_NM,
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

    def __init__(
        self,
        config: TrainingConfig | None = None,
        experiment_name: str | None = None,
        checkpoint_dir: str | None = None,
    ):
        """Args:
        config: Training configuration (None for checkpoint evaluation mode)
        experiment_name: Name for logging and checkpointing
        checkpoint_dir: Directory for saving checkpoints.
        """
        self.config = config

        # Only setup training components if config is provided
        if config is not None:
            self.experiment_name = experiment_name or config.model_config['type']
            self.checkpoint_dir = checkpoint_dir or './checkpoints'

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

    def compute_input_gradients(
        self, model: torch.nn.Module, signals: torch.Tensor
    ) -> np.ndarray:
        """Compute gradients of model output with respect to input signals.

        For each sample, computes ∂output[t]/∂input for all timesteps t.
        Returns the sum of absolute gradients across all output timesteps,
        showing which input points have the strongest overall influence.

        Args:
            model: The trained model
            signals: Input signals tensor of shape [batch_size, num_channels, signal_length]

        Returns:
            Gradients array of shape [batch_size, num_channels, signal_length]
            containing sum of |∂output[t]/∂input| across all output timesteps t
        """
        # CRITICAL FIX: The model's forward pass has torch.set_grad_enabled(self.training)
        # which disables gradients in eval mode. We need training mode for gradients!
        # model.train()  # Enable gradient computation in forward pass
        model.training = True

        # Create a completely fresh tensor that requires gradients
        # The key insight: we need a leaf tensor with requires_grad=True
        signals_input = torch.tensor(
            signals.detach().cpu().numpy(),
            dtype=torch.float32,
            requires_grad=True,
            device='cpu',
        )

        logger.info(f'signals_input.requires_grad: {signals_input.requires_grad}')
        logger.info(f'signals_input.is_leaf: {signals_input.is_leaf}')

        # Explicitly enable gradient computation
        with torch.enable_grad():
            # Forward pass
            output = model(signals_input)

            logger.info(f'output.requires_grad: {output.requires_grad}')
            logger.info(f'output.grad_fn: {output.grad_fn}')
            logger.info(f'output shape: {output.shape}')

            # output shape: [batch_size, signal_length]
            # We want: for each output point, compute gradient w.r.t. all inputs
            # Then sum the absolute gradients across output timesteps

            # Create gradient outputs: ones for all output points
            # This computes sum of gradients for all outputs at once
            grad_outputs = torch.ones_like(output)

            logger.info('About to call torch.autograd.grad...')

            # Compute gradients: d(output)/d(signals_input)
            # This gives us the jacobian summed over output dimension
            gradients = torch.autograd.grad(
                outputs=output,
                inputs=signals_input,
                grad_outputs=grad_outputs,
                create_graph=False,
                retain_graph=False,
            )[0]

            logger.info('Successfully computed gradients!')

        # Convert to numpy and take absolute value
        abs_gradients = np.abs(gradients.detach().numpy())

        return abs_gradients

    def plot_predictions_from_checkpoint(
        self,
        checkpoint_path: str,
        dataset_path: str,
        batch_size: int = 64,
        split_ratios: tuple[int, int, int] = (80, 10, 10),
        num_workers: int = 4,
    ):
        """Plot predictions from a checkpoint.

        Creates a plot with up to 7 rows:
        1. Predicted vs target velocity
        2. Velocity residuals (predicted - target)
        3. Predicted vs target displacement
        4. Displacement residuals (predicted - target)
        5-7. Input signals (up to N photodiode channels)

        When dataset_path is "synthetic", generates synthetic test data.
        Synthetic config is inferred from checkpoint hparams when available,
        otherwise defaults are used based on the model's in_channels.

        Args:
            checkpoint_path: Path to the dataset file, or "synthetic"
            dataset_path: Path to the dataset file, or "synthetic"
            batch_size: Batch size for data loader
            split_ratios: Train/val/test split ratios (real data only)
            num_workers: Number of data loader workers (real data only)
        """
        # Load the model from checkpoint
        model = LitModule.load_from_checkpoint(checkpoint_path, strict=False)
        accelerator = 'cpu' if not torch.cuda.is_available() else 'gpu'
        trainer = L.Trainer(accelerator=accelerator, logger=[])

        if dataset_path == 'synthetic':
            # Infer synthetic config from checkpoint hparams
            in_channels = model.hparams['model_hparams']['in_channels']
            ckpt = torch.load(checkpoint_path, weights_only=False)
            hp = ckpt.get('hyper_parameters', {})

            wavelengths_nm = hp.get(
                'wavelengths_nm', DEFAULT_WAVELENGTHS_NM[:in_channels]
            )
            start_freq = hp.get('start_freq', 1.0)
            end_freq = hp.get('end_freq', 1000.0)
            max_displacement_um = hp.get('max_displacement_um', 5.0)

            logger.info(
                f'Generating synthetic test data: '
                f'{len(wavelengths_nm)} channels, '
                f'freq=[{start_freq}, {end_freq}] Hz, '
                f'max_disp={max_displacement_um} μm'
            )

            test_loader = generate_synthetic_test_data(
                num_samples=batch_size * 4,
                wavelengths_nm=wavelengths_nm,
                batch_size=batch_size,
                start_freq=start_freq,
                end_freq=end_freq,
                max_displacement_um=max_displacement_um,
            )
        else:
            # Real data: setup data loaders
            self.setup_data(
                dataset_path=dataset_path,
                batch_size=batch_size,
                split_ratios=split_ratios,
                num_workers=num_workers,
            )
            test_loader = self.test_loader

        # Get predictions using trainer.predict
        predictions = trainer.predict(model, test_loader)

        # Process the first batch of predictions
        # predictions[batch_idx] returns a tuple of (velocity_hat, velocity_target,
        # displacement_hat, displacement_target, signals)
        batch_idx = 0
        velocity_hat = predictions[batch_idx][0].cpu().numpy()  # Predicted velocities
        velocity_target = predictions[batch_idx][1].cpu().numpy()  # Target velocities
        displacement_hat = (
            predictions[batch_idx][2].cpu().numpy()
        )  # Predicted displacement
        displacement_target = (
            predictions[batch_idx][3].cpu().numpy()
        )  # Target displacement
        signals = predictions[batch_idx][4].cpu().numpy()  # Input signals

        logger.debug(f'Signals shape: {signals.shape}')
        logger.debug(f'Velocity hat shape: {velocity_hat.shape}')
        logger.debug(f'Velocity target shape: {velocity_target.shape}')
        logger.debug(f'Displacement hat shape: {displacement_hat.shape}')
        logger.debug(f'Displacement target shape: {displacement_target.shape}')

        # Get the number of samples in the batch and number of PD channels
        batch_size, num_channels, signal_length = signals.shape

        # Calculate MSE losses for the entire batch
        mse_loss_fn = torch.nn.MSELoss()
        velocity_tensor = torch.tensor(velocity_hat)
        velocity_target_tensor = torch.tensor(velocity_target)
        displacement_tensor = torch.tensor(displacement_hat)
        displacement_target_tensor = torch.tensor(displacement_target)
        batch_velocity_mse = mse_loss_fn(velocity_tensor, velocity_target_tensor).item()
        batch_displacement_mse = mse_loss_fn(
            displacement_tensor, displacement_target_tensor
        ).item()

        # Move model to CPU for gradient computation
        model = model.cpu()

        # For gradient computation, use the same signals from predictions
        # Create a fresh tensor from the numpy array to enable gradient computation
        signals_for_grad = torch.tensor(
            signals, dtype=torch.float32, requires_grad=True, device='cpu'
        )

        # Compute gradients of output with respect to input signals
        # This computes sum of |∂output[t]/∂input| showing which inputs have strongest influence
        abs_gradients = self.compute_input_gradients(model, signals_for_grad)

        # Find global min/max across all channels for consistent colormap
        vmean = abs_gradients.mean()
        vstd = abs_gradients.std()
        vmin = 0
        vmax = vmean + 2 * vstd

        logger.info(f'Gradient range: [{vmin:.2e}, {vmax:.2e}]')

        # Use batch_idx for consistency with original code
        batch_idx = 0

        # Plot for each sample in the batch
        for i in range(
            min(batch_size, 10)
        ):  # Limit to 10 samples to avoid too many plots
            # Calculate sample-specific MSE losses
            sample_velocity_mse = mse_loss_fn(
                torch.tensor(velocity_hat[i]), torch.tensor(velocity_target[i])
            ).item()
            sample_displacement_mse = mse_loss_fn(
                torch.tensor(displacement_hat[i]), torch.tensor(displacement_target[i])
            ).item()

            # Calculate residuals
            displacement_residual = displacement_hat[i] - displacement_target[i]
            velocity_residual = velocity_hat[i] - velocity_target[i]

            # Create a figure with subplots - two for residuals, two for predictions, and up to 3 for signals
            # Add extra space for colorbar
            fig, axs = plt.subplots(4 + num_channels, 1, figsize=(12, 12), sharex=True)

            # Plot predicted vs target velocity
            axs[0].plot(velocity_target[i], label='Target Velocity', color='blue')
            axs[0].plot(
                velocity_hat[i], label='Predicted Velocity', color='red', linestyle='--'
            )
            axs[0].set_title(f'Velocity (MSE: {sample_velocity_mse:.2e})')
            axs[0].set_ylabel('Velocity (μm/s)')
            axs[0].grid(True, alpha=0.3)
            axs[0].legend(loc='upper right')

            # Plot velocity residuals
            axs[1].plot(velocity_residual, label='Velocity Residual', color='purple')
            axs[1].axhline(y=0, color='black', linestyle=':', alpha=0.5)
            axs[1].set_title(f'Velocity Residual (MSE: {sample_velocity_mse:.2e})')
            axs[1].set_ylabel('Residual (μm/s)')
            axs[1].grid(True, alpha=0.3)
            axs[1].legend(loc='upper right')

            # Plot predicted vs target displacement
            axs[2].plot(
                displacement_target[i], label='Target Displacement', color='green'
            )
            axs[2].plot(
                displacement_hat[i],
                label='Predicted Displacement',
                color='orange',
                linestyle='--',
            )
            axs[2].set_title(f'Displacement (MSE: {sample_displacement_mse:.2e})')
            axs[2].set_ylabel('Displacement (μm)')
            axs[2].grid(True, alpha=0.3)
            axs[2].legend(loc='upper right')

            # Plot displacement residuals
            axs[3].plot(
                displacement_residual, label='Displacement Residual', color='purple'
            )
            axs[3].axhline(y=0, color='black', linestyle=':', alpha=0.5)
            axs[3].set_title(
                f'Displacement Residual (MSE: {sample_displacement_mse:.2e})'
            )
            axs[3].set_ylabel('Residual (μm)')
            axs[3].grid(True, alpha=0.3)
            axs[3].legend(loc='upper right')

            # Plot each input signal channel with gradient-based coloring
            hw_names = ['PD1 (RP1_CH2)', 'PD2 (RP2_CH1)', 'PD3 (RP2_CH2)']
            channel_names = [
                hw_names[j] if j < len(hw_names) else f'CH{j + 1}'
                for j in range(num_channels)
            ]

            for j in range(num_channels):
                # Get signal and gradient for this channel
                signal = signals[i, j, :]
                gradient_magnitude = abs_gradients[i, j, :]

                # Create points for line segments
                x = np.arange(signal_length)
                y = signal

                # Create line segments for coloring
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                # Use gradient magnitude for coloring
                colors = gradient_magnitude[
                    :-1
                ]  # Use gradient at start of each segment

                # Create LineCollection with gradient-based colors
                lc = LineCollection(
                    segments, cmap='hot', norm=plt.Normalize(vmin=vmin, vmax=vmax)
                )
                lc.set_array(colors)
                lc.set_linewidth(2)

                # Plot the colored line
                line = axs[j + 4].add_collection(lc)

                # Set axis limits
                axs[j + 4].set_xlim(0, signal_length - 1)
                axs[j + 4].set_ylim(signal.min() * 1.1, signal.max() * 1.1)

                axs[j + 4].set_title(
                    f'Input Signal - {channel_names[j]} (colored by |∂output/∂input|)'
                )
                axs[j + 4].set_ylabel('Amplitude')
                axs[j + 4].grid(True, alpha=0.3)

            # Set x-axis label for the bottom subplot
            axs[-1].set_xlabel('Sample')

            # Add overall title with batch MSE values
            fig.suptitle(
                f'Sample {i + 1} from Batch {batch_idx + 1}\n'
                f'Batch MSE: Velocity={batch_velocity_mse:.2e}, '
                f'Displacement={batch_displacement_mse:.2e}'
            )

            # Adjust layout to make room for colorbar
            plt.tight_layout(rect=[0, 0, 0.9, 0.96])

            # Add a colorbar for the signal plots only (last num_channels plots)
            # Get the positions of the first and last signal plots
            first_signal_ax = axs[4]  # First signal plot
            last_signal_ax = axs[-1]  # Last signal plot

            # Get the bounding boxes of these axes in figure coordinates
            first_bbox = first_signal_ax.get_position()
            last_bbox = last_signal_ax.get_position()

            # Create colorbar axis spanning only the signal plots
            # Use the bottom of the last plot and top of the first plot
            cbar_bottom = last_bbox.y0
            cbar_height = first_bbox.y1 - last_bbox.y0
            cbar_ax = fig.add_axes([0.92, cbar_bottom, 0.02, cbar_height])
            cbar = fig.colorbar(
                line, cax=cbar_ax, label='Gradient Magnitude |∂output/∂input|'
            )

            plt.show()
