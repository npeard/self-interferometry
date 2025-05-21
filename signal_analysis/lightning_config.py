#!/usr/bin/env python

from typing import Any

import lightning as L
import torch
from torch import nn, optim

from redpitaya.coil_driver import CoilDriver


class Standard(L.LightningModule):
    """Base Lightning Module for all models."""

    def __init__(
        self,
        model: torch.nn.Module | None = None,
        optimizer_hparams: dict | None = None,
        scheduler_hparams: dict | None = None,
        loss_hparams: dict | None = None,
    ):
        """Args:
        model: PyTorch model to train
        optimizer_hparams: Hyperparameters for the optimizer
        scheduler_hparams: Hyperparameters for the learning rate scheduler
        loss_hparams: Hyperparameters for the loss function.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model

        # Set default optimizer hyperparameters if none provided
        self.optimizer_hparams = optimizer_hparams or {'lr': 1e-3, 'weight_decay': 1e-5}
        self.scheduler_hparams = scheduler_hparams or {
            'milestones': [250, 450],
            'gamma': 0.1,
        }
        self.loss_hparams = loss_hparams or {}

        torch.set_float32_matmul_precision('high')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and learning rate scheduler."""
        # Configure optimizer
        if self.hparams.optimizer_name == 'Adam':
            optimizer = optim.AdamW(self.parameters(), **self.optimizer_hparams)
        elif self.hparams.optimizer_name == 'SGD':
            optimizer = optim.SGD(self.parameters(), **self.optimizer_hparams)
        else:
            raise ValueError(f'Unknown optimizer: {self.hparams.optimizer_name}')

        # Configure scheduler
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **self.scheduler_hparams)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, **self.scheduler_hparams
        )

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler}}

    # TODO: what is this, where is it used?
    def _get_progress_bar_dict(self) -> dict[str, Any]:
        """Modify progress bar display."""
        items = super()._get_progress_bar_dict()
        items.pop('v_num', None)
        return items

    def loss_function(
        self, velocity_hat: torch.Tensor, velocity_target: torch.Tensor, *args, **kwargs
    ) -> dict[str, torch.Tensor]:
        """Custom loss function that returns a dictionary of loss components.

        Args:
            y_hat: Model predictions
            targets: Ground truth targets
            x: Additional input tensor for encoding loss
            *args, **kwargs: Additional arguments

        Returns:
            Dictionary of loss components with keys representing loss types
            and values representing the corresponding loss tensors.
        """
        # Base velocity loss (MSE)
        velocity_loss = nn.MSELoss()(velocity_hat, velocity_target)

        # Create a dictionary of loss components
        loss_dict = {
            'velocity': velocity_loss
            # Add more loss components as needed
        }

        # Add a total loss entry
        loss_dict['total'] = sum(loss_dict.values())

        return loss_dict

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step for model.

        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Index of current batch

        Returns:
            Total loss value for backpropagation
        """
        signals, velocity_target = batch
        velocity_hat = self(signals)
        loss_dict = self.loss_function(velocity_hat, velocity_target)

        # Log each loss component with train_ prefix
        for loss_name, loss_value in loss_dict.items():
            self.log(
                f'train_{loss_name}_loss',
                loss_value,
                prog_bar=(loss_name == 'total'),  # Only show total loss in progress bar
                on_epoch=True,
            )

        # Return the total loss for backpropagation
        return loss_dict['total']

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Validation step for model.

        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Index of current batch
        """
        signals, velocity_target = batch
        velocity_hat = self(signals)
        loss_dict = self.loss_function(velocity_hat, velocity_target)

        # Log each loss component with val_ prefix
        for loss_name, loss_value in loss_dict.items():
            self.log(
                f'val_{loss_name}_loss',
                loss_value,
                prog_bar=(loss_name == 'total'),  # Only show total loss in progress bar
                sync_dist=True,
            )

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Test step for model.

        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Index of current batch
        """
        signals, velocity_target = batch
        velocity_hat = self(signals)
        loss_dict = self.loss_function(velocity_hat, velocity_target)

        # Log each loss component with test_ prefix
        for loss_name, loss_value in loss_dict.items():
            self.log(f'test_{loss_name}_loss', loss_value, sync_dist=True)

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Prediction step for GPT model. Return all relevant quantities for plotting.

        Args:
            batch: Input tensor
            batch_idx: Index of current batch
            dataloader_idx: Index of dataloader
        """
        signals, velocity_target = batch
        return self.model(signals)


class Teacher(Standard):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_hparams: dict | None = None,
        scheduler_hparams: dict | None = None,
        loss_hparams: dict | None = None,
        interferometer_config: dict | None = None,
    ):
        super().__init__(
            model=model,
            optimizer_hparams=optimizer_hparams,
            scheduler_hparams=scheduler_hparams,
            loss_hparams=loss_hparams,
        )
        self.model = model

        # Initialize interferometer array for signal simulation
        self._setup_interferometer_array(interferometer_config)

    def _setup_interferometer_array(self, config: dict | None = None):
        """Set up the interferometer array for signal simulation.

        Args:
            config: Configuration dictionary for interferometers. If None, default
                configuration will be used with three interferometers at standard
                wavelengths.
        """
        import numpy as np

        from signal_analysis.interferometers import (
            InterferometerArray,
            MichelsonInterferometer,
        )

        # Default configuration if none provided
        if config is None:
            config = {
                'wavelengths': [0.635, 0.515, 0.675],  # Red, Green, Deep Red in microns
                'phases': [0, np.pi / 3, np.pi / 2],
            }

        # Create interferometers based on configuration
        interferometers = [
            MichelsonInterferometer(wavelength=wl, phase=ph)
            for wl, ph in zip(config['wavelengths'], config['phases'], strict=False)
        ]

        # Create and store the interferometer array
        self.interferometer_array = InterferometerArray(interferometers)

    def loss_function(
        self,
        velocity_hat: torch.Tensor,
        velocity_target: torch.Tensor,
        displacement_target: torch.Tensor,
        signal_target: torch.Tensor,
        *args,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Custom loss function for Teacher model with velocity and displacement
        components.

        Args:
            y_hat: Model predictions (velocity)
            targets: Ground truth targets (velocity)
            x: Additional input tensor for encoding loss
            *args, **kwargs: Additional arguments

        Returns:
            Dictionary of loss components including velocity, displacement, and total
            loss
        """
        # Velocity loss (MSE)
        velocity_loss = nn.MSELoss()(velocity_hat, velocity_target)

        # Calculate displacement from velocity for both predictions and targets
        displacement_hat = CoilDriver.integrate_velocity(velocity_hat)

        # Displacement loss (MSE)
        displacement_loss = nn.MSELoss()(displacement_hat, displacement_target)

        # Calculate simulated signals from inferred displacement (PyTorch mode)
        simulated_signals = self.interferometer_array.get_simulated_buffer(
            displacement_hat
        )

        # Reshape input signals for comparison
        # Assuming x has shape [batch_size, n_channels, signal_length]
        # and we need to compare with each interferometer signal
        input_signals = [signal_target[:, i, :] for i in range(len(simulated_signals))]

        # Calculate signal loss (average MSE across all interferometer signals)
        signal_losses = [
            nn.MSELoss()(sim, inp)
            for sim, inp in zip(simulated_signals, input_signals, strict=False)
        ]
        signal_loss = torch.mean(torch.stack(signal_losses))

        # Create loss dictionary
        loss_dict = {
            'velocity': velocity_loss,
            'displacement': displacement_loss,
            'signal': signal_loss,
        }

        # Add total loss (sum of all components)
        loss_dict['total'] = sum(loss_dict.values())

        return loss_dict


class Student(Standard):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_hparams: dict | None = None,
        scheduler_hparams: dict | None = None,
        loss_hparams: dict | None = None,
    ):
        super().__init__(
            model=model,
            optimizer_hparams=optimizer_hparams,
            scheduler_hparams=scheduler_hparams,
            loss_hparams=loss_hparams,
        )
        self.model = model

    # TODO: write overrides of the training steps, loss function is the same as
    # Standard (unless we decide later that a physics constraint is helpful)
