#!/usr/bin/env python


import numpy as np
import torch
from torch import nn

from self_interferometry.redpitaya.coil_driver import CoilDriver
from self_interferometry.signal_analysis.interferometers import (
    InterferometerArray,
    MichelsonInterferometer,
)
from self_interferometry.signal_analysis.lightning_standard import Standard


class Teacher(Standard):
    def __init__(
        self,
        model_hparams: dict,
        optimizer_hparams: dict | None = None,
        scheduler_hparams: dict | None = None,
        loss_hparams: dict | None = None,
        interferometer_config: dict | None = None,
    ):
        super().__init__(
            model_hparams=model_hparams,
            optimizer_hparams=optimizer_hparams,
            scheduler_hparams=scheduler_hparams,
            loss_hparams=loss_hparams,
        )
        # Initialize interferometer array for signal simulation
        self._setup_interferometer_array(interferometer_config)

    def _setup_interferometer_array(self, config: dict | None = None):
        """Set up the interferometer array for signal simulation.

        Args:
            config: Configuration dictionary for interferometers. If None, default
                configuration will be used with three interferometers at standard
                wavelengths.
        """
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

    def training_step(
        self,
        batch: tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step for Teacher model.

        Args:
            batch: Tuple of (inputs, targets) where targets is a tuple of
                  (velocity, displacement, signals)
            batch_idx: Index of current batch

        Returns:
            Total loss value for backpropagation
        """
        # Unpack the batch
        signals, velocity_target, displacement_target = batch

        # Forward pass
        velocity_hat = self(signals)

        # Calculate loss
        loss_dict = self.loss_function(
            velocity_hat, velocity_target, displacement_target, signals
        )

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
        self,
        batch: tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        batch_idx: int,
    ) -> None:
        """Validation step for Teacher model.

        Args:
            batch: Tuple of (inputs, targets) where targets is a tuple of
                  (velocity, displacement, signals)
            batch_idx: Index of current batch
        """
        # Unpack the batch
        signals, velocity_target, displacement_target = batch

        # Forward pass
        velocity_hat = self(signals)

        # For CNN with stride > 1, extract the corresponding target values
        if hasattr(self, '_output_full_signal') and not self._output_full_signal:
            # Extract targets at the same positions where predictions were made
            batch_size = velocity_target.shape[0]
            downsampled_target = torch.zeros_like(velocity_hat)

            for window_idx, i in enumerate(
                range(0, velocity_target.shape[1], self._window_stride)
            ):
                if window_idx >= self._num_windows:
                    break
                downsampled_target[:, window_idx] = velocity_target[:, i]

            velocity_target = downsampled_target

            # For prediction, we might want to interpolate back to full signal length
            # This is optional and depends on how you want to visualize the results

        # Calculate loss
        loss_dict = self.loss_function(
            velocity_hat, velocity_target, displacement_target, signals
        )

        # Log each loss component with val_ prefix
        for loss_name, loss_value in loss_dict.items():
            self.log(
                f'val_{loss_name}_loss',
                loss_value,
                prog_bar=(loss_name == 'total'),  # Only show total loss in progress bar
                sync_dist=True,
            )

    def test_step(
        self,
        batch: tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        batch_idx: int,
    ) -> None:
        """Test step for Teacher model.

        Args:
            batch: Tuple of (inputs, targets) where targets is a tuple of
                  (velocity, displacement, signals)
            batch_idx: Index of current batch
        """
        # Unpack the batch
        signals, velocity_target, displacement_target = batch

        # Forward pass
        velocity_hat = self(signals)

        # For CNN with stride > 1, extract the corresponding target values
        if hasattr(self, '_output_full_signal') and not self._output_full_signal:
            # Extract targets at the same positions where predictions were made
            downsampled_target = torch.zeros_like(velocity_hat)

            for window_idx, i in enumerate(
                range(0, velocity_target.shape[1], self._window_stride)
            ):
                if window_idx >= self._num_windows:
                    break
                downsampled_target[:, window_idx] = velocity_target[:, i]

            velocity_target = downsampled_target

            # For prediction, we might want to interpolate back to full signal length
            # This is optional and depends on how you want to visualize the results

        # Calculate loss
        loss_dict = self.loss_function(
            velocity_hat, velocity_target, displacement_target, signals
        )

        # Log each loss component with test_ prefix
        for loss_name, loss_value in loss_dict.items():
            self.log(f'test_{loss_name}_loss', loss_value, sync_dist=True)

    def predict_step(
        self,
        batch: tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prediction step for Teacher model. Return all relevant quantities for
        plotting.

        Args:
            batch: Tuple of (inputs, targets) where targets is a tuple of
                  (velocity, displacement, signals)
            batch_idx: Index of current batch
            dataloader_idx: Index of dataloader

        Returns:
            Tuple of (velocity_hat, velocity_target, displacement_hat,
            displacement_target, signals)
        """
        signals, velocity_target, displacement_target = batch
        print(f'Teacher predict_step input signals shape: {signals.shape}')

        # During inference, we want to make sure window_stride is set to 1 for CNNs
        if hasattr(self, 'model_config') and hasattr(
            self.model_config, 'window_stride'
        ):
            self.model_config.window_stride = 1

        velocity_hat = self(signals)
        displacement_hat = CoilDriver.integrate_velocity(velocity_hat)

        return (
            velocity_hat,
            velocity_target,
            displacement_hat,
            displacement_target,
            signals,
        )
