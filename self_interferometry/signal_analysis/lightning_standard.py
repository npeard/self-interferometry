#!/usr/bin/env python

from typing import Any

import lightning as L
import torch
from torch import nn, optim

from self_interferometry.redpitaya.coil_driver import CoilDriver
from self_interferometry.signal_analysis.models_cnn import BarlandCNN, BarlandCNNConfig
from self_interferometry.signal_analysis.models_fcn import FCN, FCNConfig
from self_interferometry.signal_analysis.models_tcn import TCN, TCNConfig


class Standard(L.LightningModule):
    """Base Lightning Module for all models."""

    def __init__(
        self,
        model_hparams: dict | None = None,
        optimizer_hparams: dict | None = None,
        scheduler_hparams: dict | None = None,
        loss_hparams: dict | None = None,
    ):
        """Args:
        model_hparams: Hyperparameters for the model
        optimizer_hparams: Hyperparameters for the optimizer
        scheduler_hparams: Hyperparameters for the learning rate scheduler
        loss_hparams: Hyperparameters for the loss function.
        """
        super().__init__()
        self.model_hparams = model_hparams
        self.optimizer_hparams = optimizer_hparams
        self.scheduler_hparams = scheduler_hparams
        self.loss_hparams = loss_hparams
        self.save_hyperparameters(ignore=['model'])
        self.model = self.create_model()

        torch.set_float32_matmul_precision('high')

    def _create_cnn_config(self) -> BarlandCNNConfig:
        """Create CNNConfig from model configuration."""
        return BarlandCNNConfig(
            # Common parameters
            input_size=self.model_hparams.get('input_size', 256),
            output_size=self.model_hparams.get('output_size', 1),
            in_channels=self.model_hparams.get('in_channels', 3),
            activation=self.model_hparams.get('activation', 'LeakyReLU'),
            dropout=self.model_hparams.get('dropout', 0.1),
            # BarlandCNN specific parameters
            window_stride=self.model_hparams.get('window_stride', 128),
        )

    def _create_tcn_config(self) -> TCNConfig:
        """Create TCNConfig from model configuration."""
        return TCNConfig(
            # Common parameters
            input_size=self.model_hparams.get('input_size', 16384),
            output_size=self.model_hparams.get('output_size', 16384),
            in_channels=self.model_hparams.get('in_channels', 3),
            activation=self.model_hparams.get('activation', 'LeakyReLU'),
            dropout=self.model_hparams.get('dropout', 0.1),
            # TCN specific parameters
            kernel_size=self.model_hparams.get('kernel_size', 7),
            num_channels=self.model_hparams.get('num_channels', [16, 32, 64, 64]),
            dilation_base=self.model_hparams.get('dilation_base', 2),
            stride=self.model_hparams.get('stride', 1),
        )

    def _create_fcn_config(self) -> FCNConfig:
        """Create FCNConfig from model configuration."""
        return FCNConfig(
            # Common parameters
            input_size=self.model_hparams.get('input_size', 16384),
            output_size=self.model_hparams.get('output_size', 16384),
            in_channels=self.model_hparams.get('in_channels', 3),
            activation=self.model_hparams.get('activation', 'LeakyReLU'),
            dropout=self.model_hparams.get('dropout', 0.1),
            # FCN specific parameters
            num_channels=self.model_hparams.get('num_channels', [16, 32, 64, 64]),
            kernel_size=self.model_hparams.get('kernel_size', 7),
            use_final_conv=self.model_hparams.get('use_final_conv', True),
        )

    def create_model(self) -> nn.Module:
        """Create model instance based on config.

        Returns:
            A PyTorch model (CNN or TCN) based on the configuration
        """
        if self.model_hparams == 'ensemble':
            # This indicates that we are creating an ensemble of models
            # and no model will be created at this point.
            # See Ensemble(Standard) class for more.
            return None
        model_type = self.model_hparams.get('type')
        if model_type == 'CNN':
            print('Creating CNN model...')  # noqa: T201
            self.model_config = self._create_cnn_config()
            return BarlandCNN(self.model_config)
        elif model_type == 'TCN':
            print('Creating TCN model...')  # noqa: T201
            self.model_config = self._create_tcn_config()
            return TCN(self.model_config)
        elif model_type == 'FCN':
            print('Creating FCN model...')  # noqa: T201
            self.model_config = self._create_fcn_config()
            return FCN(self.model_config)
        else:
            raise ValueError(f'Unknown model type: {model_type}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        This method handles different model architectures:
        1. For CNN models: Uses a sliding window approach to process each window
        separately
        2. For TCN and FCN models: Processes the entire sequence at once efficiently

        Args:
            x: Input tensor of shape [batch_size, num_channels, signal_length]

        Returns:
            Tensor of shape [batch_size, signal_length] containing predicted velocity
        """
        batch_size, num_channels, signal_length = x.shape

        # Check if we're using a TCN or FCN model (which can process the entire sequence at
        # once)
        if hasattr(self.model, '__class__') and (
            self.model.__class__.__name__ in {'TCN', 'FCN'}
        ):
            # TCN approach - process the entire sequence at once
            with torch.set_grad_enabled(self.training):
                # TCN returns shape [batch_size, 1, signal_length]
                output = self.model(x)

                # Reshape to [batch_size, signal_length]
                velocity_hat = output.squeeze(1)

            return velocity_hat

        else:
            # CNN approach - sliding window implementation
            # Define the window size (number of input points that produce one output
            # point)
            window_size = self.model_config.input_size
            window_stride = self.model_config.window_stride

            # Calculate number of windows and create output tensor accordingly
            num_windows = (signal_length + window_stride - 1) // window_stride

            # Create output tensor to store results - size is now based on number
            # of windows
            if window_stride == 1:
                # If stride is 1, maintain backward compatibility with full signal
                # length output
                velocity_hat = torch.zeros((batch_size, signal_length), device=x.device)
                output_full_signal = True
            else:
                # For stride > 1, output size is reduced to number of windows
                velocity_hat = torch.zeros((batch_size, num_windows), device=x.device)
                output_full_signal = False

            # Add zero padding to handle the edges
            padding = window_size // 2
            padded_x = torch.nn.functional.pad(
                x, (padding, padding), mode='constant', value=0
            )

            # Scan through the signal with the appropriate stride
            for window_idx, i in enumerate(range(0, signal_length, window_stride)):
                # Extract window centered at position i
                start_idx = i
                end_idx = i + window_size
                window = padded_x[:, :, start_idx:end_idx]

                # Skip if window is not complete (should not happen with padding)
                if window.shape[2] < window_size:
                    continue

                # Get model prediction for this window
                with torch.set_grad_enabled(self.training):
                    pred = self.model(window)

                # Store the prediction in the output tensor
                if output_full_signal:
                    velocity_hat[:, i] = pred.squeeze()
                else:
                    velocity_hat[:, window_idx] = pred.squeeze()

            # Store the stride information for use in training/validation/testing
            self._output_full_signal = output_full_signal
            self._window_stride = window_stride
            self._num_windows = num_windows

            return velocity_hat

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and learning rate scheduler."""
        # Configure optimizer
        optimizer_name = self.optimizer_hparams.pop('name')
        if optimizer_name == 'Adam':
            # Discard momentum hyperparameter for Adam optimizer
            _ = self.optimizer_hparams.pop('momentum', None)
            optimizer = optim.AdamW(self.parameters(), **self.optimizer_hparams)
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(self.parameters(), **self.optimizer_hparams)
        else:
            raise ValueError(f'Unknown optimizer: {optimizer_name}')

        # Configure scheduler
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
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step for model.

        Args:
            batch: Tuple of (signals, velocity_target, displacement_target)
            batch_idx: Index of current batch

        Returns:
            Total loss value for backpropagation
        """
        signals, velocity_target, _ = (
            batch  # Ignore displacement_target for Standard model
        )
        velocity_hat = self(signals)

        # For CNN with stride > 1, extract the corresponding target values
        if hasattr(self, '_output_full_signal') and not self._output_full_signal:
            # Extract targets at the same positions where predictions were made
            # For each window, we want the target at the center of the window
            downsampled_target = torch.zeros_like(velocity_hat)

            for window_idx, i in enumerate(
                range(0, velocity_target.shape[1], self._window_stride)
            ):
                if window_idx >= self._num_windows:
                    break
                downsampled_target[:, window_idx] = velocity_target[:, i]

            velocity_target = downsampled_target

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
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Validation step for model.

        Args:
            batch: Tuple of (signals, velocity_target, displacement_target)
            batch_idx: Index of current batch
        """
        signals, velocity_target, _ = (
            batch  # Ignore displacement_target for Standard model
        )
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
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Test step for model.

        Args:
            batch: Tuple of (signals, velocity_target, displacement_target)
            batch_idx: Index of current batch
        """
        signals, velocity_target, _ = (
            batch  # Ignore displacement_target for Standard model
        )
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

        loss_dict = self.loss_function(velocity_hat, velocity_target)

        # Log each loss component with test_ prefix
        for loss_name, loss_value in loss_dict.items():
            self.log(f'test_{loss_name}_loss', loss_value, sync_dist=True)

    def predict_step(
        self,
        batch: tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prediction step for model. Return all relevant quantities for plotting.

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

        # During inference, we want to make sure window_stride is set to 1 for CNNs
        if hasattr(self, 'model_config') and hasattr(
            self.model_config, 'window_stride'
        ):
            self.model_config.window_stride = 1

        velocity_hat = self(signals)
        # Use default sample rate determined by hardware in integration
        # Should be 125e6 / 256 - but generally ought to be read from data file.
        displacement_hat = CoilDriver.integrate_velocity(velocity_hat)

        # Shift all displacements to start at zero for easy comparison
        displacement_hat -= displacement_hat[:, 0:1]
        displacement_target -= displacement_target[:, 0:1]

        return (
            velocity_hat,
            velocity_target,
            displacement_hat,
            displacement_target,
            signals,
        )
