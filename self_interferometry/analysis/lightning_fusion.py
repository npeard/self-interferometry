#!/usr/bin/env python

import logging
from typing import Any, override

import lightning as L
import torch
from torch import nn, optim

# Conditional import for Muon optimizer
try:
    from muon import Muon, MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam

    MUON_AVAILABLE = True
except ImportError:
    MUON_AVAILABLE = False
    Muon = None
    MuonWithAuxAdam = None
    SingleDeviceMuonWithAuxAdam = None

from self_interferometry.acquisition.redpitaya.redpitaya_config import RedPitayaConfig
from self_interferometry.acquisition.simulations.coil_driver import CoilDriver
from self_interferometry.analysis.models.factory import create_model

logger = logging.getLogger(__name__)


class Fusion(L.LightningModule):
    """Lightning Module for fusing multiple sensor channels into velocity predictions.

    This module differs from the Ensemble module by fusing sensor channels at the
    feature level within a single model (CNN or TCN), rather than combining separate
    models. It supports both static and dynamic loss weighting between velocity and
    displacement loss terms for improved training stability.
    """

    def __init__(
        self,
        model_hparams: dict | None = None,
        optimizer_hparams: dict | None = None,
        scheduler_hparams: dict | None = None,
        loss_hparams: dict | None = None,
        training_hparams: dict | None = None,
        data_hparams: dict | None = None,
    ):
        """Args:
        model_hparams: Hyperparameters for the model
        optimizer_hparams: Hyperparameters for the optimizer
        scheduler_hparams: Hyperparameters for the learning rate scheduler
        loss_hparams: Hyperparameters for the loss function
        training_hparams: Hyperparameters for training configuration (includes target)
        data_hparams: Hyperparameters for data configuration (for logging only)
        """
        super().__init__()
        self.model_hparams = model_hparams
        self.optimizer_hparams = optimizer_hparams
        self.scheduler_hparams = scheduler_hparams
        self.loss_hparams = loss_hparams
        self.training_hparams = training_hparams
        self.data_hparams = data_hparams
        self.save_hyperparameters(ignore=['model'])
        self.model = self.create_model()

        # Determine what the model should target
        if training_hparams and 'target' in training_hparams:
            self.target = training_hparams['target']
            if self.target not in ['velocity', 'displacement']:
                raise ValueError(
                    f"target must be 'velocity' or 'displacement', got '{self.target}'"
                )
        else:
            # Default to velocity for backward compatibility
            self.target = 'velocity'
            logger.warning(
                "No 'target' specified in training config, defaulting to 'velocity'"
            )

        # Initialize dynamic loss weights if enabled
        if 'dynamic' not in self.loss_hparams:
            raise KeyError("'dynamic' key must be specified in loss_hparams config")
        self.dynamic_weighting = self.loss_hparams['dynamic']

        if self.dynamic_weighting:
            # Initialize learnable weight parameters for homoscedastic uncertainty
            # Start with reasonable initial values (e.g., 1.0)
            self.log_weight_velocity = nn.Parameter(torch.tensor(0.0))  # log(1.0) = 0.0
            self.log_weight_displacement = nn.Parameter(
                torch.tensor(0.0)
            )  # log(1.0) = 0.0

        torch.set_float32_matmul_precision('high')

    def create_model(self) -> nn.Module:
        """Create model instance based on config.

        Returns:
            A PyTorch model based on the configuration
        """
        return create_model(self.model_hparams)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        This method handles different model architectures:
        1. For CNN models: Uses a sliding window approach to process each window
        separately
        2. For TCN/FNO model: Processes the entire sequence at once efficiently

        Args:
            x: Input tensor of shape [batch_size, num_channels, signal_length]

        Returns:
            Tensor of shape [batch_size, signal_length] containing model predictions
            (velocity or displacement depending on self.target)
        """
        batch_size, num_channels, signal_length = x.shape

        # Check if we're using a TCN, UTCN, StemTCAN, or FNO model (which can process the entire sequence at
        # once)
        if hasattr(self.model, '__class__') and (
            self.model.__class__.__name__
            in {'TCN', 'UTCN', 'StemTCAN', 'FNO1d', 'UNO1d'}
        ):
            # TCN/FNO approach - process the entire sequence at once
            with torch.set_grad_enabled(self.training):
                # TCN/FNO returns shape [batch_size, 1, signal_length]
                output = self.model(x)

                # Reshape to [batch_size, signal_length]
            return output.squeeze(1)

        else:
            # CNN approach - sliding window implementation
            # Define the window size (number of input points that produce one output
            # point)
            window_size = self.model_config.sequence_length
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
        # Make a copy to avoid modifying the original hparams dict
        optimizer_hparams = self.optimizer_hparams.copy()

        # Configure optimizer
        optimizer_name = optimizer_hparams.pop('name')
        if optimizer_name == 'Adam':
            # Discard momentum hyperparameter for Adam optimizer
            _ = optimizer_hparams.pop('momentum', None)
            optimizer = optim.AdamW(self.parameters(), **optimizer_hparams)
        elif optimizer_name == 'SGD':
            # Discard weight_decay hyperparameter for SGD optimizer
            _ = optimizer_hparams.pop('weight_decay', None)
            optimizer = optim.SGD(self.parameters(), **optimizer_hparams)
        elif optimizer_name == 'Muon':
            if not MUON_AVAILABLE:
                raise ImportError(
                    'Muon optimizer requires the muon-optimizer library. '
                    'Install it with: pip install muon-optimizer'
                )
            # Discard momentum hyperparameter for Muon optimizer
            _ = optimizer_hparams.pop('momentum', None)

            # Muon uses MuonWithAuxAdam which applies:
            # - Muon to hidden layer weights (parameters with ndim >= 2)
            # - AdamW to other parameters (biases, norms, etc.)

            # Separate parameters by dimensionality
            # For models with body/head structure, we need to handle all model parameters
            hidden_weights = [p for p in self.model.parameters() if p.ndim >= 2]
            other_params = [p for p in self.model.parameters() if p.ndim < 2]

            # If dynamic weighting is enabled, add the dynamic weight parameters
            # These should be optimized with AdamW (not Muon) since they're scalars
            if self.dynamic_weighting:
                other_params.extend(
                    [self.log_weight_velocity, self.log_weight_displacement]
                )

            # Get hyperparameters
            lr = optimizer_hparams.pop('lr')
            weight_decay = float(optimizer_hparams.pop('weight_decay'))

            # Configure parameter groups
            # Muon for hidden weights, AdamW for biases/norms
            if len(hidden_weights) > 0:
                logger.info('Muon optimizer has found hidden weights')
            param_groups = [
                dict(
                    params=hidden_weights,
                    use_muon=True,
                    lr=lr,
                    weight_decay=weight_decay,
                ),
                dict(
                    params=other_params,
                    use_muon=False,
                    lr=lr * 0.1,  # Typically use lower LR for non-Muon params
                    betas=(0.9, 0.95),
                    weight_decay=weight_decay,
                ),
            ]

            # Use SingleDeviceMuonWithAuxAdam for single GPU/CPU training
            # Use MuonWithAuxAdam for distributed training
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                optimizer = MuonWithAuxAdam(param_groups, **optimizer_hparams)
            else:
                optimizer = SingleDeviceMuonWithAuxAdam(
                    param_groups, **optimizer_hparams
                )
        else:
            raise ValueError(f'Unknown optimizer: {optimizer_name}')

        # Configure multi-stage scheduler: linear warmup then cosine annealing with warm restarts
        warmup_epochs = self.scheduler_hparams['warmup_epochs']
        eta_min = self.scheduler_hparams['eta_min']
        T_0 = self.scheduler_hparams['T_0']
        T_mult = self.scheduler_hparams['T_mult']

        # LambdaLR for linear warmup
        def warmup_lambda(epoch: int) -> float:
            if warmup_epochs == 0:
                return 1.0
            return float(epoch + 1) / float(warmup_epochs)

        warmup_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=warmup_lambda
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eval(eta_min)
        )

        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler}}

    def loss_function(
        self,
        prediction: torch.Tensor,
        velocity_target: torch.Tensor,
        displacement_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Custom loss function that returns a dictionary of loss components.

        Supports two modes based on self.target:
        - 'velocity': Model predicts velocity, displacement loss is auxiliary (physics-informed)
        - 'displacement': Model predicts displacement, velocity loss is not used

        Args:
            prediction: Model predictions (velocity or displacement based on self.target)
            velocity_target: Ground truth velocity
            displacement_target: Ground truth displacement

        Returns:
            Dictionary of loss components with keys representing loss types
            and values representing the corresponding loss tensors.
        """
        sample_rate = RedPitayaConfig.SAMPLE_RATE_DEC1 / 256

        if self.target == 'velocity':
            # Model predicts velocity
            velocity_hat = prediction
            # Auxiliary loss: displacement (physics-informed, derived from velocity)
            displacement_hat = CoilDriver.integrate_velocity(velocity_hat, sample_rate)
            # Shift displacements to start at 0 for fair comparison
            displacement_hat -= displacement_hat[:, 0:1]
            displacement_target -= displacement_target[:, 0:1]
        elif self.target == 'displacement':
            # Model predicts displacement
            displacement_hat = prediction
            # Shift displacements to start at 0 for fair comparison
            displacement_hat -= displacement_hat[:, 0:1]
            displacement_target -= displacement_target[:, 0:1]
            # Auxiliary loss: velocity (physics-informed, derived from displacement)
            velocity_hat = CoilDriver.derivative_displacement(
                displacement_hat, sample_rate
            )
        else:
            raise ValueError(f'Unknown target: {self.target}')

        # Convert velocity to um/ms instead of um/s stored in Dataset
        velocity_hat *= 1e-3
        velocity_target *= 1e-3

        # Velocity Loss
        velocity_loss = nn.MSELoss()(velocity_hat, velocity_target)

        # Displacement Loss
        displacement_loss = nn.MSELoss()(displacement_hat, displacement_target)

        # Create loss dictionary
        loss_dict = {
            'velocity': velocity_loss,
            'displacement': displacement_loss,
            'total_unweighted': velocity_loss + displacement_loss,
        }

        # Weighted total loss
        if self.dynamic_weighting:
            weight_velocity = torch.exp(self.log_weight_velocity)
            weight_displacement = torch.exp(self.log_weight_displacement)

            loss_dict['total'] = (
                (1.0 / weight_velocity**2) * velocity_loss
                + (1.0 / weight_displacement**2) * displacement_loss
                + torch.log(weight_velocity * weight_displacement)
            )
            loss_dict['weight_velocity'] = weight_velocity
            loss_dict['weight_displacement'] = weight_displacement
        else:
            loss_dict['total'] = (
                self.loss_hparams['velocity_loss_weight'] * velocity_loss
                + self.loss_hparams['displacement_loss_weight'] * displacement_loss
            )

        return loss_dict

    @override
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
        signals, velocity_target, displacement_target = batch
        prediction = self(signals)  # Model prediction (velocity or displacement)

        # For CNN with stride > 1, extract the corresponding target values
        if hasattr(self, '_output_full_signal') and not self._output_full_signal:
            # Downsample both targets to match prediction stride
            downsampled_velocity = torch.zeros_like(prediction)
            downsampled_displacement = torch.zeros_like(prediction)

            for window_idx, i in enumerate(
                range(0, velocity_target.shape[1], self._window_stride)
            ):
                if window_idx >= self._num_windows:
                    break
                downsampled_velocity[:, window_idx] = velocity_target[:, i]
                downsampled_displacement[:, window_idx] = displacement_target[:, i]

            velocity_target = downsampled_velocity
            displacement_target = downsampled_displacement

        loss_dict = self.loss_function(prediction, velocity_target, displacement_target)

        # Log each loss component with train_ prefix
        for loss_name, loss_value in loss_dict.items():
            self.log(
                f'train/{loss_name}_loss',
                loss_value,
                prog_bar=(loss_name == 'total'),  # Only show total loss in progress bar
                on_epoch=True,
            )

        # Return the total loss for backpropagation
        return loss_dict['total']

    @override
    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Validation step for model.

        Args:
            batch: Tuple of (signals, velocity_target, displacement_target)
            batch_idx: Index of current batch
        """
        signals, velocity_target, displacement_target = batch
        prediction = self(signals)  # Model prediction (velocity or displacement)

        # For CNN with stride > 1, extract the corresponding target values
        if hasattr(self, '_output_full_signal') and not self._output_full_signal:
            # Downsample both targets to match prediction stride
            downsampled_velocity = torch.zeros_like(prediction)
            downsampled_displacement = torch.zeros_like(prediction)

            for window_idx, i in enumerate(
                range(0, velocity_target.shape[1], self._window_stride)
            ):
                if window_idx >= self._num_windows:
                    break
                downsampled_velocity[:, window_idx] = velocity_target[:, i]
                downsampled_displacement[:, window_idx] = displacement_target[:, i]

            velocity_target = downsampled_velocity
            displacement_target = downsampled_displacement

        loss_dict = self.loss_function(prediction, velocity_target, displacement_target)

        # Log each loss component with val_ prefix
        for loss_name, loss_value in loss_dict.items():
            self.log(
                f'val/{loss_name}_loss',
                loss_value,
                prog_bar=(loss_name == 'total'),  # Only show total loss in progress bar
                sync_dist=torch.distributed.is_available()
                and torch.distributed.is_initialized(),
            )

    @override
    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Test step for model.

        Args:
            batch: Tuple of (signals, velocity_target, displacement_target)
            batch_idx: Index of current batch
        """
        signals, velocity_target, displacement_target = batch
        prediction = self(signals)  # Model prediction (velocity or displacement)

        # For CNN with stride > 1, extract the corresponding target values
        if hasattr(self, '_output_full_signal') and not self._output_full_signal:
            # Downsample both targets to match prediction stride
            downsampled_velocity = torch.zeros_like(prediction)
            downsampled_displacement = torch.zeros_like(prediction)

            for window_idx, i in enumerate(
                range(0, velocity_target.shape[1], self._window_stride)
            ):
                if window_idx >= self._num_windows:
                    break
                downsampled_velocity[:, window_idx] = velocity_target[:, i]
                downsampled_displacement[:, window_idx] = displacement_target[:, i]

            velocity_target = downsampled_velocity
            displacement_target = downsampled_displacement

        loss_dict = self.loss_function(prediction, velocity_target, displacement_target)

        # Log each loss component with test_ prefix
        for loss_name, loss_value in loss_dict.items():
            self.log(
                f'test/{loss_name}_loss',
                loss_value,
                sync_dist=torch.distributed.is_available()
                and torch.distributed.is_initialized(),
            )

    @override
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

        prediction = self(signals)  # Model prediction (velocity or displacement)
        sample_rate = RedPitayaConfig.SAMPLE_RATE_DEC1 / 256

        # Derive both velocity and displacement for plotting, regardless of target
        if self.target == 'velocity':
            velocity_hat = prediction
            displacement_hat = CoilDriver.integrate_velocity(
                velocity_hat, sample_rate=sample_rate
            )
        else:  # self.target == 'displacement'
            displacement_hat = prediction
            velocity_hat = CoilDriver.derivative_displacement(
                displacement_hat, sample_rate
            )

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
