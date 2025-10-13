#!/usr/bin/env python

import logging
from typing import Any, override

import lightning as L
import torch
from torch import nn, optim

from self_interferometry.acquisition.redpitaya.redpitaya_config import RedPitayaConfig
from self_interferometry.acquisition.simulations.coil_driver import CoilDriver
from self_interferometry.analysis.barland_cnn import BarlandCNN, BarlandCNNConfig
from self_interferometry.analysis.tcn import TCN, TCNConfig
from self_interferometry.analysis.utcn import UTCN, UTCNConfig

# Conditional import for FNO and UNO - only available with torch >= 2.8
try:
    from self_interferometry.analysis.fno import NEURALOP_AVAILABLE, FNO1d, FNOConfig
    from self_interferometry.analysis.uno import UNO1d, UNOConfig
except ImportError:
    NEURALOP_AVAILABLE = False
    FNO1d = None
    FNOConfig = None
    UNO1d = None
    UNOConfig = None

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
    ):
        """Args:
        model_hparams: Hyperparameters for the model
        optimizer_hparams: Hyperparameters for the optimizer
        scheduler_hparams: Hyperparameters for the learning rate scheduler
        loss_hparams: Hyperparameters for the loss function
        training_hparams: Hyperparameters for training configuration (includes target)
        """
        super().__init__()
        self.model_hparams = model_hparams
        self.optimizer_hparams = optimizer_hparams
        self.scheduler_hparams = scheduler_hparams
        self.loss_hparams = loss_hparams
        self.training_hparams = training_hparams
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

    def _create_barland_config(self) -> BarlandCNNConfig:
        """Create BarlandCNNConfig from model configuration."""
        return BarlandCNNConfig(
            # Common parameters
            input_size=self.model_hparams['input_size'],
            output_size=self.model_hparams['output_size'],
            in_channels=self.model_hparams['in_channels'],
            activation=self.model_hparams['activation'],
            dropout=self.model_hparams['dropout'],
            # BarlandCNN specific parameters
            window_stride=self.model_hparams['window_stride'],
        )

    def _create_tcn_config(self) -> TCNConfig:
        """Create TCNConfig from model configuration."""
        return TCNConfig(
            # Common parameters
            input_size=self.model_hparams['input_size'],
            output_size=self.model_hparams['output_size'],
            in_channels=self.model_hparams['in_channels'],
            activation=self.model_hparams['activation'],
            norm=self.model_hparams['norm'],
            dropout=self.model_hparams['dropout'],
            # TCN specific parameters
            kernel_size=self.model_hparams['kernel_size'],
            num_channels=self.model_hparams['num_channels'],
            dilation_base=self.model_hparams['dilation_base'],
            stride=self.model_hparams['stride'],
        )

    def _create_utcn_config(self) -> UTCNConfig:
        """Create UTCNConfig from model configuration."""
        return UTCNConfig(
            # Common parameters
            input_size=self.model_hparams['input_size'],
            output_size=self.model_hparams['output_size'],
            in_channels=self.model_hparams['in_channels'],
            activation=self.model_hparams['activation'],
            norm=self.model_hparams['norm'],
            dropout=self.model_hparams['dropout'],
            # UTCN specific parameters
            kernel_size=self.model_hparams['kernel_size'],
            n_layers=self.model_hparams['n_layers'],
            utcn_out_channels=self.model_hparams['utcn_out_channels'],
            utcn_dilations=self.model_hparams['utcn_dilations'],
            horizontal_skips_map=self.model_hparams['horizontal_skips_map'],
            horizontal_skip=self.model_hparams['horizontal_skip'],
            stride=self.model_hparams['stride'],
        )

    def _create_fno_config(self):
        """Create FNOConfig from model configuration."""
        if not NEURALOP_AVAILABLE:
            raise ImportError(
                'FNO model requires the neuralop library, which requires PyTorch >= 2.8. '
                'Please upgrade PyTorch or use a different model (CNN/TCN).'
            )
        return FNOConfig(
            # Common parameters (required by training interface)
            input_size=self.model_hparams['input_size'],
            output_size=self.model_hparams['output_size'],
            in_channels=self.model_hparams['in_channels'],
            # FNO specific parameters
            n_modes=(self.model_hparams['n_modes'],),  # Expects tuple of length 1
            hidden_channels=self.model_hparams['hidden_channels'],
            n_layers=self.model_hparams['n_layers'],
            max_n_modes=self.model_hparams['max_n_modes'],
            fno_block_precision=self.model_hparams['fno_block_precision'],
            use_channel_mlp=self.model_hparams['use_channel_mlp'],
            channel_mlp_dropout=self.model_hparams['channel_mlp_dropout'],
            channel_mlp_expansion=self.model_hparams['channel_mlp_expansion'],
            non_linearity=self.model_hparams['non_linearity'],
            stabilizer=self.model_hparams['stabilizer'],
            norm=self.model_hparams['norm'],
            preactivation=self.model_hparams['preactivation'],
            fno_skip=self.model_hparams['fno_skip'],
            channel_mlp_skip=self.model_hparams['channel_mlp_skip'],
            separable=self.model_hparams['separable'],
            factorization=self.model_hparams['factorization'],
            rank=self.model_hparams['rank'],
            joint_factorization=self.model_hparams['joint_factorization'],
            fixed_rank_modes=self.model_hparams['fixed_rank_modes'],
            implementation=self.model_hparams['implementation'],
            decomposition_kwargs=self.model_hparams['decomposition_kwargs'],
        )

    def _create_uno_config(self):
        """Create UNOConfig from model configuration."""
        if not NEURALOP_AVAILABLE:
            raise ImportError(
                'UNO model requires the neuralop library, which requires PyTorch >= 2.8. '
                'Please upgrade PyTorch or use a different model (CNN/TCN).'
            )
        return UNOConfig(
            # Common parameters (required by training interface)
            input_size=self.model_hparams['input_size'],
            output_size=self.model_hparams['output_size'],
            in_channels=self.model_hparams['in_channels'],
            # UNO specific parameters
            hidden_channels=self.model_hparams['hidden_channels'],
            lifting_channels=self.model_hparams['lifting_channels'],
            projection_channels=self.model_hparams['projection_channels'],
            positional_embedding=self.model_hparams['positional_embedding'],
            n_layers=self.model_hparams['n_layers'],
            uno_out_channels=self.model_hparams['uno_out_channels'],
            uno_n_modes=self.model_hparams['uno_n_modes'],
            uno_scalings=self.model_hparams['uno_scalings'],
            horizontal_skips_map=self.model_hparams['horizontal_skips_map'],
            incremental_n_modes=self.model_hparams['incremental_n_modes'],
            channel_mlp_dropout=self.model_hparams['channel_mlp_dropout'],
            channel_mlp_expansion=self.model_hparams['channel_mlp_expansion'],
            non_linearity=self.model_hparams['non_linearity'],
            norm=self.model_hparams['norm'],
            preactivation=self.model_hparams['preactivation'],
            fno_skip=self.model_hparams['fno_skip'],
            horizontal_skip=self.model_hparams['horizontal_skip'],
            channel_mlp_skip=self.model_hparams['channel_mlp_skip'],
            separable=self.model_hparams['separable'],
            factorization=self.model_hparams['factorization'],
            rank=self.model_hparams['rank'],
            fixed_rank_modes=self.model_hparams['fixed_rank_modes'],
            implementation=self.model_hparams['implementation'],
            decomposition_kwargs=self.model_hparams['decomposition_kwargs'],
            domain_padding=self.model_hparams['domain_padding'],
            domain_padding_mode=self.model_hparams['domain_padding_mode'],
        )

    def create_model(self) -> nn.Module:
        """Create model instance based on config.

        Returns:
            A PyTorch model (CNN or TCN) based on the configuration
        """
        if self.model_hparams == 'ensemble':
            # This indicates that we are creating an ensemble of models
            # and no model will be created at this point.
            # See Ensemble(Fusion) class for more.
            return None
        model_type = self.model_hparams['type']
        if model_type == 'Barland':
            logger.debug('Creating BarlandCNN model...')
            self.model_config = self._create_barland_config()
            return BarlandCNN(self.model_config)
        elif model_type == 'TCN':
            logger.debug('Creating TCN model...')
            self.model_config = self._create_tcn_config()
            return TCN(self.model_config)
        elif model_type == 'UTCN':
            logger.debug('Creating UTCN model...')
            self.model_config = self._create_utcn_config()
            return UTCN(self.model_config)
        elif model_type == 'FNO':
            if not NEURALOP_AVAILABLE:
                raise ImportError(
                    'FNO model requires the neuralop library, which requires PyTorch >= 2.8. '
                    'Please upgrade PyTorch or use a different model (CNN/TCN).'
                )
            logger.debug('Creating FNO model...')
            self.model_config = self._create_fno_config()
            return FNO1d(self.model_config)
        elif model_type == 'UNO':
            if not NEURALOP_AVAILABLE:
                raise ImportError(
                    'UNO model requires the neuralop library, which requires PyTorch >= 2.8. '
                    'Please upgrade PyTorch or use a different model (CNN/TCN).'
                )
            logger.debug('Creating UNO model...')
            self.model_config = self._create_uno_config()
            return UNO1d(self.model_config)
        else:
            raise ValueError(f'Unknown model type: {model_type}')

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

        # Check if we're using a TCN or FNO model (which can process the entire sequence at
        # once)
        if hasattr(self.model, '__class__') and (
            self.model.__class__.__name__ in {'TCN', 'FNO1d', 'UNO1d'}
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
            # Discard weight_decay hyperparameter for SGD optimizer
            _ = self.optimizer_hparams.pop('weight_decay', None)
            optimizer = optim.SGD(self.parameters(), **self.optimizer_hparams)
        else:
            raise ValueError(f'Unknown optimizer: {optimizer_name}')

        # Configure multi-stage scheduler: linear warmup then cosine annealing
        warmup_epochs = self.scheduler_hparams['warmup_epochs']
        eta_min = self.scheduler_hparams['eta_min']
        T_max = self.scheduler_hparams['T_max']

        # LambdaLR for linear warmup
        def warmup_lambda(epoch: int) -> float:
            if warmup_epochs == 0:
                return 1.0
            return float(epoch + 1) / float(warmup_epochs)

        warmup_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=warmup_lambda
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eval(eta_min)
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
                sync_dist=True,
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
            self.log(f'test/{loss_name}_loss', loss_value, sync_dist=True)

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
