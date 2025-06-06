#!/usr/bin/env python3


import torch
from torch import nn

from self_interferometry.signal_analysis.lightning_standard import Standard
from self_interferometry.signal_analysis.models_cnn import BarlandCNN, BarlandCNNConfig
from self_interferometry.signal_analysis.models_fcn import FCN, FCNConfig
from self_interferometry.signal_analysis.models_tcn import TCN, TCNConfig


class Ensemble(Standard):
    """Lightning module that uses an ensemble of single-channel models.

    This module creates multiple single-channel models (one for each input channel)
    and averages their predictions. This allows comparison between multi-channel
    models and ensembles of single-channel models.
    """

    def __init__(
        self,
        model_hparams: dict | None = None,
        optimizer_hparams: dict | None = None,
        scheduler_hparams: dict | None = None,
        loss_hparams: dict | None = None,
    ):
        """Initialize the MeanSingleChannel module.

        Args:
            model_hparams: Model hyperparameters
            optimizer_hparams: Optimizer hyperparameters
        """
        # Initialize the parent class but don't create the model yet
        super().__init__(
            model_hparams='ensemble',  # Prevent model creation in parent class
            optimizer_hparams=optimizer_hparams,
            scheduler_hparams=scheduler_hparams,
            loss_hparams=loss_hparams,
        )
        self.model_hparams = model_hparams
        self.save_hyperparameters()

        # Create the ensemble of models
        self.models = self.create_ensemble()

        # Set high precision for matrix multiplication
        torch.set_float32_matmul_precision('high')

    def create_ensemble(self) -> nn.ModuleList:
        """Create an ensemble of single-channel models.

        Returns:
            ModuleList containing single-channel models
        """
        # Determine the number of input channels
        in_channels = self.model_hparams.get('in_channels', 3)

        # Create a list to store the models
        models = []

        # Create one model per input channel
        print(f'Creating {in_channels} single-channel models...')  # noqa: T201
        for _ in range(in_channels):
            # Create a copy of model_hparams with in_channels=1
            single_channel_hparams = self.model_hparams.copy()
            single_channel_hparams['in_channels'] = 1

            # Create the appropriate model type
            model_type = single_channel_hparams.get('type', 'CNN')

            if model_type == 'CNN':
                print('Creating CNN model...')  # noqa: T201
                config = self._create_cnn_config_single_channel()
                models.append(BarlandCNN(config))
            elif model_type == 'TCN':
                print('Creating TCN model...')  # noqa: T201
                config = self._create_tcn_config_single_channel()
                models.append(TCN(config))
            elif model_type == 'FCN':
                print('Creating FCN model...')  # noqa: T201
                config = self._create_fcn_config_single_channel()
                models.append(FCN(config))
            else:
                raise ValueError(f'Unknown model type: {model_type}')

            print(f'Created model {len(models)}')  # noqa: T201

        return nn.ModuleList(models)

    def _create_cnn_config_single_channel(self) -> BarlandCNNConfig:
        """Create CNNConfig for a single-channel model."""
        return BarlandCNNConfig(
            # Common parameters
            input_size=self.model_hparams.get('input_size', 256),
            output_size=self.model_hparams.get('output_size', 1),
            in_channels=1,  # Always 1 for single-channel models
            activation=self.model_hparams.get('activation', 'LeakyReLU'),
            dropout=self.model_hparams.get('dropout', 0.1),
            # BarlandCNN specific parameters
            window_stride=self.model_hparams.get('window_stride', 128),
        )

    def _create_tcn_config_single_channel(self) -> TCNConfig:
        """Create TCNConfig for a single-channel model."""
        return TCNConfig(
            # Common parameters
            input_size=self.model_hparams.get('input_size', 16384),
            output_size=self.model_hparams.get('output_size', 16384),
            in_channels=1,  # Always 1 for single-channel models
            activation=self.model_hparams.get('activation', 'LeakyReLU'),
            dropout=self.model_hparams.get('dropout', 0.1),
            # TCN specific parameters
            kernel_size=self.model_hparams.get('kernel_size', 7),
            num_channels=self.model_hparams.get('num_channels', [16, 32, 64, 64]),
        )

    def _create_fcn_config_single_channel(self) -> FCNConfig:
        """Create FCNConfig for a single-channel model."""
        return FCNConfig(
            # Common parameters
            input_size=self.model_hparams.get('input_size', 16384),
            output_size=self.model_hparams.get('output_size', 16384),
            in_channels=1,  # Always 1 for single-channel models
            activation=self.model_hparams.get('activation', 'LeakyReLU'),
            dropout=self.model_hparams.get('dropout', 0.1),
            # FCN specific parameters
            num_channels=self.model_hparams.get('num_channels', [16, 32, 64, 64]),
            kernel_size=self.model_hparams.get('kernel_size', 7),
            use_final_conv=self.model_hparams.get('use_final_conv', True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ensemble of models.

        This method:
        1. Splits the input tensor into individual channels
        2. Processes each channel through its corresponding model
        3. Averages the predictions from all models

        Args:
            x: Input tensor of shape [batch_size, num_channels, signal_length]

        Returns:
            Tensor of shape [batch_size, signal_length] containing predicted velocity
        """
        batch_size, num_channels, signal_length = x.shape

        # Check if number of channels matches number of models
        if num_channels != len(self.models):
            raise ValueError(
                f'Input has {num_channels} channels but ensemble has {len(self.models)} models'
            )

        # Initialize list to store predictions from each model
        predictions = []

        # Process each channel through its corresponding model
        for i in range(num_channels):
            # Extract single channel
            x_channel = x[:, i : i + 1, :]  # Shape: [batch_size, 1, signal_length]

            # Get the corresponding model
            model = self.models[i]

            # Check if we're using a TCN or FCN model
            if model.__class__.__name__ in {'TCN', 'FCN'}:
                # Process the entire sequence at once
                with torch.set_grad_enabled(self.training):
                    # Model returns shape [batch_size, 1, signal_length]
                    output = model(x_channel)

                    # Reshape to [batch_size, signal_length]
                    velocity_hat = output.squeeze(1)

                predictions.append(velocity_hat)

            else:
                # CNN approach - sliding window implementation
                window_size = self.model_hparams.get('input_size', 256)
                window_stride = self.model_hparams.get('window_stride', 128)

                # Calculate number of windows
                num_windows = (signal_length + window_stride - 1) // window_stride

                # Create output tensor
                if window_stride == 1:
                    velocity_hat = torch.zeros(
                        (batch_size, signal_length), device=x.device
                    )
                    output_full_signal = True
                else:
                    velocity_hat = torch.zeros(
                        (batch_size, num_windows), device=x.device
                    )
                    output_full_signal = False

                # Add padding
                padding = window_size // 2
                padded_x = torch.nn.functional.pad(
                    x_channel, (padding, padding), mode='constant', value=0
                )

                # Process windows
                for window_idx, j in enumerate(range(0, signal_length, window_stride)):
                    # Extract window
                    start_idx = j
                    end_idx = j + window_size
                    window = padded_x[:, :, start_idx:end_idx]

                    # Skip incomplete windows
                    if window.shape[2] < window_size:
                        continue

                    # Get prediction
                    with torch.set_grad_enabled(self.training):
                        pred = model(window)

                    # Store prediction
                    if output_full_signal:
                        velocity_hat[:, j] = pred.squeeze()
                    else:
                        velocity_hat[:, window_idx] = pred.squeeze()

                # Store stride information
                self._output_full_signal = output_full_signal
                self._window_stride = window_stride
                self._num_windows = num_windows

                predictions.append(velocity_hat)

        # Average predictions from all models
        return torch.stack(predictions).mean(dim=0)
