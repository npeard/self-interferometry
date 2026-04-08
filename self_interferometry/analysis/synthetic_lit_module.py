#!/usr/bin/env python

import logging
from typing import override

import torch
from torch.utils.data import Dataset

from self_interferometry.acquisition.redpitaya.redpitaya_config import RedPitayaConfig
from self_interferometry.acquisition.simulations.coil_driver import CoilDriver
from self_interferometry.acquisition.simulations.waveform import Waveform
from self_interferometry.analysis.lit_module import LitModule

logger = logging.getLogger(__name__)

DEFAULT_WAVELENGTHS_NM = [635, 674.8, 515, 780, 450]


class SyntheticIndexDataset(Dataset):
    """Dummy dataset that returns indices, for use with SyntheticLitModule.

    The actual data is generated on-device in the LightningModule's training_step.
    This dataset only provides the correct number of iterations per epoch.
    """

    def __init__(self, length: int):
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> int:
        return idx


class SyntheticLitModule(LitModule):
    """Lightning Module that generates synthetic interferometer data on-the-fly on GPU.

    Extends LitModule with on-device synthetic data generation. Each training/validation
    step generates a fresh batch of random displacement waveforms, passes them through
    simulated Michelson interferometers, and feeds the result to the parent class's
    loss computation.

    The displacement waveforms are generated using the same Rayleigh amplitude + random
    phase approach as the real experiment's Waveform class, but in PyTorch for GPU
    acceleration. The coil driver transfer function is skipped — displacement is generated
    directly.
    """

    def __init__(
        self,
        model_hparams: dict | None = None,
        optimizer_hparams: dict | None = None,
        scheduler_hparams: dict | None = None,
        loss_hparams: dict | None = None,
        training_hparams: dict | None = None,
        data_hparams: dict | None = None,
        wavelengths_nm: list[float] | None = None,
        start_freq: float = 1.0,
        end_freq: float = 1000.0,
        steps_per_epoch: int = 1000,
        max_displacement_um: float = 5.0,
    ):
        """Args:
        model_hparams: Hyperparameters for the model
        optimizer_hparams: Hyperparameters for the optimizer
        scheduler_hparams: Hyperparameters for the learning rate scheduler
        loss_hparams: Hyperparameters for the loss function
        training_hparams: Hyperparameters for training configuration
        data_hparams: Hyperparameters for data configuration
        wavelengths_nm: List of interferometer wavelengths in nanometers
        start_freq: Lower bound of displacement spectrum frequency range (Hz)
        end_freq: Upper bound of displacement spectrum frequency range (Hz)
        steps_per_epoch: Number of training batches per epoch
        max_displacement_um: Peak displacement amplitude in microns
        """
        super().__init__(
            model_hparams=model_hparams,
            optimizer_hparams=optimizer_hparams,
            scheduler_hparams=scheduler_hparams,
            loss_hparams=loss_hparams,
            training_hparams=training_hparams,
            data_hparams=data_hparams,
        )

        if wavelengths_nm is None:
            raise ValueError('wavelengths_nm must be provided for synthetic training')

        assert len(wavelengths_nm) == model_hparams['in_channels'], (
            f'Number of wavelengths ({len(wavelengths_nm)}) must match '
            f'in_channels ({model_hparams["in_channels"]})'
        )

        self.steps_per_epoch = steps_per_epoch
        self.max_displacement_um = max_displacement_um
        self.acq_sample_rate = RedPitayaConfig.SAMPLE_RATE_DEC1 / 256

        # Store wavelengths in microns as a buffer for device transfer
        # Shape [1, C, 1] for broadcasting with [B, 1, L] displacement
        wavelengths_um = [w / 1000.0 for w in wavelengths_nm]
        self.register_buffer(
            'wavelengths_um',
            torch.tensor(wavelengths_um, dtype=torch.float32).view(1, -1, 1),
        )

        # Create waveform generator (reuses existing frequency grid logic)
        self.waveform = Waveform(start_freq=start_freq, end_freq=end_freq)

        # Save synthetic-specific hyperparameters
        self.save_hyperparameters(ignore=['model'])

        logger.info(
            f'SyntheticLitModule initialized with {len(wavelengths_nm)} '
            f'interferometers: {wavelengths_nm} nm'
        )

    def _generate_synthetic_batch(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a batch of synthetic interferometer data on the given device.

        Args:
            batch_size: Number of samples to generate
            device: Target device for tensor creation

        Returns:
            Tuple of (signals, velocity, displacement) where:
            - signals: [batch_size, num_channels, seq_len] interferometer signals
            - velocity: [batch_size, seq_len] displacement velocity
            - displacement: [batch_size, seq_len] mirror displacement in microns
        """
        # 1. Generate random displacement waveforms [B, L]
        displacement = self.waveform.generate_batch_torch(batch_size, device)

        # Apply random amplitude scaling per sample
        scale = torch.rand(batch_size, 1, device=device) * self.max_displacement_um
        displacement = displacement * scale

        # Normalize each waveform to have unit max amplitude before scaling
        max_abs = displacement.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
        displacement = displacement / max_abs * scale

        # 2. Random interferometer phases per sample per channel [B, C, 1]
        num_channels = self.wavelengths_um.shape[1]
        random_phases = (
            torch.rand(batch_size, num_channels, 1, device=device) * 2.0 * torch.pi
        )

        # 3. Compute interferometer signals: cos(2*pi/lambda * 2 * displacement + phase)
        # displacement: [B, L] -> [B, 1, L] for broadcasting
        signals = torch.cos(
            2.0 * torch.pi / self.wavelengths_um * 2.0 * displacement.unsqueeze(1)
            + random_phases
        )

        # 4. Per-sample per-channel z-score normalization
        signals_mean = signals.mean(dim=-1, keepdim=True)
        signals_std = signals.std(dim=-1, keepdim=True).clamp(min=1e-8)
        signals = (signals - signals_mean) / signals_std

        # 5. Compute velocity from displacement
        velocity = CoilDriver.derivative_displacement(
            displacement, self.acq_sample_rate
        )

        return signals, velocity, displacement

    @override
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Generate synthetic data on-device and delegate to parent training_step.

        Args:
            batch: Tensor of indices from SyntheticIndexDataset (used only for batch size)
            batch_idx: Index of current batch
        """
        batch_size = len(batch)
        synthetic_batch = self._generate_synthetic_batch(batch_size, self.device)
        return super().training_step(synthetic_batch, batch_idx)

    @override
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Generate synthetic data on-device and delegate to parent validation_step.

        Args:
            batch: Tensor of indices from SyntheticIndexDataset (used only for batch size)
            batch_idx: Index of current batch
        """
        batch_size = len(batch)
        synthetic_batch = self._generate_synthetic_batch(batch_size, self.device)
        super().validation_step(synthetic_batch, batch_idx)

    @override
    def predict_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate synthetic data on-device and delegate to parent predict_step."""
        batch_size = len(batch)
        synthetic_batch = self._generate_synthetic_batch(batch_size, self.device)
        return super().predict_step(synthetic_batch, batch_idx)

    @override
    def on_validation_epoch_start(self) -> None:
        """Set a fixed random seed for reproducible validation data."""
        torch.manual_seed(42)
