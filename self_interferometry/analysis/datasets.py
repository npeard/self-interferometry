#!/usr/bin/env python

import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from self_interferometry.acquisition.redpitaya.redpitaya_config import RedPitayaConfig
from self_interferometry.acquisition.simulations.coil_driver import CoilDriver

logger = logging.getLogger(__name__)


class VelocityDataset(Dataset):
    """Dataset class for velocity prediction from photodiode signals.

    This dataset is designed to work with HDF5 files containing Red Pitaya channel data
    with the following structure:
    - RP1_CH1: Speaker drive voltage (from which velocity is computed)
    - RP1_CH2, RP2_CH1, RP2_CH2: Photodiode signals from interferometers

    The dataset automatically applies z-score normalization to photodiode signals:
    Each photodiode channel is normalized by computing the mean and standard deviation
    across all samples in the dataset, then applying (signal - mean) / std for each
    sample. This ensures stable training with balanced channel contributions.

    The dataset returns:
    - signals: Z-score normalized photodiode signals (1-3 channels based on
      num_pd_channels)
    - velocity: Velocity computed from speaker drive voltage
    - displacement: Displacement computed by integrating velocity

    For standard models that only need velocity, the displacement can be ignored in the
    training loop.

    Args:
        file_path: Path to HDF5 file
        num_pd_channels: Number of photodiode channels to use (1-3)
        cache_size: Number of items to cache in memory (0 for no caching)
        channel_dropout: Probability of dropping a channel during training
            (default: 0.0)
    """

    def __init__(
        self,
        file_path: str | Path,
        num_pd_channels: int = 3,
        cache_size: int = 0,
        channel_dropout: float = 0.0,
    ):
        self.file_path = file_path
        self.num_pd_channels = min(max(1, num_pd_channels), 3)  # Ensure between 1 and 3
        self.cache_size = cache_size
        self.channel_dropout = channel_dropout  # Probability of dropping a channel
        self.sample_rate = None  # Will be set in open_hdf5

        # Channel keys for photodiode signals
        self.pd_channel_keys = ['RP1_CH2', 'RP2_CH1', 'RP2_CH2'][: self.num_pd_channels]

        # Channel key for speaker drive voltage
        self.voltage_key = 'RP1_CH1'

        # Initialize cache
        self._cache = {}
        self._cache_keys = []

        # Initialize CoilDriver for velocity calculation
        self.coil_driver = CoilDriver()

        # Validate file and get dataset info
        with h5py.File(self.file_path, 'r') as f:
            self._validate_file_structure(f)
            self.length = self._get_dataset_length(f)

        # Compute normalization statistics for photodiode channels
        self._compute_normalization_stats()

        self.opened_flag = False

    def _validate_file_structure(self, f: h5py.File) -> None:
        """Validate the HDF5 file has required datasets."""
        if self.voltage_key not in f:
            raise ValueError(f"Voltage key '{self.voltage_key}' not found in file")

        for key in self.pd_channel_keys:
            if key not in f:
                raise ValueError(f"Photodiode channel key '{key}' not found in file")

    def _get_dataset_length(self, f: h5py.File) -> int:
        """Get the length of the dataset."""
        return len(f[self.voltage_key])

    def _compute_normalization_stats(self) -> None:
        """Compute mean and std for each photodiode channel across all samples.

        This method processes the data in chunks to avoid loading everything into
        memory simultaneously, then computes overall statistics.
        """
        logger.info('Computing normalization statistics for photodiode channels...')

        # Initialize accumulators for each channel
        channel_sums = np.zeros(self.num_pd_channels)
        channel_sq_sums = np.zeros(self.num_pd_channels)
        total_points_per_channel = 0

        # Process data in chunks to manage memory usage
        chunk_size = min(1000, self.length)

        with h5py.File(self.file_path, 'r') as f:
            pd_datasets = [f[key] for key in self.pd_channel_keys]

            for start_idx in range(0, self.length, chunk_size):
                end_idx = min(start_idx + chunk_size, self.length)

                # Load chunk data for all channels
                chunk_data = np.array(
                    [dataset[start_idx:end_idx] for dataset in pd_datasets]
                )
                # Shape: (num_channels, chunk_samples, signal_length)

                # Accumulate statistics for each channel
                for ch_idx in range(self.num_pd_channels):
                    channel_data = chunk_data[ch_idx].flatten()
                    channel_sums[ch_idx] += np.sum(channel_data)
                    channel_sq_sums[ch_idx] += np.sum(channel_data**2)

                # Update total points per channel (same for all channels)
                total_points_per_channel += chunk_data[0].size

        # Compute final statistics
        channel_means = channel_sums / total_points_per_channel
        channel_vars = channel_sq_sums / total_points_per_channel - channel_means**2
        channel_stds = np.sqrt(channel_vars)

        # Add small epsilon to prevent division by zero
        epsilon = 1e-8
        channel_stds = np.maximum(channel_stds, epsilon)

        # Store statistics for use in __getitem__ (reshape for broadcasting)
        self.pd_means = channel_means.reshape(-1, 1)
        self.pd_stds = channel_stds.reshape(-1, 1)

        logger.info(
            f'Normalization stats computed - '
            f'Means: {channel_means}, Stds: {channel_stds}'
        )

    def open_hdf5(self):
        """Open HDF5 file for reading.

        This is done lazily to support multiprocessing in DataLoader.
        Also sets the sample_rate from file attributes if available.
        """
        if not self.opened_flag:
            self.h5_file = h5py.File(self.file_path, 'r')
            self.voltage_data = self.h5_file[self.voltage_key]
            self.pd_data = [self.h5_file[key] for key in self.pd_channel_keys]

            # Get sample rate from file attributes if available
            if 'sample_rate' in self.h5_file.attrs:
                self.sample_rate = float(self.h5_file.attrs['sample_rate'])
            else:
                # Default sample rate for Red Pitaya with decimation of 256
                self.sample_rate = RedPitayaConfig.SAMPLE_RATE_DEC1 / 256
                logger.warning(
                    f'Using default sample rate of {self.sample_rate:.2f} Hz'
                )

            self.opened_flag = True

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single item from the dataset.

        Returns:
            Tuple of (signals, velocity, displacement) where:
            - signals: Tensor of shape (num_pd_channels, signal_length) containing
              photodiode signals
            - velocity: Tensor of shape (signal_length,) containing velocity
            - displacement: Tensor of shape (signal_length,) containing displacement
              computed by integrating velocity
        """
        # Check cache first
        if idx in self._cache:
            signals, velocity, displacement = self._cache[idx]
        else:
            # Lazy loading of HDF5 file
            self.open_hdf5()

            # Load photodiode signals
            pd_signals = [self.pd_data[i][idx] for i in range(len(self.pd_data))]

            # Load voltage and compute velocity
            voltage = self.voltage_data[idx]
            velocity, _, _ = self.coil_driver.get_velocity(voltage, self.sample_rate)
            displacement, _, _ = self.coil_driver.get_displacement(
                voltage, self.sample_rate
            )

            # Stack photodiode signals into a single tensor
            signals = np.stack(pd_signals)

            # Add to cache
            if self.cache_size > 0:
                self._add_to_cache(idx, (signals, velocity, displacement))

        # Apply z-score normalization to photodiode signals
        signals = (signals - self.pd_means) / self.pd_stds

        # Apply channel dropout during training if probability > 0 and multiple channels
        if (
            self.channel_dropout > 0
            and self.num_pd_channels > 1
            and np.random.random() < self.channel_dropout
        ):
            # Randomly select a channel to drop
            channel_to_drop = np.random.randint(0, self.num_pd_channels)
            # Set the selected channel to zeros
            signals[channel_to_drop, :] = 0.0

        # Convert to PyTorch tensors
        signals_tensor = torch.FloatTensor(signals)
        velocity_tensor = torch.FloatTensor(velocity)
        displacement_tensor = torch.FloatTensor(displacement)

        return signals_tensor, velocity_tensor, displacement_tensor

    def _add_to_cache(
        self, key: int, value: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Add an item to the cache, maintaining cache size limit."""
        if self.cache_size == 0:
            return

        if len(self._cache) >= self.cache_size:
            # Remove oldest item if cache is full
            oldest_key = self._cache_keys[0]
            del self._cache[oldest_key]
            self._cache_keys.pop(0)

        self._cache[key] = value
        self._cache_keys.append(key)


def get_data_loaders(
    dataset_path: str,
    split_ratios: tuple[int, int, int] = (80, 10, 10),
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    **dataset_kwargs: dict[str],
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for training, validation, and testing from a single dataset.

    Args:
        dataset_path: Path to the single HDF5 dataset file
        split_ratios: Tuple of (train, val, test) percentages that sum to 100
        batch_size: Batch size for all dataloaders
        num_workers: Number of worker processes for data loading
        seed: Random seed for reproducible splits
        **dataset_kwargs: Additional arguments to pass to the dataset class

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Validate split ratios
    if sum(split_ratios) != 100:
        raise ValueError(f'Split ratios must sum to 100, got {sum(split_ratios)}')

    # Create the full dataset to get its length
    full_dataset = VelocityDataset(dataset_path, **dataset_kwargs)
    total_length = len(full_dataset)

    # Calculate split sizes
    train_size = int(total_length * split_ratios[0] / 100)
    val_size = int(total_length * split_ratios[1] / 100)
    test_size = total_length - train_size - val_size  # Remainder goes to test

    # Create random split
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    logger.info(
        f'Dataset split - Total: {total_length}, Train: {train_size}, '
        f'Val: {val_size}, Test: {test_size}'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
