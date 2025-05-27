#!/usr/bin/env python

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from self_interferometry.redpitaya.coil_driver import CoilDriver


class StandardVelocityDataset(Dataset):
    """Dataset class for velocity prediction from photodiode signals.

    This dataset is designed to work with HDF5 files containing Red Pitaya channel data
    with the following structure:
    - RP1_CH1: Speaker drive voltage (from which velocity is computed)
    - RP1_CH2, RP2_CH1, RP2_CH2: Photodiode signals from interferometers

    The dataset returns:
    - signals: Photodiode signals (1-3 channels based on num_pd_channels)
    - velocity: Velocity computed from speaker drive voltage
    - displacement: Displacement computed by integrating velocity

    For standard models that only need velocity, the displacement can be ignored in the
    training loop. For teacher models, all three outputs can be used.

    Args:
        file_path: Path to HDF5 file
        num_pd_channels: Number of photodiode channels to use (1-3)
        cache_size: Number of items to cache in memory (0 for no caching)
    """

    def __init__(
        self, file_path: str | Path, num_pd_channels: int = 3, cache_size: int = 0
    ):
        self.file_path = file_path
        self.num_pd_channels = min(max(1, num_pd_channels), 3)  # Ensure between 1 and 3
        self.cache_size = cache_size
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
                self.sample_rate = 125e6 / 256
                print(
                    f'Warning: Using default sample rate of {self.sample_rate:.2f} Hz'
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
    train_path: str,
    val_path: str,
    test_path: str,
    batch_size: int,
    num_workers: int = 4,
    **dataset_kwargs: dict[str],
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for training, validation, and testing.

    Args:
        train_path: Path to training data HDF5 file
        val_path: Path to validation data HDF5 file
        test_path: Path to test data HDF5 file
        batch_size: Batch size for all dataloaders
        num_workers: Number of worker processes for data loading
        **dataset_kwargs: Additional arguments to pass to the dataset class

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = StandardVelocityDataset(train_path, **dataset_kwargs)
    val_dataset = StandardVelocityDataset(val_path, **dataset_kwargs)
    test_dataset = StandardVelocityDataset(test_path, **dataset_kwargs)

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
