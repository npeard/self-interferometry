#!/usr/bin/env python

from collections.abc import Callable
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from redpitaya.manager import RedPitayaManager


class H5Dataset(Dataset):
    """Base class for HDF5 datasets.

    Args:
        file_path: Path to HDF5 file
        input_key: Key for input data in HDF5 file
        target_key: Key for target data in HDF5 file
        transform: Optional transform to apply to inputs
        target_transform: Optional transform to apply to targets
        cache_size: Number of items to cache in memory (0 for no caching)
    """

    def __init__(
        self,
        file_path: str,
        input_key: str = 'inputs',
        target_key: str = 'targets',
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        cache_size: int = 0,
    ):
        self.file_path = file_path
        self.input_key = input_key
        self.target_key = target_key
        self.transform = transform
        self.target_transform = target_transform
        self.cache_size = cache_size

        # Initialize cache
        self._cache = {}
        self._cache_keys = []

        # Validate file and get dataset info
        with h5py.File(self.file_path, 'r') as f:
            self._validate_file_structure(f)
            self.length = self._get_dataset_length(f)

        self.opened_flag = False

    def _validate_file_structure(self, f: h5py.File) -> None:
        """Validate the HDF5 file has required datasets."""
        if self.input_key not in f:
            raise ValueError(f"Input key '{self.input_key}' not found in file")
        if self.target_key not in f:
            raise ValueError(f"Target key '{self.target_key}' not found in file")

    def _get_dataset_length(self, f: h5py.File) -> int:
        """Get the length of the dataset."""
        return len(f[self.target_key])

    def open_hdf5(self):
        """Open HDF5 file for reading.

        This is done lazily to support multiprocessing in DataLoader.
        """
        if not self.opened_flag:
            self.h5_file = h5py.File(self.file_path, 'r')
            self.inputs = self.h5_file[self.input_key]
            self.targets = self.h5_file[self.target_key]
            self.opened_flag = True

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single item from the dataset."""
        # Check cache first
        if idx in self._cache:
            inputs, targets = self._cache[idx]
        else:
            # Lazy loading of HDF5 file
            self.open_hdf5()

            # Load data
            inputs = self.inputs[idx]
            targets = self.targets[idx]

            # Apply transforms if specified
            if self.transform is not None:
                inputs = self.transform(inputs)
            if self.target_transform is not None:
                targets = self.target_transform(targets)

            # Add to cache
            if self.cache_size > 0:
                self._add_to_cache(idx, (inputs, targets))

        return torch.FloatTensor(inputs), torch.FloatTensor(targets)

    def _add_to_cache(self, key: int, value: tuple[np.ndarray, np.ndarray]) -> None:
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


def create_dataset(
    rp_manager,
    output_dir: str | Path,
    dataset_name: str,
    num_samples: int,
    device_idx: int = 0,
    delay_between_shots: float = 0.5,
    timeout: int = 5,
) -> str:
    """Create a dataset by acquiring multiple samples and saving them incrementally.

    This method uses run_multiple_shots to acquire data and saves it to an HDF5 file.

    Args:
        rp_manager: RedPitayaManager instance to use for data acquisition
        output_dir: Directory to save the dataset
        dataset_name: Name of the dataset file (e.g., 'train.h5')
        num_samples: Number of samples to acquire
        device_idx: Index of the device to use as primary
        delay_between_shots: Delay between shots in seconds
        timeout: Timeout for acquisition in seconds

    Returns:
        Path to the created dataset file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / dataset_name

    print(f'Acquiring {num_samples} samples for dataset: {dataset_name}')

    # Run multiple shots to acquire the data and save incrementally to HDF5
    rp_manager.run_multiple_shots(
        num_shots=num_samples,
        device_idx=device_idx,
        delay_between_shots=delay_between_shots,
        plot_data=False,  # Don't plot during dataset creation
        keep_final_plot=False,
        hdf5_file=str(file_path),  # Save directly to HDF5
        timeout=timeout,
    )

    return str(file_path)


def create_train_val_test_datasets_from_rp(
    rp_manager: RedPitayaManager,
    output_dir: str | Path,
    train_samples: int,
    val_samples: int,
    test_samples: int,
    **kwargs,
) -> tuple[str, str, str]:
    """Create train, validation, and test datasets from Red Pitaya data.

    Args:
        rp_manager: RedPitayaManager instance to use for data acquisition
        output_dir: Directory to save the datasets
        train_samples: Number of training samples
        val_samples: Number of validation samples
        test_samples: Number of test samples
        **kwargs: Additional arguments passed to create_dataset

    Returns:
        Tuple of (train_path, val_path, test_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_files = ['train.h5', 'val.h5', 'test.h5']

    # Create training dataset
    print('\nGenerating training dataset...')
    train_path = create_dataset(
        rp_manager=rp_manager,
        output_dir=output_dir,
        dataset_name=dataset_files[0],
        num_samples=train_samples,
        **kwargs,
    )

    # Create validation dataset
    print('\nGenerating validation dataset...')
    val_path = create_dataset(
        rp_manager=rp_manager,
        output_dir=output_dir,
        dataset_name=dataset_files[1],
        num_samples=val_samples,
        **kwargs,
    )

    # Create test dataset
    print('\nGenerating test dataset...')
    test_path = create_dataset(
        rp_manager=rp_manager,
        output_dir=output_dir,
        dataset_name=dataset_files[2],
        num_samples=test_samples,
        **kwargs,
    )

    return train_path, val_path, test_path


def inspect_dataset(dataset_path=None, batch_size=1):
    """Inspect a dataset by displaying samples and histograms.

    This function reads an HDF5 dataset and displays two plot windows:
    1. Raw signals from the dataset with appropriate labels
    2. Histograms of the entire dataset with overlaid histograms for the current sample

    When the plot windows are closed, it advances to the next sample.

    Args:
        dataset_path: Path to the HDF5 dataset file (defaults to train.h5 in signal_analysis/data)
        batch_size: Batch size for the DataLoader (default: 1)
    """
    # Default to train.h5 in signal_analysis/data if no path is provided
    if dataset_path is None:
        dataset_path = Path(__file__).parent / 'data' / 'train.h5'
    else:
        dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f'Dataset file not found: {dataset_path}')

    print(f'Inspecting dataset: {dataset_path}')

    # Create a custom dataset class for the HDF5 file
    class RawSignalDataset(Dataset):
        def __init__(self, file_path):
            self.file_path = file_path
            with h5py.File(file_path, 'r') as f:
                self.length = next(iter(f.values())).shape[0]
                self.keys = list(f.keys())

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            with h5py.File(self.file_path, 'r') as f:
                return {key: f[key][idx] for key in self.keys}

    # Load the entire dataset to compute histograms
    with h5py.File(dataset_path, 'r') as f:
        # Get all keys and their data
        all_data = {key: f[key][:] for key in f.keys()}
        num_samples = next(iter(all_data.values())).shape[0]
        print(
            f'Dataset contains {num_samples} samples with keys: {list(all_data.keys())}'
        )

    # Create dataset and dataloader
    dataset = RawSignalDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Function to display the plots for a sample
    def display_sample(sample_data):
        # Create two separate figures
        # Figure 1: Raw signals
        fig1, axes1 = plt.subplots(len(sample_data), 1, figsize=(6, 10), sharex=True)
        if len(sample_data) == 1:
            axes1 = [axes1]  # Make it iterable if there's only one subplot

        # Figure 2: Histograms
        fig2, axes2 = plt.subplots(len(sample_data), 1, figsize=(6, 10))
        if len(sample_data) == 1:
            axes2 = [axes2]  # Make it iterable if there's only one subplot

        # Channel labels and colors
        labels = {
            'RP1_CH1': 'Speaker Drive Voltage',
            'RP1_CH2': 'Photodiode 1',
            'RP2_CH1': 'Photodiode 2',
            'RP2_CH2': 'Photodiode 3',
        }
        colors = ['blue', 'red', 'green', 'purple']

        # Calculate time data (assuming 125MHz/256 sample rate as in RedPitayaManager)
        sample_rate = 125e6 / 256  # Default decimation in RedPitayaManager
        for i, (key, data) in enumerate(sample_data.items()):
            # Get the corresponding color
            color = colors[i % len(colors)]

            # Get label for the channel
            label = labels.get(key, key)

            # Plot raw signal
            time_data = np.linspace(0, len(data) / sample_rate, len(data))
            axes1[i].plot(time_data * 1000, data, color=color)  # Convert to ms
            axes1[i].set_ylabel(label)
            axes1[i].grid(True)

            if i == len(sample_data) - 1:
                axes1[i].set_xlabel('Time (ms)')

            # Plot histograms
            # Full dataset histogram on primary y-axis
            n, bins, patches = axes2[i].hist(
                all_data[key].flatten(),
                bins=50,
                color=color,
                alpha=1.0,
                label=f'All samples ({num_samples})',
            )
            axes2[i].set_ylabel('Frequency (all samples)', color=color)
            axes2[i].tick_params(axis='y', labelcolor=color)

            # Create a second y-axis for the current sample histogram
            ax2_twin = axes2[i].twinx()
            n_sample, _, patches_sample = ax2_twin.hist(
                data.flatten(),
                bins=bins,
                color='orange',
                alpha=0.4,
                label='Current sample',
            )
            ax2_twin.set_ylabel('Frequency (current sample)', color='orange')
            ax2_twin.tick_params(axis='y', labelcolor='orange')

            # Add a title and x-label
            axes2[i].set_title(f'{label} Histogram')
            axes2[i].set_xlabel('Value')

            # Create a combined legend
            lines1, labels1 = axes2[i].get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            axes2[i].legend(lines1 + lines2, labels1 + labels2, loc='upper right')

            # Add grid
            axes2[i].grid(True)

        # Set overall titles
        fig1.suptitle('Raw Signals', fontsize=16)
        fig2.suptitle('Signal Histograms', fontsize=16)

        # Adjust layout
        fig1.tight_layout(rect=[0, 0, 1, 0.95])
        fig2.tight_layout(rect=[0, 0, 1, 0.95])

        # Show the plots
        plt.show(block=True)

    # Create a function to handle plot close events and show the next sample
    def show_samples():
        # Iterate through the dataset
        for i, batch in enumerate(dataloader):
            print(f'\nDisplaying sample {i + 1}/{len(dataloader)}')

            # Convert batch dictionary to regular dictionary with numpy arrays
            sample = {k: v[0].numpy() for k, v in batch.items()}

            # Display the sample (this will block until both windows are closed)
            display_sample(sample)

            # If we've reached the end of the dataset, break
            if i == len(dataloader) - 1:
                break

    # Start the sample display process
    show_samples()

    print('Dataset inspection complete.')


if __name__ == '__main__':
    inspect_dataset()
