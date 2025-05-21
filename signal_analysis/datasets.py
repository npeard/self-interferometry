#!/usr/bin/env python

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from redpitaya.manager import RedPitayaManager
from signal_analysis.interferometers import InterferometerArray, MichelsonInterferometer


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

    def __len__(self) -> int:
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


class MultiKeyH5Dataset(Dataset):
    """Dataset class for HDF5 files that returns all keys as a dictionary.

    This class is useful for inspection and visualization of datasets where
    you want to access all available keys in the HDF5 file rather than just
    input/target pairs.

    Args:
        file_path: Path to HDF5 file
    """

    def __init__(self, file_path: str | Path):
        self.file_path = file_path
        with h5py.File(file_path, 'r') as f:
            self.keys = list(f.keys())
            self.length = next(iter(f.values())).shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """Get a single item from the dataset as a dictionary of arrays.

        Args:
            idx: Index of the item to get

        Returns:
            Dictionary mapping each key in the HDF5 file to its corresponding
            array at the given index
        """
        with h5py.File(self.file_path, 'r') as f:
            return {key: f[key][idx] for key in self.keys}


def create_dataset(
    rp_manager: RedPitayaManager,
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

    print(f'Acquiring {num_samples} samples for dataset: {dataset_name}')  # noqa: T201

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


def generate_training_data_from_rp(
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
    print('\nGenerating training dataset...')  # noqa: T201
    train_path = create_dataset(
        rp_manager=rp_manager,
        output_dir=output_dir,
        dataset_name=dataset_files[0],
        num_samples=train_samples,
        **kwargs,
    )

    # Create validation dataset
    print('\nGenerating validation dataset...')  # noqa: T201
    val_path = create_dataset(
        rp_manager=rp_manager,
        output_dir=output_dir,
        dataset_name=dataset_files[1],
        num_samples=val_samples,
        **kwargs,
    )

    # Create test dataset
    print('\nGenerating test dataset...')  # noqa: T201
    test_path = create_dataset(
        rp_manager=rp_manager,
        output_dir=output_dir,
        dataset_name=dataset_files[2],
        num_samples=test_samples,
        **kwargs,
    )

    return train_path, val_path, test_path


def inspect_dataset(dataset_path: str | Path, batch_size: int = 1) -> None:
    """Inspect a dataset by displaying samples and histograms.

    This function reads an HDF5 dataset and displays two plot windows:
    1. Raw signals from the dataset with appropriate labels
    2. Histograms of the entire dataset with overlaid histograms for the current sample

    When the plot windows are closed, it advances to the next sample.

    Args:
        dataset_path: Path to the HDF5 dataset file
        batch_size: Batch size for the DataLoader (default: 1)
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f'Dataset file not found: {dataset_path}')

    print(f'Inspecting dataset: {dataset_path}')  # noqa: T201

    # Load the entire dataset to compute histograms
    with h5py.File(dataset_path, 'r') as f:
        # Get all keys and their data
        all_data = {key: f[key][:] for key in f}
        num_samples = next(iter(all_data.values())).shape[0]
        print(  # noqa: T201
            f'Dataset contains {num_samples} samples with keys: {list(all_data.keys())}'
        )

    # Create dataset and dataloader using the MultiKeyH5Dataset class
    dataset = MultiKeyH5Dataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Function to display the plots for a sample
    def display_sample(sample_data: dict[str, np.ndarray]):
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
            print(f'\nDisplaying sample {i + 1}/{len(dataloader)}')  # noqa: T201

            # Convert batch dictionary to regular dictionary with numpy arrays
            sample = {k: v[0].numpy() for k, v in batch.items()}

            # Display the sample (this will block until both windows are closed)
            display_sample(sample)

            # If we've reached the end of the dataset, break
            if i == len(dataloader) - 1:
                break

    # Start the sample display process
    show_samples()

    print('Dataset inspection complete.')  # noqa: T201


def create_pretraining_dataset(
    interferometer_array: InterferometerArray,
    output_dir: str | Path,
    dataset_name: str,
    num_samples: int,
    start_freq: float = 1,
    end_freq: float = 1000,
    randomize_phase_only: bool = False,
    random_single_tone: bool = False,
    normalize_gain: bool = True,
) -> str:
    """Create a pretraining dataset using simulated interferometer data.

    This function generates simulated interferometer signals and saves them to an
    HDF5 file with the same structure as real acquisition data for seamless transfer
    learning.

    Args:
        interferometer_array: InterferometerArray instance to use for simulation
        output_dir: Directory to save the dataset
        dataset_name: Name of the dataset file (e.g., 'pretrain.h5')
        num_samples: Number of samples to generate
        start_freq: The lower bound of the valid frequency range (Hz)
        end_freq: The upper bound of the valid frequency range (Hz)
        randomize_phase_only: If True, only randomize phases while keeping spectrum
            amplitudes constant
        random_single_tone: If True, generate a single tone at a randomly selected
            frequency
        normalize_gain: If True, pre-compensate the spectrum to normalize gain
            across frequencies

    Returns:
        Path to the created dataset file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / dataset_name

    print(f'Generating {num_samples} simulated samples for dataset: {dataset_name}')  # noqa: T201

    # Create file and initialize with metadata
    with h5py.File(file_path, 'w') as f:
        # Store acquisition parameters as attributes
        sample_rate = 125e6 / 256  # Default decimation in RedPitayaManager
        f.attrs['sample_rate'] = sample_rate
        f.attrs['decimation'] = 256  # Default decimation
        f.attrs['buffer_size'] = 16384  # Default buffer size
        f.attrs['creation_time'] = datetime.now().isoformat()
        f.attrs['simulated'] = True
        f.attrs['num_samples'] = 0  # Will be updated as we add samples

        # Store wavelengths of interferometers
        wavelengths = [
            interf.wavelength for interf in interferometer_array.interferometers
        ]
        f.attrs['wavelengths'] = wavelengths

        # Create empty datasets for each channel
        # We'll use the same keys as in real acquisition for compatibility
        channel_keys = ['RP1_CH1', 'RP1_CH2', 'RP2_CH1', 'RP2_CH2']

        # First sample to determine shapes
        time, signals, acq_voltage, displacement, velocity = (
            interferometer_array.sample_simulated(
                start_freq=start_freq,
                end_freq=end_freq,
                randomize_phase_only=randomize_phase_only,
                random_single_tone=random_single_tone,
                normalize_gain=normalize_gain,
            )
        )

        # Create datasets
        # RP1_CH1: Speaker drive voltage (from which velocity is computed)
        # RP1_CH2, RP2_CH1, RP2_CH2: Photodiode signals from interferometers
        f.create_dataset(
            channel_keys[0],  # RP1_CH1 - Speaker drive voltage
            shape=(0, len(acq_voltage)),
            maxshape=(None, len(acq_voltage)),
            dtype='float32',
            chunks=(1, len(acq_voltage)),
            compression='gzip',
            compression_opts=4,
        )

        # Create datasets for interferometer signals
        for i in range(min(len(signals), 3)):
            channel_idx = i + 1  # Start from RP1_CH2
            f.create_dataset(
                channel_keys[channel_idx],
                shape=(0, len(signals[i])),
                maxshape=(None, len(signals[i])),
                dtype='float32',
                chunks=(1, len(signals[i])),
                compression='gzip',
                compression_opts=4,
            )

    # Generate and save samples
    for i in tqdm(range(num_samples)):
        # Generate a new sample
        time, signals, acq_voltage, displacement, velocity = (
            interferometer_array.sample_simulated(
                start_freq=start_freq,
                end_freq=end_freq,
                randomize_phase_only=randomize_phase_only,
                random_single_tone=random_single_tone,
                normalize_gain=normalize_gain,
            )
        )

        # Prepare data dictionary
        data = {}

        # RP1_CH1: Speaker drive voltage (from which velocity is computed)
        data[channel_keys[0]] = acq_voltage

        # Add interferometer signals
        for j in range(min(len(signals), 3)):
            channel_idx = j + 1  # Start from RP1_CH2
            data[channel_keys[channel_idx]] = signals[j]

        # Save to HDF5 file
        with h5py.File(file_path, 'a') as f:
            current_size = f.attrs['num_samples']

            # Resize datasets and add new data
            for key, array in data.items():
                if key in f:
                    dataset = f[key]
                    new_size = current_size + 1
                    dataset.resize(new_size, axis=0)
                    dataset[current_size] = array

            # Update sample count
            f.attrs['num_samples'] = current_size + 1

        if (i + 1) % 10 == 0 or i == 0 or i == num_samples - 1:
            print(f'Generated {i + 1}/{num_samples} samples')  # noqa: T201

    print(f'Pretraining dataset saved to {file_path}')  # noqa: T201
    return str(file_path)


def generate_pretraining_data(
    output_dir: str | Path,
    train_samples: int = 1000,
    val_samples: int = 200,
    test_samples: int = 100,
    start_freq: float = 1,
    end_freq: float = 1000,
) -> tuple[str, str, str]:
    """Generate pretraining datasets using simulated interferometer data.

    This function creates an interferometer array with three interferometers at
    different wavelengths and generates simulated data for pretraining.

    Args:
        output_dir: Directory to save the datasets
        train_samples: Number of training samples to generate
        val_samples: Number of validation samples to generate
        test_samples: Number of test samples to generate
        start_freq: The lower bound of the valid frequency range (Hz)
        end_freq: The upper bound of the valid frequency range (Hz)

    Returns:
        Tuple of (train_path, val_path, test_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create an array of 3 interferometers with different wavelengths
    interferometers = [
        MichelsonInterferometer(wavelength=0.635, phase=0),  # 635nm (Red)
        MichelsonInterferometer(wavelength=0.515, phase=np.pi / 3),  # 515nm (Green)
        MichelsonInterferometer(wavelength=0.675, phase=np.pi / 2),  # 675nm (Deep Red)
    ]

    interferometer_array = InterferometerArray(interferometers)

    dataset_files = ['pretrain.h5', 'preval.h5', 'pretest.h5']

    # Create training dataset
    print('\nGenerating pretraining training dataset...')  # noqa: T201
    train_path = create_pretraining_dataset(
        interferometer_array=interferometer_array,
        output_dir=output_dir,
        dataset_name=dataset_files[0],
        num_samples=train_samples,
        start_freq=start_freq,
        end_freq=end_freq,
        randomize_phase_only=False,
        random_single_tone=False,
        normalize_gain=True,
    )

    # Create validation dataset
    print('\nGenerating pretraining validation dataset...')  # noqa: T201
    val_path = create_pretraining_dataset(
        interferometer_array=interferometer_array,
        output_dir=output_dir,
        dataset_name=dataset_files[1],
        num_samples=val_samples,
        start_freq=start_freq,
        end_freq=end_freq,
        randomize_phase_only=False,
        random_single_tone=False,
        normalize_gain=True,
    )

    # Create test dataset
    print('\nGenerating pretraining test dataset...')  # noqa: T201
    test_path = create_pretraining_dataset(
        interferometer_array=interferometer_array,
        output_dir=output_dir,
        dataset_name=dataset_files[2],
        num_samples=test_samples,
        start_freq=start_freq,
        end_freq=end_freq,
        randomize_phase_only=False,
        random_single_tone=False,
        normalize_gain=True,
    )

    return train_path, val_path, test_path


if __name__ == '__main__':
    inspect_dataset('./signal_analysis/data/pretrain.h5')
