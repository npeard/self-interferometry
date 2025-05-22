#!/usr/bin/env python

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset


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


def display_sample(
    sample_data: dict[str, np.ndarray],
    all_data: dict[str, np.ndarray],
    num_samples: int,
):
    """Display plots for a sample from the dataset.

    Args:
        sample_data: Dictionary of data for the current sample
        all_data: Dictionary of all data in the dataset
        num_samples: Number of samples in the dataset
    """
    # Create a figure with 2 rows and 2 columns
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Dataset Visualization', fontsize=16)

    # Plot 1: Raw signals
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_title('Raw Signals')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Amplitude')

    # Plot the signals
    for key, data in sample_data.items():
        if key == 'RP1_CH1':
            label = 'Speaker Drive Voltage'
        elif key == 'RP1_CH2':
            label = 'PD1 (635nm Red)'
        elif key == 'RP2_CH1':
            label = 'PD2 (675nm Deep Red)'
        elif key == 'RP2_CH2':
            label = 'PD3 (515nm Green)'
        else:
            label = key

        ax1.plot(data, label=label)

    ax1.legend()
    ax1.grid(True)

    # Plot 2: Velocity and Displacement
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_title('Velocity and Displacement')
    ax2.set_xlabel('Sample')

    # Calculate velocity from speaker drive voltage
    # For simplicity, we'll just use a scaled version of the voltage
    # In a real application, you would use the proper transfer function
    voltage = sample_data.get('RP1_CH1', np.zeros(1))

    # Create a sample rate based on typical Red Pitaya settings
    sample_rate = 125e6 / 256  # Default decimation

    # Create a time array
    time = np.arange(len(voltage)) / sample_rate

    # Simple scaling for visualization
    velocity = voltage * 1000  # Scale for visibility

    # Calculate displacement by integrating velocity
    displacement = np.cumsum(velocity) / sample_rate
    displacement = displacement - np.mean(displacement)  # Remove DC offset

    # Plot velocity and displacement
    ax2.plot(time, velocity, label='Velocity (μm/s)', color='red')

    # Create a second y-axis for displacement
    ax3 = ax2.twinx()
    ax3.plot(
        time, displacement, label='Displacement (μm)', color='blue', linestyle='--'
    )

    # Add legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Set labels
    ax2.set_ylabel('Velocity (μm/s)', color='red')
    ax3.set_ylabel('Displacement (μm)', color='blue')

    ax2.grid(True)

    # Plot 3: Histogram of signal values
    ax4 = plt.subplot(2, 2, 3)
    ax4.set_title('Histogram of Signal Values')
    ax4.set_xlabel('Value')
    ax4.set_ylabel('Frequency')

    # Plot histograms for each signal
    for key, data in sample_data.items():
        if key == 'RP1_CH1':
            label = 'Speaker Drive Voltage'
        elif key == 'RP1_CH2':
            label = 'PD1 (635nm Red)'
        elif key == 'RP2_CH1':
            label = 'PD2 (675nm Deep Red)'
        elif key == 'RP2_CH2':
            label = 'PD3 (515nm Green)'
        else:
            label = key

        ax4.hist(data, bins=50, alpha=0.5, label=label)

    ax4.legend()
    ax4.grid(True)

    # Plot 4: Frequency domain analysis
    ax5 = plt.subplot(2, 2, 4)
    ax5.set_title('Frequency Domain Analysis')
    ax5.set_xlabel('Frequency (Hz)')
    ax5.set_ylabel('Magnitude')

    # Calculate and plot FFT for each signal
    for key, data in sample_data.items():
        if key == 'RP1_CH1':
            label = 'Speaker Drive Voltage'
        elif key == 'RP1_CH2':
            label = 'PD1 (635nm Red)'
        elif key == 'RP2_CH1':
            label = 'PD2 (675nm Deep Red)'
        elif key == 'RP2_CH2':
            label = 'PD3 (515nm Green)'
        else:
            label = key

        # Compute FFT
        fft_data = np.fft.rfft(data)
        fft_freq = np.fft.rfftfreq(len(data), d=1 / sample_rate)

        # Plot only up to 5000 Hz for better visibility
        max_freq_idx = np.searchsorted(fft_freq, 5000)
        ax5.plot(fft_freq[:max_freq_idx], np.abs(fft_data[:max_freq_idx]), label=label)

    ax5.legend()
    ax5.grid(True)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
    plt.show()

    # Display sample information
    print('Sample information:')  # noqa: T201
    print(f'- Total samples in dataset: {num_samples}')  # noqa: T201

    # Display statistics for each channel
    for key, data in sample_data.items():
        if key == 'RP1_CH1':
            label = 'Speaker Drive Voltage'
        elif key == 'RP1_CH2':
            label = 'PD1 (635nm Red)'
        elif key == 'RP2_CH1':
            label = 'PD2 (675nm Deep Red)'
        elif key == 'RP2_CH2':
            label = 'PD3 (515nm Green)'
        else:
            label = key

        print(f'- {label} statistics:')  # noqa: T201
        print(f'  - Mean: {np.mean(data):.4f}')  # noqa: T201
        print(f'  - Std: {np.std(data):.4f}')  # noqa: T201
        print(f'  - Min: {np.min(data):.4f}')  # noqa: T201
        print(f'  - Max: {np.max(data):.4f}')  # noqa: T201
        print(f'  - Range: {np.max(data) - np.min(data):.4f}')  # noqa: T201


def show_samples(dataloader: DataLoader, all_data: dict, num_samples: int):
    """Iterate through the dataset and display each sample.

    Args:
        dataloader: DataLoader for the dataset
        all_data: Dictionary of all data in the dataset
        num_samples: Number of samples in the dataset
    """
    for i, sample_data in enumerate(dataloader):
        print(f'\nSample {i + 1}/{len(dataloader)}')  # noqa: T201
        display_sample(sample_data, all_data, num_samples)

        # Ask if user wants to continue
        if i < len(dataloader) - 1:
            user_input = input("Press Enter to view next sample, or 'q' to quit: ")
            if user_input.lower() == 'q':
                break


def inspect_dataset(dataset_path: str | Path, batch_size: int = 1):
    """Inspect a dataset by displaying samples and histograms.

    This function reads an HDF5 dataset and displays two plot windows:
    1. Raw signals from the dataset with appropriate labels
    2. Analysis plots including:
       - Velocity and displacement
       - Histogram of signal values
       - Frequency domain analysis

    The function also prints statistics for each channel.

    Args:
        dataset_path: Path to the HDF5 dataset file
        batch_size: Batch size for the DataLoader (default: 1)
    """
    # Load the dataset
    dataset = MultiKeyH5Dataset(dataset_path)

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Get all data for statistics
    with h5py.File(dataset_path, 'r') as f:
        all_data = {key: f[key][:] for key in f.keys()}
        num_samples = next(iter(f.values())).shape[0]

    # Show samples
    show_samples(dataloader, all_data, num_samples)


if __name__ == '__main__':
    # Example usage
    inspect_dataset('./self_interferometry/signal_analysis/data/pretrain.h5')
