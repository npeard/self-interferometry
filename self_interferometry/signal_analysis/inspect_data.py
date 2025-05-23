#!/usr/bin/env python

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from self_interferometry.signal_analysis.datasets import StandardVelocityDataset


def visualize_dataset(dataset_path: str | Path, max_samples: int = 10, batch_size: int = 1):
    """Visualize samples from a dataset using StandardVelocityDataset.

    Creates plots similar to the training.py visualization but without model predictions.
    Each plot includes:
    1. Velocity and displacement on dual y-axes
    2-4. Input signals from photodiode channels

    Args:
        dataset_path: Path to the HDF5 dataset file
        max_samples: Maximum number of samples to visualize
        batch_size: Batch size for the DataLoader
    """
    # Get sample rate from HDF5 file attributes if available
    sample_rate = None
    with h5py.File(dataset_path, 'r') as f:
        if 'sample_rate' in f.attrs:
            sample_rate = float(f.attrs['sample_rate'])
            print(f"Using sample rate from HDF5 file: {sample_rate:.2f} Hz")  # noqa: T201
        else:
            # Default sample rate for Red Pitaya with decimation of 256
            sample_rate = 125e6 / 256
            print(f"Warning: Using default sample rate of {sample_rate:.2f} Hz")  # noqa: T201
            
        # Print dataset information
        print(f"Dataset: {Path(dataset_path).name}")  # noqa: T201
        print(f"Number of samples: {len(f[list(f.keys())[0]])}")  # noqa: T201
        print(f"Available channels: {list(f.keys())}")  # noqa: T201

    # Create dataset and dataloader
    dataset = StandardVelocityDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Get samples from dataloader
    samples_processed = 0
    for batch_idx, (signals, velocity, displacement) in enumerate(dataloader):
        if samples_processed >= max_samples:
            break
            
        # Convert tensors to numpy arrays
        signals = signals.numpy()
        velocity = velocity.numpy()
        displacement = displacement.numpy()
        
        # Get the number of samples in the batch and number of PD channels
        batch_size, num_channels, signal_length = signals.shape
        
        # Print shapes for debugging
        print(f"\nBatch {batch_idx + 1}:")  # noqa: T201
        print(f"Signals shape: {signals.shape}")  # noqa: T201
        print(f"Velocity shape: {velocity.shape}")  # noqa: T201
        print(f"Displacement shape: {displacement.shape}")  # noqa: T201

        # Plot each sample in the batch
        for i in range(min(batch_size, max_samples - samples_processed)):
            samples_processed += 1
            
            # Create a figure with subplots - one for velocities and up to 3 for signals
            fig, axs = plt.subplots(1 + num_channels, 1, figsize=(10, 8), sharex=True)
            
            # If there's only one subplot, make it an array for consistent indexing
            if num_channels == 0:
                axs = [axs]
                
            # Create time array
            time = np.arange(signal_length) / sample_rate

            # Plot velocity on the primary y-axis
            axs[0].plot(time, velocity[i], label='Velocity', color='blue')
            axs[0].set_title('Velocity and Displacement')
            axs[0].set_ylabel('Velocity (μm/s)', color='blue')
            axs[0].tick_params(axis='y', labelcolor='blue')
            axs[0].grid(True, alpha=0.3)
            
            # Create a twin axis for displacement
            ax_twin = axs[0].twinx()
            ax_twin.plot(time, displacement[i], label='Displacement', color='green')
            ax_twin.set_ylabel('Displacement (μm)', color='green')
            ax_twin.tick_params(axis='y', labelcolor='green')
            
            # Ensure both legends are visible
            lines1, labels1 = axs[0].get_legend_handles_labels()
            lines2, labels2 = ax_twin.get_legend_handles_labels()
            ax_twin.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            # Plot each input signal channel
            for j in range(num_channels):
                channel_names = ['PD1 (RP1_CH2)', 'PD2 (RP2_CH1)', 'PD3 (RP2_CH2)']
                axs[j + 1].plot(time, signals[i, j, :], label=f'Channel {j + 1}')
                axs[j + 1].set_title(f'Input Signal - {channel_names[j]}')
                axs[j + 1].set_ylabel('Amplitude')
                axs[j + 1].grid(True)

            # Set x-axis label for the bottom subplot
            axs[-1].set_xlabel('Time (s)')

            # Add overall title
            plt.suptitle(f'Sample {samples_processed} from Dataset')
            plt.tight_layout()
            plt.show()


def analyze_dataset(dataset_path: str | Path):
    """Perform statistical analysis on a dataset.
    
    Calculates and displays statistics for each channel in the dataset.
    
    Args:
        dataset_path: Path to the HDF5 dataset file
    """
    with h5py.File(dataset_path, 'r') as f:
        print(f"\nDataset Analysis: {Path(dataset_path).name}")  # noqa: T201
        print(f"Number of samples: {len(f[list(f.keys())[0]])}")  # noqa: T201
        
        # Calculate statistics for each channel
        for key in f.keys():
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
                
            # Get data and calculate statistics
            data = f[key][:]
            mean = np.mean(data)
            std = np.std(data)
            min_val = np.min(data)
            max_val = np.max(data)
            range_val = max_val - min_val
            
            # Print statistics
            print(f"\n{label} statistics:")  # noqa: T201
            print(f"  - Mean: {mean:.4f}")  # noqa: T201
            print(f"  - Std: {std:.4f}")  # noqa: T201
            print(f"  - Min: {min_val:.4f}")  # noqa: T201
            print(f"  - Max: {max_val:.4f}")  # noqa: T201
            print(f"  - Range: {range_val:.4f}")  # noqa: T201


def frequency_analysis(dataset_path: str | Path, max_freq: float = 5000):
    """Perform frequency domain analysis on a dataset.
    
    Creates FFT plots for each channel in the dataset.
    
    Args:
        dataset_path: Path to the HDF5 dataset file
        max_freq: Maximum frequency to display in Hz
    """
    # Get sample rate from HDF5 file attributes if available
    with h5py.File(dataset_path, 'r') as f:
        if 'sample_rate' in f.attrs:
            sample_rate = float(f.attrs['sample_rate'])
        else:
            # Default sample rate for Red Pitaya with decimation of 256
            sample_rate = 125e6 / 256
            
        # Create figure for FFT plots
        plt.figure(figsize=(12, 8))
        plt.title(f'Frequency Domain Analysis - {Path(dataset_path).name}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        
        # Calculate and plot FFT for each channel (using first sample)
        for key in f.keys():
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
                
            # Get data from first sample
            data = f[key][0]
            
            # Compute FFT
            fft_data = np.fft.rfft(data)
            fft_freq = np.fft.rfftfreq(len(data), d=1/sample_rate)
            
            # Plot only up to max_freq Hz for better visibility
            max_freq_idx = np.searchsorted(fft_freq, max_freq)
            plt.plot(fft_freq[:max_freq_idx], np.abs(fft_data[:max_freq_idx]), label=label)
        
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # Example usage
    dataset_path = './signal_analysis/data/train.h5'
    
    # Visualize samples from the dataset
    visualize_dataset(dataset_path, max_samples=5)
    
    # Analyze dataset statistics
    analyze_dataset(dataset_path)
    
    # Perform frequency analysis
    frequency_analysis(dataset_path)
