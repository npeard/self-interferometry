#!/usr/bin/env python

from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from redpitaya.manager import RedPitayaManager
from signal_analysis.interferometers import InterferometerArray, MichelsonInterferometer


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
        device_idx: Index of the device to use for acquisition
        delay_between_shots: Delay between shots in seconds
        timeout: Timeout for acquisition in seconds

    Returns:
        Path to the created dataset file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / dataset_name

    print(f'Acquiring {num_samples} samples for dataset: {dataset_name}')  # noqa: T201

    # Create file and initialize with metadata
    with h5py.File(file_path, 'w') as f:
        # Store acquisition parameters as attributes
        f.attrs['sample_rate'] = rp_manager.get_sample_rate(device_idx)
        f.attrs['decimation'] = rp_manager.get_decimation(device_idx)
        f.attrs['buffer_size'] = rp_manager.get_buffer_size(device_idx)
        f.attrs['creation_time'] = datetime.now().isoformat()
        f.attrs['num_samples'] = 0  # Will be updated as we add samples

        # Create empty datasets for each channel
        channel_keys = rp_manager.get_channel_keys()
        first_sample = rp_manager.run_one_shot(device_idx=device_idx, timeout=timeout)

        for key, array in first_sample.items():
            f.create_dataset(
                key,
                shape=(0, len(array)),
                maxshape=(None, len(array)),
                dtype='float32',
                chunks=(1, len(array)),
                compression='gzip',
                compression_opts=4,
            )

    # Acquire samples and save them incrementally
    for i in tqdm(range(num_samples)):
        # Acquire a new sample
        sample = rp_manager.run_one_shot(device_idx=device_idx, timeout=timeout)

        # Save the sample to the HDF5 file
        with h5py.File(file_path, 'r+') as f:
            current_size = f.attrs['num_samples']

            # Add each channel's data
            for key, array in sample.items():
                dataset = f[key]
                new_size = current_size + 1
                dataset.resize(new_size, axis=0)
                dataset[current_size] = array

            # Update sample count
            f.attrs['num_samples'] = current_size + 1

        # Delay between shots
        if i < num_samples - 1 and delay_between_shots > 0:
            rp_manager.delay(delay_between_shots)

    print(f'Dataset saved to {file_path}')  # noqa: T201
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
        train_samples: Number of training samples to acquire
        val_samples: Number of validation samples to acquire
        test_samples: Number of test samples to acquire
        **kwargs: Additional arguments to pass to create_dataset

    Returns:
        Tuple of (train_path, val_path, test_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_files = ['train.h5', 'val.h5', 'test.h5']

    # Create training dataset
    print('\nAcquiring training dataset...')  # noqa: T201
    train_path = create_dataset(
        rp_manager=rp_manager,
        output_dir=output_dir,
        dataset_name=dataset_files[0],
        num_samples=train_samples,
        **kwargs,
    )

    # Create validation dataset
    print('\nAcquiring validation dataset...')  # noqa: T201
    val_path = create_dataset(
        rp_manager=rp_manager,
        output_dir=output_dir,
        dataset_name=dataset_files[1],
        num_samples=val_samples,
        **kwargs,
    )

    # Create test dataset
    print('\nAcquiring test dataset...')  # noqa: T201
    test_path = create_dataset(
        rp_manager=rp_manager,
        output_dir=output_dir,
        dataset_name=dataset_files[2],
        num_samples=test_samples,
        **kwargs,
    )

    return train_path, val_path, test_path


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

        # RP1_CH2, RP2_CH1, RP2_CH2: Photodiode signals from interferometers
        # Check that they are in the expected order that matches our hardware
        # setup. RP1_CH2 is the first interferometer, RP2_CH1 is the second,
        # and RP2_CH2 is the third.
        assert len(signals) == 3, (
            f'Expected 3 interferometer signals, \
                                    got {len(signals)}'
        )
        assert [
            interferometer.wavelength
            for interferometer in interferometer_array.interferometers
        ] == [0.635, 0.675, 0.515], 'Expected wavelengths [0.635, 0.675, 0.515]'

        # Add interferometer signals
        for j in range(min(len(signals), 3)):
            channel_idx = j + 1  # Start from RP1_CH2
            data[channel_keys[channel_idx]] = signals[j]

        # Save the sample to the HDF5 file
        with h5py.File(file_path, 'r+') as f:
            current_size = f.attrs['num_samples']

            # Add each channel's data
            for key, array in data.items():
                dataset = f[key]
                new_size = current_size + 1
                dataset.resize(new_size, axis=0)
                dataset[current_size] = array

            # Update sample count
            f.attrs['num_samples'] = current_size + 1

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
        MichelsonInterferometer(wavelength=0.675, phase=np.pi / 2),  # 675nm (Deep Red)
        MichelsonInterferometer(wavelength=0.515, phase=np.pi / 3),  # 515nm (Green)
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
    # We do want normalize_gain=True, because we are using CoilDriver's transfer
    # functions to calculate voltage, displacement, and velocity in
    # InterferometerArray.sample_simulated(), so we should get a flat velocity spectrum.

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
    # We do want normalize_gain=True, because we are using CoilDriver's transfer
    # functions to calculate voltage, displacement, and velocity in
    # InterferometerArray.sample_simulated(), so we should get a flat velocity spectrum.

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
    # We do want normalize_gain=True, because we are using CoilDriver's transfer
    # functions to calculate voltage, displacement, and velocity in
    # InterferometerArray.sample_simulated(), so we should get a flat velocity spectrum.

    return train_path, val_path, test_path
