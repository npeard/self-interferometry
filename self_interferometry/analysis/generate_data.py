#!/usr/bin/env python

import logging
from pathlib import Path

from self_interferometry.acquisition.redpitaya.manager import RedPitayaManager

logger = logging.getLogger(__name__)


def generate_training_data_from_rp(
    rp_manager: RedPitayaManager,
    output_dir: str | Path,
    train_samples: int,
    val_samples: int,
    test_samples: int,
) -> tuple[str, str, str]:
    """Create train, validation, and test datasets from Red Pitaya data.
    RedPitayaManager handles saving to HDF5 internally shot by shot, to protect against
    connection loss. See RedPitayaManager.save_data for more details.

    Args:
        rp_manager: RedPitayaManager instance to use for data acquisition
        output_dir: Directory to save the datasets
        train_samples: Number of training samples to acquire
        val_samples: Number of validation samples to acquire
        test_samples: Number of test samples to acquire

    Returns:
        Tuple of (train_path, val_path, test_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_files = ['train.h5', 'val.h5', 'test.h5']
    train_path = output_dir / dataset_files[0]
    val_path = output_dir / dataset_files[1]
    test_path = output_dir / dataset_files[2]

    # Create training dataset
    logger.info('\nAcquiring training dataset...')
    _ = rp_manager.run_multiple_shots(num_shots=train_samples, hdf5_file=train_path)

    # Create validation dataset
    logger.info('\nAcquiring validation dataset...')
    _ = rp_manager.run_multiple_shots(num_shots=val_samples, hdf5_file=val_path)

    # Create test dataset
    logger.info('\nAcquiring test dataset...')
    _ = rp_manager.run_multiple_shots(num_shots=test_samples, hdf5_file=test_path)

    return train_path, val_path, test_path