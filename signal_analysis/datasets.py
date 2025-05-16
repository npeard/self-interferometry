#!/usr/bin/env python

from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from fluo.speckle1d import Fluorescence1D
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


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


def create_data_loaders(
    train_path: str,
    val_path: str,
    test_path: str,
    batch_size: int,
    num_workers: int = 4,
    **dataset_kwargs: dict[str, Any],
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
    train_dataset = H5Dataset(train_path, **dataset_kwargs)
    val_dataset = H5Dataset(val_path, **dataset_kwargs)
    test_dataset = H5Dataset(test_path, **dataset_kwargs)

    # Only using pretraining dataset for now
    # train_dataset = PreTrainingDataset(train_path, **dataset_kwargs)
    # val_dataset = PreTrainingDataset(val_path, **dataset_kwargs)
    # test_dataset = PreTrainingDataset(test_path, **dataset_kwargs)

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


def generate_pretraining_data(
    file_path: str, num_pix: int, num_samples: int, chunk_size: int | None = None
) -> None:
    """Generate pretraining data and save to HDF5 file.

    Args:
        file_path: Path to save the HDF5 file
        num_pix: Number of pixels in the detector
        num_samples: Number of samples to generate
        chunk_size: Optional chunk size for HDF5 dataset compression
    """
    # Create parent directory if it doesn't exist
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    # Default chunk size to balance between compression and access speed
    if chunk_size is None:
        chunk_size = min(100, num_samples)

    with h5py.File(file_path, 'w') as f:
        # Store generation parameters as attributes
        f.attrs['num_samples'] = num_samples
        f.attrs['num_pix'] = num_pix

        # Create datasets with chunking and compression
        f.create_dataset(
            'absPhi',
            shape=(num_samples, Phi_dim, Phi_dim),
            dtype='float32',
            chunks=(chunk_size, Phi_dim, Phi_dim),
            compression='gzip',
            compression_opts=4,
        )

        # Only storing one quadrant of phase intentionally, redundancy by antisymmetry
        f.create_dataset(
            'phase',
            shape=(num_samples, num_pix),
            dtype='float32',
            chunks=(chunk_size, num_pix),
            compression='gzip',
            compression_opts=4,
        )

        # Generate data
        # TODO: make a test case that compares the output dims of this block
        # to the output dims of plot_cosPhi (the from_data method therein)
        print(f'\nGenerating {num_samples} samples...')
        for i in tqdm(range(num_samples)):
            # Generate random phase, out to 2*num_pix - 1 where num_pix is the
            # number of pixels in the detector. We expect the triple correlation to
            # contain phase information out to 2*kmax or num_pix from origin.
            phase = np.random.uniform(-np.pi, np.pi, num_pix)
            # Set origin, always zero, needed in computing Phi
            phase[0] = 0

            # Compute Phi matrix
            Phi = Fluorescence1D.compute_Phi_from_phase(phase)

            # Store in dataset
            f['absPhi'][i] = np.abs(Phi[1:, 1:])
            # Only storing one quadrant of phase intentionally, redundancy by antisymmetry
            f['phase'][i] = phase


def create_train_val_test_datasets(
    output_dir: str,
    num_pix: int = 21,
    train_samples: int = int(1e6),
    val_samples: int = int(1e4),
    test_samples: int = int(1e4),
    **kwargs,
) -> None:
    """Create train, validation and test datasets for pretraining.

    Args:
        output_dir: Directory to save the datasets
        num_pix: Number of pixels in each sample
        train_samples: Number of training samples
        val_samples: Number of validation samples
        test_samples: Number of test samples
        **kwargs: Additional arguments passed to generate_pretraining_data
    """
    output_dir = Path(output_dir)
    dataset_files = ['train.h5', 'val.h5', 'test.h5']

    # Delete existing dataset files if they exist
    for file in dataset_files:
        file_path = output_dir / file
        if file_path.exists():
            print(f'Removing existing dataset file: {file}')
            file_path.unlink()

    print('Generating training dataset...')
    generate_pretraining_data(
        file_path=str(output_dir / 'train.h5'),
        num_pix=num_pix,
        num_samples=train_samples,
        **kwargs,
    )

    print('\nGenerating validation dataset...')
    generate_pretraining_data(
        file_path=str(output_dir / 'val.h5'),
        num_pix=num_pix,
        num_samples=val_samples,
        **kwargs,
    )

    print('\nGenerating test dataset...')
    generate_pretraining_data(
        file_path=str(output_dir / 'test.h5'),
        num_pix=num_pix,
        num_samples=test_samples,
        **kwargs,
    )


def visualize_pretraining_dataset(
    file_path: str,
    num_samples: int = 4,
    random_seed: int | None = None,
    save_path: str | None = None,
) -> None:
    """Visualize random samples from a pretraining dataset.

    Args:
        file_path: Path to HDF5 file
        num_samples: Number of samples to visualize
        random_seed: Optional random seed for reproducibility
        save_path: Optional path to save the visualization
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    with h5py.File(file_path, 'r') as f:
        # Get dataset info
        total_samples = f['Phi'].shape[0]
        indices = np.random.choice(total_samples, num_samples, replace=False)
        indices.sort()  # Sort indices for HDF5 compatibility

        # Load selected samples
        Phi_samples = f['Phi'][indices]
        phase_samples = f['phase'][indices]

        # Get metadata
        metadata = dict(f.attrs)

        # Create figure with 2 rows (Phi and phase) and num_samples columns
        fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))

        # Plot samples
        for i in range(num_samples):
            # Plot Phi matrix
            im_phi = axes[0, i].imshow(Phi_samples[i], cmap='viridis')
            axes[0, i].set_title(f'Phi Matrix {i + 1}')
            axes[0, i].axis('off')
            plt.colorbar(im_phi, ax=axes[0, i])

            # Plot phase
            im_phase = axes[1, i].plot(phase_samples[i])
            axes[1, i].set_title(f'Phase {i + 1}')
            axes[1, i].set_ylim(-np.pi, np.pi)
            axes[1, i].grid(True)

        # Add overall title
        plt.suptitle(
            f'Samples from {Path(file_path).name}\n'
            + f'Total Samples: {total_samples}, Pixels: {metadata["num_pix"]}'
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()


def inspect_pretraining_dataset(file_path: str) -> dict:
    """Calculate and return statistics about the pretraining dataset.

    Args:
        file_path: Path to HDF5 file

    Returns:
        Dictionary containing dataset statistics
    """
    with h5py.File(file_path, 'r') as f:
        Phi = f['Phi'][:]
        phase = f['phase'][:]
        metadata = dict(f.attrs)

        stats = {
            'num_samples': Phi.shape[0],
            'num_pix': metadata['num_pix'],
            'Phi_stats': {
                'min': float(Phi.min()),
                'max': float(Phi.max()),
                'mean': float(Phi.mean()),
                'std': float(Phi.std()),
            },
            'phase_stats': {
                'min': float(phase.min()),
                'max': float(phase.max()),
                'mean': float(phase.mean()),
                'std': float(phase.std()),
            },
        }
    return stats
