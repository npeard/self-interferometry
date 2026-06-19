#!/usr/bin/env python
"""Lightning DataModule for velocity/displacement prediction.

Encapsulates the train/val/test split and DataLoader construction that used to
live as boilerplate in ``datasets.get_data_loaders`` and ``TrainingInterface``.
The split is deterministic (fixed seed) for reproducible evaluation.
"""

import logging

import lightning as lightning_module
import torch
from torch.utils.data import DataLoader, random_split

from smi.analysis.datasets import VelocityDataset

logger = logging.getLogger(__name__)


class VelocityDataModule(lightning_module.LightningDataModule):
    """DataModule wrapping a single :class:`VelocityDataset` HDF5 file.

    Args:
        dataset_path: Path to the HDF5 dataset file.
        split_ratios: (train, val, test) percentages summing to 100.
        batch_size: Batch size for all dataloaders.
        num_workers: Worker processes for data loading.
        seed: Seed for the deterministic split.
        **dataset_kwargs: Forwarded to :class:`VelocityDataset` (e.g.
            ``num_pd_channels``, ``cache_size``).
    """

    def __init__(
        self,
        dataset_path: str,
        split_ratios: tuple[int, int, int] = (80, 10, 10),
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
        **dataset_kwargs: object,
    ) -> None:
        super().__init__()
        if sum(split_ratios) != 100:
            raise ValueError(f'Split ratios must sum to 100, got {sum(split_ratios)}')
        self.dataset_path = dataset_path
        self.split_ratios = split_ratios
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.dataset_kwargs = dataset_kwargs
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None) -> None:
        """Create the dataset and the deterministic train/val/test split."""
        if self.train_dataset is not None:
            return

        full_dataset = VelocityDataset(self.dataset_path, **self.dataset_kwargs)
        total = len(full_dataset)
        train_size = int(total * self.split_ratios[0] / 100)
        val_size = int(total * self.split_ratios[1] / 100)
        test_size = total - train_size - val_size

        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size], generator=generator
        )
        logger.info(
            'Dataset split - Total: %d, Train: %d, Val: %d, Test: %d',
            total,
            train_size,
            val_size,
            test_size,
        )

    def _loader(self, dataset: object, *, shuffle: bool) -> DataLoader:
        """Build a DataLoader with the shared settings."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        """Shuffled training dataloader."""
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader (unshuffled)."""
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Test dataloader (unshuffled)."""
        return self._loader(self.test_dataset, shuffle=False)
