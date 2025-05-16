#!/usr/bin/env python

from typing import Any

import lightning as L
import torch
from torch import nn, optim


class BaseLightningModule(L.LightningModule):
    """Base Lightning Module for all models"""

    def __init__(
        self,
        model: torch.nn.Module | None = None,
        optimizer_name: str = 'Adam',
        optimizer_hparams: dict | None = None,
        scheduler_hparams: dict | None = None,
        loss_hparams: dict | None = None,
    ):
        """Args:
        model: PyTorch model to train
        optimizer_name: Name of the optimizer to use
        optimizer_hparams: Hyperparameters for the optimizer
        scheduler_hparams: Hyperparameters for the learning rate scheduler
        loss_hparams: Hyperparameters for the loss function
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model

        # Set default optimizer hyperparameters if none provided
        self.optimizer_hparams = optimizer_hparams or {'lr': 1e-3, 'weight_decay': 1e-5}
        self.scheduler_hparams = scheduler_hparams or {
            'milestones': [250, 450],
            'gamma': 0.1,
        }
        self.loss_hparams = loss_hparams or {}

        torch.set_float32_matmul_precision('high')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        # Configure optimizer
        if self.hparams.optimizer_name == 'Adam':
            optimizer = optim.AdamW(self.parameters(), **self.optimizer_hparams)
        elif self.hparams.optimizer_name == 'SGD':
            optimizer = optim.SGD(self.parameters(), **self.optimizer_hparams)
        else:
            raise ValueError(f'Unknown optimizer: {self.hparams.optimizer_name}')

        # Configure scheduler
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **self.scheduler_hparams)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, **self.scheduler_hparams
        )

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler}}

    # TODO: what is this, where is it used?
    def _get_progress_bar_dict(self) -> dict[str, Any]:
        """Modify progress bar display"""
        items = super()._get_progress_bar_dict()
        items.pop('v_num', None)
        return items


class CNNDecoder(BaseLightningModule):
    """Lightning Module for training CNN models"""

    def __init__(
        self,
        model_type: str = 'CNN',
        model_hparams: dict | None = None,
        optimizer_name: str = 'Adam',
        optimizer_hparams: dict | None = None,
        scheduler_hparams: dict | None = None,
        loss_hparams: dict | None = None,
    ):
        """Args:
        model_type: Name of the model (should be "CNN")
        model_hparams: Hyperparameters for the CNN model
        optimizer_name: Name of the optimizer to use
        optimizer_hparams: Hyperparameters for the optimizer
        scheduler_hparams: Hyperparameters for the learning rate scheduler
        loss_hparams: Hyperparameters for the loss function
        """
        # Create CNN model
        if model_type != 'CNN':
            raise ValueError("model_type must be 'CNN' for CNNDecoder")

        model_hparams = model_hparams or {}
        model = CNN(**model_hparams)

        super().__init__(
            model=model,
            optimizer_name=optimizer_name,
            optimizer_hparams=optimizer_hparams,
            scheduler_hparams=scheduler_hparams,
            loss_hparams=loss_hparams,
        )

    def loss_function(
        self,
        y_hat: torch.Tensor,
        targets: torch.Tensor,
        x: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Custom loss function for GPT training.

        Args:
            y_hat: Model predictions
            targets: Ground truth targets
            x: Additional input tensor for encoding loss
            *args, **kwargs: Additional arguments
        """
        # Base loss (MSE)
        loss = nn.MSELoss()(y_hat, targets)

        return loss

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step for GPT model.

        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Index of current batch
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y, x)

        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Validation step for GPT model.

        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Index of current batch
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y, x)

        self.log('val_loss', loss, prog_bar=True)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Test step for GPT model.

        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Index of current batch
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y, x)

        self.log('test_loss', loss)

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Prediction step for GPT model. Return all relevant quantities for plotting.

        Args:
            batch: Input tensor
            batch_idx: Index of current batch
            dataloader_idx: Index of dataloader
        """
        x, y = batch
        predictions = self.model(x)
        return predictions
