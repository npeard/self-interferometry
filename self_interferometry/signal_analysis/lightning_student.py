#!/usr/bin/env python


import torch

from self_interferometry.signal_analysis.lightning_standard import Standard


class Student(Standard):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_hparams: dict | None = None,
        scheduler_hparams: dict | None = None,
        loss_hparams: dict | None = None,
    ):
        super().__init__(
            model=model,
            optimizer_hparams=optimizer_hparams,
            scheduler_hparams=scheduler_hparams,
            loss_hparams=loss_hparams,
        )
        self.model = model

    # TODO: write overrides of the training steps, loss function is the same as
    # Standard (unless we decide later that a physics constraint is helpful)
