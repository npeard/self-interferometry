#!/usr/bin/env python3

import os, sys, glob
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from decoder import VelocityDecoder
from torch import FloatTensor
import numpy as np
import matplotlib.pyplot as plt
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from itertools import product
import datetime
import time

# Define a custom Dataset class
class VelocityDataset(Dataset):
    def __init__(self, h5_file):
        self.h5_file = h5_file
        with h5py.File(self.h5_file, 'r') as f:
            self.length = len(f['Time (s)']) # num shots
        print(self.h5_file)
        self.opened_flag = False

    def open_hdf5(self, rolling=True, group_size=256, step=128):
        """Set up inputs and targets. For each shot, buffer is split into rolling data.
        Inputs include grouped photodiode trace of 'group_size', spaced interval 'step' apart.
        Targets include average velocity of each group. 
        Input shape is [num_shots, num_groups, group_size] and target shape is [num_shots, num_groups, 1],
        where num_groups = (buffer_len - group_size)/step + 1, given that buffer_len - group_size is a multiple of step. 
        If the given 'group_size' and 'step' do not satisfy the above requirement, 
        the data will not be cleanly grouped.

        Args:
            group_size (int, optional): Size of each group. buffer_len - group_size = 0 (mod step). Defaults to 256.
            step (int, optional): Size of step between group starts. buffer_len - grou_size = 0 (mod step). Defaults to 1.
        """
        # solves issue where hdf5 file opened in __init__ prevents multiple
        # workers: https://github.com/pytorch/pytorch/issues/11929
        self.file = h5py.File(self.h5_file, 'r')
        # print(torch.cuda.get_device_name(0))
        pds = torch.Tensor(np.array(self.file['PD (V)'])) # [num_shots, buffer_size]
        vels = torch.Tensor(np.array(self.file['Speaker (Microns/s)'])) # [num_shots, buffer_size]

        if rolling:
            # ROLLING INPUT INDICES
            num_groups = (pds.shape[1] - group_size) // step + 1
            start_idxs = torch.arange(num_groups) * step  # starting indices for each group
            idxs = torch.arange(group_size)[:, None] + start_idxs
            idxs = torch.transpose(idxs, dim0=0, dim1=1)
            self.inputs = pds[:, idxs]
            self.targets = torch.unsqueeze(torch.mean(vels[:, idxs], dim=2), dim=2)
        else:
            # STEP INPUT
            grouped_pds = torch.stack(torch.split(pds, group_size, dim=1))
            self.inputs = torch.transpose(grouped_pds, dim0=0, dim1=1)
            grouped_vels = torch.stack(torch.split(vels, group_size, dim=1))
            grouped_vels = torch.transpose(grouped_vels, dim0=0, dim1=1)
            self.targets = torch.unsqueeze(torch.mean(grouped_vels, dim=2), dim=2)

        # print(self.inputs.size())  # [2k, 64, 256]
        # print(self.targets.size())  # [2k, 64, 1]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # print("getitem entered")
        if not self.opened_flag: #not hasattr(self, 'h5_file'):
            self.open_hdf5()
            self.opened_flag = True
            # print("open_hdf5 finished")
        return FloatTensor(self.inputs[idx]), FloatTensor(self.targets[idx])

    # def __getitems__(self, indices: list):
    #     if not self.opened_flag:
    #         self.open_hdf5()
    #         self.opened_flag = True
    #         print("open_hdf5 in getitems")
    #     return FloatTensor(self.inputs[indices]), FloatTensor(self.targets[indices])

class TrainingRunner:
    def __init__(self, training_h5, validation_h5, testing_h5,
                 velocity_only=True):
        self.training_h5 = training_h5
        self.validation_h5 = validation_h5
        self.testing_h5 = testing_h5
        self.velocity_only = velocity_only

        # get dataloaders
        self.set_dataloaders()
        print("dataloaders set:", datetime.datetime.now())
        # dimensions
        input_ref = next(iter(self.train_loader))
        # print("loaded next(iter", datetime.datetime.now())
        self.input_size = input_ref[0].shape[1]  # num_groups
        self.output_size = input_ref[1].shape[1]  # num_groups
        print(f"input ref {len(input_ref)} , {input_ref[0].size()}")
        print(f"output ref {len(input_ref)} , {input_ref[1].size()}")
        # print(f"train.py input_size {self.input_size}")
        # print(f"train.py output_size {self.output_size}")

        # directories
        self.checkpoint_dir = "./checkpoints"
        print('TrainingRunner initialized', datetime.datetime.now())
        
    def get_custom_dataloader(self, h5_file, batch_size=128, shuffle=True,
                              velocity_only=True):
        # if velocity_only:
        dataset = VelocityDataset(h5_file)
        print("dataset initialized")
        # We can use DataLoader to get batches of data
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                num_workers=16, persistent_workers=True,
                                pin_memory=True)
        print("dataloader initialized")
        return dataloader
    
    def set_dataloaders(self, batch_size=128):
        self.batch_size = batch_size
        self.train_loader = self.get_custom_dataloader(self.training_h5, velocity_only=self.velocity_only, batch_size=self.batch_size)
        self.valid_loader = self.get_custom_dataloader(self.validation_h5, velocity_only=self.velocity_only, batch_size=self.batch_size, shuffle=False)
        self.test_loader = self.get_custom_dataloader(self.testing_h5, velocity_only=self.velocity_only, batch_size=self.batch_size, shuffle=False)


    def train_model(self, model_name, save_name=None, **kwargs):
        """Train model.

        Args:
            model_name: Name of the model you want to run. Is used to look up the class in "model_dict"
            save_name (optional): If specified, this name will be used for creating the checkpoint and logging directory.
        """
        if save_name is None:
            save_name = model_name

        # logger
        logger = WandbLogger(project='SMI',
                             group=model_name, log_model=True,
                             save_dir=os.path.join(self.checkpoint_dir, save_name))

        # callbacks
        # early stopping
        early_stop_callback = EarlyStopping(monitor="val_loss",
                                            min_delta=0.00,
                                            patience=5,
                                            verbose=True,
                                            mode="min")
        checkpoint_callback = ModelCheckpoint(save_weights_only=True,
                                              mode="min", monitor="val_loss")
        # Save the best checkpoint based on the maximum val_acc recorded.
        # Saves only weights and not optimizer

        # Create a PyTorch Lightning trainer with the generation callback
        trainer = L.Trainer(
            default_root_dir=os.path.join(self.checkpoint_dir, save_name),
            accelerator="gpu",
            devices=[0],
            max_epochs=2000,
            callbacks=[early_stop_callback, checkpoint_callback],
            check_val_every_n_epoch=10,
            logger=logger
        )

        # L.seed_everything(42)  # To be reproducible
        model = VelocityDecoder(model_name=model_name, **kwargs)
        trainer.fit(model, self.train_loader, self.valid_loader)

        # Load best checkpoint after training
        model = VelocityDecoder.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path)

        # Test best model on validation and test set
        val_result = trainer.test(model, dataloaders=self.valid_loader,
                                  verbose=False)
        test_result = trainer.test(model, dataloaders=self.test_loader,
                                   verbose=False)
        result = {"test": test_result[0]["test_acc"],
                  "val": val_result[0]["test_acc"]}

        logger.experiment.finish()

        return model, result
    
    def scan_hyperparams(self):
            for lr in [1e-4]:#[1e-3]: #, 1e-2, 3e-2]:

                model_config = {"input_size": self.input_size,
                                "output_size": self.output_size}
                optimizer_config = {"lr": lr}
                                    #"momentum": 0.9,}
                misc_config = {"batch_size": self.batch_size}

                self.train_model(model_name="CNN",
                                 model_hparams=model_config,
                                 optimizer_name="Adam",
                                 optimizer_hparams=optimizer_config,
                                 misc_hparams=misc_config)

    def load_model(self, model_tag, model_name='CNN'):
        # Check whether pretrained model exists. If yes, load it and skip training
        pretrained_filename = os.path.join(self.checkpoint_dir, model_name, "SMI", model_tag,
                                            "checkpoints", "*" + ".ckpt")
        print(pretrained_filename)
        if os.path.isfile(glob.glob(pretrained_filename)[0]):
            pretrained_filename = glob.glob(pretrained_filename)[0]
            print(
                f"Found pretrained model at {pretrained_filename}, loading...")
            # Automatically loads the model with the saved hyperparameters
            model = VelocityDecoder.load_from_checkpoint(pretrained_filename)

            # Create a PyTorch Lightning trainer with the generation callback
            trainer = L.Trainer(
                accelerator="gpu",
                devices=[0]
            )

            # Test best model on validation and test set
            val_result = trainer.test(model, dataloaders=self.valid_loader,
                                        verbose=False)
            test_result = trainer.test(model, dataloaders=self.test_loader,
                                        verbose=False)
            result = {"test": test_result[0]["test_acc"],
                        "val": val_result[0]["test_acc"]}

            return model, result