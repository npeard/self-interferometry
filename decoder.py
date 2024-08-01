#!/usr/bin/env python3

from torch import optim, nn
import torch
import lightning as L
from models import CNN

class VelocityDecoder(L.LightningModule):
    def __init__(self, model_name, model_hparams, optimizer_name,
                 optimizer_hparams, misc_hparams):
        """Decoder for the target's velocity

        Args:
            model_name: Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams: Hyperparameters for the model, as dictionary.
            optimizer_name: Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams: Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = self.create_model(model_name, model_hparams)
        # Create loss module
        self.loss_function = nn.MSELoss()

        torch.set_float32_matmul_precision('medium')
        
    def create_model(self, model_name, model_hparams):
        if model_name in model_dict:
            return model_dict[model_name](**model_hparams)
        else:
            assert False, f'Unknown model name "{model_name}". Available models are: {str(model_dict.keys())}'

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(self.parameters(),
                                    **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(),
                                  **self.hparams.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=[100, 150],
                                                   gamma=0.1)
        return [optimizer], [scheduler]
    
    def get_loss_function(self, loss_hparams):
        if loss_hparams["loss_name"] == "mse":
            self.loss_function = nn.MSELoss()
        else:
            assert False, f'Unknown loss: "{loss_hparams["loss_name"]}"'
            
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(-1, x.size(1)**2)
        preds = self.model(x)
        loss = self.loss_function(preds, y)
        acc = (preds == y).float().mean()
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # validation_step defines the validation loop.
        x, y = batch
        x = x.view(-1, x.size(1)**2)
        preds = self.model(x)
        loss = self.loss_function(preds, y)
        acc = (preds == y).float().mean()
        self.log("val_acc", acc, on_step=False, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(-1, x.size(1)**2)
        preds = self.model(x)
        loss = self.loss_function(preds, y)
        acc = (preds == y).float().mean()
        self.log("test_acc", acc, on_step=False, on_epoch=True)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        x = x.view(-1, x.size(1)**2)
        y_hat = self.model(x)
        return y_hat, y