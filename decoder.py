#!/usr/bin/env python3

from torch import optim, nn
import torch
import lightning as L
from models import CNN

model_dict = {"CNN": CNN}

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
        
        self.step = misc_hparams["step"]

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
                                                   milestones=[100, 150, 200],
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
        preds = self.model(x)
        loss = self.loss_function(preds, y)
        acc = (preds == y).float().mean()
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # validation_step defines the validation loop.
        x, y = batch
        # print(type(x))
        # print(f"x size {x.size()}")
        # print(f"y size {y.size()}")
        preds = self.model(x)
        loss = self.loss_function(preds, y)
        acc = (preds == y).float().mean()
        self.log("val_acc", acc, on_step=False, on_epoch=True)
        self.log("val_loss", loss, prog_bar=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        x_tot, y_tot = batch  # [batch_size, 1, buffer_size], [batch_size, 1, num_groups]
        num_groups = y.shape[2]
        group_size = x.shape[2] - (num_groups - 1) * self.step
        avg_loss = 0
        avg_acc = 0 
        for i in range(num_groups):
            start_idx = i*self.step
            x = x_tot[:, :, start_idx:start_idx+group_size]
            y = y_tot[:, :, i]
            preds = self.model(x)
            avg_loss += self.loss_function(preds, y)
            avg_acc += (preds == y).float().mean()
        avg_loss /= num_groups
        avg_acc /= num_groups
        self.log("test_acc", avg_acc, on_step=False, on_epoch=True)
        self.log("test_loss", avg_loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, test_mode=False, dataloader_idx=0):
        x_tot, y_tot = batch 
        if test_mode:
             # x_tot, y_tot: [batch_size, 1, buffer_size], [batch_size, 1, num_groups]
            num_groups = y_tot.shape[2]
            group_size = x_tot.shape[2] - (num_groups - 1) * self.step
            y_hat = []
            for i in range(num_groups):
                start_idx = i*self.step
                x = x_tot[:, :, start_idx:start_idx+group_size]  # [batch_size, 1, group_size]
                y = y_tot[:, :, i]  # [batch_size, 1, 1]
                y_hat.append(model(x).flatten())
            y_hat = torch.squeeze(torch.transpose(torch.stack(y_hat), dim0=0, dim1=1), dim=1)  # [batch_size, 1, num_groups]  
        else:
            y_hat = self.model(x_tot)
        return y_hat, y_tot