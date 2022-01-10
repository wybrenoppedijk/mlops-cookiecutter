from typing import Optional

import torch.nn.functional as F
from pytorch_lightning import LightningModule
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, STEP_OUTPUT
from torch import nn
import functools
import operator
import wandb
from src.visualization.visualize import Visuals


class MyAwesomeModel(LightningModule):

    def __init__(
        self,
            config
    ):
        super().__init__()
        self.config = config
        hparams = config.model
        torch.manual_seed(hparams["seed"])
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=hparams["img_channels"], out_channels=hparams["max_channel_size"] // 2,
                kernel_size=hparams["kernel"], stride=hparams["stride"],
                padding=hparams["padding"]),
            nn.BatchNorm2d(hparams["max_channel_size"] // 2),
            nn.ReLU(),
            nn.Dropout(hparams["dropout_rate"]),
            nn.Conv2d(
                in_channels=hparams["max_channel_size"] // 2,
                out_channels=hparams["max_channel_size"],
                kernel_size=hparams["kernel"], stride=hparams["stride"], padding=hparams["padding"]),
            nn.BatchNorm2d(hparams["max_channel_size"]),
            nn.ReLU(),
            nn.Dropout(hparams["dropout_rate"])
        )
        num_features = functools.reduce(operator.mul, list(self.conv(torch.rand(1, * (hparams["img_channels"],
                                                                                      hparams['img_width'],
                                                                                      hparams["img_height"]))).shape))
        self.fc = nn.Sequential(
            nn.Linear(num_features, hparams["hidden_size"]),
            nn.ReLU(),
            nn.Dropout(hparams["dropout_rate"]),
            nn.Linear(hparams["hidden_size"], hparams["hidden_size"]),
            nn.ReLU(),
            nn.Dropout(hparams["dropout_rate"]),
            nn.Linear(hparams["hidden_size"], hparams["num_classes"])
        )

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Expected 4D tensor as input, got {}D".format(x.ndim))
        if x.shape[1] != self.config.model["img_channels"]:
            raise ValueError(
                "Expected input with {} channels, got {}".format(self.config.model["img_channels"], x.shape[1]))
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        x, y = args[0]
        y_hat = self.forward(x)
        loss = F.nll_loss(y_hat, y)

        acc = (y == y_hat.argmax(dim=-1)).float().mean()

        self.log('train_loss', loss)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, *args, **kwargs) -> STEP_OUTPUT:
        x, y = args[0]
        y_hat = self.forward(x)
        loss = F.nll_loss(y_hat, y)
        acc = (y == y_hat.argmax(dim=-1)).float().mean()
        if self.current_epoch % 2 == 0:
            viz = Visuals(self, args[0])
            plt2=viz.intermediate_representation("conv.6", self.current_epoch )
            self.logger.experiment.log({"Intermediate Representation": wandb.Image(plt2)})
            plt1 = viz.intermediate_distribution("conv.6", self.current_epoch)
            self.logger.experiment.log({"Intermediate Distribution": wandb.Image(plt1)})
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return acc


    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.config.train["lr"])

