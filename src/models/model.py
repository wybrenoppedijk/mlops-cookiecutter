import torch.nn.functional as F
import torch
from torch import nn
import functools
import operator

class MyAwesomeModel(nn.Module):

    def __init__(
        self,
            hparams
    ):
        super().__init__()
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
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x
