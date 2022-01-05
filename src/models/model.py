import torch.nn.functional as F
from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.drop = nn.Dropout(0.2)

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2
        )
        self.batch1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2
        )
        self.batch2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        self.fc1 = nn.Linear(32 * 7 * 7, 150)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(150, 250)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(250, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu2(x)
        x = self.drop(x)

        x = self.fc1(x.view(-1, 32 * 7 * 7))
        x = self.relu3(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x
