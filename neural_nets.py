import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        self.convGroup = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # 28x28x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14x16
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # 14x14x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 7x7x32
            nn.Dropout(p=0.25)
        )

        self.fcGroup = nn.Sequential(
            nn.Linear(1568, 128),
            nn.ReLU(),
            nn.Dropout(p=0.50),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.convGroup(x)
        x = torch.flatten(x, 1)
        x = self.fcGroup(x)
        out = nn.functional.log_softmax(x, dim=1)
        return out
