import torch
import torch.nn as nn


class VanillaCNN(nn.Module):
    def __init__(self):
        """
        Initialize the Vanilla CNN
        Conv: 7x7 kernel, stride 1 and padding 0
        Max Pooling: 2x2 kernel, stride 2
        """
        super(VanillaCNN, self).__init__()
        ### TODO: BEGIN SOLUTION ###
        self.conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * 13 * 13, 10)
        ### END SOLUTION ###

    def forward(self, x):
        outs = None
        ### TODO: BEGIN SOLUTION ###
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        outs = self.fc(x)
        ### END SOLUTION ###

        return outs
