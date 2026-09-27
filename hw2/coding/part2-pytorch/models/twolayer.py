import torch
import torch.nn as nn


class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        """
        Initialize the TwoLayerNet, use sigmoid activation between layers
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        """
        super(TwoLayerNet, self).__init__()
        ### TODO: BEGIN SOLUTION ###
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        ### END SOLUTION ###

    def forward(self, x):
        out = None
        ### TODO: BEGIN SOLUTION ###
        x = x.view(x.size(0), -1)
        out = self.fc2(self.sigmoid(self.fc1(x)))
        ### END SOLUTION ###
        return out
