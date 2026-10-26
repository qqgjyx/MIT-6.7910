"""SimpleCNN of Jastrzebski et al. (2020, App. D), after the Keras CIFAR-10 example: two 32-filter and
two 64-filter 3x3 convolutions with max pooling after each pair, then a 128-unit dense layer.
`activation` is ReLU for the replication and GELU for our runs (input derivatives need it smooth)."""

import torch.nn as nn
from torch.nn import functional as F


class SimpleCNN(nn.Module):
    def __init__(self, width=32, hidden=128, num_classes=10, activation='relu'):
        super().__init__()
        self.act = {'relu': F.relu, 'gelu': F.gelu}[activation]
        self.conv1 = nn.Conv2d(3, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width, 3)
        self.conv3 = nn.Conv2d(width, 2 * width, 3, padding=1)
        self.conv4 = nn.Conv2d(2 * width, 2 * width, 3)
        self.fc1 = nn.Linear(2 * width * 6 * 6, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = F.max_pool2d(self.act(self.conv2(self.act(self.conv1(x)))), 2)
        x = F.max_pool2d(self.act(self.conv4(self.act(self.conv3(x)))), 2)
        return self.fc2(self.act(self.fc1(x.flatten(1))))

    def loss(self, x, y):
        return F.cross_entropy(self(x).float(), y)
